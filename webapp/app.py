"""Flask 后端：提供数据集/方法元数据、单机试验、全套评测三个 API。

启动：python -m webapp.app  然后访问 http://127.0.0.1:5050
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from detagg.bench import full_sweep
from detagg.datasets import DATASET_BUILDERS, build_dataset
from detagg.executor import (order_for_method, random_order, run_sum,
                             run_variance)
from detagg.sums import SUM_METHODS, SUM_METHOD_KEYS, make_sum_state
from detagg.stats import VAR_METHODS, VAR_METHOD_KEYS, make_var_state
from detagg.truth import compute_truth

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta")
def api_meta() -> Any:
    datasets = []
    for name in DATASET_BUILDERS:
        ds = build_dataset(name)
        finite = ds.finite_values
        datasets.append({
            "name": name,
            "description": ds.description,
            "note": ds.note,
            "n": ds.n,
            "scale": ds.scale,
            "null_count": int(ds.null_mask.sum()),
            "finite_min": float(finite.min()) if finite.size else None,
            "finite_max": float(finite.max()) if finite.size else None,
            "finite_abs_max": float(np.abs(finite).max()) if finite.size else None,
        })
    return jsonify({
        "datasets": datasets,
        "sum_methods": [
            {
                "key": k,
                "label": SUM_METHODS[k].label,
                "deterministic": SUM_METHODS[k].deterministic,
                "exact": SUM_METHODS[k].exact,
                "needs_scale": SUM_METHODS[k].needs_scale,
                "cost_note": SUM_METHODS[k].cost_note,
                "guarantee_note": SUM_METHODS[k].guarantee_note,
                "range_note": SUM_METHODS[k].range_note,
            } for k in SUM_METHOD_KEYS
        ],
        "var_methods": [
            {
                "key": k,
                "label": VAR_METHODS[k].label,
                "deterministic": VAR_METHODS[k].deterministic,
                "needs_scale": VAR_METHODS[k].needs_scale,
                "cost_note": VAR_METHODS[k].cost_note,
                "guarantee_note": VAR_METHODS[k].guarantee_note,
                "range_note": VAR_METHODS[k].range_note,
            } for k in VAR_METHOD_KEYS
        ],
    })


@app.route("/api/truth")
def api_truth() -> Any:
    name = request.args.get("dataset", "normal")
    ds = build_dataset(name)
    t = compute_truth(ds)
    return jsonify({
        "n": t.n,
        "sum": str(t.sum),
        "sum_f64": t.sum_f64,
        "mean": str(t.mean),
        "mean_f64": t.mean_f64,
        "variance": str(t.variance),
        "variance_f64": t.variance_f64,
        "stddev": str(t.stddev),
        "stddev_f64": t.stddev_f64,
    })


def _distribution(results: List[float], truth_f64: float) -> Dict[str, Any]:
    bit_patterns: Dict[str, int] = {}
    finite = [r for r in results if math.isfinite(r)]
    nan_runs = len(results) - len(finite)
    for r in results:
        key = "nan" if math.isnan(r) else repr(r)
        bit_patterns[key] = bit_patterns.get(key, 0) + 1
    errs = [abs(r - truth_f64) for r in finite]
    ulp = math.ulp(truth_f64) if truth_f64 and math.isfinite(truth_f64) \
        else math.ulp(0.0)
    return {
        "distinct": len(bit_patterns),
        "bit_patterns": sorted(
            ({"value": k, "count": c} for k, c in bit_patterns.items()),
            key=lambda d: -d["count"])[:20],
        "nan_runs": nan_runs,
        "max_abs_err": max(errs) if errs else None,
        "median_abs_err": sorted(errs)[len(errs) // 2] if errs else None,
        "max_ulps": (max(errs) / ulp) if errs and ulp else None,
        "results": results,
    }


@app.route("/api/run", methods=["POST"])
def api_run() -> Any:
    body = request.get_json(force=True)
    ds = build_dataset(body["dataset"], n=body.get("n"))
    trials = int(body.get("trials", 300))
    base_seed = int(body.get("base_seed", 700_000))
    threaded = bool(body.get("threaded", False))
    kind = body.get("kind", "sum")
    method = body["method"]
    # 显式 scale 用于演示守卫：不传时跟随数据集声明的 scale
    scale = int(body["scale"]) if body.get("scale") is not None else ds.scale
    t0 = time.perf_counter()
    errors: List[Dict[str, Any]] = []
    results: List[Any] = []
    orders: List[Dict[str, Any]] = []
    for trial in range(trials):
        order = order_for_method(ds.finite_count, seed=base_seed + trial, method=method,
                                 max_partitions=int(body.get("max_partitions", 64)))
        orders.append({
            "partitions": len(order.partitions),
            "merge_head": order.merge_perm[:8],
        })
        try:
            if kind == "sum":
                results.append(run_sum(method, ds.values, order,
                                       scale=scale, null_mask=ds.null_mask,
                                       threaded=threaded))
            else:
                var, std = run_variance(method, ds.values, order,
                                        scale=scale, null_mask=ds.null_mask,
                                        threaded=threaded)
                results.append(var)
        except (OverflowError, Exception) as e:  # 守卫触发：记录但不中断
            errors.append({"trial": trial, "error": f"{type(e).__name__}: {e}"})
            results.append(None)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    truth = compute_truth(ds)
    truth_f64 = truth.sum_f64 if kind == "sum" else truth.variance_f64
    valid = [r for r in results if r is not None]
    dist = _distribution(valid, truth_f64) if valid else {
        "distinct": 0, "bit_patterns": [], "nan_runs": 0,
        "max_abs_err": None, "median_abs_err": None,
        "max_ulps": None, "results": []}
    mean_info = None
    if kind == "sum" and valid:
        from detagg.stats import run_mean
        means = {run_mean(r, truth.n) for r in valid if math.isfinite(r)}
        mean_info = {
            "distinct": len(means),
            "truth_f64": truth.mean_f64,
            "sample": next(iter(means)),
        }
    return jsonify({
        "dataset": ds.name,
        "method": method,
        "kind": kind,
        "trials": trials,
        "elapsed_ms": elapsed_ms,
        "threaded": threaded,
        "truth_f64": truth_f64,
        "truth_decimal": str(truth.sum if kind == "sum" else truth.variance),
        "distribution": dist,
        "mean_info": mean_info,
        "errors": errors[:10],
        "error_count": len(errors),
        "orders_sample": orders[:5],
    })


@app.route("/api/sweep", methods=["POST"])
def api_sweep() -> Any:
    body = request.get_json(force=True)
    trials = int(body.get("trials", 200))
    threaded = bool(body.get("threaded", False))
    names = body.get("datasets") or list(DATASET_BUILDERS)
    datasets = [build_dataset(n) for n in names]
    report = full_sweep(datasets, trials=trials, threaded=threaded,
                        methods=body.get("sum_methods"),
                        var_methods=body.get("var_methods"))
    return jsonify(report)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
