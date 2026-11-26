"""评测台：几百种随机切分/合并顺序下的确定性、误差与开销。

每个 (数据集, 方法) 组合输出：
- distinct_results：trials 次运行里出现了多少种不同的 bit 模式
- bit_identical    ：是否逐位相同（确定性方案的验收线）
- 误差分布         ：相对十进制真值的 abs/rel 误差、ULP 数、最大与中位
- 极端误差复现     ：朴素方案在量级悬殊/相消数据上的最大偏差金额
- 速度 / 内存      ：相对朴素基线的耗时倍数、状态对象的字节数
"""

from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .datasets import Dataset
from .executor import (RunOrder, order_for_method, random_order, run_sum,
                       run_variance)
from .sums import SUM_METHODS, SUM_METHOD_KEYS, PrecisionExceeded
from .stats import VAR_METHODS, VAR_METHOD_KEYS
from .truth import compute_truth


@dataclass
class ErrorPoint:
    abs_err: float
    rel_err: Optional[float]
    ulps: Optional[float]
    result: float


@dataclass
class BenchRow:
    dataset: str
    method: str
    kind: str                       # sum / variance
    trials: int
    distinct_results: int
    bit_identical: bool
    matches_truth_f64: Optional[bool]
    max_abs_err: Optional[float]
    median_abs_err: Optional[float]
    max_rel_err: Optional[float]
    max_ulps: Optional[float]
    nan_runs: int
    error: Optional[str] = None
    time_ms: Optional[float] = None
    speedup_vs_naive: Optional[float] = None
    peak_mem_kb: Optional[float] = None
    sample_results: List[float] = field(default_factory=list)


def _bit_key(v: float) -> int:
    import struct
    if math.isnan(v):
        return -1  # 所有 nan 视为同一特殊结果
    return struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def _err_points(results: List[float], truth_f64: float,
                truth_exact) -> List[ErrorPoint]:
    from decimal import Decimal
    pts = []
    for r in results:
        if not math.isfinite(r):
            pts.append(ErrorPoint(float("nan"), None, None, r))
            continue
        abs_err = abs(r - truth_f64)
        rel_err = None
        if truth_exact != 0:
            rel_err = float(abs(truth_exact - Decimal(repr(r))) / abs(truth_exact))
        ulp = math.ulp(truth_f64) if math.isfinite(truth_f64) and truth_f64 != 0 \
            else math.ulp(0.0)
        pts.append(ErrorPoint(abs_err, rel_err,
                              abs_err / ulp if ulp else None, r))
    return pts


def benchmark_sum(ds: Dataset, method: str, trials: int = 300,
                  base_seed: int = 700_000,
                  max_partitions: int = 64,
                  threaded: bool = False) -> BenchRow:
    truth = compute_truth(ds)
    x = ds.values
    results: List[float] = []
    err: Optional[str] = None
    t0 = time.perf_counter()
    try:
        for t in range(trials):
            order = order_for_method(ds.finite_count, seed=base_seed + t, method=method,
                                     max_partitions=max_partitions)
            results.append(run_sum(method, x, order, scale=ds.scale,
                                   null_mask=ds.null_mask, threaded=threaded))
    except (PrecisionExceeded, OverflowError) as e:
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if err:
        return BenchRow(ds.name, method, "sum", trials, 0, False, None,
                        None, None, None, None, 0, error=err,
                        time_ms=elapsed_ms)

    finite_runs = [r for r in results if math.isfinite(r)]
    nan_runs = len(results) - len(finite_runs)
    keys = {_bit_key(r) for r in results}
    pts = _err_points(finite_runs, truth.sum_f64, truth.sum)
    abs_errs = sorted(p.abs_err for p in pts if math.isfinite(p.abs_err))
    rel_errs = [p.rel_err for p in pts if p.rel_err is not None]
    ulps = [p.ulps for p in pts if p.ulps is not None]
    return BenchRow(
        dataset=ds.name, method=method, kind="sum", trials=trials,
        distinct_results=len(keys),
        bit_identical=len(keys) == 1,
        matches_truth_f64=(finite_runs[0] == truth.sum_f64) if finite_runs else None,
        max_abs_err=abs_errs[-1] if abs_errs else None,
        median_abs_err=abs_errs[len(abs_errs) // 2] if abs_errs else None,
        max_rel_err=max(rel_errs) if rel_errs else None,
        max_ulps=max(ulps) if ulps else None,
        nan_runs=nan_runs,
        time_ms=elapsed_ms,
        sample_results=results[:8],
    )


def benchmark_variance(ds: Dataset, method: str, trials: int = 300,
                       base_seed: int = 800_000,
                       max_partitions: int = 64,
                       threaded: bool = False) -> BenchRow:
    truth = compute_truth(ds)
    results: List[tuple] = []
    err: Optional[str] = None
    t0 = time.perf_counter()
    try:
        for t in range(trials):
            order = random_order(ds.finite_count, seed=base_seed + t,
                                 max_partitions=max_partitions)
            results.append(run_variance(method, ds.values, order,
                                        scale=ds.scale, null_mask=ds.null_mask,
                                        threaded=threaded))
    except (PrecisionExceeded, OverflowError) as e:
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if err:
        return BenchRow(ds.name, method, "variance", trials, 0, False, None,
                        None, None, None, None, 0, error=err,
                        time_ms=elapsed_ms)

    vars_ = [r[0] for r in results]
    finite = [v for v in vars_ if math.isfinite(v)]
    keys = {_bit_key(v) for v in vars_}
    pts = _err_points(finite, truth.variance_f64, truth.variance)
    abs_errs = sorted(p.abs_err for p in pts if math.isfinite(p.abs_err))
    rel_errs = [p.rel_err for p in pts if p.rel_err is not None]
    ulps = [p.ulps for p in pts if p.ulps is not None]
    return BenchRow(
        dataset=ds.name, method=method, kind="variance", trials=trials,
        distinct_results=len(keys),
        bit_identical=len(keys) == 1,
        matches_truth_f64=(finite[0] == truth.variance_f64) if finite else None,
        max_abs_err=abs_errs[-1] if abs_errs else None,
        median_abs_err=abs_errs[len(abs_errs) // 2] if abs_errs else None,
        max_rel_err=max(rel_errs) if rel_errs else None,
        max_ulps=max(ulps) if ulps else None,
        nan_runs=len(vars_) - len(finite),
        time_ms=elapsed_ms,
        sample_results=vars_[:8],
    )


def measure_memory(ds: Dataset, method: str, kind: str = "sum",
                   scale: Optional[int] = None) -> float:
    """单次跑通的峰值内存（KB）。"""
    scale = ds.scale if scale is None else scale
    tracemalloc.start()
    order = random_order(ds.finite_count, seed=42)
    try:
        if kind == "sum":
            run_sum(method, ds.values, order, scale=scale,
                    null_mask=ds.null_mask)
        else:
            run_variance(method, ds.values, order, scale=scale,
                         null_mask=ds.null_mask)
    except (PrecisionExceeded, OverflowError):
        pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def _bench_sum_scaled(ds: Dataset, method: str, trials: int,
                      threaded: bool, scale: int) -> "BenchRow":
    """用指定 scale 跑求和评测（深小数数据集需要 scale=9 而不是默认 6）。"""
    truth = compute_truth(ds)
    results: List[float] = []
    err: Optional[str] = None
    t0 = time.perf_counter()
    try:
        for t in range(trials):
            order = order_for_method(ds.finite_count, seed=700_000 + t, method=method)
            results.append(run_sum(method, ds.values, order, scale=scale,
                                   null_mask=ds.null_mask, threaded=threaded))
    except (PrecisionExceeded, OverflowError) as e:
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if err:
        return BenchRow(ds.name, method, "sum", trials, 0, False, None,
                        None, None, None, None, 0, error=err, time_ms=elapsed_ms)
    return _finish_sum_row(ds, method, trials, results, truth, elapsed_ms)


def _bench_var_scaled(ds: Dataset, method: str, trials: int,
                      threaded: bool, scale: int) -> "BenchRow":
    truth = compute_truth(ds)
    results: List[tuple] = []
    err: Optional[str] = None
    t0 = time.perf_counter()
    try:
        for t in range(trials):
            order = random_order(ds.finite_count, seed=800_000 + t)
            results.append(run_variance(method, ds.values, order, scale=scale,
                                        null_mask=ds.null_mask, threaded=threaded))
    except (PrecisionExceeded, OverflowError) as e:
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if err:
        return BenchRow(ds.name, method, "variance", trials, 0, False, None,
                        None, None, None, None, 0, error=err, time_ms=elapsed_ms)
    vars_ = [r[0] for r in results]
    finite = [v for v in vars_ if math.isfinite(v)]
    keys = {_bit_key(v) for v in vars_}
    pts = _err_points(finite, truth.variance_f64, truth.variance)
    abs_errs = sorted(p.abs_err for p in pts if math.isfinite(p.abs_err))
    rel_errs = [p.rel_err for p in pts if p.rel_err is not None]
    ulps = [p.ulps for p in pts if p.ulps is not None]
    return BenchRow(
        dataset=ds.name, method=method, kind="variance", trials=trials,
        distinct_results=len(keys), bit_identical=len(keys) == 1,
        matches_truth_f64=(finite[0] == truth.variance_f64) if finite else None,
        max_abs_err=abs_errs[-1] if abs_errs else None,
        median_abs_err=abs_errs[len(abs_errs) // 2] if abs_errs else None,
        max_rel_err=max(rel_errs) if rel_errs else None,
        max_ulps=max(ulps) if ulps else None,
        nan_runs=len(vars_) - len(finite), time_ms=elapsed_ms,
        sample_results=vars_[:8])


def _finish_sum_row(ds, method, trials, results, truth, elapsed_ms) -> "BenchRow":
    finite_runs = [r for r in results if math.isfinite(r)]
    nan_runs = len(results) - len(finite_runs)
    keys = {_bit_key(r) for r in results}
    pts = _err_points(finite_runs, truth.sum_f64, truth.sum)
    abs_errs = sorted(p.abs_err for p in pts if math.isfinite(p.abs_err))
    rel_errs = [p.rel_err for p in pts if p.rel_err is not None]
    ulps = [p.ulps for p in pts if p.ulps is not None]
    return BenchRow(
        dataset=ds.name, method=method, kind="sum", trials=trials,
        distinct_results=len(keys), bit_identical=len(keys) == 1,
        matches_truth_f64=(finite_runs[0] == truth.sum_f64) if finite_runs else None,
        max_abs_err=abs_errs[-1] if abs_errs else None,
        median_abs_err=abs_errs[len(abs_errs) // 2] if abs_errs else None,
        max_rel_err=max(rel_errs) if rel_errs else None,
        max_ulps=max(ulps) if ulps else None,
        nan_runs=nan_runs, time_ms=elapsed_ms,
        sample_results=results[:8])


def full_sweep(datasets: List[Dataset], trials: int = 300,
               threaded: bool = False,
               methods: Optional[List[str]] = None,
               var_methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """整套评测，返回可 JSON 序列化的报告。"""
    sum_keys = methods or SUM_METHOD_KEYS
    var_keys = var_methods or VAR_METHOD_KEYS
    rows: List[BenchRow] = []
    for ds in datasets:
        # 先跑朴素基线拿到基准耗时
        base = benchmark_sum(ds, "naive", trials=trials, threaded=threaded)
        base_ms = base.time_ms or 1.0
        for m in sum_keys:
            # 定点方案必须跟随数据集声明的 scale，否则深小数数据会被错误地按 6 位拒绝
            if SUM_METHODS[m].needs_scale:
                row = _bench_sum_scaled(ds, m, trials, threaded, ds.scale)
            else:
                row = benchmark_sum(ds, m, trials=trials, threaded=threaded)
            row.speedup_vs_naive = (row.time_ms / base_ms) if row.time_ms else None
            if row.error is None:
                row.peak_mem_kb = measure_memory(
                    ds, m, "sum", scale=ds.scale if SUM_METHODS[m].needs_scale else None)
            rows.append(row)
        base_var = benchmark_variance(ds, "naive_sumsq", trials=trials,
                                      threaded=threaded)
        base_var_ms = base_var.time_ms or 1.0
        for m in var_keys:
            if VAR_METHODS[m].needs_scale:
                row = _bench_var_scaled(ds, m, trials, threaded, ds.scale)
            else:
                row = benchmark_variance(ds, m, trials=trials, threaded=threaded)
            row.speedup_vs_naive = (row.time_ms / base_var_ms) if row.time_ms else None
            if row.error is None:
                row.peak_mem_kb = measure_memory(
                    ds, m, "variance",
                    scale=ds.scale if VAR_METHODS[m].needs_scale else None)
            rows.append(row)

    truths = {ds.name: _truth_brief(ds) for ds in datasets}
    return {
        "trials": trials,
        "threaded": threaded,
        "datasets": [_dataset_brief(ds) for ds in datasets],
        "truths": truths,
        "sum_methods": {k: _method_brief(v) for k, v in SUM_METHODS.items()
                        if k in sum_keys},
        "var_methods": {k: _method_brief(v) for k, v in VAR_METHODS.items()
                        if k in var_keys},
        "rows": [asdict(r) for r in rows],
    }


def _method_brief(spec) -> Dict[str, Any]:
    return {
        "label": spec.label,
        "deterministic": spec.deterministic,
        "exact": getattr(spec, "exact", False),
        "cost_note": spec.cost_note,
        "guarantee_note": spec.guarantee_note,
        "range_note": spec.range_note,
        "needs_scale": spec.needs_scale,
    }


def _dataset_brief(ds: Dataset) -> Dict[str, Any]:
    finite = ds.finite_values
    return {
        "name": ds.name,
        "description": ds.description,
        "n": ds.n,
        "scale": ds.scale,
        "note": ds.note,
        "null_count": int(ds.null_mask.sum()),
        "finite_min": float(finite.min()) if finite.size else None,
        "finite_max": float(finite.max()) if finite.size else None,
        "finite_abs_max": float(np.abs(finite).max()) if finite.size else None,
    }


def _truth_brief(ds: Dataset) -> Dict[str, Any]:
    t = compute_truth(ds)
    return {
        "n": t.n,
        "sum": str(t.sum),
        "sum_f64": t.sum_f64,
        "mean": str(t.mean),
        "mean_f64": t.mean_f64,
        "variance": str(t.variance),
        "variance_f64": t.variance_f64,
        "stddev": str(t.stddev),
        "stddev_f64": t.stddev_f64,
    }
