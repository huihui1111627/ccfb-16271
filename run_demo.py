"""命令行演示：复现非确定性 → 各方案治理 → 成本对照。

用法：
  python3 run_demo.py                 # 快速演示（小数据、120 次随机顺序）
  python3 run_demo.py --full          # 完整对照（默认数据规模、300 次）
  python3 run_demo.py --dataset cancellation --method kahan --trials 200
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from detagg.bench import benchmark_sum, benchmark_variance
from detagg.config import deterministic_aggregation, get_config
from detagg.datasets import build_dataset
from detagg.executor import random_order, run_sum
from detagg.groupby import make_group_keys, run_group_sum
from detagg.sums import SUM_METHOD_KEYS
from detagg.stats import VAR_METHOD_KEYS
from detagg.truth import compute_truth
from detagg.window import deterministic_cumsum, parallel_cumsum

SMALL_N = {
    "normal": 30_000, "magnitude_spread": 20_000, "cancellation": 30_000,
    "deep_decimals": 15_000, "epsilon": 30_000, "special_values": 10_000,
}


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_reproduction(trials: int, full: bool) -> None:
    hr("1. 问题复现：同一数据、不同随机切分/合并顺序，结果差多少")
    for name in ["normal", "magnitude_spread", "cancellation", "epsilon"]:
        ds = build_dataset(name, n=None if full else SMALL_N[name])
        truth = compute_truth(ds)
        row = benchmark_sum(ds, "naive", trials=trials)
        print(f"\n[{name}] n={ds.n}  十进制真值={truth.sum_f64!r}")
        print(f"  朴素求和 {trials} 次随机顺序 → {row.distinct_results} 种不同 bit 结果，"
              f"最大绝对误差 {row.max_abs_err:.3e}（≈{row.max_ulps:.1f} ULP），"
              f"中位误差 {row.median_abs_err:.3e}")
        # Kahan 对照
        rk = benchmark_sum(ds, "kahan", trials=trials)
        print(f"  Kahan   {trials} 次 → {rk.distinct_results} 种结果，"
              f"最大绝对误差 {rk.max_abs_err:.3e}（≈{rk.max_ulps:.1f} ULP）")


def show_methods(trials: int, full: bool) -> None:
    hr("2/3. 七种求和 + 三种方差：确定性、与十进制真值的差距、速度")
    name = "magnitude_spread"
    ds = build_dataset(name, n=None if full else SMALL_N[name])
    base_ms = None
    print(f"\n数据集={name}, n={ds.n}, trials={trials}")
    print(f"{'方法':<14}{'bit模式':>8}{'逐位相同':>9}{'等于真值':>9}"
          f"{'最大误差':>14}{'最大ULP':>10}{'耗时ms':>10}{'倍数':>8}")
    for m in SUM_METHOD_KEYS:
        r = benchmark_sum(ds, m, trials=trials)
        if base_ms is None:
            base_ms = r.time_ms
        if r.error:
            print(f"{m:<14}{'守卫':>8}{'—':>9}{'—':>9}{'—':>14}{'—':>10}"
                  f"{r.time_ms:>10.1f}{'—':>8}  {r.error[:40]}")
            continue
        print(f"{m:<14}{r.distinct_results:>8}{'是' if r.bit_identical else '否':>9}"
              f"{'是' if r.matches_truth_f64 else '否':>9}"
              f"{r.max_abs_err:>14.3e}{r.max_ulps:>10.1f}"
              f"{r.time_ms:>10.1f}{r.time_ms / base_ms:>7.2f}x")

    print(f"\n方差/标准差（同一数据集）")
    base_v = None
    print(f"{'方法':<14}{'bit模式':>8}{'逐位相同':>9}{'等于真值':>9}"
          f"{'最大误差':>14}{'最大ULP':>10}{'耗时ms':>10}{'倍数':>8}")
    for m in VAR_METHOD_KEYS:
        r = benchmark_variance(ds, m, trials=trials)
        if base_v is None:
            base_v = r.time_ms
        if r.error:
            print(f"{m:<14} 守卫触发：{r.error[:60]}")
            continue
        print(f"{m:<14}{r.distinct_results:>8}{'是' if r.bit_identical else '否':>9}"
              f"{'是' if r.matches_truth_f64 else '否':>9}"
              f"{r.max_abs_err:>14.3e}{r.max_ulps:>10.1f}"
              f"{r.time_ms:>10.1f}{r.time_ms / base_v:>7.2f}x")


def show_extremes() -> None:
    hr("附加：极端场景的误差量级")
    # 大数加小数
    x = np.array([1e16] + [1.0] * 10_000, dtype=np.float64)
    order = random_order(x.shape[0], seed=1)
    naive = run_sum("naive", x, order)
    fsum = run_sum("fsum", x, order)
    print(f"\n[1e16 + 10000×1.0] 朴素={naive!r} fsum={fsum!r} "
          f"朴素直接吞掉全部 1.0：{'是' if naive == 1e16 else '否'}")

    # 朴素方差反例
    rng = np.random.default_rng(11)
    y = 1e8 + rng.normal(0, 0.01, size=20_000)
    from detagg.stats import NaiveSumSqState, WelfordChanState, ExactFixedVarState
    vn = NaiveSumSqState(); vn.update(y)
    vw = WelfordChanState(); vw.update(y)
    ve = ExactFixedVarState(scale=9); ve.update(y)
    print(f"[均值1e8、噪声0.01 的方差] 朴素矩法={vn.finalize()[0]:.6e}  "
          f"Welford={vw.finalize()[0]:.6e}  精确定点={ve.finalize()[0]:.6e}")


def show_window_group() -> None:
    hr("4. 窗口累计求和 与 分组聚合的确定性")
    rng = np.random.default_rng(5)
    x = np.round(rng.uniform(-100, 100, 8_000), 2)
    p_naive = {parallel_cumsum(x, seed=s, method="naive").tobytes() for s in range(20)}
    p_exact = {parallel_cumsum(x, seed=s, method="fixed_bigint",
                               scale=2).tobytes() for s in range(20)}
    det = deterministic_cumsum(x, scale=2)
    print(f"\n窗口累计求和：朴素并行 20 种随机调度 → {len(p_naive)} 种结果；"
          f"定点并行 → {len(p_exact)} 种，且与确定性实现逐位相同："
          f"{np.array_equal(parallel_cumsum(x, seed=0, method='fixed_bigint', scale=2), det)}")

    n = 20_000
    gx = np.round(rng.uniform(-1e4, 1e4, n), 2)
    keys = make_group_keys(rng, n, 20)
    g_naive = {tuple(sorted(run_group_sum(keys, gx, "naive", seed=s).items()))
               for s in range(20)}
    g_exact = {tuple(sorted(run_group_sum(keys, gx, "fixed_bigint",
                                          seed=s, scale=2).items()))
               for s in range(20)}
    print(f"分组求和：朴素 20 次随机切分 → {len(g_naive)} 种分组结果；"
          f"定点 → {len(g_exact)} 种（50 个组全部逐位相同）")


def show_switch() -> None:
    hr("6. 确定性开关：只对需要审计的查询启用，其余照旧")
    print(f"\n默认：{get_config().describe()}")
    with deterministic_aggregation(enabled=True, sum_method="fixed_bigint",
                                   var_method="exact_fixed", scale=6):
        print(f"块内：{get_config().describe()}")
    print(f"退出上下文：{get_config().describe()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="完整规模 + 300 次随机顺序")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--method", default=None)
    args = ap.parse_args()

    trials = args.trials or (300 if args.full else 120)
    t0 = time.perf_counter()
    if args.dataset and args.method:
        ds = build_dataset(args.dataset)
        r = benchmark_sum(ds, args.method, trials=trials)
        t = compute_truth(ds)
        print(f"dataset={args.dataset} method={args.method} trials={trials}")
        print(f"distinct={r.distinct_results} bit_identical={r.bit_identical} "
              f"matches_truth={r.matches_truth_f64} max_abs_err={r.max_abs_err} "
              f"max_ulps={r.max_ulps} time_ms={r.time_ms:.1f}")
        if r.error:
            print("guard:", r.error)
        print("decimal truth:", t.sum)
        return 0

    show_reproduction(trials, args.full)
    show_methods(trials, args.full)
    show_extremes()
    show_window_group()
    show_switch()
    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
