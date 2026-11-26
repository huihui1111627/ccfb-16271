"""核心正确性测试：python -m pytest tests/ -q"""

import math

import numpy as np
import pytest

from detagg.bench import benchmark_sum, benchmark_variance
from detagg.config import deterministic_aggregation, get_config
from detagg.datasets import build_dataset
from detagg.executor import random_order, run_sum, run_variance
from detagg.groupby import make_group_keys, run_group_sum, run_group_variance
from detagg.sums import (FSumState, Fixed64State, FixedBigintState,
                         NaiveState, PrecisionExceeded, KahanState,
                         DecimalState, _fsum_lmm, make_sum_state)
from detagg.stats import (ExactFixedVarState, NaiveSumSqState,
                          WelfordChanState)
from detagg.truth import compute_truth
from detagg.window import (deterministic_cumsum, parallel_cumsum,
                           sequential_cumsum)


# ---------------------------------------------------------------- fsum 移植

def test_fsum_lmm_matches_math_fsum():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.uniform(-1e9, 1e9, size=rng.integers(1, 500))
        partials: list = []
        for v in x.tolist():
            _fsum_lmm(partials, float(v))
        assert math.fsum(partials) == math.fsum(x.tolist())


def test_fsum_partials_merge_order_independent():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1e9, 1e9, size=10_000)
    outs = set()
    for seed in range(40):
        order = random_order(x.shape[0], seed=seed)
        root = None
        for lo, hi in order.partitions:
            st = FSumState()
            st.update(x[lo:hi])
            root = st if root is None else (root.merge(st) or root)
        perm = order.merge_perm
        # 上面是分区顺序合并；再按随机 merge_perm 合并一遍验证
        states = []
        for lo, hi in order.partitions:
            st = FSumState()
            st.update(x[lo:hi])
            states.append(st)
        r2 = None
        for idx in perm:
            r2 = states[idx] if r2 is None else (r2.merge(states[idx]) or r2)
        outs.add(r2.finalize())
    assert len(outs) == 1
    assert outs.pop() == math.fsum(x.tolist())


# --------------------------------------------------------------- 确定性总测

@pytest.mark.parametrize("method", ["fixed64", "fixed_bigint", "decimal"])
def test_exact_sum_methods_match_decimal_truth(method):
    # 定点/Decimal 在声明精度内直接还原十进制，结果必须等于十进制真值的最近 float64。
    # 用普通金额数据：fixed64 在量级悬殊数据上会正确触发溢出守卫（另见溢出测试）。
    ds = build_dataset("normal", n=20_000)
    vals = set()
    for t in range(60):
        order = random_order(ds.n, seed=10_000 + t)
        vals.add(run_sum(method, ds.values, order, scale=ds.scale,
                         null_mask=ds.null_mask))
    assert len(vals) == 1
    assert vals.pop() == compute_truth(ds).sum_f64


def test_fixed64_overflow_is_a_range_limit_not_a_bug():
    # 1e12 级大额 × 上万行必然撑爆 int64：守卫必须报错，交给 fixed_bigint 兜底
    ds = build_dataset("magnitude_spread", n=20_000)
    order = random_order(ds.n, seed=10_000)
    with pytest.raises(OverflowError):
        run_sum("fixed64", ds.values, order, scale=ds.scale,
                null_mask=ds.null_mask)
    # 同一份数据 fixed_bigint 正常且逐位确定、贴合真值
    vals = set()
    for t in range(30):
        order = random_order(ds.n, seed=10_000 + t)
        vals.add(run_sum("fixed_bigint", ds.values, order, scale=ds.scale,
                         null_mask=ds.null_mask))
    assert len(vals) == 1


def test_fsum_is_order_independent_but_keeps_float_input_error():
    # fsum 消除的是“累加顺序”误差，不消除输入本身的十进制->float 表示误差
    ds = build_dataset("cancellation", n=20_000)
    vals = set()
    for t in range(60):
        order = random_order(ds.n, seed=20_000 + t)
        vals.add(run_sum("fsum", ds.values, order, scale=ds.scale,
                         null_mask=ds.null_mask))
    assert len(vals) == 1                       # 任意顺序逐位相同
    got = vals.pop()
    assert got == math.fsum(ds.finite_values)   # 等于 float 输入的精确总和
    # 但与十进制真值存在输入侧的固有偏差（相消后被放大）
    assert got != compute_truth(ds).sum_f64


def test_fixed_methods_match_truth_even_on_cancellation():
    ds = build_dataset("cancellation", n=20_000)
    vals = set()
    for t in range(60):
        order = random_order(ds.n, seed=21_000 + t)
        vals.add(run_sum("fixed_bigint", ds.values, order, scale=ds.scale,
                         null_mask=ds.null_mask))
    assert len(vals) == 1
    assert vals.pop() == compute_truth(ds).sum_f64


def test_fsum_block_deterministic_but_retains_input_error():
    # 工程推荐档：全局固定块向量化 + 精确合并，任何随机切分/合并逐位相同
    for name in ["normal", "magnitude_spread", "cancellation", "epsilon",
                 "special_values"]:
        ds = build_dataset(name, n=20_000)
        vals = set()
        for t in range(60):
            order = random_order(ds.finite_count, seed=22_000 + t,
                                 align=4096)
            v = run_sum("fsum_block", ds.values, order,
                        null_mask=ds.null_mask)
            vals.add("nan" if math.isnan(v) else v)
        assert len(vals) == 1, name
    # 线程池模式下同样确定
    ds = build_dataset("normal", n=20_000)
    outs = set()
    for t in range(20):
        order = random_order(ds.finite_count, seed=23_000 + t, align=4096)
        outs.add(run_sum("fsum_block", ds.values, order,
                         null_mask=ds.null_mask, threaded=True))
    assert len(outs) == 1
    # 它不消除 float 输入侧的表示误差：相消数据上与十进制真值仍有固有偏差
    ds = build_dataset("cancellation", n=20_000)
    order = random_order(ds.finite_count, seed=1, align=4096)
    got = run_sum("fsum_block", ds.values, order, null_mask=ds.null_mask)
    assert got != compute_truth(ds).sum_f64   # 块级舍入+输入舍入，非十进制精确


def test_fsum_block_rejects_non_contiguous_group_usage():
    # 分组聚合成员在全局数组中不连续，全局块对齐不成立，必须明确拒绝
    st = make_sum_state("fsum_block")
    with pytest.raises(PrecisionExceeded):
        st.update(np.array([1.0, 2.0]))  # 无 global_lo


def test_mean_is_deterministic_when_sum_is():
    # 平均值 = 确定的和 / 计数：一次除法，顺序无关
    from detagg.stats import run_mean
    ds = build_dataset("normal", n=20_000)
    n = compute_truth(ds).n
    means = set()
    for t in range(40):
        order = random_order(ds.finite_count, seed=24_000 + t)
        s = run_sum("fixed_bigint", ds.values, order, scale=ds.scale,
                    null_mask=ds.null_mask)
        means.add(run_mean(s, n))
    assert len(means) == 1
    got = means.pop()
    # 定点路径 (整数和/n/factor) 与 Decimal 直除的最后一次舍入可能差 1 ULP
    assert abs(got - compute_truth(ds).mean_f64) <= math.ulp(
        compute_truth(ds).mean_f64)


def test_naive_is_nondeterministic_on_spread():
    ds = build_dataset("magnitude_spread", n=50_000)
    vals = set()
    for t in range(80):
        order = random_order(ds.n, seed=30_000 + t)
        vals.add(run_sum("naive", ds.values, order, null_mask=ds.null_mask))
    assert len(vals) > 1  # 复现：同一数据不同顺序，朴素求和结果跳动


def test_kahan_stays_close_to_truth_under_all_orders():
    # 逐元素 Neumaier 补偿：这几组数据上随机顺序已经逐位相同；
    # 但 deterministic 标志仍为 False——它不是数学证明，极端构造下可能差 1 ULP。
    # 与十进制真值之间只允许存在 float 输入侧的固有表示误差（相消后被放大）。
    for dsname, tol in [("cancellation", 1e-6), ("magnitude_spread", 1e-6),
                        ("normal", 1e-6), ("epsilon", 1e-9)]:
        ds = build_dataset(dsname, n=50_000)
        truth = compute_truth(ds).sum_f64
        vals = set()
        for t in range(60):
            order = random_order(ds.n, seed=40_000 + t)
            v = run_sum("kahan", ds.values, order, null_mask=ds.null_mask)
            vals.add(v)
            assert abs(v - truth) <= tol
        assert len(vals) == 1


# --------------------------------------------------------------- 边界守卫

def test_fixed_scale_guard_rejects_deep_decimals():
    ds = build_dataset("deep_decimals", n=2_000)
    order = random_order(ds.n, seed=1)
    with pytest.raises(PrecisionExceeded):
        run_sum("fixed64", ds.values, order, scale=6, null_mask=ds.null_mask)
    with pytest.raises(PrecisionExceeded):
        run_sum("fixed_bigint", ds.values, order, scale=6,
                null_mask=ds.null_mask)
    # scale=9 时可以正常跑且与真值一致
    vals = set()
    for t in range(20):
        order = random_order(ds.n, seed=500 + t)
        vals.add(run_sum("fixed_bigint", ds.values, order, scale=9,
                         null_mask=ds.null_mask))
    assert len(vals) == 1
    assert vals.pop() == compute_truth(ds).sum_f64


def test_fixed64_overflow_guard():
    # scale=6 下单个 1e15 就是 1e21 个最小单位，直接超 int64
    x = np.array([1e15] * 20, dtype=np.float64)
    order = random_order(x.shape[0], seed=1)
    with pytest.raises(OverflowError):
        run_sum("fixed64", x, order, scale=6)
    # 大整数方案不溢出
    assert run_sum("fixed_bigint", x, order, scale=6) == \
        pytest.approx(2e16)


def test_decimal_precision_guard():
    st = DecimalState(precision=6)  # 故意给很小的精度
    # 12 位整数 + 小数，精确总和需要约 14 位有效数字，6 位上下文必被吃满
    st.update(np.array([123456789012.0, 0.987654321, 0.123456789],
                       dtype=np.float64))
    with pytest.raises(PrecisionExceeded):
        st.finalize()
    # 精度充足时正常，且顺序无关
    ok = DecimalState(precision=50)
    ok.update(np.array([123456789012.0, 0.987654321, 0.123456789],
                       dtype=np.float64))
    assert math.isfinite(ok.finalize())


# --------------------------------------------------------------- 特殊值

def test_special_value_semantics():
    x_inf = np.array([1.0, 2.0, np.inf, 3.0])
    x_both = np.array([1.0, np.inf, -np.inf])
    x_nan = np.array([1.0, np.nan, 2.0])
    order3 = random_order(3, seed=2)
    order4 = random_order(4, seed=3)
    for m in ["naive", "kahan", "fsum", "fixed64", "fixed_bigint"]:
        assert math.isinf(run_sum(m, x_inf, order4, scale=6))
        assert math.isnan(run_sum(m, x_both, order3, scale=6))
        assert math.isnan(run_sum(m, x_nan, order3, scale=6))


def test_nulls_are_skipped():
    x = np.array([np.nan, 1.5, np.nan, 2.5])
    mask = np.array([True, False, True, False])
    order = random_order(4, seed=4)
    for m in ["naive", "fsum", "fixed64", "fixed_bigint"]:
        assert run_sum(m, x, order, scale=6, null_mask=mask) == 4.0
    ds = build_dataset("special_values", n=5_000)
    truth = compute_truth(ds)
    for t in range(10):
        order = random_order(ds.n, seed=900 + t)
        assert math.isnan(run_sum("fsum", ds.values, order,
                                  null_mask=ds.null_mask))  # 含真 nan
    # 真值计数 = 非空 且 有限的元素个数（NULL 与 inf/nan 都不计入）
    assert truth.n == int((np.isfinite(ds.values) & ~ds.null_mask).sum())


# --------------------------------------------------------------- 方差

def test_welford_matches_truth():
    ds = build_dataset("normal", n=20_000)
    order = random_order(ds.n, seed=7)
    var, std = run_variance("welford_chan", ds.values, order,
                            null_mask=ds.null_mask)
    t = compute_truth(ds)
    assert math.isclose(var, t.variance_f64, rel_tol=1e-12)
    assert math.isclose(std, t.stddev_f64, rel_tol=1e-12)


def test_naive_sumsq_fails_on_large_mean_small_variance():
    # 经典数值分析反例：均值 1e8、噪声仅 1e-2 量级，E[x²] 与 E[x]² 相消
    # 后有效数字被抹掉。真值方差约 1e-4，朴素矩法可能给出 0 或错几个数量级。
    rng = np.random.default_rng(11)
    x = 1e8 + rng.normal(0, 0.01, size=20_000)
    st_naive = NaiveSumSqState()
    st_welford = WelfordChanState()
    st_exact = ExactFixedVarState(scale=9)
    st_naive.update(x)
    st_welford.update(x)
    st_exact.update(x)
    var_naive, _ = st_naive.finalize()
    var_welford, _ = st_welford.finalize()
    var_exact, _ = st_exact.finalize()
    # Welford 与精确方案一致（相对误差 <1%）
    assert math.isclose(var_welford, var_exact, rel_tol=1e-2)
    # 朴素矩法显著偏离真值（相对误差 >50%，常见结果直接是 0）
    assert abs(var_naive - var_exact) / var_exact > 0.5


def test_exact_variance_deterministic_and_truthful():
    ds = build_dataset("normal", n=10_000)
    vals = set()
    for t in range(40):
        order = random_order(ds.n, seed=60_000 + t)
        vals.add(run_variance("exact_fixed", ds.values, order,
                              scale=ds.scale, null_mask=ds.null_mask)[0])
    assert len(vals) == 1
    assert vals.pop() == compute_truth(ds).variance_f64


def test_exact_variance_scale_guard():
    ds = build_dataset("deep_decimals", n=1_000)
    order = random_order(ds.finite_count, seed=1)
    with pytest.raises(PrecisionExceeded):
        run_variance("exact_fixed", ds.values, order, scale=6,
                     null_mask=ds.null_mask)


def test_decimal_variance_matches_truth_and_is_deterministic():
    # Decimal 方差消除 float 计算误差，逐位确定且贴合十进制真值
    ds = build_dataset("magnitude_spread", n=5_000)
    truth = compute_truth(ds).variance_f64
    vals = set()
    for t in range(30):
        order = random_order(ds.finite_count, seed=61_000 + t)
        vals.add(run_variance("decimal_var", ds.values, order,
                              null_mask=ds.null_mask)[0])
    assert len(vals) == 1
    assert vals.pop() == truth


def test_exact_fixed_variance_keeps_input_representation_error():
    # 重要语义：定点方差计算过程精确（顺序无关），但不消除 float 存储误差，
    # 在量级悬殊数据上可能与十进制真值差 1 ULP——它保证可复现，不保证还原真值
    ds = build_dataset("magnitude_spread", n=5_000)
    truth = compute_truth(ds).variance_f64
    vals = set()
    for t in range(30):
        order = random_order(ds.finite_count, seed=62_000 + t)
        vals.add(run_variance("exact_fixed", ds.values, order,
                              scale=ds.scale, null_mask=ds.null_mask)[0])
    assert len(vals) == 1                       # 逐位可复现
    got = vals.pop()
    assert abs(got - truth) <= 2 * math.ulp(truth)  # 仅差输入侧 1~2 ULP


# --------------------------------------------------------------- 分组/窗口

def test_group_sum_determinism():
    rng = np.random.default_rng(3)
    n = 20_000
    x = np.round(rng.uniform(-1e4, 1e4, n), 2)
    keys = make_group_keys(rng, n, n_groups=20)
    naive = [tuple(sorted(run_group_sum(keys, x, "naive", seed=s).items()))
             for s in range(30)]
    exact = [tuple(sorted(run_group_sum(keys, x, "fixed_bigint",
                                        seed=s, scale=2).items()))
             for s in range(30)]
    assert len(set(naive)) > 1          # 朴素分组求和会跳动
    assert len(set(exact)) == 1         # 定点分组求和逐位相同


def test_group_variance_exact():
    rng = np.random.default_rng(4)
    n = 10_000
    x = np.round(rng.uniform(-1e4, 1e4, n), 2)
    keys = make_group_keys(rng, n, n_groups=10)
    runs = [tuple(sorted(run_group_variance(keys, x, "exact_fixed",
                                            seed=s, scale=2).items()))
            for s in range(20)]
    assert len(set(runs)) == 1


def test_window_cumsum():
    rng = np.random.default_rng(5)
    x = np.round(rng.uniform(-100, 100, 5_000), 2)
    seq_naive = sequential_cumsum(x, "naive")
    seq_exact = sequential_cumsum(x, "fixed_bigint", scale=2)
    det = deterministic_cumsum(x, scale=2)
    assert np.array_equal(seq_exact, det)   # 两条精确路径逐位相同
    # 并行 + 精确状态：不同随机顺序结果一致，且等于顺序精确参照
    p1 = parallel_cumsum(x, seed=11, method="fixed_bigint", scale=2)
    p2 = parallel_cumsum(x, seed=12, method="fixed_bigint", scale=2)
    assert np.array_equal(p1, p2)
    assert np.array_equal(p1, det)
    # 并行 + 朴素状态：随机顺序之间（或与顺序参照之间）出现末位差异
    pn1 = parallel_cumsum(x, seed=11, method="naive")
    pn2 = parallel_cumsum(x, seed=12, method="naive")
    assert (not np.array_equal(pn1, pn2)) or (not np.array_equal(pn1, seq_naive))


# --------------------------------------------------------------- 开关

def test_config_switch():
    assert get_config().enabled is False
    with deterministic_aggregation(enabled=True, sum_method="fsum") as cfg:
        assert cfg.enabled and get_config().sum_method == "fsum"
    assert get_config().enabled is False
