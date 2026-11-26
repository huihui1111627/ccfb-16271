"""并行聚合执行器：随机切分 + 随机合并顺序。

执行模型
--------
1. 数据按随机边界切成 p 个分区（每个分区至少 1 个元素，边界用
   “p-1 个随机切点”生成，模拟引擎每次不同的 work stealing / 分片）；
2. 每个分区在独立状态机上 update；
3. 分区状态按随机排列两两合并，合并树也是随机的（左结合随机链），
   模拟真实引擎里“先算完的分区先进入 reduce”；
4. finalize 输出结果。

两种驱动方式
------------
- simulated：顺序执行，纯靠随机数制造不同切分/合并顺序，便于重复试验；
- threaded ：ThreadPoolExecutor 真并行，任务以随机乱序提交，合并按
             实际完成顺序（as_completed）进行——和真实引擎的
             非确定性来源同构（GIL 不影响 numpy/C 计算段的并行释放）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .sums import BLOCK, SumState, make_sum_state
from .stats import VarState, make_var_state


@dataclass
class RunOrder:
    """一次随机执行的切分与合并描述（可复现）。"""

    seed: int
    partitions: List[Tuple[int, int]]   # 半开区间 [lo, hi)
    merge_perm: List[int]                # 分区进入 reduce 的顺序

    def describe(self) -> str:
        sizes = [hi - lo for lo, hi in self.partitions]
        return (f"seed={self.seed}, p={len(sizes)}, "
                f"分区大小 min/med/max={min(sizes)}/{int(np.median(sizes))}/{max(sizes)}, "
                f"合并顺序前8={self.merge_perm[:8]}")


def random_order(n: int, seed: int, max_partitions: int = 64,
                 align: Optional[int] = None) -> RunOrder:
    """生成一次随机切分与随机合并顺序。

    align：若给定（如 BLOCK=4096），分区边界强制对齐到 align 的整数倍。
    这样“块内向量化求和”的块就是全局固定的物理切片，随机只作用在块如何
    分组到分区、以及分区如何归并——分块 fsum 这类方案因此逐位确定。
    """
    if n < 2:
        return RunOrder(seed=seed, partitions=[(0, n)], merge_perm=[0])
    rng = np.random.default_rng(seed)
    p = int(rng.integers(2, min(max_partitions, n) + 1))
    # 随机切点：在 1..n-1 中取 p-1 个不重复点
    cuts = sorted(rng.choice(np.arange(1, n), size=p - 1, replace=False).tolist())
    if align is not None and align > 1:
        cuts = sorted({min(n, max(1, int(round(c / align)) * align))
                       for c in cuts if 0 < c < n})
    bounds = [0] + cuts + [n]
    # 相邻边界可能因对齐重合，去掉空分区
    partitions = []
    for i in range(len(bounds) - 1):
        if bounds[i] < bounds[i + 1]:
            partitions.append((bounds[i], bounds[i + 1]))
    merge_perm = rng.permutation(len(partitions)).tolist()
    return RunOrder(seed=seed, partitions=partitions, merge_perm=merge_perm)


def _prepare(x: np.ndarray,
             null_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, dict]:
    """预扫描：剔除 NULL，分离特殊值标志，返回压缩后的有限值数组。

    状态机此后只看有限值，分区下标即压缩数组下标——这让 fsum_block
    的全局块对齐在含 NULL/inf/nan 的数据上依然成立。特殊值标志与
    切分/合并顺序无关，在 finalize 阶段统一按 IEEE 规则处理。
    """
    x = np.ascontiguousarray(x)
    if null_mask is not None and null_mask.any():
        x = x[~null_mask]
    finite = np.isfinite(x)
    specials = {
        "nan": bool(np.isnan(x).any()),
        "pos_inf": bool((x == np.inf).any()),
        "neg_inf": bool((x == -np.inf).any()),
    }
    return np.ascontiguousarray(x[finite]), specials


def _apply_specials(specials: dict, finite_result: float,
                    has_data: bool) -> float:
    if specials["nan"] or (specials["pos_inf"] and specials["neg_inf"]):
        return float("nan")
    if specials["pos_inf"]:
        return float("inf")
    if specials["neg_inf"]:
        return float("-inf")
    if not has_data:
        return float("nan")
    return finite_result


def run_sum(method: str, x: np.ndarray, order: RunOrder,
            scale: Optional[int] = None,
            null_mask: Optional[np.ndarray] = None,
            threaded: bool = False,
            max_workers: int = 8) -> float:
    """按给定随机顺序跑一次求和。"""
    fx, specials = _prepare(x, null_mask)
    if fx.shape[0] == 0:
        return _apply_specials(specials, 0.0, False)

    def build(lo: int, hi: int) -> SumState:
        st = make_sum_state(method, scale=scale)
        st.update(fx[lo:hi], global_lo=lo)
        return st
    states: List[Optional[SumState]] = [None] * len(order.partitions)
    if threaded:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(build, lo, hi): i
                for i, (lo, hi) in enumerate(order.partitions)
            }
            for fut in as_completed(futs):
                states[futs[fut]] = fut.result()
    else:
        for i, (lo, hi) in enumerate(order.partitions):
            states[i] = build(lo, hi)

    root: Optional[SumState] = None
    for idx in order.merge_perm:
        if root is None:
            root = states[idx]
        else:
            root.merge(states[idx])
    assert root is not None
    return _apply_specials(specials, root.finalize(), True)


def run_variance(method: str, x: np.ndarray, order: RunOrder,
                 scale: Optional[int] = None,
                 null_mask: Optional[np.ndarray] = None,
                 threaded: bool = False,
                 max_workers: int = 8) -> Tuple[float, float]:
    """按给定随机顺序跑一次方差/标准差，返回 (variance, stddev)。"""
    fx, specials = _prepare(x, null_mask)
    if specials["nan"] or specials["pos_inf"] or specials["neg_inf"] \
            or fx.shape[0] == 0:
        return float("nan"), float("nan")

    def build(lo: int, hi: int) -> VarState:
        st = make_var_state(method, scale=scale)
        st.update(fx[lo:hi], global_lo=lo)
        return st

    states: List[Optional[VarState]] = [None] * len(order.partitions)
    if threaded:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(build, lo, hi): i
                for i, (lo, hi) in enumerate(order.partitions)
            }
            for fut in as_completed(futs):
                states[futs[fut]] = fut.result()
    else:
        for i, (lo, hi) in enumerate(order.partitions):
            states[i] = build(lo, hi)

    root: Optional[VarState] = None
    for idx in order.merge_perm:
        if root is None:
            root = states[idx]
        else:
            root.merge(states[idx])
    assert root is not None
    return root.finalize()


def order_for_method(n: int, seed: int, method: str,
                     max_partitions: int = 64) -> RunOrder:
    """按方法选择切分方式：块向量化方案强制全局块对齐，其余纯随机。"""
    align = BLOCK if method == "fsum_block" else None
    return random_order(n, seed=seed, max_partitions=max_partitions, align=align)


def repeated_runs(runner: Callable[[RunOrder], object], n: int,
                  trials: int, base_seed: int,
                  max_partitions: int = 64) -> List[object]:
    """同一数据、trials 个不同随机顺序各跑一次。"""
    out = []
    for t in range(trials):
        order = random_order(n, seed=base_seed + t, max_partitions=max_partitions)
        out.append(runner(order))
    return out
