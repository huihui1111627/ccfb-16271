"""窗口函数：累计求和 SUM(x) OVER (ORDER BY ... ROWS ...)。

对比两种执行方式
----------------
- sequential_cumsum：严格左到右逐行累计，结果只由排序决定（排序固定即确定），
  是单线程引擎的答案；并行引擎若按块切分再合并前缀，答案可能不同。
- parallel_cumsum  ：随机分块 -> 各块本地 cumsum -> 块和按随机调度归并 ->
  按原始行序组装前缀。朴素/补偿方法下跨运行末位跳动；精确状态下逐位相同。

确定性方案的思路：块内与块间都用可合并的精确求和状态机（定点或 fsum），
前缀值 = 前面所有块的精确状态 + 本块内的精确状态，任何归并顺序结果一致。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .executor import random_order
from .sums import PrecisionExceeded, SumState, make_sum_state


def sequential_cumsum(x: np.ndarray, method: str = "naive",
                      scale: Optional[int] = None) -> np.ndarray:
    """单线程参照实现：逐行累计。"""
    st = make_sum_state(method, scale=scale)
    out = np.empty(x.shape[0], dtype=np.float64)
    for i, v in enumerate(x):
        st.update(np.array([v], dtype=np.float64), global_lo=i)
        out[i] = st.finalize()
    return out


def parallel_cumsum(x: np.ndarray, seed: int, method: str = "naive",
                    scale: Optional[int] = None,
                    max_partitions: int = 32) -> np.ndarray:
    """模拟并行引擎的累计求和：随机分块、随机调度，按行序组装前缀。

    窗口函数有严格的行序语义，因此“随机”只作用在块的计算与块和的
    归并调度上；最终前缀必须按原始块顺序组装：
      块 k 的前缀 = 块 0..k-1 的块和（以随机合并树归并）+ 块内前缀。

    精确类状态机（fixed64/fixed_bigint/fsum/decimal）下，块和归并树
    随机不影响结果，任意 seed 逐位相同；朴素/Kahan 等浮点状态机会暴露
    归并顺序导致的末位差异——这正是并行引擎窗口结果跳动的来源。
    """
    n = x.shape[0]
    order = random_order(n, seed=seed, max_partitions=max_partitions)
    rng = np.random.default_rng(seed + 1)
    schedule = rng.permutation(len(order.partitions)).tolist()

    # 1) 各块本地累计与块和状态（块内顺序固定）
    block_sum: List[SumState] = []
    for lo, hi in order.partitions:
        st = make_sum_state(method, scale=scale)
        for j in range(lo, hi):
            st.update(np.array([x[j]], dtype=np.float64), global_lo=j)
        block_sum.append(st)

    # 2) 块和按随机调度顺序归并（精确状态下顺序无关），保存各块最终块和
    merged: List[SumState] = [st.clone() for st in block_sum]
    root: Optional[SumState] = None
    for block_idx in schedule:
        if root is None:
            root = block_sum[block_idx].clone()
        else:
            root.merge(block_sum[block_idx])

    # 3) 按原始块顺序求前缀状态：prefix[k] = 块 0..k-1 之和
    prefix_state: List[SumState] = []
    running: Optional[SumState] = None
    for k in range(len(order.partitions)):
        prefix_state.append(
            make_sum_state(method, scale=scale) if running is None
            else running.clone())
        if running is None:
            running = merged[k].clone()
        else:
            running.merge(merged[k])

    # 4) 组装：全局前缀值 = 前缀块状态 + 块内前 j+1 个值
    out = np.empty(n, dtype=np.float64)
    for k, (lo, hi) in enumerate(order.partitions):
        for j in range(hi - lo):
            cur = prefix_state[k].clone()
            cur.update(x[lo:lo + j + 1], global_lo=lo)
            out[lo + j] = cur.finalize()
    return out


def deterministic_cumsum(x: np.ndarray, scale: int = 6) -> np.ndarray:
    """确定性开关下推荐的窗口实现：定点大整数，O(n)、逐位可复现。"""
    factor = 10 ** scale
    scaled = x * factor
    rounded = np.round(scaled)
    back = rounded / factor
    tol = np.maximum(np.abs(x) * np.finfo(np.float64).eps,
                     np.finfo(np.float64).tiny)
    if (np.abs(back - x) > tol).any():
        raise PrecisionExceeded(
            f"累计求和：存在超过 scale={scale} 位小数的值；"
            "提高 scale 或改用 fsum 窗口实现。")
    # object dtype 让 cumsum 走 Python 大整数，避免 int64 前缀回绕
    prefix = np.cumsum(rounded.astype(object), dtype=object)
    return np.array([int(v) / factor for v in prefix.tolist()],
                    dtype=np.float64)
