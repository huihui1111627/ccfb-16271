"""分组聚合 GROUP BY：随机切分下保证确定结果。

执行方式
--------
1. 数据随机切成 p 个分区；
2. 每个分区各自维护 {group_key: 状态机} 的本地哈希表；
3. 合并阶段按随机顺序把分区表两两 merge；
4. 逐组 finalize。

朴素浮点状态机：同一分组的成员来自不同分区、合并顺序随机 => 末位跳动。
精确状态机（定点/fsum/decimal）：任意切分与合并顺序下逐位相同。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .executor import random_order
from .sums import SumState, make_sum_state
from .stats import VarState, make_var_state


@dataclass
class GroupResult:
    key: Any
    value: float
    variance: Optional[float] = None
    stddev: Optional[float] = None
    count: int = 0


def _group_sum_tables(keys: np.ndarray, x: np.ndarray, order,
                      method: str, scale: Optional[int]):
    tables: List[Dict[Any, SumState]] = []
    for lo, hi in order.partitions:
        tab: Dict[Any, SumState] = {}
        for k, v in zip(keys[lo:hi], x[lo:hi]):
            st = tab.get(k)
            if st is None:
                st = make_sum_state(method, scale=scale)
                tab[k] = st
            st.update(np.array([v], dtype=np.float64))
        tables.append(tab)
    return tables


def run_group_sum(keys: np.ndarray, x: np.ndarray, method: str,
                  seed: int, scale: Optional[int] = None,
                  max_partitions: int = 32) -> Dict[Any, float]:
    """一次随机分组求和，返回 {组键: 组和}。"""
    n = x.shape[0]
    order = random_order(n, seed=seed, max_partitions=max_partitions)
    tables = _group_sum_tables(keys, x, order, method, scale)

    root: Dict[Any, SumState] = {}
    for idx in order.merge_perm:
        for k, st in tables[idx].items():
            if k in root:
                root[k].merge(st)
            else:
                root[k] = st.clone()
    return {k: st.finalize() for k, st in root.items()}


def run_group_variance(keys: np.ndarray, x: np.ndarray, method: str,
                       seed: int, scale: Optional[int] = None,
                       max_partitions: int = 32) -> Dict[Any, tuple]:
    """一次随机分组方差，返回 {组键: (variance, stddev)}。"""
    n = x.shape[0]
    order = random_order(n, seed=seed, max_partitions=max_partitions)
    tables: List[Dict[Any, VarState]] = []
    for lo, hi in order.partitions:
        tab: Dict[Any, VarState] = {}
        for k, v in zip(keys[lo:hi], x[lo:hi]):
            st = tab.get(k)
            if st is None:
                st = make_var_state(method, scale=scale)
                tab[k] = st
            st.update(np.array([v], dtype=np.float64))
        tables.append(tab)

    root: Dict[Any, VarState] = {}
    for idx in order.merge_perm:
        for k, st in tables[idx].items():
            if k in root:
                root[k].merge(st)
            else:
                root[k] = st.clone()
    return {k: st.finalize() for k, st in root.items()}


def make_group_keys(rng: np.random.Generator, n: int,
                    n_groups: int = 50) -> np.ndarray:
    """造一组分组键（定长字符串，模拟账户号）。"""
    base = rng.integers(0, n_groups, size=n)
    return np.array([f"ACC-{i:04d}" for i in base])
