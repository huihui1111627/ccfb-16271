"""造数模块。

所有数据集都是 (n, 2) 的结构：
- values: np.float64 数组，参与浮点计算
- decimals: 与 values 一一对应的精确十进制字符串（或 None 表示空值），
  真值模块据此做 Decimal 计算，避免“用浮点当真值”的循环论证。

特殊值（inf / nan）以 np.float64 原样给出，对应的 decimals 为 None。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# 金额类数据统一保留到 1e-6（百万分之一元，覆盖大多数对账系统的分/厘精度）
MONEY_DECIMALS = 6
MIN_INT64 = np.iinfo(np.int64).min
MAX_INT64 = np.iinfo(np.int64).max


@dataclass
class Dataset:
    """一份测试数据。"""

    name: str
    description: str
    values: np.ndarray            # float64，含 inf/nan；空值位置为 nan（由 null_mask 区分）
    decimals: List[Optional[str]]  # 精确十进制文本，空值/特殊值位置为 None
    null_mask: np.ndarray          # bool，True 表示该位置是 SQL 意义上的 NULL
    scale: int = MONEY_DECIMALS    # 定点方案使用的小数位数（10**scale）
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @property
    def finite_values(self) -> np.ndarray:
        """去掉 NULL 与特殊值后的有限 float64 数组。"""
        finite = np.isfinite(self.values)
        return self.values[finite]

    @property
    def finite_count(self) -> int:
        """参与聚合的有限非空元素个数（执行器压缩后的实际长度）。"""
        return int((np.isfinite(self.values) & ~self.null_mask).sum())

    def finite_decimals(self) -> List[str]:
        return [d for d, v in zip(self.decimals, self.values)
                if d is not None and np.isfinite(v)]


def _from_decimal_text(
    name: str,
    description: str,
    texts: List[Optional[str]],
    scale: int = MONEY_DECIMALS,
    note: str = "",
) -> Dataset:
    """由十进制文本构造数据集；None 表示 NULL，'inf'/'-inf'/'nan' 表示特殊值。"""
    values = np.empty(len(texts), dtype=np.float64)
    decimals: List[Optional[str]] = []
    null_mask = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        if t is None:
            values[i] = np.nan
            decimals.append(None)
            null_mask[i] = True
        elif t.lower() in ("inf", "+inf", "infinity"):
            values[i] = np.inf
            decimals.append(None)
        elif t.lower() in ("-inf", "-infinity"):
            values[i] = -np.inf
            decimals.append(None)
        elif t.lower() == "nan":
            values[i] = np.nan
            decimals.append(None)
        else:
            values[i] = float(t)
            decimals.append(t)
    return Dataset(name=name, description=description, values=values,
                   decimals=decimals, null_mask=null_mask, scale=scale, note=note)


def _money_text(rng: np.random.Generator, low: float, high: float,
                decimals: int = 2) -> str:
    return f"{rng.uniform(low, high):.{decimals}f}"


def make_normal(rng: np.random.Generator, n: int = 200_000) -> Dataset:
    """正常对账金额：-1万 ~ 10万，两位小数，约 2% 空值。"""
    texts: List[Optional[str]] = []
    for _ in range(n):
        if rng.random() < 0.02:
            texts.append(None)
        else:
            texts.append(_money_text(rng, -10_000, 100_000, 2))
    return _from_decimal_text(
        "normal", "正常金额：-1万~10万元、两位小数、约2%空值", texts,
        note="绝大多数生产表的形态，误差通常在 1e-9 量级，但已经会触发逐位比对告警。",
    )


def make_magnitude_spread(rng: np.random.Generator, n: int = 100_000) -> Dataset:
    """量级悬殊：1e12 的大数与 1e-6 的小额流水混在一起。"""
    texts: List[Optional[str]] = []
    for i in range(n):
        r = rng.random()
        if r < 0.01:                       # 1% 大额（企业间结算）
            texts.append(_money_text(rng, 1e11, 5e12, 2))
        elif r < 0.03:                     # 2% 空值
            texts.append(None)
        else:                              # 97% 几元到几百元的小额流水
            texts.append(f"{rng.uniform(0.01, 500.0):.6f}")
    return _from_decimal_text(
        "magnitude_spread", "量级悬殊：1e12 级大额与 1e-6 级小额混排", texts,
        note="大数吞小数：朴素累加时小额可能在舍入中完全丢失，是最坏情况之一。",
    )


def make_cancellation(rng: np.random.Generator, n: int = 200_000) -> Dataset:
    """大量正负相消：成对生成几乎相等的借贷，真实余额远小于单笔金额。"""
    texts: List[Optional[str]] = []
    for _ in range(n // 2):
        base = rng.uniform(1e5, 1e8)
        delta = rng.uniform(-0.05, 0.05)          # 每对净值只有几分钱
        texts.append(f"{base:.2f}")
        texts.append(f"{-(base + delta):.2f}")
    for _ in range(n % 2):
        texts.append(None)
    rng.shuffle(texts)
    return _from_decimal_text(
        "cancellation", "正负相消：借贷成对、单笔 1e5~1e8，真实净值仅几十元", texts,
        note="灾难性抵消：相对误差被放大到 O(1)，方差/标准差的二遍算法也会失真。",
    )


def make_deep_decimals(rng: np.random.Generator, n: int = 100_000) -> Dataset:
    """深小数：9 位小数的汇率/利息数据，用来暴露 scale=6 定点方案的越界守卫。"""
    texts: List[Optional[str]] = []
    for _ in range(n):
        if rng.random() < 0.02:
            texts.append(None)
        else:
            sign = "-" if rng.random() < 0.3 else ""
            texts.append(f"{sign}{rng.uniform(0, 1e6):.9f}")
    return _from_decimal_text(
        "deep_decimals", "深小数：9 位小数（scale=6 的定点方案会拒绝并提示）", texts,
        scale=9,
        note="用来演示定点方案的适用边界：超过声明 scale 时应报错而不是静默截断。",
    )


def make_epsilon(rng: np.random.Generator, n: int = 200_000) -> Dataset:
    """经典 1.0 + eps 累加：演示大数加小数的精度天花板。"""
    texts: List[Optional[str]] = ["1.0"]
    for _ in range(n - 1):
        texts.append("0.0000001")  # 1e-7，远小于 1 附近的 ULP（~2.2e-16）……累计后可见
    return _from_decimal_text(
        "epsilon", "大数加小数：1 个 1.0 加 20 万个 1e-7（真值 1.02）", texts,
        note="朴素 float 累加能保住部分，但不同切分/合并顺序给出的舍入不同。",
    )


def make_special_values(rng: np.random.Generator, n: int = 50_000) -> Dataset:
    """含 NULL / +inf / -inf / nan：验证特殊值语义在所有方案下一致。"""
    texts: List[Optional[str]] = []
    for i in range(n):
        r = rng.random()
        if r < 0.05:
            texts.append(None)          # NULL：参与计数时跳过
        elif r < 0.055:
            texts.append("inf")         # +inf：求和结果应为 +inf
        elif r < 0.058:
            texts.append("-inf")        # -inf：与 +inf 同时存在时按 IEEE 规则得 nan
        elif r < 0.060:
            texts.append("nan")         # nan：任何含 nan 的聚合结果为 nan
        else:
            texts.append(_money_text(rng, -1000, 1000, 2))
    return _from_decimal_text(
        "special_values", "特殊值：NULL/±inf/nan 混合正常金额", texts,
        note="NULL 必须跳过；inf/nan 按 IEEE-754 传播，且各方案行为一致。",
    )


DATASET_BUILDERS: Dict[str, Callable[[np.random.Generator, int], Dataset]] = {
    "normal": make_normal,
    "magnitude_spread": make_magnitude_spread,
    "cancellation": make_cancellation,
    "deep_decimals": make_deep_decimals,
    "epsilon": make_epsilon,
    "special_values": make_special_values,
}


def build_dataset(name: str, seed: int = 20261127,
                  n: Optional[int] = None) -> Dataset:
    """按名字构造数据集。造数用独立 RNG，和执行期的切分/合并随机数互不影响。"""
    if name not in DATASET_BUILDERS:
        raise KeyError(f"未知数据集 {name!r}，可选：{sorted(DATASET_BUILDERS)}")
    rng = np.random.default_rng(seed)
    builder = DATASET_BUILDERS[name]
    if n is None:
        return builder(rng)
    return builder(rng, n)


def all_datasets(seed: int = 20261127) -> List[Dataset]:
    return [build_dataset(name, seed=seed) for name in DATASET_BUILDERS]
