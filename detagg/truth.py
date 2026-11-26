"""高精度十进制真值。

全部基于 decimal.Decimal，默认 60 位精度，直接吃 datasets 里保存的精确
十进制文本，不经过 float64，因此可以作为衡量各 float 方案误差的标尺。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import List, Optional

import numpy as np

from .datasets import Dataset

DEFAULT_PRECISION = 60


@dataclass
class Truth:
    n: int                      # 参与聚合的有限非空值个数
    sum: Decimal
    mean: Decimal
    variance: Decimal           # 总体方差（分母 n）
    stddev: Decimal
    min: Decimal
    max: Decimal
    # 用 float64 表达的真值（Decimal -> float64 的最近舍入，只此一次舍入）
    sum_f64: float
    mean_f64: float
    variance_f64: float
    stddev_f64: float


def _decimals_of(ds: Dataset) -> List[str]:
    out: List[str] = []
    for text, val in zip(ds.decimals, ds.values):
        if text is not None and np.isfinite(val):
            out.append(text)
    return out


def compute_truth(ds: Dataset, precision: int = DEFAULT_PRECISION) -> Truth:
    """对有限非空值计算十进制真值。"""
    texts = _decimals_of(ds)
    with localcontext() as ctx:
        ctx.prec = precision
        nums = [Decimal(t) for t in texts]
        n = len(nums)
        if n == 0:
            zero = Decimal(0)
            return Truth(0, zero, zero, zero, zero, zero, zero,
                         0.0, 0.0, 0.0, 0.0)

        s = sum(nums, Decimal(0))
        mean = s / n
        # 两遍式方差：真值层面不存在舍入问题
        var = sum(((x - mean) ** 2 for x in nums), Decimal(0)) / n
        # Decimal 平方根（精度足够时正确舍入）
        std = var.sqrt() if var >= 0 else Decimal(0)
        mn = min(nums)
        mx = max(nums)

    return Truth(n=n, sum=s, mean=mean, variance=var, stddev=std,
                 min=mn, max=mx,
                 sum_f64=float(s), mean_f64=float(mean),
                 variance_f64=float(var), stddev_f64=float(std))


def decimal_ulp_f64(x: float) -> float:
    """float64 在 x 量级上的 ULP，用于把误差换算成“差了几个 ULP”。"""
    if not math.isfinite(x) or x == 0.0:
        return math.ulp(0.0)
    return math.ulp(x)


def ulps_between(a: float, b: float) -> Optional[float]:
    """两个有限 float64 之间相差多少个 ULP（按 b 的量级计）；非有限值返回 None。"""
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if a == b:
        return 0.0
    # 用 nextafter 步数衡量，跨量级也成立
    step = 0
    lo, hi = (a, b) if abs(a) < abs(b) else (b, a)
    cur = lo
    bound = 1 << 62
    while cur != hi and step < bound:
        nxt = math.nextafter(cur, hi)
        if nxt == cur:
            break
        cur = nxt
        step += 1
    return float(step)
