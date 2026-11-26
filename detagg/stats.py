"""方差 / 标准差状态机。

三种方案，全部支持分区并行合并：
- naive_sumsq : E[x²] - E[x]² 单遍矩法，数值不稳定，刻意保留作为反例；
- welford_chan: Welford 在线均值 + M2，分区用 Chan 公式合并，数值稳定
                但合并树随机时末位仍可能跳动；
- exact_fixed : 定点大整数保存 Σx、Σx²，整数域精确，顺序无关（受 scale 守卫保护）。

finalize 返回 (variance, stddev)，统一为总体口径（分母 n），与真值模块一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Dict, List, Optional, Type

import numpy as np

from .sums import PrecisionExceeded, _iter_blocks


class VarState:
    method = "base"
    deterministic = False
    label = ""
    cost_note = ""
    guarantee_note = ""
    range_note = ""

    def __init__(self) -> None:
        self._special = False
        self._count = 0

    def update(self, x: np.ndarray, global_lo: Optional[int] = None) -> None:
        finite = np.isfinite(x)
        if not finite.all():
            self._special = True
        self._count += int(finite.sum())
        if finite.any():
            self._update_finite(np.ascontiguousarray(x[finite]),
                                global_lo=global_lo)

    def merge(self, other: "VarState") -> None:
        assert type(self) is type(other)
        self._special |= other._special
        self._count += other._count
        if other._count:
            self._merge_finite(other)

    def finalize(self) -> tuple:
        """返回 (variance, stddev)；空集或含特殊值时为 (nan, nan)。"""
        if self._special or self._count == 0:
            return float("nan"), float("nan")
        return self._finalize_finite()

    def clone(self) -> "VarState":
        import copy
        return copy.deepcopy(self)

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        raise NotImplementedError

    def _merge_finite(self, other: "VarState") -> None:
        raise NotImplementedError

    def _finalize_finite(self) -> tuple:
        raise NotImplementedError


class NaiveSumSqState(VarState):
    """单遍矩法：保存 Σx、Σx²，方差 = Σx²/n - (Σx/n)²。

    正负相消时两个大数相减可能抹掉全部有效数字，是教科书级反例。
    """

    method = "naive_sumsq"
    label = "朴素矩法 E[x²]-E[x]²（不稳定，反例）"
    deterministic = False
    cost_note = "一次读两个向量化累加，O(1) 内存，最快"
    guarantee_note = "无稳定性保证，灾难性抵消下结果可能为负或差几个数量级"
    range_note = "Σx² 溢出 float64（>1.8e308）时直接得 inf"

    def __init__(self) -> None:
        super().__init__()
        self.s1 = 0.0
        self.s2 = 0.0

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        for blk in _iter_blocks(x):
            self.s1 += float(blk.sum())
            self.s2 += float((blk * blk).sum())

    def _merge_finite(self, other: "NaiveSumSqState") -> None:
        self.s1 += other.s1
        self.s2 += other.s2

    def _finalize_finite(self) -> tuple:
        n = self._count
        mean = self.s1 / n
        var = self.s2 / n - mean * mean
        # 刻意不钳位负值：负的“方差”正是抵消灾难的可见信号，报告里要暴露
        return var, math.sqrt(var) if var >= 0.0 else float("nan")


class WelfordChanState(VarState):
    """Welford 在线算法 + Chan 并行合并。

    分区状态 (n, mean, M2)，合并公式：
        delta = mean_B - mean_A
        mean  = mean_A + delta * n_B / n_T
        M2    = M2_A + M2_B + delta² * n_A n_B / n_T
    数值稳定，是 Spark/DuckDB 等引擎并行方差的标准做法；
    但公式内浮点运算仍依赖合并顺序，随机合并树下末位可能不同。
    """

    method = "welford_chan"
    label = "Welford + Chan 并行合并（稳定，顺序敏感）"
    deterministic = False
    cost_note = "每值几次标量运算，约 2~4x；状态仅 3 个 float"
    guarantee_note = "误差 ~ ε·方差量级，与 n 近似无关；不保证逐位一致"
    range_note = "任意有限 float64 输入"

    def __init__(self) -> None:
        super().__init__()
        self.mean = 0.0
        self.m2 = 0.0

    @staticmethod
    def _welford_update(mean: float, m2: float, n: int,
                        x: np.ndarray) -> tuple:
        for v in x.tolist():
            n += 1
            delta = v - mean
            mean += delta / n
            delta2 = v - mean
            m2 += delta * delta2
        return mean, m2, n

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        # 向量化 Welford：按块更新，块内用 (x-mean) 形式保持稳定性
        n = 0
        mean, m2 = self.mean, self.m2
        for blk in _iter_blocks(x):
            bn = blk.shape[0]
            bmean = float(blk.sum()) / bn
            bm2 = float(((blk - bmean) ** 2).sum())
            nt = n + bn
            if nt == bn:
                mean, m2 = bmean, bm2
            else:
                delta = bmean - mean
                mean = mean + delta * bn / nt
                m2 = m2 + bm2 + delta * delta * n * bn / nt
            n = nt
        self.mean, self.m2 = mean, m2

    def _merge_finite(self, other: "WelfordChanState") -> None:
        na, nb = self._count - other._count, other._count
        nt = na + nb
        delta = other.mean - self.mean
        self.mean = self.mean + delta * nb / nt
        self.m2 = self.m2 + other.m2 + delta * delta * na * nb / nt

    def _finalize_finite(self) -> tuple:
        var = self.m2 / self._count
        return var, math.sqrt(max(var, 0.0))


class ExactFixedVarState(VarState):
    """定点精确方差：scale 位小数内，Σx、Σx² 在整数域精确，顺序无关。

    每个分区累加 Python 大整数（逐元素平方和可能超 int64，不能用定长）。
    finalize 用 Decimal 完成除法与开方，唯一一次浮点舍入发生在转 float64 时。
    """

    method = "exact_fixed"
    label = "定点大整数方差（顺序无关，scale 守卫）"
    deterministic = True
    cost_note = "逐元素大整数平方累加，约 5~15x，O(1) 状态（位数随数据增长）"
    guarantee_note = "scale 位小数内与十进制真值的最近 float64 逐位相同"
    range_note = "小数位超 scale 抛 PrecisionExceeded；大整数仅受内存限制"

    def __init__(self, scale: int = 6, precision: int = 60) -> None:
        super().__init__()
        self.scale = scale
        self.factor = 10 ** scale
        self.precision = precision
        self.s1 = 0   # Σx（定点整数）
        self.s2 = 0   # Σx²（定点整数平方）

    def _scaled_floats(self, x: np.ndarray) -> np.ndarray:
        scaled = x * self.factor
        rounded = np.round(scaled)
        # ULP 容差的往返判据：超 scale 的取整误差远超 1 ULP，必被拦下
        back = rounded / self.factor
        tol = np.maximum(np.abs(x) * np.finfo(np.float64).eps,
                         np.finfo(np.float64).tiny)
        bad = np.abs(back - x) > tol
        if bad.any():
            examples = x[bad][:5]
            raise PrecisionExceeded(
                f"方差定点方案：输入含超过 scale={self.scale} 位小数的值"
                f"（示例 {examples.tolist()}）；提高 scale 或改用 decimal 方案。")
        return rounded

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        rounded = self._scaled_floats(x)
        # 全程 Python 大整数：逐元素平方和轻松超过 int64，不能用定长类型
        s1 = 0
        s2 = 0
        for blk in _iter_blocks(rounded):
            for fv in blk.tolist():
                v = int(fv)
                s1 += v
                s2 += v * v
        self.s1 += s1
        self.s2 += s2

    def _merge_finite(self, other: "ExactFixedVarState") -> None:
        if other.scale != self.scale:
            raise PrecisionExceeded(
                f"定点 scale 不一致：{self.scale} vs {other.scale}，无法合并")
        self.s1 += other.s1
        self.s2 += other.s2

    def _finalize_finite(self) -> tuple:
        n = self._count
        f2 = Decimal(self.factor * self.factor)
        with localcontext() as ctx:
            ctx.prec = self.precision
            s1 = Decimal(self.s1)
            s2 = Decimal(self.s2)
            nd = Decimal(n)
            # var = s2/n/f² - (s1/n/f)²，整理成一次分式以减少中间舍入
            num = s2 * nd - s1 * s1          # 整数，精确
            var = num / (nd * nd * f2)
            std = var.sqrt() if var >= 0 else Decimal(0)
        return float(var), float(std)


class DecimalVarState(VarState):
    """Decimal 宽精度方差：逐值转 Decimal 保存，合并只拼接，finalize 统一算。

    与 exact_fixed 的关键区别：它直接吃 str(float) 的最短往返十进制文本，
    因此消除的是“float 存储值”层面的全部计算误差；若数据源自十进制文本
    （本项目数据集正是如此），结果与十进制真值一致。代价是每值一个
    Decimal 对象、纯 Python 运算，最慢（约 30~100x）。
    """

    method = "decimal_var"
    label = "Decimal 宽精度方差（顺序无关，消除计算误差）"
    deterministic = True
    cost_note = "每值一个 Decimal 对象，约 30~100x，O(n) 对象内存"
    guarantee_note = "上下文精度内结果精确、任意顺序逐位相同；精度不足即报错"
    range_note = "有效数字超 precision 抛 PrecisionExceeded"

    def __init__(self, precision: int = 60) -> None:
        super().__init__()
        self.precision = precision
        self.values: List[Decimal] = []

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        self.values.extend(Decimal(str(v)) for v in x.tolist())

    def _merge_finite(self, other: "DecimalVarState") -> None:
        self.values.extend(other.values)

    def _finalize_finite(self) -> tuple:
        n = len(self.values)
        with localcontext() as ctx:
            ctx.prec = self.precision + 20
            nd = Decimal(n)
            mean_hi = sum(self.values, Decimal(0)) / nd
            num_hi = sum(((v - mean_hi) ** 2 for v in self.values), Decimal(0))
            var_hi = num_hi / nd
            # 用高精度结果做基准，在声明精度上重算；两者末位不一致即说明
            # 声明精度被吃满，拒绝静默输出
            ctx.prec = self.precision
            mean = sum(self.values, Decimal(0)) / nd
            num = sum(((v - mean) ** 2 for v in self.values), Decimal(0))
            var = num / nd
            if var_hi != 0 and abs(var_hi - var) / abs(var_hi) > \
                    Decimal(10) ** (-(self.precision - 4)):
                raise PrecisionExceeded(
                    f"Decimal 方差精度 {self.precision} 位不足，请提高 precision。")
            std = var_hi.sqrt() if var_hi >= 0 else Decimal(0)
        return float(var_hi), float(std)


@dataclass
class VarMethodSpec:
    key: str
    cls: Type[VarState]
    label: str
    deterministic: bool
    cost_note: str
    guarantee_note: str
    range_note: str
    needs_scale: bool = False


def _var_spec(state: VarState) -> VarMethodSpec:
    return VarMethodSpec(
        key=state.method, label=state.label, cls=type(state),
        deterministic=state.deterministic, cost_note=state.cost_note,
        guarantee_note=state.guarantee_note, range_note=state.range_note,
        needs_scale=isinstance(state, ExactFixedVarState),
    )


VAR_METHODS: Dict[str, VarMethodSpec] = {}
for _v in (NaiveSumSqState(), WelfordChanState(), ExactFixedVarState(),
           DecimalVarState()):
    _s = _var_spec(_v)
    VAR_METHODS[_s.key] = _s

VAR_METHOD_KEYS: List[str] = list(VAR_METHODS)


def make_var_state(method: str, scale: Optional[int] = None) -> VarState:
    if method not in VAR_METHODS:
        raise KeyError(f"未知方差方法 {method!r}，可选：{VAR_METHOD_KEYS}")
    if method == "exact_fixed":
        return ExactFixedVarState(6 if scale is None else scale)
    return VAR_METHODS[method].cls()


def run_mean(sum_value: float, count: int) -> float:
    """平均值由求和结果导出：sum 确定 => mean 确定（一次除法舍入，顺序无关）。"""
    if count <= 0:
        return float("nan")
    return sum_value / count
