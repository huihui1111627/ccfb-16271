"""求和状态机：每种方法一个可 merge 的分区状态。

统一接口
--------
- update(x):  吃掉一个分区的 float64 数组（含 NULL/特殊值处理）
- merge(other): 与另一个分区的状态合并（执行器会以随机顺序两两合并）
- finalize(): 输出最终 float64
- clone():    深拷贝，供重复试验复用初始状态

设计要点
--------
1. NULL（np.nan 且 null_mask 标记）在进入状态机前由执行器剔除；
   真正的 nan / ±inf 走统一的 IEEE-754 特殊值吸收逻辑。
2. 每个状态机只负责“有限值”的算法，特殊值语义在基类里保证各方案一致。
3. deterministic=True 表示在任意切分/合并顺序下 finalize() 逐位相同；
   exact_for_scale 表示对声明小数位内的十进制输入，结果与十进制真值一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Dict, List, Optional, Type

import numpy as np

from .datasets import MAX_INT64, MIN_INT64

BLOCK = 4096


class PrecisionExceeded(ValueError):
    """定点/Decimal 方案声明的精度或位宽不够，拒绝静默给出错误答案。"""


# --------------------------------------------------------------------------
# 特殊值处理
# --------------------------------------------------------------------------

class SumState:
    """求和状态机基类，统一处理 nan / ±inf。"""

    method: str = "base"
    deterministic: bool = False
    exact_for_scale: bool = False
    label: str = ""
    cost_note: str = ""
    guarantee_note: str = ""
    range_note: str = ""

    def __init__(self) -> None:
        self._nan = False
        self._pos_inf = False
        self._neg_inf = False
        self._count = 0

    # ---- 公共接口 --------------------------------------------------------
    def update(self, x: np.ndarray, global_lo: Optional[int] = None) -> None:
        finite = np.isfinite(x)
        if not finite.all():
            for v in x[~finite]:
                if np.isnan(v):
                    self._nan = True
                elif v > 0:
                    self._pos_inf = True
                else:
                    self._neg_inf = True
        self._count += int(finite.sum())
        if finite.any():
            self._update_finite(np.ascontiguousarray(x[finite]),
                                global_lo=global_lo)

    def merge(self, other: "SumState") -> None:
        assert type(self) is type(other)
        self._nan |= other._nan
        self._pos_inf |= other._pos_inf
        self._neg_inf |= other._neg_inf
        self._count += other._count
        self._merge_finite(other)

    def finalize(self) -> float:
        if self._nan or (self._pos_inf and self._neg_inf):
            return float("nan")
        if self._pos_inf:
            return float("inf")
        if self._neg_inf:
            return float("-inf")
        if self._count == 0:
            return float("nan")  # 空集上的 SUM：SQL 语义为 NULL，这里用 nan 表达
        return self._finalize_finite()

    def clone(self) -> "SumState":
        import copy
        return copy.deepcopy(self)

    # ---- 子类实现 --------------------------------------------------------
    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        raise NotImplementedError

    def _merge_finite(self, other: "SumState") -> None:
        raise NotImplementedError

    def _finalize_finite(self) -> float:
        raise NotImplementedError


def _iter_blocks(x: np.ndarray):
    for i in range(0, x.shape[0], BLOCK):
        yield x[i:i + BLOCK]


# --------------------------------------------------------------------------
# 1. 朴素累加
# --------------------------------------------------------------------------

class NaiveState(SumState):
    """朴素左到右累加；分区合并也是简单相加，顺序不同舍入不同。

    生产引擎默认行为的替身：每个分区内部顺序固定，但分区切分和合并树
    每次随机，因此每次运行的结果可能在末位跳动。
    """

    method = "naive"
    label = "朴素累加（引擎默认，顺序敏感）"
    deterministic = False
    cost_note = "一次读，O(1) 额外内存，向量化块循环，速度基准 1.0x"
    guarantee_note = "无顺序保证；误差上界 ~ n·ε·Σ|x|"
    range_note = "任意 float64 输入都能跑，但大数加小数会丢精度"

    def __init__(self) -> None:
        super().__init__()
        self.total = 0.0

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        for blk in _iter_blocks(x):
            self.total += float(blk.sum())  # 块间朴素累加

    def _merge_finite(self, other: "NaiveState") -> None:
        self.total += other.total

    def _finalize_finite(self) -> float:
        return self.total


# --------------------------------------------------------------------------
# 2. 成对求和
# --------------------------------------------------------------------------

class PairwiseState(SumState):
    """成对（树形）求和：保留各分区和，最后按当前顺序做平衡二叉树归并。

    树结构由分区到达顺序决定，执行器随机洗牌后每次树不同，因此仍可能
    末位不同；但误差界从 O(nε) 降到 O(ε log n)，实际跳动幅度小得多。
    """

    method = "pairwise"
    label = "成对求和（误差 O(εlog n)，仍顺序敏感）"
    deterministic = False
    cost_note = "O(p) 额外内存（p=分区数），归并多一层树，约 1.0~1.2x"
    guarantee_note = "误差上界 ~ ε·log n·Σ|x|，不保证逐位一致"
    range_note = "任意 float64 输入"

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[float] = []

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        self.parts.append(float(x.sum()))  # numpy 分区内本身就是 pairwise

    def _merge_finite(self, other: "PairwiseState") -> None:
        self.parts.extend(other.parts)

    @staticmethod
    def _pair_reduce(vals: List[float]) -> float:
        if len(vals) == 1:
            return vals[0]
        mid = len(vals) // 2
        return PairwiseState._pair_reduce(vals[:mid]) + \
            PairwiseState._pair_reduce(vals[mid:])

    def _finalize_finite(self) -> float:
        return self._pair_reduce(self.parts)


# --------------------------------------------------------------------------
# 3. 补偿求和（Neumaier/Kahan 变体）
# --------------------------------------------------------------------------

class KahanState(SumState):
    """Neumaier 补偿求和：保存总和 s 与补偿项 c，合并时同样做补偿。

    绝大多数数据下跨顺序逐位一致，但 c 本身也是 float64，极端抵消时
    仍可能残留 1~2 ULP 的顺序差异——它是“误差极小”而不是“数学保证”。
    """

    method = "kahan"
    label = "补偿求和 Kahan/Neumaier（误差 ~ε，非数学保证）"
    deterministic = False
    cost_note = "逐元素标量循环（向量化会让补偿失效），约 20~50x，O(1) 额外内存"
    guarantee_note = "误差界 ~ ε·Σ|x|（与 n 无关）；极端抵消下仍可能差 1 ULP"
    range_note = "任意 float64 输入"

    def __init__(self) -> None:
        super().__init__()
        self.s = 0.0
        self.c = 0.0

    def _compensated_add(self, b: float) -> None:
        a = self.s
        t = a + b
        if abs(a) >= abs(b):
            self.c += (a - t) + b
        else:
            self.c += (b - t) + a
        self.s = t

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        # 必须逐元素：块和在进入补偿前就已舍入，块级补偿看不到块内丢失的小额，
        # 极端相消数据上会退化到普通朴素求和
        for v in x.tolist():
            self._compensated_add(float(v))

    def _merge_finite(self, other: "KahanState") -> None:
        self._compensated_add(other.s)
        self._compensated_add(other.c)

    def _finalize_finite(self) -> float:
        return self.s + self.c


# --------------------------------------------------------------------------
# 4. 精确部分和（math.fsum 同款算法，可并行合并）
# --------------------------------------------------------------------------

def _fsum_lmm(partials: List[float], x: float) -> None:
    """CPython math.fsum 的部分和维护算法（long-multiply 简化版移植）。

    partials 中每个元素都是精确的 float（无舍入残留），任何子集的
    math.fsum 都等于原始输入的精确总和（最后一次舍入除外）。
    """
    i = 0
    for y in partials:
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        lo = y - (hi - x)
        if lo:
            partials[i] = lo
            i += 1
        x = hi
    del partials[i:]
    partials.append(x)


class FSumState(SumState):
    """精确部分和求和（math.fsum 同源算法）。

    每个分区维护一个“精确部分和”列表；分区合并就是把两边的部分和
    重新过一遍 LMM——因为部分和本身都是精确 float，合并不丢任何信息。
    最终结果只在最后一步做一次舍入，与切分/合并顺序完全无关。
    """

    method = "fsum"
    label = "精确部分和 fsum（顺序无关，一次最终舍入）"
    deterministic = True
    exact_for_scale = True  # 对 float64 能表达的输入，给出正确舍入的总和
    cost_note = "纯 Python 部分和循环，约 3~8x；部分和列表 O(有效量级数)"
    guarantee_note = "结果等于精确实数总和的最近 float64，任意顺序逐位相同"
    range_note = "任意 float64 输入；不改变输入本身的表示误差"

    def __init__(self) -> None:
        super().__init__()
        self.partials: List[float] = []

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        for v in x.tolist():  # 需要逐个值进入部分和算法，不能用块和代替
            _fsum_lmm(self.partials, v)

    def _merge_finite(self, other: "FSumState") -> None:
        for p in other.partials:
            _fsum_lmm(self.partials, p)

    def _finalize_finite(self) -> float:
        return math.fsum(self.partials)  # 与部分和算法自洽的最终舍入


class FSumBlockState(FSumState):
    """工程化混合方案：全局固定块向量化 + 块间精确部分和。

    关键不变量：喂给 numpy 的块永远是全数组 [k*BLOCK, (k+1)*BLOCK) 的
    物理切片——即使分区边界落在块中间，状态机也会先把“本分区起始的
    残块”暂存、等相邻分区到达后拼成完整全局块再做 numpy 求和。
    因此每个全局块的块和与分区切分完全无关，随机只作用在块的分组与
    归并顺序上，而块间走精确部分和，最终逐位确定。

    块内残留一次 ε 级 pairwise 舍入（不随 n 放大），块间合并完全精确，
    是“向量化速度”与“顺序无关”之间的实用折中，生产引擎推荐档。
    """

    method = "fsum_block"
    label = "分块 fsum（固定块向量化+精确合并，推荐）"
    deterministic = True
    exact_for_scale = False  # 块内有一次 ε 级舍入，不保证等于十进制真值
    cost_note = "向量化块循环，约 1.5~3x；部分和列表 O(块数)，每分区 O(1) 残块缓存"
    guarantee_note = "任意切分/合并顺序逐位相同；块内残留 ~ε 级误差，不随 n 放大"
    range_note = "任意 float64 输入；需要十进制级精确时用 fixed_bigint"

    def __init__(self) -> None:
        super().__init__()
        # 全局块号 -> [(残片全局起始下标, 残片)]；拼接前按起始下标排序，
        # 保证块内容是规范的全局顺序，与分区合并顺序无关
        self._pending: Dict[int, List[tuple]] = {}

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        if global_lo is None:
            raise PrecisionExceeded(
                "fsum_block 依赖全局块对齐，只能用于连续分区的全表聚合；"
                "分组/窗口等非连续场景请改用 fsum 或 fixed_bigint。")
        lo0 = int(global_lo)
        hi0 = lo0 + x.shape[0]
        first_block = lo0 // BLOCK
        last_block = (hi0 - 1) // BLOCK
        for b in range(first_block, last_block + 1):
            g_lo, g_hi = b * BLOCK, (b + 1) * BLOCK
            s = max(lo0, g_lo) - lo0
            e = min(hi0, g_hi) - lo0
            frag = np.ascontiguousarray(x[s:e])
            start = max(lo0, g_lo)
            if lo0 == g_lo and hi0 == g_hi:
                self._emit_block(frag)          # 完整全局块，直接发射
            else:
                self._pending.setdefault(b, []).append((start, frag))
                self._try_flush(b)

    def _try_flush(self, b: int) -> None:
        frags = self._pending.get(b)
        if not frags:
            return
        if sum(f.shape[0] for _, f in frags) == BLOCK:
            # 残片按全局位置排序后拼齐：numpy 只在规范物理块上求和
            frags.sort(key=lambda t: t[0])
            full = (frags[0][1] if len(frags) == 1
                    else np.concatenate([f for _, f in frags]))
            del self._pending[b]
            self._emit_block(full)

    def _emit_block(self, full_block: np.ndarray) -> None:
        _fsum_lmm(self.partials, float(full_block.sum(dtype=np.float64)))

    def _merge_finite(self, other: "FSumBlockState") -> None:
        for b, frags in other._pending.items():
            self._pending.setdefault(b, []).extend(frags)
        for b in list(self._pending.keys()):
            self._try_flush(b)
        for p in other.partials:
            _fsum_lmm(self.partials, p)

    def _finalize_finite(self) -> float:
        # 所有分区合并后，末尾不足一个 BLOCK 的全局残块必然拼齐
        for b, frags in list(self._pending.items()):
            frags.sort(key=lambda t: t[0])
            tail = (frags[0][1] if len(frags) == 1
                    else np.concatenate([f for _, f in frags]))
            _fsum_lmm(self.partials, float(tail.sum(dtype=np.float64)))
        self._pending.clear()
        return math.fsum(self.partials)


# --------------------------------------------------------------------------
# 5. 定点求和：int64 累加器（带越界守卫）
# --------------------------------------------------------------------------

class _FixedBase(SumState):
    """定点公共逻辑：按 10**scale 转整数，超出 scale 的输入直接拒绝。"""

    def __init__(self, scale: int = 6) -> None:
        super().__init__()
        self.scale = scale
        self.factor = 10 ** scale
        self.total = self._zero()

    @staticmethod
    def _zero():
        raise NotImplementedError

    def _scaled_floats(self, x: np.ndarray) -> np.ndarray:
        """按 scale 取整，并守卫小数位越界；返回 float64 整数数组。

        判据：round(x*factor)/factor 与 x 的差不超过 x 的 1 ULP。
        - 合法的 scale 位内十进制值：乘除的表示误差都在 1 ULP 内，通过；
        - 超 scale 的值：取整至少丢掉 0.1 个定点单位，远超 1 ULP，拒绝；
        - 极少数深小数值在该量级上 float64 已无法区分第 scale+1 位
          （ULP 比丢弃量还大），此时取整不损失 float 中实际存在的信息，
          予以通过——这是 float64 输入的固有极限，不是守卫漏洞。
        """
        scaled = x * self.factor
        rounded = np.round(scaled)
        back = rounded / self.factor
        tol = np.maximum(np.abs(x) * np.finfo(np.float64).eps,
                         np.finfo(np.float64).tiny)
        bad = np.abs(back - x) > tol
        if bad.any():
            examples = x[bad][:5]
            raise PrecisionExceeded(
                f"输入含超过 scale={self.scale} 位小数的值（示例 {examples.tolist()}）；"
                "提高 scale，或改用 fixed_bigint / decimal / fsum 方案。"
            )
        return rounded

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        rounded = self._scaled_floats(x)
        for blk in _iter_blocks(rounded):
            self._absorb_block(blk)

    def _absorb_block(self, block_rounded: np.ndarray) -> None:
        raise NotImplementedError

    def _merge_finite(self, other: "_FixedBase") -> None:
        if other.scale != self.scale:
            raise PrecisionExceeded(
                f"定点 scale 不一致：{self.scale} vs {other.scale}，无法合并")
        self._absorb_python_int(other.total)

    def _finalize_finite(self) -> float:
        # 整数除法转 float，全程只有这一次浮点舍入
        return int(self.total) / self.factor


class Fixed64State(_FixedBase):
    """定点 int64：金额乘 10^scale 后用 64 位整数累加，每次合并检查溢出。"""

    method = "fixed64"
    label = "定点 int64（顺序无关，溢出即报错）"
    deterministic = True
    exact_for_scale = True
    cost_note = "一次取整+向量化整数加，约 1.1~1.5x，O(1) 额外内存"
    guarantee_note = "scale 位小数内的十进制数据，结果与十进制真值逐位相同"
    range_note = "累加和必须落在 int64（±9.22e18）；超出抛 OverflowError"

    def __init__(self, scale: int = 6) -> None:
        super().__init__(scale)

    @staticmethod
    def _zero() -> int:
        return 0

    def _absorb_block(self, block_rounded: np.ndarray) -> None:
        # 转换前先做 int64 范围守卫，拒绝依赖 NumPy 版本相关的溢出回绕行为
        if block_rounded.size and (
                np.max(block_rounded) > MAX_INT64
                or np.min(block_rounded) < MIN_INT64):
            raise OverflowError(
                "定点 int64：单值已超出 int64 范围；请改用 fixed_bigint。")
        chunk_sum = int(block_rounded.astype(np.int64).sum(dtype=np.int64))
        t = self.total
        if t > 0 and chunk_sum > MAX_INT64 - t:
            raise OverflowError(
                "定点 int64 累加溢出；请改用 fixed_bigint（任意位宽 Python 整数）。")
        if t < 0 and chunk_sum < MIN_INT64 - t:
            raise OverflowError(
                "定点 int64 累加溢出（负向）；请改用 fixed_bigint。")
        self.total = t + chunk_sum

    def _absorb_python_int(self, v: int) -> None:
        t = self.total
        if t > 0 and v > MAX_INT64 - t or t < 0 and v < MIN_INT64 - t:
            raise OverflowError(
                "定点 int64 分区合并溢出；请改用 fixed_bigint。")
        self.total = t + v


# --------------------------------------------------------------------------
# 6. 定点求和：任意位宽 Python 整数
# --------------------------------------------------------------------------

class FixedBigintState(_FixedBase):
    """定点大整数：与 fixed64 同样的取整守卫，累加器换成 Python 任意精度 int。"""

    method = "fixed_bigint"
    label = "定点大整数（顺序无关，永不溢出）"
    deterministic = True
    exact_for_scale = True
    cost_note = "块和转 Python int，约 1.5~3x；累加器位数随总和缓慢增长"
    guarantee_note = "scale 位小数内与十进制真值逐位相同，无溢出上限"
    range_note = "仅受内存限制；小数位超 scale 同样拒绝"

    @staticmethod
    def _zero() -> int:
        return 0

    def _absorb_block(self, block_rounded: np.ndarray) -> None:
        # 走 Python 任意精度 int（object 累加），不经 int64，永不回绕
        self.total += sum(int(v) for v in block_rounded.tolist())

    def _absorb_python_int(self, v: int) -> None:
        self.total += v  # Python 大整数，任意位宽


# --------------------------------------------------------------------------
# 7. 宽精度 Decimal 累加器
# --------------------------------------------------------------------------

class DecimalState(SumState):
    """Decimal 宽精度累加器，分区保存 Decimal 部分和，可任意顺序合并。"""

    method = "decimal"
    label = "Decimal 宽精度（顺序无关，精度不足即报错）"
    deterministic = True
    exact_for_scale = True
    cost_note = "纯 Python 十进制运算，约 30~100x；每值一个 Decimal 对象"
    guarantee_note = "上下文精度内结果精确；finalize 做双精度复核，不足即报错"
    range_note = "有效数字超过 context.precision 时抛 PrecisionExceeded"

    def __init__(self, precision: int = 50, check_precision: bool = True) -> None:
        super().__init__()
        self.precision = precision
        self.check_precision = check_precision
        # 保存逐个精确 Decimal（str(float) 是最短往返文本），
        # 不在 update 阶段做有限精度求和，避免部分和提前被截断
        self.values: List[Decimal] = []

    def _update_finite(self, x: np.ndarray,
                       global_lo: Optional[int] = None) -> None:
        self.values.extend(Decimal(str(v)) for v in x.tolist())

    def _merge_finite(self, other: "DecimalState") -> None:
        self.values.extend(other.values)

    def _finalize_finite(self) -> float:
        with localcontext() as ctx:
            ctx.prec = self.precision
            total = sum(self.values, Decimal(0))
            if self.check_precision:
                # 双精度复核：提高 20 位重算，结果变化说明原精度被吃满，
                # 当前答案可能已被舍入——拒绝静默输出
                ctx.prec = self.precision + 20
                verify = sum(self.values, Decimal(0))
                if total != verify:
                    raise PrecisionExceeded(
                        f"Decimal 精度 {self.precision} 位不足以容纳精确结果，"
                        "请提高 precision 或改用 fixed_bigint。")
        return float(total)


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------

@dataclass
class MethodSpec:
    key: str
    cls: Type[SumState]
    label: str
    deterministic: bool
    exact: bool
    cost_note: str
    guarantee_note: str
    range_note: str
    needs_scale: bool = False


def _spec(state: SumState) -> MethodSpec:
    return MethodSpec(
        key=state.method, label=state.label, cls=type(state),
        deterministic=state.deterministic, exact=state.exact_for_scale,
        cost_note=state.cost_note, guarantee_note=state.guarantee_note,
        range_note=state.range_note,
        needs_scale=isinstance(state, _FixedBase),
    )


_SPECS: Dict[str, MethodSpec] = {}


def _register(state: SumState) -> None:
    s = _spec(state)
    _SPECS[s.key] = s


for _s in (NaiveState(), PairwiseState(), KahanState(), FSumState(),
           FSumBlockState(), Fixed64State(), FixedBigintState(), DecimalState()):
    _register(_s)

SUM_METHODS: Dict[str, MethodSpec] = dict(_SPECS)
SUM_METHOD_KEYS: List[str] = list(_SPECS.keys())


def make_sum_state(method: str, scale: Optional[int] = None) -> SumState:
    """工厂方法。定点方案需要 scale（默认取 6，覆盖分/厘级金额）。"""
    if method not in _SPECS:
        raise KeyError(f"未知求和方法 {method!r}，可选：{SUM_METHOD_KEYS}")
    if method == "fixed64":
        return Fixed64State(6 if scale is None else scale)
    if method == "fixed_bigint":
        return FixedBigintState(6 if scale is None else scale)
    return _SPECS[method].cls()
