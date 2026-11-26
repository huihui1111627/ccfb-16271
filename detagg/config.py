"""确定性开关。

引擎接入方式（概念示例）：

    cfg = get_config()
    with deterministic_aggregation(enabled=True, sum_method="fixed_bigint",
                                   var_method="exact_fixed", scale=6):
        # 块内的聚合执行器会读到本线程的配置，对账类查询走精确路径，
        # 普通分析查询不包这个上下文，照旧用引擎原生浮点聚合。
        result = run_sum(cfg.sum_method, values, order, scale=cfg.scale, ...)

开关只影响“查询规划期选择哪套状态机”，执行器、切分、合并逻辑完全共用，
因此精确方案与普通方案的运维路径一致，不需要两套引擎。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional


@dataclass
class AggregationConfig:
    enabled: bool = False              # 总开关：False 时一切照旧（原生浮点）
    sum_method: str = "fixed_bigint"   # naive / pairwise / kahan / fsum /
                                       # fixed64 / fixed_bigint / decimal
    var_method: str = "exact_fixed"    # naive_sumsq / welford_chan / exact_fixed
    scale: int = 6                     # 定点小数位（金额到 1e-6）
    window_method: str = "fixed_bigint"
    fail_on_overflow: bool = True      # 定点溢出/越界时报错，而不是静默降级
    decimal_precision: int = 50

    def describe(self) -> str:
        if not self.enabled:
            return "确定性开关关闭：使用引擎原生浮点聚合（顺序敏感、照旧）"
        return (f"确定性开关开启：sum={self.sum_method}, var={self.var_method}, "
                f"scale={self.scale}, window={self.window_method}, "
                f"溢出策略={'报错' if self.fail_on_overflow else '自动降级 fsum'}")


_GLOBAL = AggregationConfig()
_LOCK = threading.RLock()
_LOCAL = threading.local()


def get_config() -> AggregationConfig:
    """取当前生效配置（线程内覆盖优先，否则全局）。"""
    cfg = getattr(_LOCAL, "config", None)
    if cfg is not None:
        return cfg
    return _GLOBAL


def set_global_config(cfg: AggregationConfig) -> None:
    global _GLOBAL
    with _LOCK:
        _GLOBAL = cfg


@contextmanager
def deterministic_aggregation(enabled: bool = True,
                              sum_method: Optional[str] = None,
                              var_method: Optional[str] = None,
                              scale: Optional[int] = None,
                              fail_on_overflow: Optional[bool] = None):
    """查询级开关：只在 with 块内生效，退出自动还原。"""
    base = get_config()
    local = AggregationConfig(
        enabled=enabled,
        sum_method=sum_method or base.sum_method,
        var_method=var_method or base.var_method,
        scale=base.scale if scale is None else scale,
        fail_on_overflow=base.fail_on_overflow
        if fail_on_overflow is None else fail_on_overflow,
        window_method=base.window_method,
        decimal_precision=base.decimal_precision,
    )
    prev = getattr(_LOCAL, "config", None)
    _LOCAL.config = local
    try:
        yield local
    finally:
        _LOCAL.config = prev
