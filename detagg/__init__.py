"""detagg — 可复现非确定性、可切换确定性方案的并行聚合执行器。

模块一览
--------
- datasets: 造数（正常金额 / 量级悬殊 / 正负相消 / 深小数 / 特殊值）
- truth:    高精度十进制真值
- sums:     七种求和状态机（朴素、成对、Kahan、fsum、定点 int64、定点大整数、Decimal）
- stats:    方差/标准差（朴素矩、Welford-Chan 并行合并、精确定点）
- executor: 随机分区 + 随机合并顺序（顺序 / 线程池）执行器
- window:   累计求和（顺序 vs 并行前缀）
- groupby:  分组聚合
- config:   确定性开关
- bench:    数百次随机顺序的确定性 / 误差 / 速度 / 内存评测
"""

from .config import AggregationConfig, deterministic_aggregation, get_config
from .datasets import Dataset, build_dataset, DATASET_BUILDERS
from .executor import RunOrder, random_order, run_sum, run_variance
from .sums import SUM_METHODS, make_sum_state
from .stats import VAR_METHODS, make_var_state

__all__ = [
    "AggregationConfig",
    "deterministic_aggregation",
    "get_config",
    "Dataset",
    "build_dataset",
    "DATASET_BUILDERS",
    "RunOrder",
    "random_order",
    "run_sum",
    "run_variance",
    "SUM_METHODS",
    "make_sum_state",
    "VAR_METHODS",
    "make_var_state",
]
