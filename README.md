# 确定性聚合执行器（detagg）

复现并治理**并行浮点聚合的非确定性**：同一条汇总 SQL，因为引擎每次对数据的
分区切分和分区合并顺序不同，float64 求和/方差的末位舍入不同，对账系统逐位
比对时误报“数据被改”。本项目用 Python + NumPy 造了一个可并行的聚合执行器，
把问题量化复现，并给出可开关的顺序无关方案。

## 一分钟看懂结论

| 现象（实测，300 次随机切分/合并） | 数据 | 结果 |
|---|---|---|
| 朴素求和末位跳动 | 正常金额 n=20万 | 出现 4~7 种不同结果，差 2~3 ULP |
| 大数加小数 | 1e12 级总额 + 小额流水 | 不同顺序最大差 **1.5 元** |
| 正负相消 | 借贷成对、真实净值几十元 | 60 种结果，相对误差被放大到 O(1) |
| 朴素矩法方差 | 均值 1e8、噪声 0.01 | 算出**负方差**（直接失效） |
| 定点/fsum/Decimal 方案 | 全部数据集 | **每次逐位相同**，守卫触发时报错而非静默出错 |

## 目录结构

```
detagg/
  datasets.py   6 类数据集（正常/量级悬殊/正负相消/深小数/大数加小数/特殊值）
  truth.py      60 位 Decimal 十进制真值（不经过 float64）
  sums.py       8 种求和状态机（均可分区 merge）
  stats.py      4 种方差/标准差状态机 + 平均值
  executor.py   随机分区 + 随机合并树，顺序模拟 / 线程池真并行
  window.py     窗口累计求和（顺序 vs 并行随机调度 vs 确定性）
  groupby.py    分组聚合
  config.py     确定性开关（查询级上下文，默认关闭、其余照旧）
  bench.py      数百次随机顺序的确定性/误差/速度/内存评测
webapp/         Flask 页面：选数据和方法、一键跑数百次、看分布和真值差距
tests/          28 个 pytest 测试
run_demo.py     命令行演示
```

## 运行

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 28 个测试
python run_demo.py                  # 命令行快速演示
python run_demo.py --full           # 完整规模 + 300 次随机顺序
python -m webapp.app                # 页面：http://127.0.0.1:5050
```

## 求和方案：代价、保证、适用范围

| 方法 | 顺序无关 | 等于十进制真值 | 相对耗时 | 额外内存 | 保证与边界 |
|---|---|---|---|---|---|
| `naive` 朴素累加 | ✗ | ✗ | 1.0x（基准） | O(1) | 引擎默认行为，误差上界 n·ε·Σ\|x\| |
| `pairwise` 成对求和 | ✗ | ✗ | ~1.0x | O(分区数) | 误差降到 ε·log n，但合并树随机仍跳动 |
| `kahan` Neumaier 补偿 | ✗（非数学保证） | 多数数据贴合 | ~20~50x | O(1) | 必须**逐元素**补偿；块和会让补偿失效 |
| `fsum` 精确部分和 | ✓ | ✗* | ~40~250x | O(有效量级数) | float 输入的精确总和、一次最终舍入 |
| **`fsum_block` 分块 fsum** | ✓ | ✗* | **~0.6~3x** | O(块数) | 全局固定块向量化+精确合并，**生产推荐** |
| `fixed64` 定点 int64 | ✓ | ✓** | ~2~3x | O(1) | 累加和超 int64（±9.22e18）即报错 |
| **`fixed_bigint` 定点大整数** | ✓ | ✓** | ~6~30x | O(累加器位数) | 永不溢出，金额类对账的主选 |
| `decimal` 宽精度 Decimal | ✓ | ✓ | ~30~200x | O(n) 对象 | 精度不足即报错；最慢，留给审计/仲裁 |

\* `fsum` 系列消除的是**累加顺序**误差，不消除数据进 float64 时的十进制→二进制
表示误差。正负相消会把这个输入侧误差放大，所以它们逐位可复现，但不等于十进制真值。
\*\* 定点方案在声明 `scale` 位小数内与十进制真值逐位相同；超出 scale 直接抛
`PrecisionExceeded`，绝不静默截断。

### 超出保证范围怎么办（守卫，而非静默错误）

- 小数位超过 `scale` → `PrecisionExceeded`：提高 scale，或换 `fixed_bigint`/`decimal`。
- int64 累加溢出 → `OverflowError`：自动升级到 `fixed_bigint`（Python 任意精度整数）。
- Decimal 有效数字超上下文精度 → `PrecisionExceeded`：提高 precision。
- `fsum_block` 用在非连续的分组场景 → `PrecisionExceeded`：分组用 `fsum`/定点。

## 方差 / 标准差

| 方法 | 顺序无关 | 说明 |
|---|---|---|
| `naive_sumsq` E[x²]-E[x]² | ✗ | 单遍矩法，灾难抵消时给出负方差，刻意保留作反例 |
| `welford_chan` | ✗ | Welford 在线均值 + Chan 分区合并公式，数值稳定但合并树随机仍跳动 |
| `exact_fixed` 定点大整数 | ✓ | Σx、Σx² 整数域精确；**计算**顺序无关，但不消除 float 存储误差（量级悬殊时差真值 1~2 ULP） |
| `decimal_var` | ✓ | 逐值 Decimal，消除计算误差并贴合十进制真值，30~100x |

平均值由确定的和除以计数得到，一次除法、顺序无关。

## 两个容易被忽略的坑（本项目都踩给你看）

1. **Kahan 不能吃块和**。如果先用 numpy 对 4096 个值求和再做补偿，块内已经
   发生的舍入补偿项根本看不到，极端相消数据上退化到朴素求和（实测误差 6.5e-6）。
   逐元素补偿后误差降到输入固有的 3.5e-7。
2. **“计算精确”不等于“结果等于十进制真值”**。定点方差的整数运算完全精确，
   但输入 float64 本身带表示误差，`Σx²` 把它放大后仍可能差 1 ULP。要从源头
   消除，只能让引擎直接吃十进制文本（Decimal 方案），或在入库时就用定点列。

## 确定性开关

默认关闭，普通分析查询完全照旧；只给对账/审计类查询包一层上下文：

```python
from detagg.config import deterministic_aggregation
from detagg.executor import run_sum, random_order

with deterministic_aggregation(enabled=True, sum_method="fixed_bigint",
                               var_method="exact_fixed", scale=6):
    order = random_order(n, seed=...)
    total = run_sum("fixed_bigint", values, order, scale=6, null_mask=nulls)
```

开关只在查询规划期选择状态机，执行器、切分、合并路径与普通查询完全共用，
不需要两套引擎。
