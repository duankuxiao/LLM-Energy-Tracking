<p align="center">
  <img src="logo.png" alt="项目标志" width="180"/>
</p>

# 全球 AI 数据中心能耗与碳排放核算

[English README](README.md)

本仓库是全球人工智能（AI）数据中心用电与运行碳排放研究的配套代码。模型综合数据中心容量路径、国家分配、年度 AI 容量因子、公开 GPU 集群负荷轨迹、分部件功率模型、PUE，以及年度或小时级电网碳强度。

仓库术语与原稿保持一致：核心比较连接“**年度 CPU 模型**”和“**小时级工作负载解析模型**”，并通过有序 bridge 依次引入分部件功率、轨迹派生的平均工作负载、小时负荷变化和小时碳强度匹配。这里的端点和中间步骤均是模型配置；小时级端点不被视为实测真值。

> **研究范围。** 本研究核算电力消耗、运行负荷和运行碳排放，不开展用水量或 WUE 分析。

## 核算框架

### 两个端点模型

| 端点 | 功率与负荷表征 | 电网碳强度 | 用途 |
| --- | --- | --- | --- |
| 年度 CPU 模型 | 汇总的 CPU 风格利用率—功率方程，采用固定任务比例和利用率假设 | 年度均值 | 传统年度核算基线 |
| 小时级工作负载解析模型 | 轨迹驱动的 CPU、GPU、内存、存储和 IT 风扇分部件功率 | 小时级 | 年度结果与运行峰值、爬坡指标的参考配置 |

两个端点共用相同的需求路径、国家容量边界、年度 AI 容量因子、PUE 输入和电力系统路径。由于二者之间同时改变了多项表征选择，代码使用有序 bridge 显式分解差异。

### 有序 bridge

图 3 使用五个连续配置。每项效应均定义为后一配置减去前一配置。

| 顺序 | 源数据中的配置标识 | 新引入的变化 |
| --- | --- | --- |
| 1 | `annual_cpu_baseline` | 年度 CPU 起始基线 |
| 2 | `component_power_with_legacy_load` | 在保留年度模型任务比例和利用率假设的同时，将汇总功率替换为分部件功率 |
| 3 | `flat_trace_annual_carbon` | 使用轨迹派生的平均任务—资源负荷，保持负荷平坦并继续使用年度碳强度 |
| 4 | `hourly_trace_annual_carbon` | 恢复完整小时负荷形状，继续使用年度碳强度 |
| 5 | `hourly_trace_hourly_carbon` | 将小时用电与小时电网碳强度匹配 |

四项效应标识为 `power_configuration`、`mean_workload_representation`、`load_temporal_variability` 和 `hourly_carbon_matching`。代码检查四项效应之和是否等于“小时端点减年度基线”的直接差值。该分解依赖替换顺序，不应解释为与顺序无关的因果归因。

## 审稿导航

| 原稿内容 | 主要实现 | 供审稿人核对的输出 |
| --- | --- | --- |
| 需求路径、国家分配、任务与部件能耗（图 1） | [`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py)、[`scripts/figure1_data.py`](scripts/figure1_data.py) | [`results/figure1_data.xlsx`](results/figure1_data.xlsx) |
| 电网路径与减排量（图 2） | [`dataset/Factors.py`](dataset/Factors.py)、[`scripts/figure2_data.py`](scripts/figure2_data.py) | [`results/figure2_data.xlsx`](results/figure2_data.xlsx) |
| 年度—小时端点比较、有序 bridge、峰值和爬坡（图 3） | [`core/annual_cpu_model.py`](core/annual_cpu_model.py)、[`core/hourly_workload_resolved_model.py`](core/hourly_workload_resolved_model.py)、[`scripts/figure3_data.py`](scripts/figure3_data.py) | [`results/figure3_data.xlsx`](results/figure3_data.xlsx) |
| 单因素敏感性分析（图 4） | [`scripts/figure4_data.py`](scripts/figure4_data.py) | [`results/figure4_data.xlsx`](results/figure4_data.xlsx) |
| MLPerf 功率诊断与轨迹边界审查（补充图 S1） | [`scripts/validate_energy_model.py`](scripts/validate_energy_model.py)、[`dataset/validation/`](dataset/validation/) | 运行后写入 `results/model_validation/` |

## 研究范围与情景

### 国家范围

默认模型边界包含 24 个国家：

`USA`、`China`、`Japan`、`France`、`India`、`Singapore`、`Canada`、`Germany`、`United_Kingdom`、`Australia`、`Italy`、`South_Korea`、`South_Africa`、`Ireland`、`UAE`、`Brazil`、`Israel`、`Netherlands`、`Spain`、`Sweden`、`Belgium`、`Norway`、`Poland` 和 `Switzerland`。

这些标识是 [`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) 与 [`dataset/Factors.py`](dataset/Factors.py) 使用的规范键。当前国家分配没有表示剩余的全球容量份额，因此输出应表述为“模型覆盖国家合计”，而不是完整全球总量。

### AI 数据中心需求路径

| 标识 | 含义 |
| --- | --- |
| `Base` | 基准容量增长路径 |
| `Lift-Off` | 更快需求增长路径 |
| `High Efficiency` | 效率改善路径 |
| `Headwinds` | 受约束增长路径 |

向代码传入情景名称时必须使用表中的准确拼写。

### 电力系统路径

| 标识 | 含义 |
| --- | --- |
| `CP` | 当前政策（Current Policies） |
| `NDC` | 国家自主贡献（Nationally Determined Contributions） |
| `NZ` | 净零（Net Zero） |

## 数据与代码可用性

| 资源 | 仓库状态 | 获取方式或来源 |
| --- | --- | --- |
| 两个端点模型与有序 bridge 源代码 | 已包含 | 本仓库 |
| 年度碳强度和 PUE 输入 | 已包含 | [`dataset/Factors.py`](dataset/Factors.py) |
| 容量与需求路径输入 | 已包含 | [`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) |
| 外部诊断输入 | 已包含 | [`dataset/validation/`](dataset/validation/) |
| 图 1—4 源数据工作簿 | 已包含 | [`results/`](results/) |
| Electricity Maps 小时数据及其派生情景文件 | **不公开分发** | 需从 Electricity Maps 合法取得 |
| Alibaba GPU Pod 和服务器轨迹 | 未打包 | 需从 Alibaba Cluster Trace Program 获取 |
| IEA 数据附件 | 不重新分发 | 需从 International Energy Agency 获取 |

年度 CPU 模型可使用仓库内的年度输入独立运行。完整运行小时级工作负载解析模型和有序 bridge，需要准备下述外部负荷轨迹与小时碳强度文件。仓库内已有论文图表源数据工作簿，审稿人无需获得受限数据即可核对已报告数值。

### 容量与增长输入

[`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) 存放 2025—2030 年四类需求路径下的总容量、IT 容量和非 IT 容量，以及固定国家份额和年度 AI 容量因子。

默认 AI 容量校准依据 IEA 报告 [Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai) 讨论的 AI 专用数据中心用电路径。校准端点与中间年份插值假设记录在源代码中。国家份额和中间情景值属于模型输入或派生参数，不应理解为原始数据提供方直接发布的逐国预测。

### GPU 工作负载与服务器轨迹

小时级工作负载解析模型读取两类 Alibaba Serverless Infrastructure 数据：

- `dataset/asi_opensource_pod_hourly/`：GPU Pod 小时记录；
- `dataset/asi_opensource_server_hourly/`：服务器资源清单小时记录。

轨迹来自 [Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata)，系统背景参见 [Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba](https://www.usenix.org/conference/osdi26/presentation/li-suyi)。

能耗边界保留所有活跃 Pod，包括 GPU 请求为零的 CPU-only Pod。原稿中的任务结构审查使用独立且更严格的过滤条件：论文时间窗口内的已分类任务必须具有 GPU 显存请求和有效 GPU 使用小时。`training` 保持为训练，在线与离线推理合并为推理，`dev` 与明确标注的 `other` 合并为其他，空值、`unknown` 和异常标签列为 `unclassified`。未分类活动计入总能耗，但不进入已分类任务比例的分母。

公开轨迹缺少完整存储活动和主机内存容量，因此实现使用空闲存储功率和 CPU 加权的内存活动代理。这些假设记录在返回的 `trace_resource_capacity` 表中。

### 年度碳强度与 PUE

[`dataset/Factors.py`](dataset/Factors.py) 提供 24 国在 `CP`、`NDC` 和 `NZ` 路径下的年度电网碳强度，以及按国家、年份和需求路径区分的 PUE。碳强度单位为 `kg CO₂/MWh`。

年度 CPU 模型直接使用年度因子；有序 bridge 的年度碳强度配置也使用这些因子。当小时文件缺失且允许回退时，同一年度因子作为回退值。

### 小时级电网碳强度

小时模型预期在 `dataset/EM-CPNDCNZ/` 下读取 360 个本地 CSV：24 个国家 × 3 类电力系统路径 × 2026—2030 五年。文件包含 UTC 时间戳、直接碳强度和生命周期碳强度列。

研究使用经授权的 2025 年 Electricity Maps 数据作为相对小时形状，并缩放至 [`dataset/Factors.py`](dataset/Factors.py) 中的年度路径。处理后的小时文件不公开分发。合资格研究人员需通过 Electricity Maps [数据页面](https://www.electricitymaps.com/data)或[学术访问说明](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)独立申请，并遵守最新许可条款。

## 项目结构

```text
.
├── core/
│   ├── annual_cpu_model.py
│   ├── hourly_workload_resolved_model.py
│   ├── past_research_data_center_energy_carbon_model.py
│   └── task_model.py
├── dataset/
│   ├── Factors.py
│   ├── Installed_capacity_data.py
│   ├── EM-CPNDCNZ/                    # 本地授权小时输入
│   ├── validation/
│   ├── asi_opensource_pod_hourly/     # 外部本地输入
│   └── asi_opensource_server_hourly/  # 外部本地输入
├── results/                           # 生成结果与论文图源数据
├── scripts/
│   ├── figure1_data.py
│   ├── figure2_data.py
│   ├── figure3_data.py
│   ├── figure4_data.py
│   └── validate_energy_model.py
├── run.py                             # 两端点比较入口
├── LICENSE
├── README.md
└── README.zh-CN.md
```

## 环境与依赖

投稿复现包使用 Python 3.12.3，并固定以下版本：

- `numpy==2.1.3`
- `pandas==2.2.3`
- `pyarrow==17.0.0`
- `openpyxl==3.1.5`
- `matplotlib==3.9.2`

在仓库根目录创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==2.1.3 pandas==2.2.3 pyarrow==17.0.0 openpyxl==3.1.5 matplotlib==3.9.2
```

Linux 或 macOS：

```bash
source .venv/bin/activate
```

## 外部数据准备

下列受限或外部本地输入由 `.gitignore` 排除：

```text
dataset/EM-CPNDCNZ.zip
dataset/EM-CPNDCNZ/
dataset/asi_opensource_pod_hourly/
dataset/asi_opensource_server_hourly/
```

预期目录：

```text
dataset/
├── EM-CPNDCNZ/
│   ├── USA/
│   │   ├── ...-US-CP-2026-hourly.csv
│   │   ├── ...-US-NDC-2026-hourly.csv
│   │   └── ...-US-NZ-2026-hourly.csv
│   └── Great Britain/
│       └── ...-GB-CP-2026-hourly.csv
├── asi_opensource_pod_hourly/
│   └── day=0/hour=00/part-000.parquet
└── asi_opensource_server_hourly/
    └── day=0/hour=00/part-000.parquet
```

模型键 `United_Kingdom` 对应小时数据目录 `Great Britain`。

## 运行分析

所有命令均从仓库根目录执行。

### 比较两个端点模型

```powershell
python run.py --policy CP --scenarios Base --year-start 2026 --years 5
```

入口程序读取一次工作负载轨迹，运行年度 CPU 端点和小时级工作负载解析端点，并将结果写入 `results/model_comparison/`。可同时指定多个需求路径：

```powershell
python run.py --policy NDC --scenarios Base "Lift-Off" "High Efficiency"
```

快速执行检查可限制国家和轨迹小时数：

```powershell
python run.py --countries USA --year-start 2026 --years 1 --max-intervals 24
```

`max_intervals` 会改变轨迹采样基础，仅用于 smoke test 和调试；生成论文结果时必须省略。

### 生成论文图源数据

```powershell
python scripts/figure1_data.py
python scripts/figure2_data.py
python scripts/figure3_data.py
python scripts/figure4_data.py
```

图 3 脚本是有序 bridge 的权威实现。写入工作簿前，它会检查配置覆盖、国家聚合、小时—年度能耗闭合以及 bridge 加和闭合。

### 运行外部诊断

```powershell
python scripts/validate_energy_model.py --bootstrap-replicates 10000
```

诊断流程使用固定随机种子 `20260819`。输入来源和筛选规则记录在 [`dataset/validation/`](dataset/validation/) 中。

## Python 接口

### 年度 CPU 模型

```python
from core.annual_cpu_model import run_annual_cpu_model

result = run_annual_cpu_model(
    renewable_energy_policy="CP",
    scenarios=["Base", "Lift-Off"],
    year_start=2026,
    years=5,
)
```

### 小时级工作负载解析模型

```python
from core.hourly_workload_resolved_model import run_workload_component_footprint

result = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    save_hourly_outputs=False,
)
```

小时接口的 `hourly_carbon_scope` 可设为 `direct` 或 `life_cycle`。小时 CSV 默认不保存，以避免生成过大的文件。

## 主要参数

| 参数 | 含义 | 常用值或默认值 |
| --- | --- | --- |
| `renewable_energy_policy` | 电力系统路径 | `CP`、`NDC` 或 `NZ` |
| `scenarios` | 需求路径 | `Base`、`Lift-Off`、`High Efficiency` 或 `Headwinds` |
| `year_start` | 输出起始年份 | 原稿分析为 2026 |
| `years` | 连续年份数 | 5 |
| `countries` | 模型覆盖国家 | 默认全部 24 国 |
| `pue_scale` | PUE 敏感性乘数 | `1.0` |
| `ai_capacity_factors` | 年度 AI 容量校准 | `DEFAULT_AI_CAPACITY_FACTORS` |
| `hourly_carbon_scope` | 小时碳强度边界 | `direct` 或 `life_cycle` |
| `hourly_carbon_fallback_to_annual` | 小时文件缺失时使用年度因子 | `True` |
| `capacity_quantile` | 轨迹容量清单分位数 | `0.96` |
| `max_resource_utilization` | 归一化资源利用率上限 | `1.0` |
| `max_intervals` | smoke test 的轨迹小时限制 | 默认使用完整轨迹 |
| `include_zero_gpu_pods` | 能耗边界是否保留 CPU-only Pod | `True` |

年度 CPU 模型还提供固定任务比例、任务利用率、空闲功率率和最大功率率参数。小时模型通过 `HardwarePowerConfig` 与 `Alibaba2026TraceConfig` 暴露部件功率和轨迹处理假设。

## 输出结果

端点比较入口写入：

```text
results/model_comparison/
├── annual_cpu_model/
├── hourly_workload_resolved_model/
└── summary/
    ├── Endpoint_Models_Country_Annual.csv
    ├── Endpoint_Models_Global_Annual.csv
    ├── Endpoint_Comparison_Country_Annual.csv
    └── Endpoint_Comparison_Global_Annual.csv
```

比较表报告年度 CPU 模型相对于小时级工作负载解析端点的差值。正百分比表示年度结果高于小时参考配置。

小时模型可返回或保存：

- `annual_summary`：国家—年份用电、排放和资源利用；
- `component_energy`：CPU、GPU、内存、存储和 IT 风扇能耗；
- `task_type_energy`：训练、推理、其他和未分类任务能耗；
- `task_demand`、`task_execution`：按任务来源与执行位置统计的资源小时；
- `capacity_overflow`：资源容量溢出诊断；
- `hourly_carbon`：小时用电、碳强度和排放；
- `workload_profile_summary`：不同报告口径下的工作负载结构；
- `trace_resource_capacity`：轨迹容量估计及来源。

### 单位

| 字段后缀 | 单位 |
| --- | --- |
| `_mw` | MW |
| `_mwh` | MWh |
| `_twh` | TWh |
| `_kg_per_mwh` | kg CO₂/MWh |
| `_tco2` | t CO₂ |
| `_mtco2` | Mt CO₂ |

## 解释边界

- 容量输入覆盖 2025—2030 年，没有新证据时不应外推。
- 固定国家份额不表达跨地区迁移、国家份额变化或剩余全球容量份额。
- 所有国家共用相同的归一化相对负荷形状；模型不表达时区和逐国任务结构差异。
- 4,440 小时 Alibaba 轨迹会重复至 8,760 小时。拼接边界不改变年度能耗，但可能决定未过滤的最大爬坡。
- 未来小时电网序列保留 2025 年相对形状，不预测调度、储能、进口和可变电源结构变化。
- 部件份额、单位功率、空闲比例、风扇响应、内存活动和存储活动均为可配置参考假设。
- MLPerf 诊断仅检查功率量级和偏差，不是校准拟合，也不验证未来国家级设施用电。
- 缺失小时碳数据时可以回退到年度因子；解释时间结果前应检查 `carbon_factor_source`。
- 小时级工作负载解析端点是透明参考配置，不是实测真值。

## 许可证

项目代码采用 [MIT License](LICENSE)。第三方数据仍受各原始提供方的访问、引用和再分发条款约束。
