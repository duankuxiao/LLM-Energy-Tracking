<p align="center">
  <img src="logo.png" alt="Logo" width="180"/>
</p>

# 全球 AI 数据中心能耗与碳排放核算

本项目面向全球主要人工智能（AI）数据中心，构建年度及小时级电力消耗与碳排放核算模型。研究以 24 个主要数据中心国家为对象，综合考虑全球数据中心容量增长、国家容量分配、AI 容量占比、服务器功率特征、PUE，以及年度或小时级电网碳强度，评估 2025—2030 年不同需求与电力系统情景下的能源和碳排放结果。

项目的核心是由“时间分辨率 × 硬件功率模型”形成的 M1—M4 核算矩阵，用于识别年度平均核算造成的时间聚合偏差，以及传统 CPU 功率近似造成的 GPU 硬件表征偏差。

> 当前论文与 M1—M4 模型聚焦电力消耗、负荷和碳排放，不开展水足迹或 WUE 分析。

## 核算模型

| 模型 | 能耗模型 | 电网碳因子 | 主要用途 |
| --- | --- | --- | --- |
| M1 | 传统 CPU 利用率—功率模型 | 年度平均 | 传统简化核算基线 |
| M2 | 传统 CPU 利用率—功率模型 | 小时级 | 单独修正碳因子的时间分辨率 |
| M3 | GPU 工作负载与分部件功率模型 | 年度平均 | 单独修正 AI 硬件能耗表征 |
| M4 | GPU 工作负载与分部件功率模型 | 小时级 | 完整的小时级 GPU 核算基准 |

四个模型使用一致的数据中心容量、国家分配、AI 容量边界、PUE 和政策情景。模型间的主要差别仅来自服务器能耗表征和电网碳因子的时间分辨率。

### M1：年度 CPU 模型

`core/m1_annual_cpu_model.py` 调用 `core/past_research_data_center_energy_carbon_model.py` 中的过去研究核算方法，采用空闲功率率、最大功率率和综合利用率计算年度 IT 用电，再通过 PUE 转换为数据中心总用电，最后匹配 `dataset/Factors.py` 中的年度碳因子。

M1 不引入小时任务负荷，也不区分 CPU、GPU、内存、存储等设备能耗组成。其年度简化模型使用训练、推理、其他和未分类四类统一任务口径；默认有效任务比例为训练 20%、推理 75%、其他 5%，未分类比例为 0。各任务继续采用传统利用率—功率系数计算，并输出任务年度能耗。

### M2：小时碳因子 CPU 模型

`core/m2_hourly_carbon_cpu_model.py` 使用与 M1 完全相同的年度任务分类和能耗算法，因此 M1 与 M2 的国家年度总用电量及任务能耗按设计保持一致。M2 不引入小时任务曲线和设备分项，而是将国家年度用电表示为恒定小时功率，再逐小时匹配电网碳强度。

M2 用于隔离年度平均碳因子与小时级碳因子之间的时间聚合效应。

### M3：年度 GPU 模型

`core/m3_annual_gpu_model.py` 调用 `core/m4_hourly_gpu_model.py` 中的 GPU 工作负载与分部件能耗算法，但关闭小时电网匹配，仅汇总年度用电并采用 `dataset/Factors.py` 中的年度碳因子。

该模型基于公开 GPU 集群轨迹估算 CPU、GPU、内存、存储和 IT 风扇功率，并将轨迹负载按国家 AI IT 容量进行缩放。

### M4：小时级 GPU 模型

`core/m4_hourly_gpu_model.py` 是完整核算模型。它读取 GPU Pod 和服务器轨迹，构建训练、推理、其他和未分类任务的小时资源负载，计算各硬件组成的小时功率，通过 PUE 得到设施用电，再与国家小时碳强度逐时匹配。

M4 是比较 M1—M3 核算偏差时的高精度基准。

## 研究范围与情景

### 国家范围

默认研究以下 24 个国家：

`USA`、`China`、`Japan`、`France`、`India`、`Singapore`、`Canada`、`Germany`、`United_Kingdom`、`Australia`、`Italy`、`South_Korea`、`South_Africa`、`Ireland`、`UAE`、`Brazil`、`Israel`、`Netherlands`、`Spain`、`Sweden`、`Belgium`、`Norway`、`Poland` 和 `Switzerland`。

代码使用上述英文标识作为国家键。自定义国家列表时，名称必须与 `dataset/Installed_capacity_data.py` 和 `dataset/Factors.py` 中的键一致。

### AI 数据中心需求情景

代码支持四种容量增长情景：

- `Base`：基准增长路径；
- `Lift-Off`：较快需求增长路径；
- `High Efficiency`：效率提升路径；
- `Headwinds`：增长受限路径。

传入函数的情景名称必须使用上面的精确拼写。

### 电力系统情景

- `CP`：Current Policies，当前政策路径；
- `NDC`：Nationally Determined Contributions，国家自主贡献路径；
- `NZ`：Net Zero，净零路径。

年度碳因子、小时碳因子的未来缩放以及模型输出均按这三类政策路径组织。

## 数据来源

### AI 数据中心容量与增长路径

- `dataset/Installed_capacity_data.py`：存放 2025—2030 年四类情景下的总容量、IT 容量、非 IT 容量、24 国容量份额和默认 AI 容量校准因子。
- 默认 AI 容量因子根据 IEA 2026 年报告 [Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai) 中的 AI 专用数据中心用电路径进行校准；代码注释记录了校准端点和中间年份假设。

IEA 数据附件作为外部来源使用，不随本仓库重新分发；论文运行所需的已转录容量路径和校准输入已经保存在 `dataset/Installed_capacity_data.py`。

需要注意，国家容量份额和情景插值属于项目的模型输入或派生参数，不应被理解为原始数据提供方直接发布的逐国预测值。

### GPU 工作负载与服务器轨迹

- `dataset/asi_opensource_pod_hourly/`：Alibaba Serverless Infrastructure（ASI）GPU Pod 小时轨迹；
- `dataset/asi_opensource_server_hourly/`：ASI 服务器资源与设备清单小时轨迹；
- 官方项目入口：[Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata)；
- 轨迹背景与系统说明：[Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba](https://www.usenix.org/conference/osdi26/presentation/li-suyi)。

模型读取 `asi_opensource_pod_hourly` 中的 CPU 请求与利用率、GPU 请求、GPU SM 利用率、GPU 显存使用、任务类型和运行状态等字段。能耗边界默认纳入全部活跃 Pod：即使 `gpu_request == 0`，CPU-only Pod 的 CPU 和内存活动仍属于 AI 集群能耗。论文任务比例采用独立的严格筛选口径，仅统计论文时间窗口内具有 GPU 显存请求和有效 GPU 使用小时的已分类任务；该筛选不删除能耗边界内的 CPU-only 负载。`training` 保持为训练，在线与离线推理合并为推理，`dev` 与明确标注的 `other` 合并为其他，空值、`unknown` 及异常标签单列为 `unclassified`。未分类负荷保留在总能耗中，但不计入论文口径的已分类任务比例。公开轨迹不含完整的存储活动和主机内存容量，因此当前实现分别使用空闲存储功率和 CPU 加权的内存活动代理；这些假设会写入 M4 的轨迹容量来源输出。

### 年度碳因子与 PUE

`dataset/Factors.py` 提供：

- 24 国在 `CP`、`NDC` 和 `NZ` 路径下的年度电网碳排放因子；
- 不同国家、年份和需求情景对应的 PUE；
- 碳因子单位为 `kg CO₂/MWh`。

这些因子用于 M1 和 M3 的年度碳排放计算，也用于缺失小时数据时的回退处理。

### 小时级电网碳强度

M2 和 M4 在本地从 `dataset/EM-CPNDCNZ/` 读取 24 国 2026—2030 年、三类电力政策路径下的小时数据。CSV 包含 UTC 时间、直接碳强度、生命周期碳强度、无碳电力比例和可再生能源比例等字段；完整研究输入由 360 个小时 CSV 构成，即 24 个国家、CP/NDC/NZ 三类路径和 2026—2030 五个年份的完整组合。

项目内文件以 2025 年 Electricity Maps 下载数据为历史小时形状，并根据 `dataset/Factors.py` 的年度路径进行相对缩放，构造未来情景。Electricity Maps 的数据说明和获取方式参见其[数据页面](https://www.electricitymaps.com/data)和[学术数据说明](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)。

Electricity Maps 的许可不允许本项目公开再分发这些小时数据，因此 GitHub 仓库不包含 `dataset/EM-CPNDCNZ.zip` 或其解压内容。合资格研究人员应通过 Electricity Maps 官方渠道独立申请访问，并确保用途符合其最新条款。小时数据不适用本项目的 MIT 软件许可证。

## 项目结构

```text
.
├── core/
│   ├── past_research_data_center_energy_carbon_model.py
│   │                                      # 过去研究中的能耗碳排放方法
│   ├── m1_annual_cpu_model.py          # M1：年度 CPU + 年度碳因子
│   ├── m2_hourly_carbon_cpu_model.py   # M2：年度 CPU + 小时碳因子
│   ├── m3_annual_gpu_model.py          # M3：GPU 能耗 + 年度碳因子
│   ├── m4_hourly_gpu_model.py          # M4：GPU 能耗 + 小时碳因子
│   └── task_model.py                   # M1—M4 共用任务分类口径
├── dataset/
│   ├── Factors.py                      # 年度碳因子与 PUE
│   ├── Installed_capacity_data.py      # 容量、国家份额与 AI 校准因子
│   ├── EM-CPNDCNZ/                     # 经许可取得并在本地准备的小时碳强度；不纳入 Git
│   ├── validation/                     # MLPerf 与国家统计验证输入及说明
│   ├── asi_opensource_pod_hourly/      # GPU Pod 小时轨迹
│   └── asi_opensource_server_hourly/   # 服务器小时轨迹
├── results/                            # 默认模型输出及 Figure 1–4 源数据工作簿
├── scripts/                            # 分析、绘图和外部验证脚本
├── run.py                              # 一次运行 M1—M4 并生成对比汇总
├── LICENSE                             # MIT 许可证
└── README.md
```

## 环境与依赖

论文提交包已在 Python 3.12.3 下验证。为固定复现环境，请使用以下精确依赖版本：

- `numpy==2.1.3`；
- `pandas==2.2.3`；
- `pyarrow==17.0.0`，用于读取 Parquet 格式的 Alibaba 轨迹；
- `openpyxl==3.1.5`，用于读取论文图表源数据工作簿；
- `matplotlib==3.9.2`，用于生成验证图。

可在项目根目录创建独立环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==2.1.3 pandas==2.2.3 pyarrow==17.0.0 openpyxl==3.1.5 matplotlib==3.9.2
```

Linux 或 macOS 下激活环境：

```bash
source .venv/bin/activate
```

## 数据准备

受许可限制的小时碳强度数据及体积较大的 GPU 轨迹不随仓库分发。以下本地输入已被 `.gitignore` 排除：

```text
dataset/EM-CPNDCNZ.zip
dataset/EM-CPNDCNZ/
dataset/asi_opensource_pod_hourly/
dataset/asi_opensource_server_hourly/
```

运行 M2 或 M4 前，用户需通过 Electricity Maps 的[数据页面](https://www.electricitymaps.com/data)或[学术数据申请说明](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)自行取得许可数据，并按上述研究方法准备未来情景文件后放入 `dataset/EM-CPNDCNZ/`。运行 M3 或 M4 还需按 Alibaba Cluster Trace Program 的公开说明准备两类 ASI Parquet 轨迹。预期目录示例如下：

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

`United_Kingdom` 在模型国家键中对应小时数据目录 `Great Britain`。

## 快速开始

所有命令均建议从项目根目录执行。

### 一次运行 M1—M4 并汇总

使用根目录下的统一入口运行四个 case：

```powershell
python run.py --policy CP --scenarios Base --year-start 2026 --years 5
```

多个情景可以同时传入；带空格的情景名需要加引号：

```powershell
python run.py --policy NDC --scenarios Base "Lift-Off" "High Efficiency"
```

统一入口先把 Alibaba GPU 轨迹构建为一个内存中的 `WorkloadProfile`，随后将同一个对象传给 M3 和 M4。因此两类大型 Parquet 轨迹只读取一次，M3 的年度碳因子计算和 M4 的小时碳因子计算仍分别执行。默认结果写入 `results/m1_m4_comparison/`。

快速检查环境时可以限制轨迹小时数和国家范围：

```powershell
python run.py --countries USA --year-start 2026 --years 1 --max-intervals 24
```

`max_intervals` 会改变轨迹统计口径，只能用于调试，正式结果应省略该参数。运行 `python run.py --help` 可查看完整参数。

### 外部验证与论文图表源数据

外部验证输入位于 `dataset/validation/`，验证脚本为 `scripts/validate_energy_model.py`。从项目根目录运行：

```powershell
python scripts\validate_energy_model.py --bootstrap-replicates 10000
```

脚本使用固定随机种子 `20260819`，将诊断表写入 `results/model_validation/`，并将补充验证图写入 `figures/`。验证数据来源、筛选口径和国家统计边界见 `dataset/validation/README.md`。

论文主图的现有源数据工作簿保存在：

- `results/figure1_data.xlsx`：Figure 1a–g；
- `results/figure2_data.xlsx`：Figure 2a–d 及 Figure 2b 的完整国家级来源表；
- `results/figure3_data.xlsx`：Figure 3a–d；
- `results/figure4_data.xlsx`：Figure 4a。

这四个工作簿是论文图表的分析后源数据；`results/` 下其他运行输出仍由 `.gitignore` 排除。

### 运行 M1

```powershell
python -m core.m1_annual_cpu_model
```

或在 Python 中指定情景：

```python
from core.m1_annual_cpu_model import run_m1_annual_cpu_model

result = run_m1_annual_cpu_model(
    renewable_energy_policy="CP",
    scenarios=["Base", "Lift-Off"],
    year_start=2026,
    years=5,
)
```

### 运行 M2

```python
from core.m2_hourly_carbon_cpu_model import run_m2_hourly_carbon_cpu_model

result = run_m2_hourly_carbon_cpu_model(
    renewable_energy_policy="NDC",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    hourly_carbon_scope="direct",
    save_hourly_outputs=False,
)
```

`hourly_carbon_scope` 可选 `direct` 或 `life_cycle`。默认只保存国家和全球年度汇总；设置 `save_hourly_outputs=True` 后会额外保存逐小时结果，文件可能较大。

### 运行 M3

```python
from core.m3_annual_gpu_model import run_m3_annual_gpu_model

result = run_m3_annual_gpu_model(
    renewable_energy_policy="NZ",
    scenarios=["High Efficiency"],
    year_start=2026,
    years=5,
)
```

M3 需要读取 GPU 轨迹，但碳排放只使用年度碳因子，不生成小时碳排放表。

### 运行 M4

```python
from core.m4_hourly_gpu_model import run_workload_component_footprint

result = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    save_hourly_outputs=False,
)
```

完整轨迹包含大量 Parquet 分区。首次检查环境或调试接口时，可用 `max_intervals` 限制读取的轨迹小时数：

```python
result = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    countries=["USA"],
    year_start=2026,
    years=1,
    max_intervals=24,
    save_outputs=False,
)
```

`max_intervals` 仅适合烟雾测试和调试，不应直接用于正式论文结果。

## 主要参数

| 参数 | 含义 | 常用值或默认值 |
| --- | --- | --- |
| `renewable_energy_policy` | 电力系统路径 | `CP`、`NDC`、`NZ` |
| `scenarios` | AI 数据中心需求情景 | `Base`、`Lift-Off`、`High Efficiency`、`Headwinds` |
| `year_start` | 输出起始年份 | M1/M2 支持 2025—2030，M3/M4 默认 2026 |
| `years` | 连续计算年份数 | 默认 5 |
| `countries` | 参与核算的国家 | 默认 24 国 |
| `pue_scale` | PUE 敏感性缩放系数 | `1.0` |
| `ai_capacity_factors` | 各年份 AI 容量校准因子 | 默认读取 `DEFAULT_AI_CAPACITY_FACTORS` |
| `hourly_carbon_scope` | 小时碳强度口径 | `direct` 或 `life_cycle` |
| `hourly_carbon_fallback_to_annual` | 小时数据缺失时是否回退到年度因子 | `True` |
| `max_intervals` | 限制 GPU 轨迹读取小时数 | 默认读取全部轨迹 |
| `default_p_infer` | M1/M2 默认推理任务比例 | `0.75` |
| `default_p_other` | M1/M2 默认其他任务比例 | `0.05` |
| `task_ratio_by_country` | M1/M2 国家级四类任务比例覆盖 | 默认不覆盖 |
| `include_zero_gpu_pods` | M3/M4 是否把零 GPU 请求的 CPU-only Pod 纳入能耗边界 | `True`；设为 `False` 仅用于 GPU-bearing Pod 敏感性分析 |

M1/M2 还允许调整训练、推理和其他任务占比及利用率，以及空闲功率率与最大功率率；M3/M4 允许通过 `HardwarePowerConfig` 和 `Alibaba2026TraceConfig` 调整硬件功率与轨迹处理参数。

## 输出结果

默认输出写入 `results/` 下的模型专属目录。

### 统一运行与模型对比

`run.py` 将每个模型的原始结果分别写入 `results/m1_m4_comparison/m1_annual_cpu/` 至 `m4_hourly_gpu/`，并在 `summary/` 中生成：

- `All_Models_Country_Annual.csv`：M1—M4 国家年度统一长表；
- `All_Models_Global_Annual.csv`：M1—M4 全球年度统一长表；
- `Model_Comparison_Country_Annual.csv`：国家年度并排对比表；
- `Model_Comparison_Global_Annual.csv`：全球年度并排对比表。

并排对比表包含四个模型的用电量、负荷加权碳因子和碳排放，并为 M1—M3 提供相对 M4 的绝对差与百分比差。百分比按 `(模型结果 - M4) / M4 × 100%` 计算，正值表示高于 M4。

### M1

- `M1_Country_Annual.csv`：国家年度用电和碳排放；
- `M1_Global_Annual.csv`：全球年度汇总；
- `M1_Country_TaskType_Energy.csv`：国家年度任务分类能耗和碳排放。

### M2

- `M2_Country_Annual.csv`：国家年度用电和小时匹配后的碳排放；
- `M2_Global_Annual.csv`：全球年度汇总；
- `M2_Country_TaskType_Energy.csv`：国家年度任务分类能耗和小时匹配后的碳排放；
- `M2_Country_Hourly.csv`：可选的国家逐小时结果。

### M3

- `M3_Country_Annual.csv`：国家年度 GPU 模型结果；
- `M3_Global_Annual.csv`：全球年度汇总。

### M4

M4 返回并可保存以下主要结果表：

- `annual_summary`：国家年度能耗、碳排放和资源利用率；
- `component_energy`：CPU、GPU、内存、存储和 IT 风扇分项能耗；
- `task_type_energy`：训练、推理、其他和未分类任务能耗；
- `task_demand` 与 `task_execution`：任务来源和执行地的资源小时；
- `capacity_overflow`：资源容量溢出诊断；
- `hourly_carbon`：可选的国家小时用电、碳因子和碳排放；
- `workload_profile_summary`：轨迹工作负载汇总，同时报告全部任务占比、排除未分类后的 `used_gpu_hours` 占比，以及 Alibaba 论文复现窗口口径的验证占比；
- `trace_resource_capacity`：轨迹容量估算及其来源。

### 单位

| 字段后缀 | 单位 |
| --- | --- |
| `_mw` | MW |
| `_mwh` | MWh |
| `_twh` | TWh |
| `_kg_per_mwh` | kg CO₂/MWh |
| `_tco2` | t CO₂ |
| `_mtco2` | Mt CO₂ |

## 方法假设与注意事项

- 研究期容量数据仅覆盖 2025—2030 年，请勿在没有新增输入数据的情况下外推到范围之外。
- 24 国 IT 容量份额在当前实现中来自 `IT_RATIO`；如采用固定份额，应在论文中说明其局限性。
- M2 将年度用电表示为恒定小时功率，因此只检验小时碳因子变化，不表达训练或推理的日内负荷变化。
- Alibaba 公开轨迹是相对时间轨迹。M3/M4 将其年化到 8760 小时，并在匹配未来日历年份时重复相对小时形状。
- GPU 模型中的硬件功率份额、单位满载功率、空闲功率和风扇曲线是可配置参数，应结合实测数据开展敏感性和不确定性分析。
- M4 的年度结果可能掩盖国家小时级正负偏差抵消，正式分析应同时检查国家与全球尺度。
- 小时因子缺失时默认回退到年度因子。正式核算前应检查输出中的 `carbon_factor_source`，避免将回退结果误认为真实小时数据。
- 完整 GPU 轨迹体积很大，正式运行需要足够的磁盘空间、内存和处理时间。

## 许可证

本项目代码采用 [MIT License](LICENSE)。外部数据集不自动适用本项目的软件许可证；Alibaba、IEA、Electricity Maps 及其他数据来源的使用、引用和再分发应分别遵循其原始许可与服务条款。
