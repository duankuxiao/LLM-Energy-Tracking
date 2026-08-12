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

M1 不引入小时任务负荷，也不区分 CPU、GPU、内存、存储等设备能耗组成。

### M2：小时碳因子 CPU 模型

`core/m2_hourly_carbon_cpu_model.py` 使用与 M1 完全相同的年度能耗算法，因此 M1 与 M2 的国家年度总用电量按设计保持一致。M2 不引入任务负荷和设备分项，而是将国家年度用电表示为恒定小时功率，再逐小时匹配电网碳强度。

M2 用于隔离年度平均碳因子与小时级碳因子之间的时间聚合效应。

### M3：年度 GPU 模型

`core/m3_annual_gpu_model.py` 调用 `core/m4_hourly_gpu_model.py` 中的 GPU 工作负载与分部件能耗算法，但关闭小时电网匹配，仅汇总年度用电并采用 `dataset/Factors.py` 中的年度碳因子。

该模型基于公开 GPU 集群轨迹估算 CPU、GPU、内存、存储和 IT 风扇功率，并将轨迹负载按国家 AI IT 容量进行缩放。

### M4：小时级 GPU 模型

`core/m4_hourly_gpu_model.py` 是完整核算模型。它读取 GPU Pod 和服务器轨迹，构建训练、推理和其他任务的小时资源负载，计算各硬件组成的小时功率，通过 PUE 得到设施用电，再与国家小时碳强度逐时匹配。

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

- `dataset/Data_annex_Energy_and_AI.xlsx`：IEA 能源与 AI 数据附件，包含世界和区域数据表。
- `dataset/Installed_capacity_data.py`：存放 2025—2030 年四类情景下的总容量、IT 容量、非 IT 容量、24 国容量份额和默认 AI 容量校准因子。
- 默认 AI 容量因子根据 IEA 2026 年报告 [Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai) 中的 AI 专用数据中心用电路径进行校准；代码注释记录了校准端点和中间年份假设。

需要注意，国家容量份额和情景插值属于项目的模型输入或派生参数，不应被理解为原始数据提供方直接发布的逐国预测值。

### GPU 工作负载与服务器轨迹

- `dataset/asi_opensource_pod_hourly/`：Alibaba Serverless Infrastructure（ASI）GPU Pod 小时轨迹；
- `dataset/asi_opensource_server_hourly/`：ASI 服务器资源与设备清单小时轨迹；
- 官方项目入口：[Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata)；
- 轨迹背景与系统说明：[Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba](https://www.usenix.org/conference/osdi26/presentation/li-suyi)。

模型读取 `asi_opensource_pod_hourly` 中的 CPU 请求与利用率、GPU 请求、GPU SM 利用率、GPU 显存使用、任务类型和运行状态等字段。公开轨迹不含完整的存储活动和主机内存容量，因此当前实现分别使用空闲存储功率和 CPU 加权的内存活动代理；这些假设会写入 M4 的轨迹容量来源输出。

### 年度碳因子与 PUE

`dataset/Factors.py` 提供：

- 24 国在 `CP`、`NDC` 和 `NZ` 路径下的年度电网碳排放因子；
- 不同国家、年份和需求情景对应的 PUE；
- 碳因子单位为 `kg CO₂/MWh`。

这些因子用于 M1 和 M3 的年度碳排放计算，也用于缺失小时数据时的回退处理。

### 小时级电网碳强度

`dataset/EM-estimate/` 保存 24 国 2026—2030 年、三类电力政策路径下的小时数据。CSV 包含 UTC 时间、直接碳强度、生命周期碳强度、无碳电力比例和可再生能源比例等字段。

项目内文件以 2025 年 Electricity Maps 下载数据为历史小时形状，并根据 `dataset/Factors.py` 的年度路径进行相对缩放，构造未来情景。Electricity Maps 的数据说明和获取方式参见其[数据页面](https://www.electricitymaps.com/data)和[学术数据说明](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)。

小时数据可能受原始提供方的访问、引用和再分发条款约束。使用者应自行确认用途符合 Electricity Maps 的最新许可，并在论文或其他公开成果中按要求署名。

### 其他辅助数据

- `dataset/Carbon_emission_factors_2010_2018.csv`：历史电力碳排放因子；
- `dataset/climate_data_2025.csv`：2025 年气候辅助数据；
- `dataset/result_df_full_year_2020.pkl`：历史全年中间数据。

这些文件并非所有 M1—M4 默认运行路径都必须读取，其用途应结合具体分析脚本和论文方法说明判断。

## 项目结构

```text
.
├── core/
│   ├── past_research_data_center_energy_carbon_model.py
│   │                                      # 过去研究中的能耗碳排放方法
│   ├── m1_annual_cpu_model.py          # M1：年度 CPU + 年度碳因子
│   ├── m2_hourly_carbon_cpu_model.py   # M2：年度 CPU + 小时碳因子
│   ├── m3_annual_gpu_model.py          # M3：GPU 能耗 + 年度碳因子
│   └── m4_hourly_gpu_model.py          # M4：GPU 能耗 + 小时碳因子
├── dataset/
│   ├── Factors.py                      # 年度碳因子与 PUE
│   ├── Installed_capacity_data.py      # 容量、国家份额与 AI 校准因子
│   ├── Data_annex_Energy_and_AI.xlsx   # IEA 数据附件
│   ├── EM-estimate/                    # 24 国未来小时碳强度
│   ├── asi_opensource_pod_hourly/      # GPU Pod 小时轨迹
│   └── asi_opensource_server_hourly/   # 服务器小时轨迹
├── paper/                              # 论文大纲与研究设计
├── results/                            # 默认模型输出目录
├── scripts/                            # 分析和批处理脚本目录
├── LICENSE                             # MIT 许可证
└── README.md
```

## 环境与依赖

建议使用 Python 3.9 或更高版本。核心依赖为：

- `numpy`；
- `pandas`；
- `pyarrow`，用于读取 Parquet 格式的 Alibaba 轨迹；
- `openpyxl`，仅在需要直接读取 Excel 数据附件时使用。

可在项目根目录创建独立环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas pyarrow openpyxl
```

Linux 或 macOS 下激活环境：

```bash
source .venv/bin/activate
```

## 数据准备

GPU 轨迹和小时碳强度数据体积较大，以下目录已被 `.gitignore` 排除，不会随普通 Git 克隆自动提供：

```text
dataset/EM-estimate/
dataset/asi_opensource_pod_hourly/
dataset/asi_opensource_server_hourly/
dataset/result_df_full_year_2020.pkl
```

运行 M2 或 M4 前需要准备 `EM-estimate`；运行 M3 或 M4 前需要准备两类 ASI Parquet 轨迹。预期目录示例如下：

```text
dataset/
├── EM-estimate/
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

M1/M2 还允许调整训练和推理占比、利用率、空闲功率率与最大功率率；M3/M4 允许通过 `HardwarePowerConfig` 和 `Alibaba2026TraceConfig` 调整硬件功率与轨迹处理参数。

## 输出结果

默认输出写入 `results/` 下的模型专属目录。

### M1

- `M1_Country_Annual.csv`：国家年度用电和碳排放；
- `M1_Global_Annual.csv`：全球年度汇总。

### M2

- `M2_Country_Annual.csv`：国家年度用电和小时匹配后的碳排放；
- `M2_Global_Annual.csv`：全球年度汇总；
- `M2_Country_Hourly.csv`：可选的国家逐小时结果。

### M3

- `M3_Country_Annual.csv`：国家年度 GPU 模型结果；
- `M3_Global_Annual.csv`：全球年度汇总。

### M4

M4 返回并可保存以下主要结果表：

- `annual_summary`：国家年度能耗、碳排放和资源利用率；
- `component_energy`：CPU、GPU、内存、存储和 IT 风扇分项能耗；
- `task_type_energy`：训练、推理和其他任务能耗；
- `task_demand` 与 `task_execution`：任务来源和执行地的资源小时；
- `capacity_overflow`：资源容量溢出诊断；
- `hourly_carbon`：可选的国家小时用电、碳因子和碳排放；
- `workload_profile_summary`：轨迹工作负载汇总；
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

## 论文材料

`paper/Outline of a Paper on Decarbonization Pathways for Global AI Data Centers_v2.md` 包含论文研究大纲、章节结构、M1—M4 方法定义、偏差分解框架和建议图表。`paper/` 已在 `.gitignore` 中排除，公开仓库若需要附带论文材料，应单独确认其版本与发布权限。

## 许可证

本项目代码采用 [MIT License](LICENSE)。外部数据集不自动适用本项目的软件许可证；Alibaba、IEA、Electricity Maps 及其他数据来源的使用、引用和再分发应分别遵循其原始许可与服务条款。
