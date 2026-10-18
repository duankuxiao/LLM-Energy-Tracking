<p align="center">
  <img src="logo.png" alt="项目标志" width="180"/>
</p>

# LLM-Energy-Tracking

[English README](README.md)

用于估算 AI 数据中心电力消耗与运行碳排放的 Python 工具。项目结合容量情景、GPU 集群工作负载轨迹、分部件功率模型、电能使用效率（PUE）和电网碳强度，支持年度与小时级计算。

主要功能包括模型比较、情景分析、参数敏感性分析及负荷峰值与爬坡诊断，结果可导出为 CSV、Excel 和图件。

## 主要功能

| 功能 | 说明 | 实现入口 |
| --- | --- | --- |
| 年度模型 | 根据容量和利用率假设，估算各国年度用电与排放 | [年度 CPU 模型](core/annual_cpu_model.py) |
| 小时级模型 | 根据工作负载轨迹，计算 CPU、GPU、内存、存储和 IT 风扇能耗 | [小时级工作负载解析模型](core/hourly_workload_resolved_model.py) |
| 需求情景分析 | 比较容量、用电、排放及任务和部件的能耗构成 | [需求分析](scripts/experiment_figure1.py) |
| 电网情景分析 | 比较排放、避免排放和负荷加权碳强度 | [电网分析](scripts/experiment_figure2.py) |
| 模型差异分解 | 分析功率配置、平均负荷、负荷波动和小时碳匹配的贡献，并计算峰值及爬坡 | [模型与负荷形状分析](scripts/experiment_figure3.py) |
| 敏感性分析 | 改变功率预算、空闲功率、响应曲线、负荷参数和碳强度波动 | [敏感性分析](scripts/experiment_figure4.py) |
| 外部诊断 | 与 MLPerf Power 数据比较参考功率配置，并检查国家电量数据的统计范围 | [功率诊断](scripts/validate_energy_model.py) |

默认配置覆盖 2026–2030 年及 24 个国家。需求情景为 `Base`、`Lift-Off`、`High Efficiency` 和 `Headwinds`；电网情景为 `CP`、`NDC` 和 `NZ`。国家标识与容量输入见 [Installed_capacity_data.py](dataset/Installed_capacity_data.py)。

## 安装

已有运行记录使用 Python 3.9.25 和以下依赖版本。所有命令均从仓库根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pandas==2.3.3 pyarrow==21.0.0 openpyxl==3.1.5 matplotlib==3.9.2
```

Linux 或 macOS 使用以下命令激活环境：

```bash
source .venv/bin/activate
```

## 输入数据

| 输入 | 位置 | 准备方式 |
| --- | --- | --- |
| 容量路径、国家份额和 AI 容量因子 | [dataset/Installed_capacity_data.py](dataset/Installed_capacity_data.py) | 仓库已包含 |
| 年度电网碳强度和 PUE | [dataset/Factors.py](dataset/Factors.py) | 仓库已包含 |
| GPU Pod 工作负载记录 | `dataset/asi_opensource_pod_hourly/` | 从 Alibaba Cluster Trace Program 获取 |
| 服务器资源清单 | `dataset/asi_opensource_server_hourly/` | 从 Alibaba Cluster Trace Program 获取 |
| 小时电网源文件 | [configs/hourly_grid_sources.csv](configs/hourly_grid_sources.csv) 中列出的路径 | 按数据提供方的访问条件准备本地文件 |
| MLPerf Power 和国家电量输入 | [dataset/validation/](dataset/validation/) | 仓库已包含，来源见该目录说明 |
| 预置图源数据 | [results/](results/) | 用于查看已有图件数值的精简工作簿 |

工作负载数据来自 [Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata)，电网数据可通过 [Electricity Maps](https://www.electricitymaps.com/data) 获取。外部轨迹和小时电网文件不纳入版本控制。

工作负载目录示例：

```text
dataset/
├── asi_opensource_pod_hourly/
│   └── day=0/hour=00/part-000.parquet
├── asi_opensource_server_hourly/
│   └── day=0/hour=00/part-000.parquet
└── EM-CPNDCNZ/
    └── ... 小时电网源 CSV 文件
```

小时源清单包含 `country`、`path`、`source_year`、`source_kind` 和 `zone_id`。清单中的路径相对于该 CSV 所在目录。仓库提供的清单选择已处理的 2026 年 CP 模板；使用时按本地输入调整路径与来源信息。

电网构造程序将源曲线映射到完整 UTC 日历，并归一化至 `dataset/Factors.py` 中的年度因子。默认配置生成 360 个直接碳因子文件，2028 年为 8,784 小时，其余年份为 8,760 小时。

## 快速开始

### 运行年度模型

年度计算使用仓库已有输入，无需准备工作负载轨迹或小时电网文件。

```powershell
python -B -m core.annual_cpu_model --policy CP --scenarios Base --year-start 2026 --years 5 --activity-mode calendar --output-dir results/annual_run
```

输出包括 `Annual_CPU_Model_Country_Annual.csv`、`Annual_CPU_Model_Selected_Countries_Annual.csv`、任务能耗表，以及实际活动因子和参数设置。

### 运行完整分析

先准备外部数据，再构造小时电网文件并执行配置中的分析：

```powershell
python -B scripts/build_hourly_grid_scenarios.py --source-manifest configs/hourly_grid_sources.csv --output-dir results/grid_hourly_calendar --scopes direct
python -B scripts/run_experiments.py --config configs/experiments.json --output-dir results/example_run
```

若小时电网目录已生成，且源输入与设置没有变化，可直接复用。每次实验应使用新的输出目录。

计算入口输出数据，不直接绘图。各项分析在同一进程中顺序运行，共享内存中的轨迹聚合结果和小时电网输入。

### 绘制图件

使用完整分析生成的工作簿：

```powershell
python -B scripts/plot_figures.py --input-dir results/example_run/data --output-dir results/example_run/figures --figures 1 2 3 4 --style original --formats svg pdf png --dpi 600
python -B scripts/plot_revision_supplements.py --input-dir results/example_run/data --output-dir results/example_run/diagnostics --dpi 600
```

第一条命令绘制四组主要图件；第二条绘制任务能量闭合、额外敏感性和爬坡诊断图。绘图入口读取已有工作簿，不重新运行模型。

`results/` 根目录下的四份精简工作簿用于查看图中数值，不包含绘图所需的全部数据表。绘图输入应指定为完整运行目录中的 `data/`。

### 选择单项分析

```powershell
python -B scripts/run_experiments.py --config configs/experiments.json --only figure1 figure4 --output-dir results/sensitivity_run
```

可选项为 `figure1`、`figure2`、`figure3`、`figure4`、`comparison`、`boundary` 和 `validation`。未指定 `--only` 时，运行 `enabled: true` 的条目；显式选择时也可启用配置中关闭的条目。

每组主要图件都有独立的计算和绘图入口。例如：

```powershell
python -B scripts/experiment_figure4.py --config configs/experiments.json --output-dir results/figure4_run
python -B scripts/plot_figure4.py --input-dir results/figure4_run/data --output-dir results/figure4_run/figures
```

将 `4` 换为 `1`、`2` 或 `3`，即可使用对应入口。

### 比较年度与小时级模型

```powershell
python -B run.py --policy CP --scenarios Base --year-start 2026 --years 5 --annual-activity-mode calendar --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_run
```

输出包含两个模型各自的结果，以及 `summary/` 下的比较表。差值百分比为正表示年度模型结果高于小时级参考配置。有序模型差异分解由 `figure3` 分析生成。

### 运行功率诊断

以下命令使用已有基准和国家电量输入，不读取工作负载轨迹，也不绘图：

```powershell
python -B scripts/validate_energy_model.py --skip-trace --skip-figure --output-dir results/power_diagnostics
```

移除 `--skip-trace` 可加入工作负载轨迹边界诊断；移除 `--skip-figure` 可生成对应图件，图件路径通过 `--figure-stem` 指定。

## 配置说明

[configs/experiments.json](configs/experiments.json) 控制完整分析流程。

| 配置区块 | 内容 |
| --- | --- |
| `inputs` | 工作负载、服务器、电网、基准及国家数据路径，可选容量因子文件 |
| `common` | 国家、年份、PUE 乘数、容量分位数、利用率上限、空闲归属、严格小时模式及轨迹读取设置 |
| `hardware` | 部件功率预算、单位功率、空闲比例和响应曲线 |
| `trace` | 轨迹过滤与资源活动设置 |
| `annual_activity` | 年度模型的活动因子配置 |
| `experiments` | 分析开关、需求和电网情景、敏感性取值及诊断设置 |
| `output` | 计算结果目录，可通过 `--output-dir` 覆盖 |

JSON 配置中的路径相对于仓库根目录。默认从 `results/grid_hourly_calendar` 读取小时电网数据，将计算结果写入 `results/all_experiments_calendar`。

常用设置如下：

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `common.year_start` / `common.years` | `2026` / `5` | 计算时间范围 |
| `common.countries` | `null` | 使用容量输入表中的全部国家 |
| `common.pue_scale` | `1.0` | 对 PUE 输入应用统一乘数 |
| `common.hourly_carbon_scope` | `direct` | 配置分析使用直接电网碳因子 |
| `common.strict_hourly_carbon` | `true` | 拒绝缺失的小时电网文件和不完整日历 |
| `common.max_intervals` | `null` | 默认使用完整轨迹，设定有限值可用于调试 |
| `common.hash_inputs` | `false` | 对工作负载文件额外执行内容哈希读取 |
| `inputs.ai_capacity_factors_file` | `null` | 默认使用内置容量因子，可提供替代 JSON |

`figure3` 分析要求默认的 24 国、2026–2030 年覆盖范围。小范围计算应选择适用的分析子集。截短轨迹进行调试时，应关闭 `experiments.validation.trace_diagnostic`，或保留足够小时用于两个诊断窗口。

内置容量因子将所选国家的电量合计校准到外部全球参考。国家份额、PUE、功率曲线和电网路径均为模型输入，汇总输出对应所选国家。电网情景采用确定性假设，默认国家负荷共享同一归一化轨迹形状。

## 输出结果

完整分析和上述绘图命令生成以下目录：

```text
results/example_run/
├── data/
│   ├── figure1_data.xlsx
│   ├── figure2_data.xlsx
│   ├── figure3_data.xlsx
│   └── figure4_data.xlsx
├── comparison/
│   ├── annual_cpu_model/
│   ├── hourly_workload_resolved_model/
│   └── summary/
├── boundary/
├── validation/
├── provenance/
│   └── <run-id>/
│       ├── resolved_experiment_config.json
│       ├── Experiment_Status.csv
│       ├── input_manifest.csv
│       ├── output_manifest.csv
│       └── run_metadata.json
├── figures/
└── diagnostics/
```

`figures/` 和 `diagnostics/` 由独立绘图命令创建。

| 输出 | 内容 |
| --- | --- |
| `data/figure1_data.xlsx` | 容量、年度用电和排放，以及任务和部件能量归属 |
| `data/figure2_data.xlsx` | 电网因子、情景排放、避免排放和负荷加权碳强度 |
| `data/figure3_data.xlsx` | 模型比较、差异分解、峰值和 P99 功率、爬坡及时序对照 |
| `data/figure4_data.xlsx` | 参数敏感性、能量构成、资源诊断及守恒检查 |
| `comparison/summary/` | 两个模型的国别和所选国家合计结果及差异 |
| `boundary/` | 校准因子对照、国家总量及容量因子 JSON |
| `validation/` | 基准功率汇总、误差指标、国家数据范围检查及已启用的轨迹诊断 |
| `provenance/<run-id>/` | 完整配置、实验状态与耗时、输入输出清单和运行元数据 |
| `figures/` | 主要图件、实际绘制的图源表和图件元数据 |
| `diagnostics/` | 任务与部件闭合、额外敏感性、爬坡诊断图及图源表 |

功率使用 MW 或 GW，电量使用 MWh 或 TWh，排放使用 tCO₂ 或 MtCO₂，碳强度使用 kgCO₂/MWh。字段后缀 `_mw`、`_mwh`、`_twh`、`_tco2` 和 `_mtco2` 表示对应单位。小时爬坡图使用 GW/h。年度电量按 8,760 小时年化，再分配到完整 UTC 日历用于小时碳匹配。

## Python 接口

模型函数以字典形式返回结果表。

```python
from core.annual_cpu_model import run_annual_cpu_model

annual = run_annual_cpu_model(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    save_outputs=False,
)
country_totals = annual["annual_summary"]
```

小时级计算需要先准备外部数据：

```python
from core.hourly_workload_resolved_model import run_workload_component_footprint

hourly = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    year_start=2026,
    years=5,
    hourly_carbon_factors_dir="results/grid_hourly_calendar",
    hourly_carbon_fallback_to_annual=False,
    save_outputs=False,
)
component_energy = hourly["component_energy"]
```

小时 CSV 导出由 `save_hourly_outputs` 控制，默认关闭。

## 项目结构

```text
configs/       分析配置和小时数据来源清单
core/          年度与小时级模型、功率曲线、核算及数据处理
dataset/       容量、PUE、电网常量及外部数据位置
scripts/       分析入口、电网构造、诊断和绘图
results/       预置图源数据与生成结果
manuscript/    LaTeX 稿件和图件资源
run.py         年度与小时级模型比较入口
```

## 许可证

项目代码采用 [MIT License](LICENSE)。外部数据遵循各提供方的访问、引用与再分发条件。
