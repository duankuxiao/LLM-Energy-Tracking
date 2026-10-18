<p align="center">
  <img src="logo.png" alt="项目标志" width="180"/>
</p>

# 世界主要 AI 国家（24 国）的数据中心能耗与碳排放估算

[English README](README.md)

本仓库研究世界主要 AI 国家中 24 个国家的数据中心用电与运行碳排放。选国依据是能够获取国家容量信息和历史电网碳强度数据；其他国家缺少可比的详细输入。模型综合数据中心容量路径、原始国家分配、年度 AI 容量因子、公开 GPU 集群轨迹、部件功率、PUE 和电网碳强度。历史来源数据、未来情景、作者插值及代理假设分别标注。

仓库术语与原稿保持一致：核心比较连接“**年度 CPU 模型**”和“**小时级工作负载解析模型**”，并通过有序 bridge 依次引入分部件功率、轨迹派生的平均工作负载、小时负荷变化和小时碳强度匹配。这里的端点和中间步骤均是模型配置；小时级端点不被视为实测真值。

> **研究范围。** 本研究核算 2026–2030 年的电力消耗、运行负荷和基于所在地电网的运营碳排放，不包括隐含排放、水使用、合同购电、市场法 Scope 2 声明或严格的 24/7 无碳电力匹配。

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
| MLPerf 功率诊断与轨迹边界审查（补充图 S1） | [`scripts/validate_energy_model.py`](scripts/validate_energy_model.py)、[`dataset/validation/`](dataset/validation/) | 总入口写入 `<output-dir>/validation/`，下文独立示例写入 `results/validation_run/` |
| 全任务/部件闭合、额外敏感性与爬坡诊断（补充图 S2–S4） | [`scripts/plot_revision_supplements.py`](scripts/plot_revision_supplements.py) | 分别从图 1、4、3 的完整实验工作簿绘制 |

`results/` 根目录下的四份工作簿仅供审稿人检查主图中实际展示的数据，保留绘图数值及对应坐标、标签和单位。图 1–4 分别包含 7、4、4、2 张工作表，不包含完整实验矩阵、未展示的结果或审计元数据。图 4 仅保留主图展示的六个参数，另外九个机制控制参数见补充图 S3。

完整实验结果需要运行代码生成：每次配置运行将完整工作簿写入 `<output-dir>/data/`，验证表写入 `validation/`，运行记录写入 `provenance/`。绘图脚本要求完整工作簿及其 W08 契约，不能将精简审稿表作为绘图输入。当前修订图件对应的完整本地运行目录为 `results/all_experiments_calendar_20261017_092038_285/`；该生成目录不能替代外部输入准备与代码复现。

当前正文与图件说明见[原稿](manuscript/sn-article-template/sn-article.tex)和[补充信息](manuscript/sn-article-template/supplementary-information.tex)。

## 研究范围与情景

### 国家范围

默认模型边界为以下 24 个世界主要 AI 国家；这一称谓描述研究范围，不代表按 AI 使用量确定的排名：

`USA`、`China`、`Japan`、`France`、`India`、`Singapore`、`Canada`、`Germany`、`United_Kingdom`、`Australia`、`Italy`、`South_Korea`、`South_Africa`、`Ireland`、`UAE`、`Brazil`、`Israel`、`Netherlands`、`Spain`、`Sweden`、`Belgium`、`Norway`、`Poland` 和 `Switzerland`。

这些标识是 [`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) 与 [`dataset/Factors.py`](dataset/Factors.py) 使用的规范键。研究结果统一称为“24 个世界主要 AI 国家合计”。原始 IT 容量份额之和为 0.9318338，保留原始分母；该比例不直接代表用电覆盖率。其他国家不纳入汇总，全球 IEA 数值作为单独注明范围的外部参考。

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
| 图 1—4 展示面板的精简图源表 | 已包含 | [`results/`](results/)；完整实验数据需运行代码生成 |
| Electricity Maps 小时数据及其派生情景文件 | **不公开分发** | 需从 Electricity Maps 合法取得 |
| Alibaba GPU Pod 和服务器轨迹 | 未打包 | 需从 Alibaba Cluster Trace Program 获取 |
| IEA 数据附件 | 不重新分发 | 需从 International Energy Agency 获取 |

年度 CPU 模型可使用仓库内的年度输入独立运行。完整运行小时级工作负载解析模型和有序 bridge，需要准备下述外部负荷轨迹与小时碳强度文件。仓库内已有论文图表源数据工作簿，审稿人无需获得受限数据即可核对已报告数值。

### 容量与增长输入

[`dataset/Installed_capacity_data.py`](dataset/Installed_capacity_data.py) 存放 2025—2030 年四类需求路径下的总容量、IT 容量和非 IT 容量，以及固定国家份额和年度 AI 容量因子。

默认 AI 容量校准依据 IEA 报告 [Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai) 讨论的 AI 专用数据中心用电路径。保留的因子将全球 Base 参考电量拟合到 24 国合计，2026–2030 年目标依次为 195、249.600、314.496、386.830、465 TWh。约 465 TWh 的端点是外部全球预测，中间目标为作者定义的插值输入；该校准不代表实测用电覆盖率。两个模型端点和所有需求路径使用同一组因子，敏感性实验不重新校准。国家份额和中间情景值属于模型输入或派生参数，不应理解为原始数据提供方直接发布的逐国预测。

### GPU 工作负载与服务器轨迹

小时级工作负载解析模型读取两类 Alibaba Serverless Infrastructure 数据：

- `dataset/asi_opensource_pod_hourly/`：GPU Pod 小时记录；
- `dataset/asi_opensource_server_hourly/`：服务器资源清单小时记录。

轨迹来自 [Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata)，系统背景参见 [Heterogeneity at Hyperscale: Characterization and Scheduling of Large Production AI Clusters at Alibaba](https://www.usenix.org/conference/osdi26/presentation/li-suyi)。

能耗边界保留所有活跃 Pod，包括 GPU 请求为零的 CPU-only Pod。原稿中的任务结构审查使用独立且更严格的过滤条件：论文时间窗口内的已分类任务必须具有 GPU 显存请求和有效 GPU 使用小时。`training` 保持为训练，在线与离线推理合并为推理，`dev` 与明确标注的 `other` 合并为其他，空值、`unknown` 和异常标签列为 `unclassified`。未分类活动计入总能耗，但不进入已分类任务比例的分母。

公开轨迹缺少完整存储活动和主机内存容量，因此实现使用空闲存储功率和 CPU 加权的内存活动代理。这些假设记录在返回的 `trace_resource_capacity` 表中。

### 年度碳强度与 PUE

[`dataset/Factors.py`](dataset/Factors.py) 提供 24 国在 `CP`、`NDC` 和 `NZ` 路径下的年度电网碳强度，以及按国家、年份和需求路径区分的 PUE。碳强度单位为 `kg CO₂/MWh`。

2026–2030 年 CP 数值是存储在 `CF_CP` 中的固定输入，运行时不从历史观测重新外推。年度 CPU 模型直接使用年度因子；有序 bridge 的年度碳强度配置也使用这些因子。这套年度表采用直接排放边界，仅在 `direct` 模式下可作为缺失小时文件的回退值；生命周期模式缺文件会报错。

### 小时级电网碳强度

当前配置从 `results/grid_hourly_calendar/` 读取生成的 360 个本地 CSV：24 个国家 × 3 类电力系统路径 × 2026–2030 五年。原稿采用直接电网碳因子；小时模型的独立接口在输入适用时也支持生命周期因子。

当前来源清单从 `dataset/EM-CPNDCNZ/` 中完整的处理后 2026 年 CP 文件重建相对小时形状，这些文件归属于 Electricity Maps 来源。原始 2025 年数据获取文件不可用，重建不能恢复或认证那些原始观测。模板映射到完整 UTC 日历后，归一化至 [`dataset/Factors.py`](dataset/Factors.py) 中的固定年度路径。处理后的小时文件不公开分发。研究人员需通过 Electricity Maps [数据页面](https://www.electricitymaps.com/data)或[学术访问说明](https://help.electricitymaps.com/en/articles/13168512-academic-data-access-and-availability)取得授权输入，并遵守提供方条款。

### W03 电网情景构造与审计

现有 NDC/NZ 是简化情景（stylized scenarios）：2025 年与 CP 相同，随后按 `endpoint**((year-2025)/5)` 缩放，2030 年端点分别为 0.89 和 0.53。年度表保留两位小数，因此实际逐国比率有舍入差异。11%/47% 是端点输入隐含的约束，不能表述为模型独立识别的政策效果。原有年度数值保持不变。

`scripts/audit_grid_scenarios.py` 检查逐国逐年的小时政策恒等式、UTC 日历、直接碳强度年均闭合，并导出固定合成负荷下的均值保持/相位压力控制。这些合成控制用于检查代数关系；实际负荷的协变由小时模型导出。审计失败返回非零退出码，并保留报告。

```powershell
python -B scripts/audit_grid_scenarios.py --output-dir results/revision_audit/grid_baseline
```

旧输入的 2028 年文件采用从 1 月 1 日起连续 8760 小时，截断了 12 月 31 日。普通模式保留兼容性，但会警告并标为 `legacy_8760_truncated_leap_year`；`--strict-hourly-carbon` 同时拒绝缺文件和不完整日历。新生成器按 UTC 月/日/小时映射完整日历（平年 8760、闰年 8784 小时）；模板没有 2 月 29 日时复制 2 月 28 日相同时刻，然后重新归一化至年度目标。小时模型仍保持已有的 8760 小时轨迹年化电量约定，再将该年电量分配到完整日历上。

生成前准备 CSV 来源清单，字段为 `country,path,source_year,source_kind,zone_id`。每国一行，`path` 相对清单所在目录或使用绝对路径，`zone_id` 必须与源文件一致，英国目录映射为 `Great Britain`。原始小时数据标为 `raw_hourly`；从已有情景文件提取形状时标为 `reconstructed_shape`，不能冒称恢复了原始数据。源文件必须完整覆盖声明年份，包含声明为 UTC 或显式时区的时间戳和带单位的直接碳强度列。可选生命周期列同样需要明确单位。

```powershell
python -B scripts/build_hourly_grid_scenarios.py --source-manifest source_manifest.csv --output-dir results/revision_audit/grid_hourly --scopes direct life_cycle
python -B scripts/audit_grid_scenarios.py --hourly-carbon-factors-dir results/revision_audit/grid_hourly --output-dir results/revision_audit/grid_checked
python -B scripts/figure2_data.py --hourly-carbon-factors-dir results/revision_audit/grid_hourly --strict-hourly-carbon --output results/revision_audit/figure2_grid.xlsx
```

生成器拒绝缺测、重复/错序时间、负值和非有限值；保留源文件已有估算值并记录估算占比。`construction_audit.csv` 和 `construction_metadata.json` 记录来源/输出哈希、区域映射、模板年份、构造参数和均值。生命周期目标按“直接年度目标 × 模板生命周期/直接均值之比”构造并单独标注，不宣称与直接年度表闭合。`--variability-scale 0` 生成平坦 CI，`--variability-scale 2` 放大波动，`--phase-hours 12` 循环平移 12 小时；截断负值后恢复各自目标均值。这些是压力测试，不是电网预测。

生成和审计输出目录必须不存在，原始输入不会被覆盖。模型返回 `grid_carbon_audit` 表并保存 `Grid_Carbon_Audit_*.csv`，即使不保存小时明细，也会记录实际来源、缺失回退、日历完整性、年电量守恒、CI 均值与负荷—CI 协变恒等式。图 2 完整实验工作簿包含 `Grid_Scenario_Definition`、`Grid_Input_Audit`、`Annual_Policy_Identity`、`Policy_Effect_Audit` 工作表，区分设定乘数和实际排放比例，并记录轨迹截取参数。生命周期模式的图 2a 仍是明确标注的直接年度 CP 参考图。

回归验证：`python -B -m unittest discover -s tests -v`（使用 README 所列 Python 依赖）。`--max-intervals` 输出仅供调试，不能作为正式论文重算结果。

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
├── configs/                           # 实验配置与电网来源清单
├── results/
│   ├── figure1_data.xlsx ... figure4_data.xlsx  # 精简审稿图源表
│   ├── grid_hourly_calendar/          # 生成的完整日历电网输入
│   └── <run>/                        # 完整实验、审计与运行记录
├── scripts/
│   ├── figure1_data.py
│   ├── figure2_data.py
│   ├── figure3_data.py
│   ├── figure4_data.py
│   ├── run_experiments.py
│   ├── experiment_figure1.py ... experiment_figure4.py
│   ├── plot_figures.py
│   ├── plot_revision_supplements.py
│   └── validate_energy_model.py
├── run.py                             # 两端点比较入口
├── LICENSE
├── README.md
└── README.zh-CN.md
```

## 环境与依赖

已完成的原稿运行在 `provenance/<run-id>/run_metadata.json` 中记录 Python 3.9.25 及以下依赖版本：

- `numpy==1.26.4`
- `pandas==2.3.3`
- `pyarrow==21.0.0`
- `openpyxl==3.1.5`
- `matplotlib==3.9.2`

使用 Python 3.9.25，在仓库根目录创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pandas==2.3.3 pyarrow==21.0.0 openpyxl==3.1.5 matplotlib==3.9.2
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

### 推荐运行方式

完整分析优先使用下述共享数据入口；原有独立命令仍可用于单项分析。

### 一次读取数据后完成全部实验

先在仓库根目录运行一次小时电网数据构造，再运行全部实验：

```powershell
python -B scripts/build_hourly_grid_scenarios.py --source-manifest configs/hourly_grid_sources.csv --output-dir results/grid_hourly_calendar --scopes direct
python -B scripts/run_experiments.py --config configs/experiments.json --output-dir results/reproduction_run
python -B scripts/plot_figures.py --input-dir results/reproduction_run/data --output-dir results/reproduction_run/figures --figures 1 2 3 4 --style original --formats svg pdf png tiff jpg --dpi 600
python -B scripts/plot_revision_supplements.py --input-dir results/reproduction_run/data --output-dir results/reproduction_run/figures --dpi 600
```

第一条命令构造电网输入，第二条计算实验，最后两条绘制主图和补充图 S2–S4；补充图 S1 由下文的外部诊断命令生成。每次实验使用新的输出目录。若电网目录已成功生成且输入未变，可复用并从实验命令开始。绘图始终显式指定完整运行的 `data/` 目录。

[`configs/hourly_grid_sources.csv`](configs/hourly_grid_sources.csv) 已列出 24 国
完整的 2026 年 CP 文件，来源类型均标为 `reconstructed_shape`。现有 2028 年
旧文件只到 12 月 30 日，不能直接用于严格模式。构造脚本以各国的 2026 年
形状按 UTC 月/日/小时映射到 2026–2030 年，2 月 29 日复制 2 月 28 日，
再归一化至 CP/NDC/NZ 各年直接碳强度均值；2028 年生成 8784 小时。
这些是重建情景输入，构造记录和来源哈希写入新目录；原始文件不覆盖。
若 `results/grid_hourly_calendar` 已成功生成且输入未变，后续无需重复构造。
重跑实验时通过 `--output-dir results/新的目录` 指定新结果目录。

默认启用图 1–4、双端点比较、边界/校准对照和外部验证，只输出计算结果。
各实验在同一进程中顺序运行，共享一次流式聚合得到的轨迹和一份小时电网输入
缓存。原始 Parquet 每批处理后释放；敏感性实验基于聚合轨迹与变换数组副本
计算，各配置对应的模型计算仍分别执行。

所有 JSON 配置路径相对于仓库根目录，CSV 来源路径相对于清单所在目录。
[`configs/experiments.json`](configs/experiments.json) 已将电网输入指向
`results/grid_hourly_calendar`，结果输出指向 `results/all_experiments_calendar`，
可按上例通过 `--output-dir` 覆盖结果目录。若已采用新校准因子，设置
`inputs.ai_capacity_factors_file`。默认要求严格小时直接碳因子、
24 国和 2026–2030 年，与图 3 的既有范围一致；日历及 CI 均值校验在读取大轨迹
前执行。小范围分析通过 `--only` 选择适用实验：

```powershell
python -B scripts/run_experiments.py --config configs/experiments.json --only figure1 figure4 --output-dir results/my_sensitivity_run
```

| 配置区块 | 控制内容 |
| --- | --- |
| `inputs` | 轨迹、服务器清单、电网路径，可选校准因子 JSON，MLPerf 和国家级对照输入 |
| `common` | 国家、年份、直接碳因子严格模式、容量分位数、利用率上限、PUE 乘数、空闲归属、轨迹上限及 Parquet 批大小 |
| `hardware`、`trace`、`annual_activity` | 对应 dataclass 的部分覆盖项；共同基线一致，明确的实验对照可改变相应参数 |
| `experiments` | 实验开关、需求/电网情景、展示国家数、验证随机种子/抽样次数/窗口、图 4 敏感性取值 |
| `output` | 本次完整实验的新输出目录；已有目录会拒绝使用 |

未指定 `--only` 时按顺序执行 `enabled=true` 的实验；显式指定 `--only` 可选择
任何已配置实验，包括开关为 false 的条目。配置读取只保留直接碳因子与图 3
研究范围约束；物理参数与核算恒等式由模型函数检查。调试截断时须关闭验证
的 `trace_diagnostic`，或保留足够小时供两个窗口统计。验证沿用自身的活动
筛选规则，其逐小时统计来自首次读取的同一批数据，不借用不同口径的模型负载。

每张图都有独立实验入口和独立绘图入口：

| 图 | 实验代码 | 绘图代码 |
| --- | --- | --- |
| 1 | [`scripts/experiment_figure1.py`](scripts/experiment_figure1.py) | [`scripts/plot_figure1.py`](scripts/plot_figure1.py) |
| 2 | [`scripts/experiment_figure2.py`](scripts/experiment_figure2.py) | [`scripts/plot_figure2.py`](scripts/plot_figure2.py) |
| 3 | [`scripts/experiment_figure3.py`](scripts/experiment_figure3.py) | [`scripts/plot_figure3.py`](scripts/plot_figure3.py) |
| 4 | [`scripts/experiment_figure4.py`](scripts/experiment_figure4.py) | [`scripts/plot_figure4.py`](scripts/plot_figure4.py) |

例如，单独计算图 4，再用另一条命令绘图：

```powershell
python -B scripts/experiment_figure4.py --config configs/experiments.json --output-dir results/figure4_recalculated
python -B scripts/plot_figure4.py --input-dir results/figure4_recalculated/data --output-dir results/figure4_drawn
```

图 1/2/3 使用相应编号。独立实验的默认目录为 `output.directory` 加图名后缀。
独立绘图仅读取导出的工作簿，不重新运行模型或读取原始轨迹。本配置的图源
目录为 `<output-dir>/data`（例如 `results/reproduction_run/data`），绘图时使用 `--input-dir` 指定；
绘图脚本自身的默认目录仍为 `results/all_experiments/data`。
原有 `figure*_data.py` 接口保留；需要总配置时使用
`experiment_figure*.py`。

默认计算结果结构为：

```text
results/all_experiments_calendar/
├── data/figure1_data.xlsx ... figure4_data.xlsx
├── comparison/
├── boundary/
├── validation/
└── provenance/<run-id>/
    ├── resolved_experiment_config.json
    ├── Experiment_Status.csv
    ├── Shared_Data_IO.csv
    ├── Shared_Grid_Source_Selection.csv
    ├── Shared_Validation_Hour_Aggregates.csv
    ├── input_manifest.csv
    ├── output_manifest.csv
    └── run_metadata.json
```

完整实验工作簿在 W08 图源契约之外包含 `Shared_Experiment_Configuration`。总运行记录
包含实际配置、逐实验状态和耗时、数据读取计数、源码版本及输入/输出清单。
实验失败时停止后续实验并记录已完成产物；端点比较复用总入口的同一份
运行记录；独立运行端点比较时仍生成自身的 W09 记录。边界实验比较自己的原/重建校准因子，不自动替换主图使用的因子。

数据只在单次进程内共享，不再使用磁盘缓存、缓存版本管理或刷新选项。需要
一次读取原始数据后输出全部结果时，使用总入口；分别运行单图实验命令会启动
独立进程，因此各自重新读取输入。`common.hash_inputs=true` 会额外读取全部
Parquet 计算哈希；保持默认 false 可避免此开销。电网 CSV 从解析所用的同一份
字节缓冲计算哈希。共享上下文仅保留聚合输入，不累积所有敏感性实验的大结果表。

下文 W03–W09 工作表名称均指完整实验输出，不指 `results/` 根目录下的精简审稿表。

### 独立边界与校准对照

默认因子保留原校准结果：它把全球参考电量目标拟合到了 24 国合计。下列命令比较原因子、重新计算的全球参考因子，以及“覆盖地区与其他地区单位 IT 容量设施用电相同”的条件对照。两端点使用同一组因子。条件对照不会自动替代正式主分析。

```powershell
python -B scripts/analyze_boundary_calibration.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/revision_audit/boundary
```

先检查 `calibration_audit.csv`、`endpoint_comparison.csv` 和 `invariant_checks.csv`。输出目录必须是新目录。`--max-intervals` 仅供调试，调试生成的因子会被正式入口拒绝读取。

确认使用同质强度条件对照后，可将导出的共同因子传给比较入口：

```powershell
python -B run.py --policy CP --scenarios Base --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --ai-capacity-factors-file results/revision_audit/boundary/ai_capacity_factors_homogeneous_intensity_boundary.json --output-dir results/revision_audit/boundary_models
```

图 1–4 数据脚本同样支持 `--ai-capacity-factors-file`，完整工作簿包含 `Accounting_Boundary` 和 `AI_Capacity_Factors` 工作表。图 1a 容量列为覆盖国家合计，`source_global_*` 列保留外部参考；图 1b 完整表区分原始分配份额和 24 国内部份额，主图和精简审稿表仅使用原始全球分母。图 2c 完整表名为 `Fig2c_Selected_Countries_Carbon`，精简审稿表名为 `Fig2c_Annual_Emissions`。审稿表仅保留图中展示的数据。

### 比较两个端点模型

```powershell
python -B run.py --policy CP --scenarios Base --year-start 2026 --years 5 --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_run
```

入口程序读取一次工作负载轨迹，运行年度 CPU 端点和小时级工作负载解析端点，并将结果写入指定的 `--output-dir`（省略时为 `results/model_comparison/`）。可同时指定多个需求路径：

```powershell
python -B run.py --policy NDC --scenarios Base "Lift-Off" "High Efficiency" --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_ndc_run
```

快速执行检查可限制国家和轨迹小时数：

```powershell
python -B run.py --countries USA --year-start 2026 --years 1 --max-intervals 24 --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_debug_run
```

`max_intervals` 会改变轨迹采样基础，仅用于 smoke test 和调试；生成论文结果时必须省略。

### 独立图源数据接口

上述配置入口生成全部四份完整工作簿。若使用保留的独立数据接口，必须显式指定其他输出文件；其默认路径是 `results/` 根目录下的精简审稿表，直接无参数运行会将其替换为完整实验工作簿。

```powershell
python -B scripts/figure1_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure1_data.xlsx
python -B scripts/figure2_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure2_data.xlsx
python -B scripts/figure3_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure3_data.xlsx
python -B scripts/figure4_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/standalone_data/figure4_data.xlsx
```

图 3 脚本是有序 bridge 的权威实现。写入工作簿前，它会检查配置覆盖、国家聚合、小时—年度能耗闭合以及 bridge 加和闭合。

### W04 年度活动与公平比较

年度模型及图 3 共用 [`core/annual_activity.py`](core/annual_activity.py)。默认 `calendar` 定义以 2025 年为起点、2030 年为终点，分母固定为 5，2030 年训练活动为 0.925，推理/其他/未分类活动为 0.7。输出区间仅用于切片，不改变同一年活动系数。`fixed` 将活动保持在 2025 年的 0.9/0.5；`legacy_six_year` 固定使用分母 6，复现旧 2025–2030 完整运行的 0.9208333/0.6666667 终点。三者均为作者设定的参考假设。

以下命令均在仓库根目录执行，使用上述 Python 环境。首先独立导出三个年度活动配置，无需读取工作负载轨迹：

```powershell
foreach ($activityMode in @("calendar", "fixed", "legacy_six_year")) {
    python -B -m core.annual_cpu_model --policy CP --scenarios Base --year-start 2026 --years 5 --activity-mode $activityMode --output-dir "results/revision_w04/annual_$activityMode"
}
```

各目录保存 `Annual_Activity_Factors.csv`、`Annual_Reference_Parameters.csv` 和国家/选定国家合计的年度结果。任务明细保留活动系数、假设利用率和实际利用率贡献，国家结果另含单位容量电量和排放。

正式生成图 3 及完整 W04 对照，使用 W03 已生成的完整日历小时输入：

```powershell
python -B scripts/figure3_data.py --annual-activity-mode calendar --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output results/revision_w04/figure3_w04.xlsx
```

该命令读取完整轨迹一次，自动计算三种年度活动配置，以及共同均值负荷的固定活动和对称活动趋势配置。共同均值先按小时模型相同的规则截断资源利用率，再对每个任务和资源取均值；年度聚合模型使用相同的 CPU 均值，分部件模型另使用明确记录的 GPU/内存/存储均值。对称趋势按任务的 `a(year)/a(2025)` 同时施加到两端共同负荷，截断前后数值均导出。两种共同均值配置使用平坦负荷与相同的年度直接碳强度，不夹带小时 CI 匹配差异。运行时检查共同 CPU 活动和安装容量一致性。

新增工作表如下：

| 工作表 | 内容 |
|---|---|
| `W04_Activity_Factors` | 三种活动模式的完整 2025–2030 系数、锚点与分母 |
| `W04_Parameter_Assumptions`、`W04_Configuration_Matrix` | 作者设定参数及各比较实际改变的维度 |
| `W04_Control_Comparison` | 国别与选定国家合计的绝对差、分别以两个端点为分母的相对差、单位容量指标和逐年变化 |
| `W04_Baseline_Impact` | 相对固定活动及旧六年分母的年度电量/排放变化，区分公式修正与活动 ramp |
| `W04_Common_Mean_Load` | 共同负荷、任务乘数、资源截断和实际资源活动 |
| `W04_Run_Metadata`、`W04_Grid_Input_Audit` | 轨迹截取、共同校准配置、小时因子来源和缺失回退 |
| `Source_Bridge_Variants`、`Source_Country_Bridge` | 五个 bridge 配置和有序贡献的完整来源 |

所有对照使用同一套容量因子，不分别重新校准。若 W02/W01 已确定新的因子文件，向年度、图 3 和 `run.py` 命令传入同一个 `--ai-capacity-factors-file` 路径。未指定时继续采用仓库默认因子。百分比列名明确标注分母；模型间差值表示给定配置的表示差异。有序 bridge 的贡献依赖替换顺序。生命周期没有对应的年度比较表，因此年度—小时比较入口要求 `direct`；单独的小时模型仍可按其接口使用生命周期文件。

图 3 小时覆盖检查兼容完整 UTC 日历（2028 年 8784 小时），普通模式也会保留并标注旧 8760 小时截断输入。运行 `run.py` 时可用 `--annual-activity-mode calendar|fixed|legacy_six_year` 选择年度端点。只需查看缩短轨迹的运行输出时，可为图 3 增加 `--max-intervals 24` 并另选输出文件；此类结果不可作为正式论文结果。

### W05 全任务能量归属、空闲/动态拆分与守恒

小时模型通过 [`core/energy_attribution.py`](core/energy_attribution.py) 直接导出四任务（training/inference/other/unclassified）×五部件（CPU/GPU/memory/storage/it_fan）的完整矩阵。`task_component_energy` 是所选规则的国别年度矩阵；`task_component_energy_all_rules` 包含三种空闲归属规则。两者均给出空闲、动态、IT 和 PUE 后设施能量的绝对值，以及明确分母的任务内、部件内和全系统份额。总功率仍按现有部件功率方程计算，归属比较共用同一次功率结果。

| `--idle-allocation-rule` | 空闲能量归属 | 闭合定义 |
|---|---|---|
| `hourly_driver`（默认） | 按每小时部件驱动活动份额分摊 | 四任务求和等于部件和 IT 总能量 |
| `annual_driver` | 按部件全年驱动活动份额固定分摊 | 四任务求和等于部件和 IT 总能量 |
| `separate_pool` | 空闲能量保留在独立池；四任务仅分摊动态能量 | 四任务动态能量加空闲池等于部件和 IT 总能量 |

三种规则的动态能量均按小时驱动活动分摊。无部件驱动活动时，回退改用各资源按容量归一化后的代理份额，避免混加 CPU 核时、GPU 时和内存等原始单位；所有资源均无活动时采用显式的四任务均分参考。`allocation_audit` 记录驱动是否可用、零驱动时段和回退依据。storage 缺少动态观测，memory 是 CPU 加权代理；这类归属为模型假设。GPU-memory 字段不能替代主机内存容量或该代理。

图 1 完整工作簿直接读取模型矩阵，包含 storage 和 `Fig1_Unclassified` 工作表。跨国家和年度的份额均先汇总绝对能量，再用相应能量总量作分母；`display_share_pct` 包含任务的全部五个部件。独立空闲池模式下，任务能量占系统比例之和小于 100%，剩余部分在 `W05_Idle_Pool` 显式列出。设施能量按同一国家年度 PUE 比例归属，不能解释为实测部件冷却能量。

在已配置依赖的项目 Python 环境中，从仓库根目录手动生成图 1 图源及三种归属对照：

```powershell
python -B scripts/figure1_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --idle-allocation-rule hourly_driver --output results/revision_w05/figure1_w05.xlsx
```

该命令默认包含四个需求情景，轨迹只读取一次，三种归属对照自动随每个情景输出；不需要分别重复运行三种规则。若要用独立空闲池作为任务面板的主配置，改为 `--idle-allocation-rule separate_pool` 并选择另一个输出文件。

图 1 的主要新增工作表如下：

| 工作表 | 内容 |
|---|---|
| `W05_Task_Component_Country`、`W05_Task_Component_All_Rules` | 所选规则及所有规则的国别完整矩阵 |
| `W05_Task_Component_Annual` | 先汇总国家绝对能量的年度矩阵和份额 |
| `W05_Task_System_Shares`、`W05_Component_System_Shares` | 四任务占全系统比例及五部件总体构成 |
| `W05_Physical_Component_Energy`、`W05_Idle_Pool` | 部件模型功率对应的能量、空闲/动态拆分及未分摊空闲池 |
| `W05_Allocation_Audit`、`W05_Energy_Conservation`、`W05_System_Closure` | 回退规则、逐部件/国家能量闭合及设施总量闭合 |
| `W05_Resource_Audit`、`W05_Global_Resource_Audit` | 原始资源量、全局/国别截断、分配后实际活动及守恒残差 |
| `W05_Power_Nonlinearity`、`W05_Proxy_Metadata` | 截断后平均活动控制、功率曲线及资源代理的单位/边界 |

生成图 4 敏感性和机制诊断：

```powershell
python -B scripts/figure4_data.py --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --idle-allocation-rule hourly_driver --output results/revision_w05/figure4_w05.xlsx
```

图 4 的 `W05_Resource_Changes` 比较各资源的源活动、容量归一化活动、截断和实际资源量；`W05_Control_Checks` 分别记录源资源量、截断后归一化活动、电量和 CI 均值是否保持。任务混合调整须保留所有源资源总量及 unclassified 活动；CI-only 控制须逐国逐年保留电量和 CI 均值。其他参数下，电量变化不因资源总量保持而被判为错误。

`W05_Power_Nonlinearity` 比较实际小时功率积分与“在相同截断后平均利用率下计算功率”的控制：GPU 的 log2 曲线、IT 风扇的三次曲线与 CPU/内存/存储线性曲线分别标注。这一控制固定实际均值，用于区分非线性和截断机制，不将原始资源量、电量和平均 CI 混称 activity。敏感性主表还直接列出模型的总体 GPU、unclassified、空闲和空闲池能量比例；GPU 预算 OAT 同时改变非 GPU 预算，不能据其斜率直接反推真实 GPU 份额。

图 4 同时导出逐实验的完整矩阵、部件能量、空闲池、归属/守恒、电网和代理诊断。`run.py` 也支持 `--idle-allocation-rule`，并在小时模型输出目录保存同名诊断 CSV。若已选定 W02/W01 的容量因子文件，向各入口传入同一 `--ai-capacity-factors-file`。正式结果使用完整轨迹；截取轨迹的输出参数记录在 `W05_Run_Metadata` 中。

### 运行外部诊断

```powershell
python -B scripts/validate_energy_model.py --bootstrap-replicates 10000 --seed 20260819 --output-dir results/validation_run --figure-stem results/validation_run/figures/supplementary_energy_validation
```

该独立命令生成补充图 S1 和诊断表，并自行读取轨迹窗口；配置总入口则在共享数据读取时导出验证表，不绘图。诊断采用固定随机种子 `20260819`，按 Public ID 进行 10,000 次 bootstrap。输入来源和筛选规则记录在 [`dataset/validation/`](dataset/validation/) 中。

### W06 接缝、国别形状与时序稳健性

小时模型新增 `temporal_alignment`、`temporal_metrics`、`trace_seams` 三张结果表，保存结果时写入同名 CSV。它们记录源小时数量、重复次数、每个重复边界的目标小时和功率跳变、日历复制/位置重采样、年度电量归一化因子及闭合残差。`hourly_carbon` 增加源小时位置、重复周期、接缝和对齐异常标记。即使采用年度 CI，时序诊断仍使用轨迹负荷，不用平坦负荷代替。

爬坡使用相邻 UTC 小时功率之差的绝对值；不计算年末到年初的循环爬坡。原始最大值、排除重复接缝的最大值、进一步排除对齐异常的最大值，以及原始/去接缝的 P95、P99 均单独导出。若过滤后没有可用边，指标为缺失值。图 3c/d 保留原始列，同时增加上述指标、接缝是否决定最大值的标记，名称为 `Maximum absolute hourly ramp`，单位为 `GW h^-1`；功率单位为 `GW`。共享归一化形状现在报告为诊断，不再要求所有国别曲线必须相同。聚合峰值先按同一 UTC 小时加总功率，再取最大值；各国各自峰值之和仅作为同时性对照。

图 3 和图 4 复用已计算的小时电量和 CI，各导出包含基准在内的 17 个时序配置：负荷平移 −6/+6/+12 h、国别相位分散、国别形状分散、负荷波动缩放 0.5/1.5、3 h 平滑、CI 平移 −6/+6 h、负荷与 CI 同向/反向平移、平坦 CI、CI 波动缩放，以及 CI 均值 ±10%。正相位表示向较晚 UTC 小时循环移动，产生的额外年末—年初连接单独标为对齐异常。国别相位按排序后国家依次赋值 −12 至 +12 h，国别波动系数循环取 0.5、1、1.5；这是明确的合成压力情景，不是国别时区或行为证据。平滑仅为敏感性处理，受平滑影响的异常邻边在额外对齐过滤指标中排除。

| 工作表 | 内容 |
|---|---|
| `W06_Trace_Alignment` | 基准/逐实验的对齐规则、接缝数量、年电量归一化和闭合 |
| `W06_Baseline_Peak_Ramp`、`W06_Baseline_Seams` | 图 3 基准的空闲底座、峰均比、完整日历爬坡和接缝事件 |
| `W06_Timing_Country`、`W06_Timing_Aggregate` | 逐国与同一 UTC 时刻合成后的压力结果，含峰值、爬坡、碳排放和协变恒等式 |
| `W06_Timing_Parameters`、`W06_Timing_Metadata` | 每国实际相位/波动/平滑/CI 均值参数、保持量、日历和假设范围 |
| `W06_Seam_Events` | 时序压力情景中每个重复接缝、相位连接及平滑影响位置 |
| `W06_Model_Temporal_Metrics`、`W06_Model_Seams`、`W06_Metric_Changes` | 图 4 各功率/负荷实验的空闲底座、峰均比、爬坡和相对基准变化 |

两类控制的保持量分别定义。上述时序压力在功率计算后逐国保持年度设施电量；负荷变形不推断资源小时守恒，也不强制物理空闲底座。除明确的 CI 均值 ±10% 情景，CI 年均值保持，但负荷加权排放可以因协变改变。图 4 另增加 `pre_power_resource_variability_scale=0.5/1/1.5`：在功率计算前逐任务、逐资源保持源资源总量，固定容量分母和硬件参数，重新计算容量截断和非线性功率。截断后资源量、电量及空闲占比是否保持由 W05/W06 表分别报告，不预设相等；已有 GPU 空闲比例实验也导出时序指标，供检查空闲底座对峰均比和爬坡的影响。

采用完整 UTC 日历匹配，2028 年为 8,784 h；年度电量仍是完整轨迹按 8,760 h 年化后分布到目标日历的总量，不重新校准。旧的截短闰年输入和年度 CI 回退继续显式标注。正式运行使用图 3/图 4 入口，并指定 `--hourly-carbon-factors-dir results/grid_hourly_calendar` 和 `--strict-hourly-carbon`。补充图 S4 展示原始、异常边过滤及 P95/P99 爬坡诊断；完整 W06 表保留时序控制和接缝记录。

`validate_energy_model.py` 的接缝检查仍只读取两个端点窗口，新增局部代理 P95/P99、接缝标记和适用范围。它不声称验证全轨迹功率或国家爬坡；完整日历的模型指标应读取上述 W06 表。

### W07 参数作用机制、验证分层与守恒

新增 `core/power_parameter_audit.py`，小时模型返回并保存 `hardware_parameters`、`parameter_dependencies`、`capacity_parameter_audit`、`mechanism_conservation`。前两张表记录实际配置与方程；容量表逐国逐年记录部件 MW 预算、单位瓦数、源轨迹容量分母、模型资源容量、空闲比例、响应指数及实际利用率。`capacity×unit_W` 回算预算只是配置恒等式，明确标记为非外部验证。

CPU/GPU 功率响应新增 `cpu_dynamic_exponent`、`gpu_dynamic_exponent`，默认均为 1，分别保持原有 CPU 线性响应与 GPU `log2(1+u)` 响应。压力配置使用 `u**alpha` 和 `log2(1+u**alpha)`，端点仍为 0、1；指数仅作假设对照，没有用 MLPerf 整机功率拟合。非线性诊断记录实际指数与公式，不再固定标为线性/凹曲线。

图 4 复用一次加载的轨迹，新增 CPU/GPU 单位瓦数 0.8/1/1.2 倍的两类控制：

- `*_unit_power_scale_fixed_idle_ratio` 同步缩放 full watts 和 idle watts。固定预算、轨迹分母、响应曲线和执行权重后，模型资源容量及模型资源小时应按瓦数反比变化，归一化活动、各部件电量、设施电量及排放应保持；运行时逐国逐年检查。
- `*_unit_power_scale_fixed_idle_watts` 仅改变 full watts，空闲瓦数保持，idle/full 比例随之变化。仍检查容量及资源小时的反比关系，但不要求电量或排放保持。

正对照包括 CPU/GPU 预算、空闲比例及动态响应指数。预算变化时其他部件份额按比例调整，以保持总 IT 预算。指数默认对照为 0.75/1/1.25；自定义基准时按基准指数缩放。`W07_Control_Evidence` 分别报告部件电量和设施电量是否出现可检出的响应，不将某个运行条件下无响应自动解释为参数普遍无效。`configured_system_W_per_GPU` 同时显示外部比较参考值：它可能随单位 GPU 瓦数改变，而容量归一化模型的电量在固定 idle/full 条件下保持。

| 图 4 工作表 | 内容 |
|---|---|
| `W07_Hardware_Parameters`、`W07_Parameter_Dependencies` | 每个实验的实际参数和依赖方程 |
| `W07_Capacity_Parameters`、`W07_Parameter_Changes` | 国别容量、源分母、利用率、idle/full、指数及相对基准变化 |
| `W07_Control_Evidence` | 正负对照类型、保持量、反比缩放检查及实际电量响应 |
| `W07_Mechanism_Conservation` | 资源分配/截断、任务加空闲池、部件/IT/设施/PUE、小时电量及碳协变恒等式的残差与容差 |

守恒检查区分源轨迹资源小时与模型资源小时，不要求两种单位直接相等。模型资源量须满足“容量×平均实际利用率×8,760 h”；设施电量按 PUE 闭合。默认容差为绝对 `1e-6` 加相对 `1e-10`，非法闭合会中止运行并报告失败记录。这些检查用于模型核算一致性，不代表外部预测精度。

MLPerf 诊断在资格筛选前去除完全重复的原始行，并审查同一 Public ID 内组织、系统名、加速器型号/数量、节点数和版本是否一致；冲突 ID 导出后排除，不通过取加速器数量中位数处理差异。所有统计在各分层内先按 Public ID 汇总，再重采样；同一个 ID 可以出现在多个任务分层，分层之间不声称相互独立。Public ID 是分析单位，不保证不同 ID 对应不同物理硬件。

新增输出包括 `mlperf_public_id_audit.csv`、`mlperf_stratified_metrics.csv`、`mlperf_hardware_coverage.csv`、`mlperf_a100_h100_comparison.csv`、`validation_claim_scope.csv` 和 `validation_run_metadata.csv`。分层覆盖硬件族、型号中显式报告的 SXM/PCI 标签、单/多节点、Server/Offline 及推理任务；整机单位筛选不推断冷却或 GPU 部件测量边界。A100/H100 覆盖、各族误差和区间分别输出。单个独立 ID 只报告点估计，95% 区间为缺失；A100/H100 任一组不足两个 ID 时，组间中位数差不生成区间。固定种子和批量 bootstrap 保持结果可复现，并控制内存占用。

保留旧文件名和统计字段以便后续衔接，但 `observed_power_max_w` 的含义是“该 ID 保留的任务报告功率中的最大值”，不是实测时序峰值，也不假设 GPU 满利用率。补图入口的标签相应更新。整体样本或硬件族误差仅属于 `external configuration diagnostic`，不验证 GPU 空闲/动态曲线、训练功率、PUE/设施能量或未来国家级精度，也不据样本百分位宣称配置保守。没有拟合曲线；未来拟合需要独立系统训练/验证划分及利用率—功率观测。

图 4 和验证入口均支持 `--hardware-config-file`（`HardwarePowerConfig` 字段的 JSON 覆盖），实际配置完整导出；不传时沿用默认参数。自定义配置的对照仅保留合法份额和 idle/full 范围内的取值。验证入口支持 `--skip-figure`，可只导出诊断表，配合 `--skip-trace` 避免读取轨迹。配置总入口将 `hardware` 设置同时用于模型实验和外部诊断。

### W08 从完整实验工作簿绘图

完成配置运行后，从该运行的完整工作簿绘图：

```powershell
python -B scripts/plot_figures.py --input-dir results/reproduction_run/data --figures 1 2 3 4 --style original --formats svg pdf png tiff jpg --dpi 600 --output-dir results/reproduction_run/figures
python -B scripts/plot_revision_supplements.py --input-dir results/reproduction_run/data --output-dir results/reproduction_run/figures --dpi 600
```

绘图器要求工作簿包含 `W08_Figure_Contract`、`W08_Panel_Metadata`、
`W08_Display_Values` 和 `Accounting_Boundary`。精简审稿表不含这些工作表，
不能作为绘图输入。可通过 `--figure1-workbook` 至
`--figure4-workbook` 分别指定其他修订目录中的图源。默认导出可编辑 SVG/PDF
及 PNG 预览；`--formats svg pdf tiff` 改为输出 TIFF。每张图同时导出图注、
源数值、面板说明和哈希清单。已有输出需要 `--overwrite` 或新的输出目录。

`--style original` 是默认风格，与当前主图布局一致。图 1a 的深色 AI 专用 IT 容量区域位于总设施容量内，二者不相加；c–e 面板展示训练、推理和其他任务各自的五部件能量比例。补充图 S2 保留四类任务、系统部件占比和空闲池闭合。图 2 展示年度 CP 参考、避免排放及负荷加权碳强度。图 3 分开标注峰值功率（GW）与最大小时绝对爬坡（GW h⁻¹）。

完整敏感性实验变化 15 个参数：图 4 分别展示六个参数的用电和排放响应，补充图 S3 展示另外九个机制控制。每个百分比使用该指标自身的基线作为分母，先逐年加总研究国家，再取 2026–2030 年均值。这些是确定性的单因素响应，不是联合不确定性区间。`--style expanded` 提供可选诊断布局，不复现当前主图面板。导出后应按最终尺寸检查版面。

`scripts/refresh_manuscript_figures.py --sync-manuscript` 还会把完整实验工作簿复制到根目录的四个审稿表路径，并重写 LaTeX 图注，不保留精简审稿表。当前数据布局应采用上述绘图命令，再单独按展示数值准备审稿表。

### W09 统一端点配置与运行记录

[`run.py`](run.py) 保留双端点比较和原有 Python 调用接口，通过
[`core/run_provenance.py`](core/run_provenance.py) 将 W04 年度活动、W05 空闲归属
和 W07 硬件配置纳入同一份实际运行记录。模型结果保留原目录结构；每次保存
结果时，另建 `output-dir/provenance/<UTC 时间戳及唯一编号>/`。

准备好 W03 完整日历小时电网输入后，完整轨迹运行示例为：

```powershell
python -B run.py --policy CP --scenarios Base --year-start 2026 --years 5 --annual-activity-mode calendar --idle-allocation-rule hourly_driver --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --hash-inputs --output-dir results/revision_w09/base_cp
```

采用 W01/W02 选定的新因子时，显式附加 `--ai-capacity-factors-file`；省略时
继续使用**将全球参考电量拟合到覆盖国家合计的原校准因子**。更换硬件配置不会
自动重新校准因子。双端点比较只接受 `direct`；`life_cycle` 可由小时模型的
独立接口运行，但不能当作与现有年度模型同口径的比较。

| 参数 | 含义 |
| --- | --- |
| `--hardware-config-file` | `HardwarePowerConfig` 的 JSON 覆盖项，与图 4 和验证入口采用相同字段 |
| `--trace-config-file` | `Alibaba2026TraceConfig` 的 JSON 覆盖项，包括 CPU-only Pod、活动状态和 GPU 负载截断 |
| `--annual-activity-mode` | `calendar`（默认）、`fixed` 或 `legacy_six_year` |
| `--annual-activity-config-file` | 年度活动端点及可选模式的 JSON，与 `--annual-activity-mode` 互斥 |
| `--idle-allocation-rule` | `hourly_driver`、`annual_driver` 或 `separate_pool`，沿用现有归属规则 |
| `--parquet-batch-size` | Parquet 每批读取行数，默认 65,536，与轨迹小时截断参数不同 |
| `--trace-read-log` | 可选的新 JSONL 文件路径；默认位于本次运行记录目录，拒绝覆盖已有日志 |
| `--hash-inputs` | 对实际打开的每个 Parquet 文件额外计算 SHA-256，会增加全文件读取开销 |
| `--max-intervals` | 调试用轨迹小时上限，记录为截断；正式分析省略 |

配置 JSON 可只覆盖部分字段，其余使用对应 dataclass 默认值；未知字段会报错。
轨迹配置示例：

```json
{"trace_anchor_year": 2001, "include_zero_gpu_pods": true, "active_states": ["Running"], "cap_gpu_sm_by_request": true}
```

年度活动配置示例：

```json
{"mode": "calendar", "training_start": 0.90, "training_target": 0.925, "inference_start": 0.50, "inference_target": 0.70}
```

命令行语法错误（包括互斥选项）在开始运行前拒绝。配置文件在运行记录范围内
加载，JSON 错误和对象/文件冲突会记录失败阶段。Python 调用仍可直接传配置
对象，但同类设置不能同时传对象与
文件。`save_outputs=False` 时运行记录和表格只在返回值中提供，并拒绝额外指定
`trace_read_log`。

| 运行记录文件 | 内容 |
| --- | --- |
| `run_metadata.json` | 请求与实际配置、UTC 起止时间、耗时、状态、失败阶段及 traceback、Python/依赖版本、命令行 argv 或 API 标识、Git HEAD/本地改动/未跟踪源码、覆盖及回退计数 |
| `input_manifest.csv` | 本地源码及配置、选定电网 CSV、读取回调实际观测到的 Parquet；大小、mtime、哈希状态及运行后变化检查 |
| `Resolved_Model_Inputs.csv` | 国家×情景×年份的原 PUE 与实际 PUE、IT 容量及分配份额、AI 因子、年度直接碳因子 |
| `AI_Capacity_Factors.csv`、`Annual_Activity_Factors.csv`、`Annual_Reference_Parameters.csv`、`Hardware_Parameters.csv` | 本次实际使用的数值参数 |
| `Grid_Input_Selection.csv`、`Grid_Input_Audit.csv` | 运行前文件选择，以及实际小时/年度来源、日历覆盖、CI 闭合与模型报告的文件哈希 |
| `Trace_Resource_Capacity.csv`、`Workload_Trace_Metadata.csv`、`Temporal_Metrics.csv` | 实际轨迹规模、年化系数、容量清单/代理来源及时间指标 |
| `Allocation_Audit.csv`、`Fallback_Summary.csv` | 电网回退、日历限制、CPU/GPU 容量清单回退、内存/存储代理及空闲能量归属回退 |
| `events.jsonl`、`trace_read.jsonl` | 及时刷新的阶段事件和逐文件/逐批轨迹读取进度，属于结构化日志，不是 stdout 副本 |
| `output_manifest.csv` | 本次观察到的新建/改写模型输出及运行记录文件的 SHA-256；此前已有且未变化的模型文件另行标识，不计为本次产物 |

源码、配置文件和选定电网 CSV 始终计算哈希。未启用 `--hash-inputs` 时，Parquet
明确标记未请求内容哈希，文件大小和 mtime 不能替代内容哈希。轨迹清单只包含
实际打开的文件，调试截断不代表审计了完整轨迹。电网覆盖按政策/国家/年份/
范围的唯一输入计数，避免在多个需求情景间重复统计；资源代理与缺失输入回退
分别标注。

正式配置应使用新的 `--output-dir`。复用目录会保留历史运行记录，但原模型和
summary 子目录中的 CSV 仍可能被覆盖，旧清单的哈希对应旧内容。可捕获的
Python 异常记录为 `failed`，键盘中断记录为 `interrupted`；原生库崩溃或强制
结束可能留下没有完成时间的 `running`，应结合最后一条已刷新的读取事件判断。
哈希或 mtime 变化标记用于识别运行中被改动的输入。`run_metadata.json` 与
`output_manifest.csv` 不做自引用哈希。

这些记录说明本地实际用了哪些文件和设置，不证明来源许可、投稿包完整性、
远程可访问性或科学有效性。`audit_revision_inputs.py` 的来源账本继续记录未
解决的来源缺口。图源和验证脚本仍为独立入口，应显式传入适用配置并检查各自
元数据；`run.py` 不自动执行它们，也不静默传递它们不支持的选项。

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
    hourly_carbon_factors_dir="results/grid_hourly_calendar",
    hourly_carbon_fallback_to_annual=False,
    output_dir="results/hourly_api_run",
    save_hourly_outputs=False,
)
```

小时接口的 `hourly_carbon_scope` 可设为 `direct` 或 `life_cycle`。设定 `hourly_carbon_fallback_to_annual=False` 会拒绝缺失文件和不完整日历，对应命令行的 `--strict-hourly-carbon`。小时 CSV 默认不保存，以避免生成过大的文件。

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
| `hourly_carbon_factors_dir` | 处理后的小时电网输入目录 | 当前配置为 `results/grid_hourly_calendar` |
| `hourly_carbon_fallback_to_annual` | 小时文件缺失时使用年度直接因子 | API 默认为 `True`，正式严格运行关闭 |
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
├── provenance/
│   └── <run-id>/
│       ├── run_metadata.json
│       ├── input_manifest.csv
│       └── output_manifest.csv
└── summary/
    ├── Endpoint_Models_Country_Annual.csv
    ├── Endpoint_Models_Selected_Countries_Annual.csv
    ├── Endpoint_Comparison_Country_Annual.csv
    ├── Endpoint_Comparison_Selected_Countries_Annual.csv
    └── Accounting_Boundary.csv
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
- 默认容量分配下，各国继承同一相对轨迹的缩放形状；W06 国别异质性配置是合成压力情景，不是独立国别行为预测。
- 4,440 小时 Alibaba 轨迹重复到 CI 的 UTC 日历（2028 年 8,784 小时），再保持 8,760 小时年化电量。拼接边界可能决定未过滤的最大爬坡，应同时查看接缝标记和过滤指标。
- 未来小时电网序列继承处理后 2026 年 CP 模板的重建形状；原始 2025 年获取文件不可用。不预测调度、储能、进口和可变电源结构变化。
- 部件份额、单位功率、空闲比例、风扇响应、内存活动和存储活动均为可配置参考假设。
- MLPerf 诊断仅检查功率量级和偏差，不是校准拟合，也不验证未来国家级设施用电。
- 兼容模式允许回退到年度直接因子；正式配置采用严格小时输入，拒绝缺失文件和不完整日历。解释时间结果前应检查 `carbon_factor_source`。
- 小时级工作负载解析端点是透明参考配置，不是实测真值。

## 许可证

项目代码采用 [MIT License](LICENSE)。第三方数据仍受各原始提供方的访问、引用和再分发条款约束。
