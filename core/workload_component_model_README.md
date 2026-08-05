# workload_component_model.py 说明

本文档说明 `core/workload_component_model.py` 的计算流程、使用的数据、主要假设、输出结果和运行方法。该文件实现的是一个基于任务负载 trace 的数据中心组件级能耗、碳排和用水计算模型。

## 1. 模型定位

`workload_component_model.py` 的目标是：

- 使用 Alibaba 风格任务 trace 构建 AI 工作负载的时间变化形状。
- 将该工作负载按未来全球 IT 装机容量情景扩展到全球。
- 按国家 IT 装机容量或任务来源权重分配任务执行负载。
- 按 CPU、GPU、内存、存储、IT 风扇等组件计算 IT 能耗。
- 结合国家年度 PUE、WUE、grid water factor 和小时级或年度级碳因子，计算国家级电力、碳排和水足迹。

该模型不同于 `core/Carbon_water_footprint.py` 中的年度利用率模型。这里的负载不是固定年度利用率，而是来自任务 trace 的时间序列。

## 2. 主要入口

核心入口函数是：

```python
from core.workload_component_model import run_workload_component_footprint
```

典型调用：

```python
results = run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    years=6,
    year_start=2025,
    save_outputs=True,
)
```

返回值是一个字典，包含多个 `pandas.DataFrame`。

## 3. 使用的数据

### 3.1 任务 trace

默认文件：

```text
dataset/result_df_full_year_2020.pkl
```

默认由参数 `workload_profile_path` 指定。该 pickle 至少需要包含两列：

- `interval_15m`：15 分钟时间戳。
- `tasks_matrix`：该 15 分钟 interval 内开始的任务矩阵。

任务矩阵默认按以下字段解析：

```text
job_id
start_time
end_time
start_dt
duration_min
cpu_usage
gpu_wrk_util
avg_mem_gb
avg_gpu_wrk_mem_gb
bandwidth_gb
weekday_name
weekday_num
```

资源单位转换：

- CPU：`cpu_usage / 100`，转换为 CPU core 数。
- GPU：`gpu_wrk_util / 100`，转换为 GPU 数。
- 内存：`avg_mem_gb`，单位 GB。
- GPU 显存：`avg_gpu_wrk_mem_gb`，用于任务分类。
- 存储或数据负载：`bandwidth_gb / 1024`，转换为 TB 等价负载。

### 3.2 装机容量和国家份额

来自：

```text
dataset/Installed_capacity_data.py
```

主要变量：

- `countries`：默认 24 个国家。
- `it_capacity`：2025-2030 年全球 IT 容量矩阵，单位 GW，列对应四个情景。
- `it_ratio`：国家 IT 容量份额。
- `total_ratio`：国家总容量或需求相关份额。

情景列对应关系：

```text
Base -> 0
Lift-Off -> 1
High Efficiency -> 2
Headwinds -> 3
```

代码中 `DATA_YEAR_START = 2025`，所以年度数据索引按实际年份对齐：

```text
data_year_idx = year - 2025
```

例如 `year_start=2026, years=5` 会使用 2026-2030 对应的容量、PUE 和年度因子。

### 3.3 年度 PUE、WUE、水因子和年度碳因子

来自：

```text
dataset/Factors.py
```

主要变量：

- `PUE`：国家、年份、情景级 PUE。
- `WUE`：国家基础 WUE。
- `carbon_emissions_factors_CP/NDC/NZ`：年度电网碳因子，单位 kg CO2/MWh。
- `grid_water_factors_CP/NDC/NZ`：年度电网水因子，单位 m3/MWh。

PUE、WUE 和 grid water factor 当前仍保持年度级，不做小时变化。

### 3.4 小时级碳因子

默认目录：

```text
dataset/EM-estimate
```

目录结构示例：

```text
dataset/EM-estimate/USA/snapshots_2026-02-10_US-CP-2026-hourly.csv
dataset/EM-estimate/China/snapshots_2026-02-10_CN-NDC-2030-hourly.csv
dataset/EM-estimate/Great Britain/snapshots_2026-02-10_GB-CP-2026-hourly.csv
```

文件匹配规则：

```text
<hourly_carbon_dir>/<country_dir>/*-<policy>-<year>-hourly.csv
```

其中：

- `policy` 为 `CP`、`NDC` 或 `NZ`。
- `year` 为实际输出年份。
- `United_Kingdom` 会映射到目录名 `Great Britain`。

默认使用列：

```text
Carbon intensity gCO2eq/kWh (direct)
```

代码按列名模糊匹配 `carbon intensity` 和 `direct`。也可以通过 `hourly_carbon_scope="life_cycle"` 使用 life cycle 碳强度列。

注意：`gCO2eq/kWh` 与 `kg CO2/MWh` 在数值上等价，所以代码直接将小时 CSV 中的数值作为 `kg/MWh` 使用。

## 4. 任务分类

任务类型共有三类：

```text
training
inference
cpu_data
```

默认分类规则由 `TaskClassificationConfig` 控制：

- 如果 `gpu_count < 0.05`，任务归为 `cpu_data`。
- 默认先将非 CPU-only 任务视为 `inference`。
- 如果满足以下条件之一，则归为 `training`：
  - 任务时长 `>= 60` 分钟，并且 GPU 数 `>= 0.5` 或 GPU 显存 `>= 8 GB`。
  - 非 CPU-only 且任务时长 `>= 120` 分钟。

这些规则是启发式分类，因为 trace 本身不一定提供显式 workload label。

## 5. Alibaba trace 到全球负载的扩展

代码没有使用一个固定的全局倍数把 Alibaba 单个数据中心任务量扩展到全球，而是按资源类型计算动态缩放。

### 5.1 构建 trace 负载

`build_workload_profile()` 将任务按 15 分钟 interval 展开。每个任务从开始 interval 持续：

```text
ceil(duration_min / interval_minutes)
```

个时间步。最终得到：

```text
load[task_type, resource, time]
```

其中：

- `task_type` 为 `training/inference/cpu_data`。
- `resource` 为 `cpu/gpu/memory/storage`。
- `time` 为 15 分钟时间步。

### 5.2 估计 trace 参考容量

对所有任务类型求和后，得到每个资源的总负载时间序列：

```text
total_load[resource, time]
```

然后取默认 95 分位数作为 trace 参考容量：

```text
trace_capacity[resource] = quantile(total_load[resource], 0.95)
```

该值不是 Alibaba 数据中心真实装机容量，而是从 trace 负载估计出的参考容量。参数 `capacity_quantile` 可调整，默认 `0.95`。

### 5.3 缩放到全球资源容量

未来全球 IT 容量来自 `it_capacity`：

```text
global_it_mw = it_capacity[data_year_idx, scenario_col] * 1000
```

再按 `it_ratio` 分配到国家：

```text
country_it_mw = global_it_mw * normalize(it_ratio)
```

每个国家 IT MW 按硬件功率份额拆到组件：

```text
CPU: 30%
GPU: 50%
Memory: 12%
Storage: 3%
IT fan: 5%
```

再按单位满载功率换算为资源容量：

```text
CPU cores = CPU MW * 1e6 / 12 W
GPU count = GPU MW * 1e6 / 250 W
Memory GB = Memory MW * 1e6 / 0.07 W
Storage TB = Storage MW * 1e6 / 6.5 W
```

所有国家资源容量求和得到：

```text
global_resource_capacity[resource]
```

最终缩放公式为：

```text
normalized_load = trace_load / trace_capacity
global_load = normalized_load * global_resource_capacity
```

因此，实际扩展系数是分资源的：

```text
scale_factor[resource] = global_resource_capacity[resource] / trace_capacity[resource]
```

此外，代码会应用 `max_resource_utilization` 上限，默认 `1.0`，避免缩放后的负载超过设定最大资源利用率。

## 6. 国家任务量和任务执行位置

代码区分两个概念：

- `task_demand`：任务来源或需求归属。
- `task_execution`：任务实际在哪个国家的数据中心执行。

### 6.1 默认任务来源权重

如果不传入 `task_origin_weights`，默认：

```text
training -> it_ratio
inference -> total_ratio
cpu_data -> total_ratio
```

也就是说，训练任务来源默认更接近 IT 容量分布，推理和 CPU/data 任务来源默认更接近 `total_ratio`。

### 6.2 默认任务执行权重

默认参数：

```python
execution_policy="capacity"
```

在该模式下，所有任务类型都按国家 IT 装机容量比例执行：

```text
execution_weight[country] = country_it_mw / sum(country_it_mw)
```

所以默认情况下，国家执行任务量与国家 IT 装机容量成正比。

### 6.3 其他执行策略

`execution_policy` 可选：

- `capacity`：按国家 IT 装机容量执行，默认。
- `origin`：按任务来源权重执行。
- `hybrid`：混合模式。

`hybrid` 默认规则：

```text
training -> capacity
inference -> 75% origin + 25% capacity
cpu_data -> 50% origin + 50% capacity
```

对应参数：

```python
inference_origin_fraction=0.75
cpu_data_origin_fraction=0.50
```

也可以通过 `task_execution_weights` 直接传入每类任务的国家执行权重。

## 7. 组件功率计算

组件包括：

```text
cpu
gpu
memory
storage
it_fan
```

先得到国家资源利用率：

```text
resource_utilization = country_resource_load / resource_capacities
```

超出容量的部分记录为 `capacity_overflow`，但用于功率计算的利用率会被裁剪到 `max_resource_utilization`。

组件功率按以下方式计算：

### 7.1 CPU

```text
cpu_power = cpu_full_mw * (cpu_idle_fraction + (1 - cpu_idle_fraction) * cpu_util)
```

其中：

```text
cpu_idle_fraction = 2.5 / 12
```

### 7.2 GPU

```text
gpu_power = gpu_full_mw * (gpu_idle_fraction + (1 - gpu_idle_fraction) * log2(1 + gpu_util))
```

其中：

```text
gpu_idle_fraction = 25 / 250
```

GPU 不是线性函数，而是用 `log2(1 + gpu_util)` 表示利用率提升时的非线性功耗变化。

### 7.3 内存

```text
memory_power = memory_full_mw * (0.80 + 0.20 * memory_util)
```

### 7.4 存储

```text
storage_power = storage_full_mw * (0.60 + 0.40 * storage_util)
```

### 7.5 IT 风扇

风扇功率由 CPU、GPU、内存利用率构造热负载：

```text
effective_heat_load =
    0.35 * cpu_util
  + 0.50 * gpu_util
  + 0.15 * memory_util
```

再计算：

```text
it_fan_power = it_fan_full_mw * (0.20 + 0.80 * effective_heat_load^3)
```

## 8. 能耗、碳排和用水计算

### 8.1 IT 能耗

组件功率按时间积分：

```text
component_it_energy_mwh = sum(component_power_mw over time) * interval_hours
country_it_energy_mwh = sum(component_it_energy_mwh over components)
```

默认 `interval_hours = 0.25`，因为 workload trace 是 15 分钟粒度。

### 8.2 设施总用电

PUE 保持年度级：

```text
facility_energy_mwh = country_it_energy_mwh * annual_pue
```

输出中的 `power_twh` 为：

```text
power_twh = facility_energy_mwh / 1e6
```

### 8.3 碳排

如果 `hourly_carbon_factors_dir is None`，使用年度碳因子：

```text
carbon_tco2 = facility_energy_mwh * annual_carbon_factor_kg_per_mwh / 1000
```

如果启用小时级碳因子，默认启用，则：

1. 读取对应国家、政策、年份的小时 CSV。
2. 将模型的 15 分钟设施用电聚合或对齐到小时级。
3. 保持年度设施总用电不变，只使用 workload trace 提供小时用电形状。
4. 逐小时计算碳排：

```text
hourly_carbon_tco2 =
    hourly_facility_energy_mwh
  * hourly_carbon_factor_kg_per_mwh
  / 1000
```

5. 年度国家碳排为小时碳排求和：

```text
carbon_tco2 = sum(hourly_carbon_tco2)
```

如果某个年份或国家没有小时级碳因子，默认会 fallback 到年度碳因子。可通过 `hourly_carbon_fallback_to_annual=False` 或命令行 `--strict-hourly-carbon` 改为缺失时报错。

### 8.4 用水

WUE 和 grid water factor 保持年度级：

```text
direct_water_m3 = facility_energy_mwh * annual_wue
grid_water_m3 = facility_energy_mwh * annual_grid_water_factor
water_m3 = direct_water_m3 + grid_water_m3
```

其中 WUE 会根据 DLC 参数调整：

```text
dlc_rate = dlc_rate_0 * (1 + dlc_increase)^data_year_idx
adjusted_wue = base_wue * (1 - dlc_rate) + (base_wue - 0.137) * dlc_rate
```

## 9. 任务类型能耗分摊

`_allocate_energy_to_task_types()` 将组件能耗分摊到三类任务。

分摊依据：

- CPU 组件按 CPU 资源负载分摊。
- GPU 组件按 GPU 资源负载分摊。
- Memory 组件按内存资源负载分摊。
- Storage 组件按存储资源负载分摊。
- IT fan 按 CPU、GPU、Memory 的加权热负载分摊。

如果某个时间步没有 driver，则使用年度 driver 占比作为 fallback。

## 10. 输出结果

`run_workload_component_footprint()` 返回字典：

```text
annual_summary
component_energy
task_demand
task_execution
task_type_energy
capacity_overflow
hourly_carbon
workload_profile_summary
trace_resource_capacity
```

### 10.1 annual_summary

国家年度汇总。主要字段：

- `installed_it_mw`：国家 IT 装机容量，MW。
- `it_energy_mwh`：IT 侧能耗，MWh。
- `facility_energy_mwh`：乘 PUE 后的设施总用电，MWh。
- `power_twh`：设施总用电，TWh。
- `carbon_tco2`、`carbon_mtco2`：碳排。
- `water_m3`、`water_million_m3`：总水足迹。
- `direct_water_m3`：直接用水。
- `grid_water_m3`：电网相关用水。
- 平均和峰值资源利用率。

### 10.2 component_energy

国家、年度、组件级能耗。包含：

- `component`
- `full_power_mw`
- `it_energy_mwh`
- `facility_energy_mwh`
- `facility_energy_twh`

### 10.3 task_demand

按任务来源权重计算的国家任务需求资源量，单位为 resource-hours。

### 10.4 task_execution

按执行策略计算的国家实际执行资源量，单位为 resource-hours。

### 10.5 task_type_energy

国家、年度、任务类型级能耗。

### 10.6 capacity_overflow

记录负载超过国家资源容量的部分：

```text
overflow_resource_hours
```

### 10.7 hourly_carbon

小时级碳排输出。主要字段：

- `scenario`
- `year`
- `country`
- `hour_index`
- `timestamp_utc`
- `facility_energy_mwh`
- `carbon_factor_kg_per_mwh`
- `carbon_tco2`
- `carbon_factor_source`：`hourly` 或 `annual_fallback`。

默认返回该表，但默认不保存为 CSV，因为完整结果较大。命令行可用 `--save-hourly-carbon` 保存。

### 10.8 workload_profile_summary

trace 任务分类和资源小时数汇总。

### 10.9 trace_resource_capacity

trace 参考容量。包含每类资源的 `trace_capacity_at_quantile`。

## 11. 保存的 CSV

当 `save_outputs=True` 时，默认保存到：

```text
results/workload_component_model
```

主要文件：

```text
Country_Annual_Summary_<policy>_<scenario>.csv
Country_Component_Energy_<policy>_<scenario>.csv
Country_Task_Demand_<policy>_<scenario>.csv
Country_Task_Execution_<policy>_<scenario>.csv
Country_TaskType_Energy_<policy>_<scenario>.csv
Country_Capacity_Overflow_<policy>_<scenario>.csv
Workload_Profile_Summary.csv
Trace_Resource_Capacity.csv
```

如果 `save_hourly_outputs=True`，额外保存：

```text
Country_Hourly_Carbon_<policy>_<scenario>.csv
```

## 12. 命令行使用

脚本入口：

```text
scripts/Run_workload_component_model.py
```

### 12.1 默认运行

```bash
python scripts/Run_workload_component_model.py
```

默认参数：

- `policy=CP`
- `scenarios=Base`
- `years=6`
- `year_start=2025`
- 使用 `dataset/result_df_full_year_2020.pkl`
- 使用 `dataset/EM-estimate` 中的小时级碳因子
- 不保存 `hourly_carbon` 大表

### 12.2 运行 2026-2030 并强制使用小时级碳因子

```bash
python scripts/Run_workload_component_model.py ^
  --policy CP ^
  --scenarios Base ^
  --year-start 2026 ^
  --years 5 ^
  --strict-hourly-carbon
```

如果任一国家或年份缺少小时级碳因子，该命令会报错。

### 12.3 保存小时级碳排表

```bash
python scripts/Run_workload_component_model.py ^
  --policy CP ^
  --scenarios Base ^
  --save-hourly-carbon
```

### 12.4 改用 life cycle 碳强度

```bash
python scripts/Run_workload_component_model.py ^
  --policy NDC ^
  --scenarios Base ^
  --hourly-carbon-scope life_cycle
```

### 12.5 关闭小时级碳因子，恢复年度碳因子

```bash
python scripts/Run_workload_component_model.py ^
  --policy CP ^
  --scenarios Base ^
  --disable-hourly-carbon
```

### 12.6 快速测试

```bash
python scripts/Run_workload_component_model.py ^
  --policy CP ^
  --scenarios Base ^
  --years 2 ^
  --max-intervals 96 ^
  --no-save
```

`--max-intervals` 只建议用于 smoke test。它会截断 workload trace，不代表完整年度结果。

## 13. Python API 参数说明

常用参数：

```python
run_workload_component_footprint(
    renewable_energy_policy="CP",
    scenarios=["Base"],
    years=6,
    countries=None,
    workload_profile_path="dataset/result_df_full_year_2020.pkl",
    workload_year=2020,
    year_start=2025,
    output_dir="results/workload_component_model",
    save_outputs=True,
    verbose=True,
    execution_policy="capacity",
    inference_origin_fraction=0.75,
    cpu_data_origin_fraction=0.50,
    capacity_quantile=0.95,
    max_resource_utilization=1.0,
    pue_scale=1.0,
    dlc_rate_0=0.05,
    dlc_increase=0.20,
    hourly_carbon_factors_dir="dataset/EM-estimate",
    hourly_carbon_scope="direct",
    hourly_carbon_fallback_to_annual=True,
    save_hourly_outputs=False,
    max_intervals=None,
)
```

重点参数：

- `renewable_energy_policy`：`CP`、`NDC` 或 `NZ`。
- `scenarios`：`Base`、`Lift-Off`、`High Efficiency`、`Headwinds`。
- `years`：输出年份数。
- `year_start`：输出起始年份，当前年度数据支持 2025-2030。
- `execution_policy`：任务执行地分配策略。
- `capacity_quantile`：将 trace 负载的哪个分位数视作参考容量。
- `max_resource_utilization`：缩放后资源利用率上限。
- `hourly_carbon_factors_dir`：小时级碳因子目录。设为 `None` 时使用年度碳因子。
- `hourly_carbon_scope`：`direct` 或 `life_cycle`。
- `hourly_carbon_fallback_to_annual`：小时数据缺失时是否回退到年度碳因子。
- `save_hourly_outputs`：是否保存小时级碳排 CSV。

## 14. 重要假设和注意事项

1. Alibaba trace 被用作全球 AI 工作负载的时间形状和任务结构模板。

2. trace 的 95 分位负载默认被当作参考容量，不代表真实 Alibaba 数据中心装机容量。

3. 默认国家任务执行量与国家 IT 装机容量成正比，因为 `execution_policy="capacity"`。

4. 小时级碳因子只影响碳排计算。PUE、WUE 和 grid water factor 仍为年度级。

5. 小时级碳排中，年度设施总用电保持与年度 PUE 计算结果一致；小时 trace 只决定年度用电在小时之间的分配形状。

6. 如果从 2020 trace 构建负载，2020 是闰年。代码会按目标小时碳因子的时间戳对齐或重采样小时用电形状，确保可以匹配 8760 小时级碳因子。

7. `--max-intervals` 会截断 trace，只用于调试，不应作为正式结果。

8. `hourly_carbon` 完整输出可能很大。例如 24 个国家、5 年、1 个情景约为：

```text
24 * 5 * 8760 = 1,051,200 行
```

多情景运行时行数会继续按情景数增加。

## 15. 推荐检查

正式运行后建议检查：

- `annual_summary` 中各国 `facility_energy_mwh` 是否与预期年度能耗量级一致。
- `hourly_carbon` 中 `carbon_factor_source` 是否为 `hourly`，避免误用年度 fallback。
- `capacity_overflow` 是否过大。如果过大，说明任务分配或容量假设可能导致某些国家资源不足。
- `trace_resource_capacity` 中各资源参考容量是否合理，尤其是 GPU 和 storage。

