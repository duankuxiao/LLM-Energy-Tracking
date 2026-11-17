# Electricity Maps 年度因子与小时情景输入

本文档对应当前原稿和 Supplementary Information（SI）的电网输入约定。

## 年度输入与单位

- 年度直接碳强度及 PUE 数值存于 `dataset/Factors.py`。
- 历史年份为 2017–2025；报告实验使用 2026–2030 年的固定年度情景输入。
- 碳强度单位为 `kgCO2eq/MWh`，与 `gCO2eq/kWh` 数值相同。
- `get_carbon_factor(policy, country, year)` 按实际年份读取年度数组；运行模型时不会重新拟合 CP 趋势。
- 年度数据的 Electricity Maps 来源归属不意味着未来小时情景是该机构发布的观测或预测。

## NDC 与 NZ 的设定

NDC、NZ 是相对于 CP 的统一、作者设定的情景代理。其预期倍率为：

```text
NDC_multiplier(t) = 0.89 ** ((t - 2025) / 5)
NZ_multiplier(t)  = 0.53 ** ((t - 2025) / 5)
```

2030 年预期倍率分别为 0.89、0.53。它们不是归属于某项 IEA 政策出版物的数值，也不是逐国政策实施或电网调度预测。代码中的 `POLICY_ENDPOINT_MULTIPLIERS` 明确保留这些设定。

实际计算读取存储的两位小数年度数组，因此：

```text
stored_CI(policy, country, t)
    = stored_CI(CP, country, t) * multiplier(policy, t) + rounding_residual
abs(rounding_residual) <= 0.005 kgCO2eq/MWh
```

`annual_policy_audit` 分别报告预期倍率、实际比值和舍入残差。CP、NDC、NZ 比较保持容量、工作负载、硬件和 PUE 一致。

## 报告实验实际使用的小时模板

`configs/hourly_grid_sources.csv` 明确列出 24 个处理后的 **2026 CP 模板**。所有条目的 `source_kind` 均为 `reconstructed_shape`。这些文件保留来源列结构，但不能因此称为 2025 年逐小时原始观测。

清单字段为 `country,path,source_year,source_kind,zone_id`；路径相对于清单目录。输入文件需要使用者按供应方访问和再分发条件自行准备，仓库不再分发小时原始输入。模型不能通过年度均值反推出缺失的小时形状。复现报告中的时间匹配效应需要同一模板，或明确标注所替换的来源和形状。

清单中的源年份是处理后模板的年份；它不证明更早的原始观测年份。生成脚本记录实际源文件、SHA-256、区域映射、来源类型和可用的估算标记，不补造原始数据链条。

## 完整 UTC 日历与年均值保持

构造脚本为 `scripts/build_hourly_grid_scenarios.py`：

1. 验证源模板的完整 UTC 日历、区域标识、碳强度单位及非负有限值。
2. 按 UTC 月、日、小时映射到目标年。
3. 源模板没有 2 月 29 日时，复制 2 月 28 日相同小时。
4. 在完成日历映射后归一化到对应国家、政策和年份的存储年度均值。

```text
phi(country, year, hour) = mapped_template(hour) / mean(mapped_template)
hourly_CI = stored_annual_CI * phi
```

2028 年必须包含 **8784 小时**，其他报告年份包含 8760 小时。2028 年保留 12 月 31 日，不截断成年初起算的 8760 小时。24 国 × 3 政策 × 5 年共生成 360 个 direct 文件。

小时匹配的日历长度与能耗年化约定不同：4,440 小时相对工作负载按 `8760/4440` 年化；映射到每年完整日历后，再保持该年度设施电量总和。扩充闰日日历不会额外增加年化电量。

```powershell
python -B scripts/build_hourly_grid_scenarios.py --source-manifest configs/hourly_grid_sources.csv --output-dir results/grid_hourly_calendar --scopes direct
python -B run.py --policy CP --scenarios Base --year-start 2026 --years 5 --annual-activity-mode calendar --hourly-carbon-factors-dir results/grid_hourly_calendar --strict-hourly-carbon --output-dir results/comparison_run
```

输出目录须为新目录。生成记录为 `construction_audit.csv` 和 `construction_metadata.json`。正式实验的 `configs/experiments.json` 启用严格小时模式，拒绝缺失文件或不完整日历；单独接口的年度因子回退仅用于明确选择该行为的其他计算。

## 排放边界与解释

报告实验仅使用 direct 通道。旧模板中的 Life cycle、CFE% 和 RE% 列不能支持独立的未来生命周期或全天候清洁电力情景结论。生命周期范围没有可用的年度回退表。

小时排放按国家分别匹配电量和碳因子，再汇总国家排放。年度因子与小时平均值一致不意味着电量加权因子一致；两者差别来自负荷和碳强度的时间协变。

这些结果是年度均值约束下的确定性情景计算，不是未来逐小时电力系统调度预测。更换模板后必须重新计算时间匹配、峰值和爬坡结果。
