import numpy as np

countries = [
    "USA", "China", "Japan", "France", "India", "Singapore",
    "Canada", "Germany", "United_Kingdom", "Australia", "Italy", "South_Korea",
    "South_Africa", "Ireland", "UAE", "Brazil", "Israel",
    "Netherlands", "Spain", "Sweden", "Belgium", "Norway",
    "Poland", "Switzerland"
]

# 1. Total ratio 列的数据
total_ratio = {
    "USA": 0.43595, "China": 0.28795, "Canada": 0.02259, "Australia": 0.02464, "Japan": 0.02304, "Germany": 0.0217,
    "United_Kingdom": 0.02098, "Norway": 0.01929, "India": 0.01982, "France": 0.01554, "South_Korea": 0.00893, "Italy": 0.00714,
    "Spain": 0.00705, "Switzerland": 0.00688, "Netherlands": 0.00652, "Singapore": 0.00661, "Poland": 0.00598, "Ireland": 0.00455,
    "Sweden": 0.0042, "South_Africa": 0.00268, "Brazil": 0.00205, "UAE": 0.00152, "Belgium": 0.0008, "Israel": 0.00089
}

# 2. IT 列的数据
it_ratio = {
    "USA": 0.4611513, "China": 0.2471825, "Canada": 0.023875, "Australia": 0.023, "Japan": 0.0215, "Germany": 0.021,
    "United_Kingdom": 0.02025, "Norway": 0.018625, "India": 0.0185, "France": 0.015, "South_Korea": 0.008375, "Italy": 0.006875,
    "Spain": 0.00675, "Switzerland": 0.006625, "Netherlands": 0.00625, "Singapore": 0.00625, "Poland": 0.00575, "Ireland": 0.004375,
    "Sweden": 0.004125, "South_Africa": 0.002125, "Brazil": 0.00175, "UAE": 0.001125, "Belgium": 0.00075, "Israel": 0.000625
}

# 3. Non-IT 列的数据
non_it_ratio = {
    "USA": 0.384990323, "China": 0.402432258, "Canada": 0.02, "Australia": 0.029677419, "Japan": 0.027741935, "Germany": 0.024193548,
    "United_Kingdom": 0.023548387, "Norway": 0.021612903, "India": 0.023870968, "France": 0.017419355, "South_Korea": 0.010645161, "Italy": 0.008064516,
    "Spain": 0.008064516, "Switzerland": 0.007741935, "Netherlands": 0.007419355, "Singapore": 0.007741935, "Poland": 0.006774194, "Ireland": 0.00516129,
    "Sweden": 0.004516129, "South_frica": 0.004193548, "Brazil": 0.002903226, "UAE": 0.002580645, "Belgium": 0.000967742, "Israel": 0.001612903
}

# 1. Total Capacity Matrix (6x4) Base, Lift-Off, High Efficiency, Headwinds  GW
total_capacity = np.array([
    [111.8458, 120.3735, 107.4201, 104.5056],  # 2025
    [129.7040, 147.3266, 120.5580, 114.5351],  # 2026
    [150.0514, 179.2578, 134.8937, 124.9117],  # 2027
    [172.8882, 216.1671, 150.4270, 135.6355],  # 2028
    [198.2144, 258.0546, 167.1581, 146.7064],  # 2029
    [226.0299, 304.9201, 185.0869, 158.1244]  # 2030
])

# 2. IT Capacity Matrix (6x4)
# 对应图片中的 IT 列
it_capacity = np.array([
    [80.3618, 86.7306, 78.0950, 74.7487],  # 2025
    [94.9924, 108.1536, 90.3079, 83.3926],  # 2026
    [111.6727, 133.4851, 103.9090, 92.4482],  # 2027
    [130.4028, 162.7250, 118.8983, 101.9155],  # 2028
    [151.1828, 195.8735, 135.2759, 111.7943],  # 2029
    [174.0125, 232.9305, 153.0417, 122.0848]  # 2030
])

# 3. Non-IT Capacity Matrix (6x4)
# 对应图片中的 Non-IT 列
non_it_capacity = np.array([
    [31.4840, 33.6429, 29.3251, 29.7569],  # 2025
    [34.7116, 39.1730, 30.2501, 31.1425],  # 2026
    [38.3787, 45.7727, 30.9847, 32.4635],  # 2027
    [42.4854, 53.4421, 31.5287, 33.7200],  # 2028
    [47.0316, 62.1811, 31.8822, 34.9121],  # 2029
    [52.0174, 71.9896, 32.0452, 36.0396]  # 2030
])

if __name__ == '__main__':
    print(total_ratio['China'])
    # 验证矩阵形状
    print(f"Total Matrix Shape: {total_capacity.shape}")
    print(f"IT Matrix Shape: {it_capacity.shape}")
    print(f"Non-IT Matrix Shape: {non_it_capacity.shape}")

    # 简单验证一个数据点：2030年 Lift-Off 情景下的 IT 数据
    # 应该是 232.9305
    print(f"Validation (2030, Lift-Off, IT): {it_capacity[5, 1]}")

