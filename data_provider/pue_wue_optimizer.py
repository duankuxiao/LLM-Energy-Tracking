"""
Best & Worst PUE and WUE Optimization Script
=============================================

This script optimizes data center cooling system parameters to find the best (minimum)
and worst (maximum) Power Usage Effectiveness (PUE) and Water Usage Effectiveness (WUE)
values for different data center configurations.

The optimization uses IPOPT (Interior Point Optimizer) solver via cyipopt to solve
nonlinear programming problems. Four types of data center cooling configurations
are analyzed:
    1. Airside Economizer (AE) with Chiller
    2. Waterside Economizer (WE) with Chiller
    3. Airside Economizer with Immersion Cooling
    4. Waterside Economizer with Immersion Cooling

Reference:
    Xiao, T., et al. (2025). Environmental impact and net-zero pathways for sustainable
    artificial intelligence servers in the USA. Nature Sustainability, 1-13.

    Simulation functions are based on the open-source model:
    https://github.com/nuoaleon/Data-Center-Water-footprint

Author: Tianqi Xiao et al.
"""

import os
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from cyipopt import minimize_ipopt

from cooling_system_simulator import (
    pue_wue_ae_chiller,
    pue_wue_chiller_waterside_economizer,
    pue_wue_ae_immersion_chiller,
    pue_wue_immersion_chiller_waterside_economizer,
)

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Resolve OpenMP library conflicts

# =============================================================================
# Initial Input Parameters (Example Values)
# =============================================================================
# This array contains example input values for the optimization.
# The values correspond to the parameters defined in the bounds arrays below.
# Note: This array is not directly used in the current optimization run;
# instead, the midpoint of bounds is used as the starting point.

EXAMPLE_INPUTS = [9, 10, 101325, 8.94746094e-01,
     2.84082031e-02, 4.72167969e-02, 9.51835937e-01, 9.49707031e+00,
     6.50488281e+02, 6.13867188e-01, 6.41894531e+06, 6.50976563e-01,
     6.90283203e-01, 6.91894531e+00, 3.35986328e+00, 2.10712891e+00,
     1.56284191e+05, 6.68945312e-01, 1.71884552e+05, 7.45117188e-01,
     3.08984375e-01, 5.09960938e+00, 2.46235102e+05, 7.08007812e-01,
     2.93588867e-03, 1.11650391e+01, 2.72070313e-01, 3.95898438e+02,
     6.11914063e-01, 2.87041016e+01, 1.54423828e+01, 1.67460938e+01,
     -1.11123047e+01, 7.18554688e+01, 2.60351562e+01, -9.17968750e-02]

# =============================================================================
# Parameter Bounds Definition
# =============================================================================
# The optimization parameters are divided into categories based on data center type:
#   - AE_LOWER_BOUNDS / AE_UPPER_BOUNDS: Airside economizer configuration.
#   - WE_LOWER_BOUNDS / WE_UPPER_BOUNDS: Waterside economizer configuration.
#   - IMMERSION_LOWER_BOUNDS / IMMERSION_UPPER_BOUNDS: Immersion cooling parameters.
#
# Parameter index mapping for AE_LOWER_BOUNDS / AE_UPPER_BOUNDS:
# --------------------------------------------------------
# Index  Parameter                Description                                  Unit
# -----  ---------                -----------                                  ----
#   0    t_oa                     Outside air dry-bulb temperature             °C
#   1    rh_oa                    Outside air relative humidity                %
#   2    p_oa                     Outside air atmospheric pressure             Pa
#   3    ups_efficiency           Uninterruptible Power Supply efficiency      -
#   4    power_distribution_loss_rate Power distribution loss rate              -
#   5    lighting_percentage      Lighting power as percentage of IT load      -
#   6    delta_t_air              Supply-return air temperature difference     °C
#   7    fan_pressure_crac        CRAC (Computer Room Air Conditioner) fan     Pa
#                                 static pressure
#   8    fan_efficiency_crac      CRAC fan efficiency                          -
#   9    pump_pressure_hd         Humidification pump pressure                 Pa
#  10    pump_efficiency_hd       Humidification pump efficiency               -
#  11    at_ct                    Cooling tower approach temperature           °C
#  12    chiller_load             Chiller load ratio (partial load factor)     -
#  13    delta_t_water            Chilled water temperature difference         °C
#  14    pump_pressure_cw         Chilled water pump pressure                  Pa
#  15    pump_efficiency_cw       Chilled water pump efficiency                -
#  16    delta_t_ct               Cooling tower water temperature difference   °C
#  17    pump_pressure_ct         Cooling tower water pump pressure            Pa
#  18    pump_efficiency_ct       Cooling tower water pump efficiency          -
#  19    windage_rate             Cooling tower drift/windage loss percentage  -
#  20    cycles_of_concentration  Cooling tower cycles of concentration        -
#  21    fan_pressure_ct          Cooling tower fan static pressure            Pa
#  22    fan_efficiency_ct        Cooling tower fan efficiency                 -
#  23    sensible_heat_ratio      Sensible heat ratio                          -
#  24    liquid_gas_ratio         Cooling tower liquid-to-gas ratio            -
#  25    t_up                     Upper supply air temperature setpoint        °C
#  26    t_lw                     Lower supply air temperature setpoint        °C
#  27    dp_up                    Upper dew point temperature setpoint         °C
#  28    dp_lw                    Lower dew point temperature setpoint         °C
#  29    rh_up                    Upper relative humidity setpoint             %
#  30    rh_lw                    Lower relative humidity setpoint             %
#  31    cop_adjustment           Chiller COP adjustment factor                -
#
# Additional parameters for waterside economizer bounds:
# -------------------------------------------------------------------------------
#  32    heat_transfer_effectiveness Heat transfer effectiveness               -
#  33    at_he                    Heat exchanger approach temperature          °C
#  34    pump_pressure_we         Waterside economizer pump pressure           Pa
#  35    pump_efficiency_we       Waterside economizer pump efficiency         -
#
# Additional parameters for immersion cooling bounds:
# ---------------------------------------------------------
#   0    coolant_density          Coolant density                              kg/m³
#   1    coolant_flow_rate        Coolant volumetric flow rate                 L/min
#   2    pump_pressure_cl         Immersion coolant pump pressure              Pa
#   3    pump_efficiency_cl       Immersion coolant pump efficiency            -

# Lower bounds for Airside Economizer configuration (32 parameters)
AE_LOWER_BOUNDS = [-10, 10, 101325, 0.90, 0, 0, 13.9, 300, 0.65, 6300, 0.60, 2.8, 0.2, 5, 114.9, 0.60, 4, 166.9, 0.60, 0.005 / 100,
         3, 100, 0.65, 0.95, 0.2, 27, 10, 15, -12, 95, 60, -0.11]

# Upper bounds for Airside Economizer configuration (32 parameters)
AE_UPPER_BOUNDS = [+35, 100, 101325, 0.99, 0.02, 0.002, 19.4, 1000, 0.90, 7700, 0.90, 6.7, 0.8, 10, 172.4, 0.80, 6, 250.4, 0.80, 0.5 / 100,
         15, 400, 0.90, 0.99, 4, 35, 18, 27, -9, 99, 95, 0.11]

# Lower bounds for Waterside Economizer configuration (36 parameters)
# Includes 4 additional parameters for waterside economizer heat exchanger
WE_LOWER_BOUNDS = [-10, 10, 101325, 0.90, 0, 0, 13.9, 300, 0.65, 6300, 0.60, 2.8, 0.2, 5, 114.9, 0.60, 4, 166.9, 0.60, 0.005 / 100,
         3, 100, 0.65, 0.95, 0.2, 27, 10, 15, -12, 95, 60, -0.11, 0.7, 1.7, 114.9, 0.60]

# Upper bounds for Waterside Economizer configuration (36 parameters)
WE_UPPER_BOUNDS = [+35, 100, 101325, 0.99, 0.02, 0.002, 19.4, 700, 0.90, 7700, 0.90, 6.7, 0.8, 10, 172.4, 0.80, 6, 250.4, 0.80, 0.5 / 100,
         15, 400, 0.90, 0.99, 4, 35, 18, 27, -9, 99, 90, 0.11, 0.9, 2.8, 172.4, 0.80]

# Additional bounds for immersion cooling parameters (4 parameters)
# [coolant_density, flow_rate, pump_pressure, pump_efficiency]
IMMERSION_LOWER_BOUNDS = [1400, 0.5, 114.9, 60]  # Lower bounds: density 1400 kg/m³, flow 0.5 L/min, pressure 114.9 Pa, efficiency 60%
IMMERSION_UPPER_BOUNDS = [1855, 2.5, 172.4, 80]  # Upper bounds: density 1855 kg/m³, flow 2.5 L/min, pressure 172.4 Pa, efficiency 80%

# =============================================================================
# Fix Certain Parameters (Reduce Optimization Dimensions)
# =============================================================================
# Some parameters are fixed to their extreme values to simplify the optimization
# or to represent specific operating conditions.

# Fix these parameters to their upper bounds (minimize their impact or set to max efficiency)
# Indices 4, 5, 19: distribution loss, lighting, and drift loss.
for index in [4, 5, 19]:
    AE_LOWER_BOUNDS[index] = AE_UPPER_BOUNDS[index]
    WE_LOWER_BOUNDS[index] = WE_UPPER_BOUNDS[index]

# Fix these parameters to their lower bounds (maximize efficiency or minimize losses)
# Indices 3, 8, 10, 12, 15, 18, 22: efficiency and load parameters.
for index in [3, 8, 10, 12, 15, 18, 22]:
    AE_UPPER_BOUNDS[index] = AE_LOWER_BOUNDS[index]
    WE_UPPER_BOUNDS[index] = WE_LOWER_BOUNDS[index]

# Additional fixed parameters for Waterside Economizer
# Indices 32, 35: heat transfer effectiveness and pump efficiency.
for index in [32, 35]:
    WE_UPPER_BOUNDS[index] = WE_LOWER_BOUNDS[index]

# Fix immersion cooling pump efficiency (index 3)
for index in [3]:
    IMMERSION_UPPER_BOUNDS[index] = IMMERSION_LOWER_BOUNDS[index]

# =============================================================================
# Combine Bounds for Immersion Cooling Configurations
# =============================================================================
# Create combined bounds arrays for configurations with immersion cooling

# Airside Economizer + Immersion Cooling (36 parameters total)
AE_IMMERSION_LOWER_BOUNDS = AE_LOWER_BOUNDS + IMMERSION_LOWER_BOUNDS  # Concatenate AE bounds with immersion cooling bounds
AE_IMMERSION_UPPER_BOUNDS = AE_UPPER_BOUNDS + IMMERSION_UPPER_BOUNDS

# Waterside Economizer + Immersion Cooling (40 parameters total)
WE_IMMERSION_LOWER_BOUNDS = WE_LOWER_BOUNDS + IMMERSION_LOWER_BOUNDS
WE_IMMERSION_UPPER_BOUNDS = WE_UPPER_BOUNDS + IMMERSION_UPPER_BOUNDS

# Store parameter counts for reference
NUM_AE_PARAMS = len(AE_LOWER_BOUNDS)  # 32 parameters for Airside Economizer
NUM_WE_PARAMS = len(WE_LOWER_BOUNDS)  # 36 parameters for Waterside Economizer
NUM_IMMERSION_PARAMS = 4            # 4 additional parameters for immersion cooling

# Preserve baseline bounds after fixed-parameter adjustments.
AE_BASE_LOWER_BOUNDS = AE_LOWER_BOUNDS.copy()
AE_BASE_UPPER_BOUNDS = AE_UPPER_BOUNDS.copy()
WE_BASE_LOWER_BOUNDS = WE_LOWER_BOUNDS.copy()
WE_BASE_UPPER_BOUNDS = WE_UPPER_BOUNDS.copy()


# =============================================================================
# Objective Function Definitions
# =============================================================================
# The optimization objective functions are defined for each data center type
# and optimization goal (minimize or maximize PUE/WUE).
#
# maximize_* objectives return negative values so IPOPT can maximize by minimization.
# minimize_* objectives return direct PUE/WUE values.
#
# Note: IPOPT minimizes the objective, so to find maximum PUE/WUE (worst case),
# we minimize -PUE or -WUE.

# -----------------------------------------------------------------------------
# Worst-case objectives (maximize PUE/WUE by minimizing negative values)
# -----------------------------------------------------------------------------

def maximize_ae_chiller_pue(inputs):
    """
    Objective 1: Maximize PUE for Airside Economizer with Chiller.
    Used to find the worst-case (highest) PUE.

    Args:
        inputs: Array of 32 parameters for AE configuration
    Returns:
        Negative PUE value (for minimization to achieve maximization)
    """
    [pue, wue] = pue_wue_ae_chiller(inputs)
    return -pue


def maximize_ae_chiller_wue(inputs):
    """
    Objective 2: Maximize WUE for Airside Economizer with Chiller.
    Used to find the worst-case (highest) WUE.

    Args:
        inputs: Array of 32 parameters for AE configuration
    Returns:
        Negative WUE value (for minimization to achieve maximization)
    """
    [pue, wue] = pue_wue_ae_chiller(inputs)
    return -wue


def maximize_we_chiller_pue(inputs):
    """
    Objective 3: Maximize PUE for Waterside Economizer with Chiller.

    Args:
        inputs: Array of 36 parameters for WE configuration
    Returns:
        Negative PUE value
    """
    [pue, wue] = pue_wue_chiller_waterside_economizer(inputs)
    return -pue


def maximize_we_chiller_wue(inputs):
    """
    Objective 4: Maximize WUE for Waterside Economizer with Chiller.

    Args:
        inputs: Array of 36 parameters for WE configuration
    Returns:
        Negative WUE value
    """
    [pue, wue] = pue_wue_chiller_waterside_economizer(inputs)
    return -wue


def maximize_ae_immersion_pue(inputs):
    """
    Objective 5: Maximize PUE for Airside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 36 parameters (32 AE + 4 immersion cooling)
    Returns:
        Negative PUE value (single-phase immersion)
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_ae_immersion_chiller(inputs)
    return -pue


def maximize_ae_immersion_wue(inputs):
    """
    Objective 6: Maximize WUE for Airside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 36 parameters (32 AE + 4 immersion cooling)
    Returns:
        Negative WUE value (single-phase immersion)
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_ae_immersion_chiller(inputs)
    return -wue


def maximize_we_immersion_pue(inputs):
    """
    Objective 7: Maximize PUE for Waterside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 40 parameters (36 WE + 4 immersion cooling)
    Returns:
        Negative PUE value
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_immersion_chiller_waterside_economizer(inputs)
    return -pue


def maximize_we_immersion_wue(inputs):
    """
    Objective 8: Maximize WUE for Waterside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 40 parameters (36 WE + 4 immersion cooling)
    Returns:
        Negative WUE value
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_immersion_chiller_waterside_economizer(inputs)
    return -wue


# -----------------------------------------------------------------------------
# Best-case objectives (minimize PUE/WUE directly)
# -----------------------------------------------------------------------------

def minimize_ae_chiller_pue(inputs):
    """
    Objective 9: Minimize PUE for Airside Economizer with Chiller.
    Used to find the best-case (lowest) PUE.

    Args:
        inputs: Array of 32 parameters for AE configuration
    Returns:
        PUE value (positive, for direct minimization)
    """
    [pue, wue] = pue_wue_ae_chiller(inputs)
    return pue


def minimize_ae_chiller_wue(inputs):
    """
    Objective 10: Minimize WUE for Airside Economizer with Chiller.
    Used to find the best-case (lowest) WUE.

    Args:
        inputs: Array of 32 parameters for AE configuration
    Returns:
        WUE value
    """
    [pue, wue] = pue_wue_ae_chiller(inputs)
    return wue


def minimize_we_chiller_pue(inputs):
    """
    Objective 11: Minimize PUE for Waterside Economizer with Chiller.

    Args:
        inputs: Array of 36 parameters for WE configuration
    Returns:
        PUE value
    """
    [pue, wue] = pue_wue_chiller_waterside_economizer(inputs)
    return pue


def minimize_we_chiller_wue(inputs):
    """
    Objective 12: Minimize WUE for Waterside Economizer with Chiller.

    Args:
        inputs: Array of 36 parameters for WE configuration
    Returns:
        WUE value
    """
    [pue, wue] = pue_wue_chiller_waterside_economizer(inputs)
    return wue


def minimize_ae_immersion_pue(inputs):
    """
    Objective 13: Minimize PUE for Airside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 36 parameters (32 AE + 4 immersion cooling)
    Returns:
        PUE value
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_ae_immersion_chiller(inputs)
    return pue


def minimize_ae_immersion_wue(inputs):
    """
    Objective 14: Minimize WUE for Airside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 36 parameters (32 AE + 4 immersion cooling)
    Returns:
        WUE value
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_ae_immersion_chiller(inputs)
    return wue


def minimize_we_immersion_pue(inputs):
    """
    Objective 15: Minimize PUE for Waterside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 40 parameters (36 WE + 4 immersion cooling)
    Returns:
        PUE value
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_immersion_chiller_waterside_economizer(inputs)
    return pue


def minimize_we_immersion_wue(inputs):
    """
    Objective 16: Minimize WUE for Waterside Economizer with Immersion Cooling.

    Args:
        inputs: Array of 40 parameters (36 WE + 4 immersion cooling)
    Returns:
        WUE value
    """
    [pue, _pue_two_phase, wue, _wue_two_phase] = pue_wue_immersion_chiller_waterside_economizer(inputs)
    return wue


# =============================================================================
# IPOPT Optimization Solver Function
# =============================================================================

# Map objective IDs to functions for clarity when calling the optimizer.
OBJ_MAP = {
    1: maximize_ae_chiller_pue,
    2: maximize_ae_chiller_wue,
    3: maximize_we_chiller_pue,
    4: maximize_we_chiller_wue,
    5: maximize_ae_immersion_pue,
    6: maximize_ae_immersion_wue,
    7: maximize_we_immersion_pue,
    8: maximize_we_immersion_wue,
    9: minimize_ae_chiller_pue,
    10: minimize_ae_chiller_wue,
    11: minimize_we_chiller_pue,
    12: minimize_we_chiller_wue,
    13: minimize_ae_immersion_pue,
    14: minimize_ae_immersion_wue,
    15: minimize_we_immersion_pue,
    16: minimize_we_immersion_wue,
}

# Objective IDs that return negative values to represent maximization.
MAX_OBJECTIVES = {1, 2, 3, 4, 5, 6, 7, 8}

IPOPT_OPTIONS = {
    'tol': float(os.getenv('IPOPT_TOL', '0.01')),
    'dual_inf_tol': float(os.getenv('IPOPT_DUAL_INF_TOL', '0.1')),
    'constr_viol_tol': float(os.getenv('IPOPT_CONSTR_VIOL_TOL', '0.1')),
    'compl_inf_tol': float(os.getenv('IPOPT_COMPL_INF_TOL', '0.1')),
    'acceptable_tol': float(os.getenv('IPOPT_ACCEPTABLE_TOL', '0.1')),
    'disp': int(os.getenv('IPOPT_DISP', '0')),
    'maxiter': int(os.getenv('IPOPT_MAXITER', '5000')),
    'nlp_scaling_method': os.getenv('IPOPT_SCALING', 'none'),
}

LAST_INPUTS = {}


def optimize_with_ipopt(objective, lower_bounds, upper_bounds, inputs):
    """
    Solve nonlinear optimization problem using IPOPT solver.

    This function uses the Interior Point Optimizer (IPOPT) to find optimal
    data center operating parameters that minimize/maximize PUE or WUE.

    Args:
        objective: Objective function or objective ID (1-16).
        lower_bounds: Lower bounds array for optimization parameters
        upper_bounds: Upper bounds array for optimization parameters
        inputs: Initial guess for optimization (starting point)

    Returns:
        solution: Optimization result object containing:
            - solution.x: Optimal parameter values
            - solution.fun: Optimal objective function value
            - solution.success: Whether optimization succeeded

    IPOPT Options:
        - tol: Overall convergence tolerance (0.01)
        - dual_inf_tol: Dual infeasibility tolerance (0.1)
        - constr_viol_tol: Constraint violation tolerance (0.1)
        - compl_inf_tol: Complementarity tolerance (0.1)
        - acceptable_tol: Acceptable tolerance for early termination (0.1)
        - disp: Display optimization progress (1 = verbose)
        - maxiter: Maximum number of iterations (5000)
        - nlp_scaling_method: Disable automatic scaling ('none')

    """
    # Define optimization bounds as list of (min, max) tuples
    bnds = [(0, 1) for _ in range(len(lower_bounds))]  # Initialize with placeholder bounds
    for i in range(len(lower_bounds)):
        max_value = upper_bounds[i]
        min_value = lower_bounds[i]
        bnds[i] = (min_value, max_value)

    # Resolve the objective function
    if isinstance(objective, int):
        objective_fn = OBJ_MAP[objective]
    else:
        objective_fn = objective

    # Solve optimization problem using IPOPT
    solution = minimize_ipopt(objective_fn, x0=inputs, bounds=bnds, options=IPOPT_OPTIONS)
    return solution


# =============================================================================
# Country batch utilities
# =============================================================================

def _midpoint(lower_bounds, upper_bounds):
    return np.average([upper_bounds, lower_bounds], axis=0).tolist()


def _pressure_to_pa(value):
    # climate_data_2025.csv uses kPa; convert to Pa when needed.
    return value * 1000 if value < 2000 else value


def _apply_climate_bounds(lower_bounds, upper_bounds, t_oa, rh_oa, p_oa):
    local_lower_bounds = lower_bounds.copy()
    local_upper_bounds = upper_bounds.copy()
    local_lower_bounds[0] = t_oa
    local_upper_bounds[0] = t_oa
    local_lower_bounds[1] = rh_oa
    local_upper_bounds[1] = rh_oa
    local_lower_bounds[2] = p_oa
    local_upper_bounds[2] = p_oa
    return local_lower_bounds, local_upper_bounds


def _solve_value(obj_id, lower_bounds, upper_bounds):
    inputs = LAST_INPUTS.get(obj_id, _midpoint(lower_bounds, upper_bounds))
    solution = optimize_with_ipopt(obj_id, lower_bounds, upper_bounds, inputs)
    if solution.success:
        LAST_INPUTS[obj_id] = solution.x.tolist()
    value = solution.fun
    if obj_id in MAX_OBJECTIVES:
        value = -value
    return value


def _best_values(ae_lower_bounds, ae_upper_bounds, we_lower_bounds, we_upper_bounds):
    # Best practice: minimize PUE/WUE for AE and WE, then average.
    best_pue = (_solve_value(9, ae_lower_bounds, ae_upper_bounds) + _solve_value(11, we_lower_bounds, we_upper_bounds)) / 2
    best_wue = (_solve_value(10, ae_lower_bounds, ae_upper_bounds) + _solve_value(12, we_lower_bounds, we_upper_bounds)) / 2
    return best_pue, best_wue


def _worst_values(ae_lower_bounds, ae_upper_bounds, we_lower_bounds, we_upper_bounds):
    # Worst practice: maximize PUE/WUE for AE and WE, then average.
    worst_pue = (_solve_value(1, ae_lower_bounds, ae_upper_bounds) + _solve_value(3, we_lower_bounds, we_upper_bounds)) / 2
    worst_wue = (_solve_value(2, ae_lower_bounds, ae_upper_bounds) + _solve_value(4, we_lower_bounds, we_upper_bounds)) / 2
    return worst_pue, worst_wue


def _base_values(ae_lower_bounds, ae_upper_bounds, we_lower_bounds, we_upper_bounds):
    # Base case uses the mean of best and worst practices at the same climate.
    best_pue, best_wue = _best_values(ae_lower_bounds, ae_upper_bounds, we_lower_bounds, we_upper_bounds)
    worst_pue, worst_wue = _worst_values(ae_lower_bounds, ae_upper_bounds, we_lower_bounds, we_upper_bounds)
    base_pue = (best_pue + worst_pue) / 2
    base_wue = (best_wue + worst_wue) / 2
    return base_pue, base_wue


def process_chunk(chunk_df, chunk_id, output_dir, output_mode='base_wue_only'):
    """
    Process one chunk of country climate records and write a worker-specific CSV.

    The default output mode writes only base WUE because the manuscript uses PUE
    from another dataset and only needs this model's base WUE estimates.
    """
    # Each worker writes its own partial result file.
    output_filename = f"output_part_{chunk_id}.csv"
    output_path = os.path.join(output_dir, output_filename)

    results = []
    total_in_chunk = len(chunk_df)

    print(f"[Worker {chunk_id}] Started processing {total_in_chunk} countries...")

    for i, (index, row) in enumerate(chunk_df.iterrows()):
        try:
            country = row['country']

            # Print worker-level progress for long optimization runs.
            print(f"[Worker {chunk_id}] {i + 1}/{total_in_chunk}: {country}")

            rh_mean = float(row['RH_mean'])
            t_mean = float(row['T_oa_mean'])
            p_oa = _pressure_to_pa(float(row['P_oa']))

            if output_mode == 'base_wue_only':
                # Base case uses mean climate conditions.
                print(f"[Worker {chunk_id}] {i + 1}/{total_in_chunk}: {country} Base Case")
                ae_base_lower_bounds, ae_base_upper_bounds = _apply_climate_bounds(AE_BASE_LOWER_BOUNDS, AE_BASE_UPPER_BOUNDS, t_mean, rh_mean, p_oa)
                we_base_lower_bounds, we_base_upper_bounds = _apply_climate_bounds(WE_BASE_LOWER_BOUNDS, WE_BASE_UPPER_BOUNDS, t_mean, rh_mean, p_oa)
                base_pue, base_wue = _base_values(ae_base_lower_bounds, ae_base_upper_bounds, we_base_lower_bounds, we_base_upper_bounds)

                # Store only the WUE value used by the manuscript workflow.
                results.append({
                    'country': country,
                    'base_wue': round(base_wue, 4),
                })
            else:
                # Full output keeps the original best/base/worst PUE and WUE table.
                t_min = float(row['T_oa_min'])
                t_max = float(row['T_oa_max'])
                rh_min = float(row['RH_min'])
                rh_max = float(row['RH_max'])

                # Best case
                print(f"[Worker {chunk_id}] {i + 1}/{total_in_chunk}: {country} Best Case")
                ae_best_lower_bounds, ae_best_upper_bounds = _apply_climate_bounds(AE_BASE_LOWER_BOUNDS, AE_BASE_UPPER_BOUNDS, t_min, rh_min, p_oa)
                we_best_lower_bounds, we_best_upper_bounds = _apply_climate_bounds(WE_BASE_LOWER_BOUNDS, WE_BASE_UPPER_BOUNDS, t_min, rh_min, p_oa)
                best_pue, best_wue = _best_values(ae_best_lower_bounds, ae_best_upper_bounds, we_best_lower_bounds, we_best_upper_bounds)

                # Base case
                print(f"[Worker {chunk_id}] {i + 1}/{total_in_chunk}: {country} Base Case")
                ae_base_lower_bounds, ae_base_upper_bounds = _apply_climate_bounds(AE_BASE_LOWER_BOUNDS, AE_BASE_UPPER_BOUNDS, t_mean, rh_mean, p_oa)
                we_base_lower_bounds, we_base_upper_bounds = _apply_climate_bounds(WE_BASE_LOWER_BOUNDS, WE_BASE_UPPER_BOUNDS, t_mean, rh_mean, p_oa)
                base_pue, base_wue = _base_values(ae_base_lower_bounds, ae_base_upper_bounds, we_base_lower_bounds, we_base_upper_bounds)

                # Worst case
                print(f"[Worker {chunk_id}] {i + 1}/{total_in_chunk}: {country} Worst Case")
                ae_worst_lower_bounds, ae_worst_upper_bounds = _apply_climate_bounds(AE_BASE_LOWER_BOUNDS, AE_BASE_UPPER_BOUNDS, t_max, rh_max, p_oa)
                we_worst_lower_bounds, we_worst_upper_bounds = _apply_climate_bounds(WE_BASE_LOWER_BOUNDS, WE_BASE_UPPER_BOUNDS, t_max, rh_max, p_oa)
                worst_pue, worst_wue = _worst_values(ae_worst_lower_bounds, ae_worst_upper_bounds, we_worst_lower_bounds, we_worst_upper_bounds)

                results.append({
                    'country': country,
                    'best_pue': round(best_pue, 4),
                    'base_pue': round(base_pue, 4),
                    'worst_pue': round(worst_pue, 4),
                    'best_wue': round(best_wue, 4),
                    'base_wue': round(base_wue, 4),
                    'worst_wue': round(worst_wue, 4),
                })

            # Persist partial results after each country.
            pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8')

        except Exception as e:
            print(f"[Worker {chunk_id}] Error on {country}: {e}")
            continue

    print(f"[Worker {chunk_id}] Finished! Saved to {output_filename}")


def run_parallel_processing(input_csv, output_dir='.', output_mode='base_wue_only'):
    """
    Split the climate table and run country-level PUE/WUE optimization in parallel.

    Args:
        input_csv: Country climate input table.
        output_dir: Directory for worker-level CSV outputs.
        output_mode: 'base_wue_only' writes country/base_wue only, which matches
            the manuscript workflow. Use 'full' to output best/base/worst PUE
            and WUE values for each country.
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    if output_mode not in {'base_wue_only', 'full'}:
        raise ValueError("output_mode must be 'base_wue_only' or 'full'")

    # Load all country climate records.
    df_all = pd.read_csv(input_csv)
    total_records = len(df_all)

    # Use at most one process per input record.
    num_processes = min(cpu_count(), total_records)
    print(f"Total records: {total_records}. Using {num_processes} parallel processes.")

    # Split the input table into roughly equal chunks.
    df_chunks = np.array_split(df_all, num_processes)

    # Build worker argument tuples.
    tasks = []
    for i, chunk in enumerate(df_chunks):
        if not chunk.empty:
            tasks.append((chunk, i, output_dir, output_mode))

    # Start the worker pool.
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_chunk, tasks)

    print("\nAll processes completed. Please merge the 'output_part_*.csv' files manually.")


if __name__ == "__main__":
    input_path = os.path.join('../dataset', 'climate_data_2025.csv')
    output_dir = '../dataset/PUE_WUE'
    run_parallel_processing(input_path, output_dir)
