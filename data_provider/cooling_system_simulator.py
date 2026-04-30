"""
Data center thermodynamic simulation functions.

This module adapts the open-source Data-Center-Water-footprint model to
estimate PUE and WUE for airside economizer, waterside economizer, chiller,
and immersion cooling configurations.

Source model: https://github.com/nuoaleon/Data-Center-Water-footprint
"""

import os
import pickle

import numpy as np

from CoolProp.HumidAirProp import HAPropsSI

# Load pretrained COP models used by the cooling-system calculations.
_COP_DIR = os.path.dirname(__file__)
COP_GP = pickle.load(open(os.path.join(_COP_DIR, 'COP_2.pkl'), 'rb'))

COP_DX_GP = pickle.load(open(os.path.join(_COP_DIR, 'COP_DX.pkl'), 'rb'))

COP_AIR_GP = pickle.load(open(os.path.join(_COP_DIR, 'COP_AC.pkl'), 'rb'))



# Cooling-system helper functions
def airside_economizer(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa, delta_t_air):
    """
    Evaluate airside economizer operation with adiabatic humidification.

    Returns economizer flags, supply-air state, moisture change, and cooling or heating loads.
    """
    h_oa = HAPropsSI('H', 'T', t_oa + 273.15, 'P', p_oa, 'RH', rh_oa / 100)
    d_oa = HAPropsSI('W', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa)
    dp_oa = HAPropsSI('Tdp', 'T', t_oa + 273.15, 'P', p_oa, 'RH', rh_oa / 100) - 273.15

    if (t_lw <= t_oa < t_up):
        d_up = np.minimum(
            HAPropsSI('W', 'T', t_oa + 273.15, 'RH', rh_up / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_oa + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
        )
        d_lw = np.maximum(
            HAPropsSI('W', 'T', t_oa + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_oa + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
        )

        if (d_lw <= d_oa <= d_up):
            ae_use = 1
            hd_use = 0
            steam_use = 0
            dhd_use = 0
            delta_d = 0
            d_sa = d_oa
            t_cd = t_oa
            h_cd = h_oa
            chiller_energy = 0
            heating_energy = 0
            t_sa = t_oa
            h_sa = h_oa

        if (d_oa > d_up):
            ae_use = 0
            hd_use = 0
            steam_use = 0
            dhd_use = 1
            d_sa = d_up
            delta_d = d_oa - d_up
            t_cd = dp_up
            h_cd = HAPropsSI('H', 'T', dp_up + 273.15, 'P', p_oa, 'W', d_up)
            chiller_energy = h_oa - h_cd
            heating_energy = 0
            t_sa = np.maximum(t_lw, t_cd)
            h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_up)

        if (d_oa < d_lw):
            ae_use = 1
            hd_use = 1
            steam_use = 0
            dhd_use = 0
            delta_d = d_lw - d_oa
            d_sa = d_lw
            h_cd = h_oa
            t_cd = HAPropsSI('T', 'H', h_oa, 'W', d_lw, 'P', p_oa) - 273.15
            chiller_energy = 0
            heating_energy = 0
            t_sa = np.maximum(t_lw, t_cd)
            h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_lw)

    if (t_oa < t_lw):
        d_up = np.minimum(
            HAPropsSI('W', 'T', t_lw + 273.15, 'RH', rh_up / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_lw + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
        )
        d_lw = np.maximum(
            HAPropsSI('W', 'T', t_lw + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_lw + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
        )

        if (d_oa < d_lw):
            ae_use = 1
            hd_use = 1
            steam_use = 0
            dhd_use = 0
            delta_d = d_lw - d_oa
            d_sa = d_lw
            h_cd = h_oa
            t_cd = HAPropsSI('T', 'H', h_oa, 'W', d_lw, 'P', p_oa) - 273.15
            chiller_energy = 0
            heating_energy = 0
            t_sa = np.maximum(t_lw, t_oa)
            h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_lw)

        if (d_lw <= d_oa <= d_up):
            ae_use = 1
            hd_use = 0
            steam_use = 0
            dhd_use = 0
            delta_d = 0
            d_sa = d_oa
            t_cd = t_oa
            h_cd = h_oa
            chiller_energy = 0
            heating_energy = 0
            t_sa = np.maximum(t_lw, t_oa)
            h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_oa)

        if (d_oa > d_up):
            d_up_at_upsasp = HAPropsSI('W', 'T', t_up + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
            if (d_oa <= d_up_at_upsasp):
                ae_use = 1
                hd_use = 0
                steam_use = 0
                dhd_use = 0
                delta_d = 0
                t_cd = t_oa
                d_sa = d_oa
                h_cd = h_oa
                chiller_energy = 0
                heating_energy = 0
                t_sa = t_lw
                h_sa = HAPropsSI('H', 'T', t_lw + 273.15, 'P', p_oa, 'W', d_oa)
            else:
                ae_use = 1
                hd_use = 0
                steam_use = 0
                dhd_use = 1
                delta_d = d_oa - d_up_at_upsasp
                t_cd = dp_up
                d_sa = d_up_at_upsasp
                h_cd = HAPropsSI('H', 'T', dp_up + 273.15, 'P', p_oa, 'W', d_up_at_upsasp)
                chiller_energy = np.maximum(h_cd - h_oa, 0)
                heating_energy = 0
                t_sa = np.maximum(t_lw, t_cd)
                h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_up_at_upsasp)

    if (t_oa >= t_up):
        d_up = np.minimum(
            HAPropsSI('W', 'T', t_up + 273.15, 'RH', rh_up / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_up + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
        )
        d_lw = np.maximum(
            HAPropsSI('W', 'T', t_up + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_up + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
        )
        h_threshold_up = HAPropsSI('H', 'T', t_up + 273.15, 'P', p_oa, 'W', d_up)
        h_threshold_lw = HAPropsSI('H', 'T', t_up + 273.15, 'P', p_oa, 'W', d_lw)

        ae_use = 0
        hd_use = 0
        steam_use = 0
        dhd_use = 0
        t_cd = t_up
        h_cd = HAPropsSI('H', 'T', t_up + 273.15, 'P', p_oa, 'W', d_up)
        d_sa = d_up
        delta_d = 0

        h_sa = h_cd
        h_ra = h_sa + 1.01 * delta_t_air * 1000

        chiller_energy = h_ra - h_cd
        heating_energy = 0
        t_sa = t_cd
        h_sa = h_cd
        h_oa = h_sa + 1.01 * delta_t_air * 1000

    return ae_use, hd_use, dhd_use, steam_use, delta_d, t_sa, h_sa, t_cd, h_cd, d_sa, h_oa, chiller_energy, heating_energy


def airside_economizer_colo(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa):
    """
    Evaluate colocation airside economizer operation without humidification or dehumidification.
    """
    h_oa = HAPropsSI('H', 'T', t_oa + 273.15, 'P', p_oa, 'RH', rh_oa / 100)
    d_oa = HAPropsSI('W', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa)
    dp_oa = HAPropsSI('Tdp', 'T', t_oa + 273.15, 'P', p_oa, 'RH', rh_oa / 100) - 273.15

    t_sa = np.max([t_up, t_lw])

    if (t_lw <= t_oa <= t_up):
        d_up = np.minimum(
            HAPropsSI('W', 'T', t_oa + 273.15, 'RH', rh_up / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_oa + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
        )
        d_lw = np.maximum(
            HAPropsSI('W', 'T', t_oa + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_oa + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
        )
        if (d_lw <= d_oa <= d_up):
            ae_use = 1
            h_cd = h_oa
            t_cd = t_oa
            d_sa = d_oa
            t_sa = t_oa
            h_sa = h_oa
        else:
            ae_use = 0
            t_sa = t_sa
            d_up = np.minimum(
                HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_up / 100, 'P', p_oa),
                HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
            )
            d_lw = np.maximum(
                HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
                HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
            )
            d_sa = np.random.uniform(d_up, d_lw, 1)[0]
            h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_sa)
            h_cd = h_sa
            t_cd = t_sa

    elif (t_oa < t_lw):
        d_up = np.minimum(
            HAPropsSI('W', 'T', t_lw + 273.15, 'RH', rh_up / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_lw + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
        )
        d_lw = np.maximum(
            HAPropsSI('W', 'T', t_lw + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_lw + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
        )
        if (d_lw <= d_oa <= d_up):
            ae_use = 1
            h_cd = h_oa
            t_cd = t_oa
            d_sa = d_oa
            t_sa = np.maximum(t_lw, t_oa)
            h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_oa)
        else:
            ae_use = 0
            t_sa = t_sa
            d_up = np.minimum(
                HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_up / 100, 'P', p_oa),
                HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
            )
            d_lw = np.maximum(
                HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
                HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
            )
            d_sa = np.random.uniform(d_up, d_lw, 1)[0]
            h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_sa)
            h_cd = h_sa
            t_cd = t_sa

    else:
        ae_use = 0
        t_sa = t_sa
        d_up = np.minimum(
            HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_up / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
        )
        d_lw = np.maximum(
            HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
            HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
        )
        d_sa = np.random.uniform(d_up, d_lw, 1)[0]
        h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_sa)
        h_cd = h_sa
        t_cd = t_sa

    return ae_use, d_sa, t_sa, h_sa, t_cd, h_cd


def cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, heat_load, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio):
    """
    Estimate cooling tower water use from evaporation, drift, and blowdown.
    """
    h_oa = HAPropsSI('H', 'T', t_oa + 273.15, 'P', p_oa, 'R', rh_oa / 100)
    wet_bulb_oa = HAPropsSI('Twb', 'H', h_oa, 'T', t_oa + 273.15, 'P', p_oa) - 273.15
    d_oa = HAPropsSI('W', 'H', h_oa, 'R', rh_oa / 100, 'P', p_oa)
    d_oasa = HAPropsSI('W', 'H', h_oa, 'R', 1, 'P', p_oa)

    latent_heat_vaporization = lambda t_water: -0.0013 * (t_water) ** 2 - 2.3097 * (t_water) + 2500.5

    ct_water_mass_flow = heat_load / (4.184 * delta_t_ct)

    t_ct = wet_bulb_oa + at_ct + delta_t_ct / 2

    ct_evaporated_water = heat_load / latent_heat_vaporization(t_ct)

    m_air = ct_evaporated_water / liquid_gas_ratio

    ct_windage_water = ct_water_mass_flow * windage_rate

    ct_blowdown_water = np.maximum(ct_evaporated_water / (cycles_of_concentration - 1) - ct_windage_water, 0)

    wue = (ct_evaporated_water + ct_windage_water + ct_blowdown_water) * 3600 / power_it

    return wue, m_air, ct_water_mass_flow / m_air, ct_evaporated_water, ct_windage_water, ct_blowdown_water


def waterside_economizer(t_sfw, t_rfw, wet_bulb_oa, at_ct, at_he, cooling_required):
    """
    Estimate whether the waterside economizer can remove all, part, or none of the cooling load.
    """
    t_ct_out = wet_bulb_oa + at_ct + at_he

    if t_ct_out <= t_sfw:
        use = 1
        we_heat_removed = cooling_required
    elif t_ct_out > t_sfw and t_ct_out < t_rfw:
        use = 1
        we_heat_removed = cooling_required * ((t_rfw - t_ct_out) / (t_rfw - t_sfw))
    else:
        use = 0
        we_heat_removed = 0

    return use, we_heat_removed


def chiller_system(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa, delta_t_air):
    """
    Return supply-air temperature, humidity ratio, and enthalpy for chiller-based cooling.
    """
    t_sa = np.max([t_up, t_lw])

    d_up = np.minimum(
        HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_up / 100, 'P', p_oa),
        HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
    )
    d_lw = np.maximum(
        HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
        HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
    )

    d_sa = np.random.uniform(d_up, d_lw, 1)[0]
    h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_sa)

    return t_sa, d_sa, h_sa


def chiller_system_dx(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa):
    """
    Return supply-air state for a direct-expansion cooling system.
    """
    t_sa = np.mean([t_up, t_lw])

    d_up = np.minimum(
        HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_up / 100, 'P', p_oa),
        HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_up + 273.15, 'P', p_oa)
    )
    d_lw = np.maximum(
        HAPropsSI('W', 'T', t_sa + 273.15, 'RH', rh_lw / 100, 'P', p_oa),
        HAPropsSI('W', 'T', t_sa + 273.15, 'Tdp', dp_lw + 273.15, 'P', p_oa)
    )
    d_sa = np.random.uniform(d_up, d_lw, 1)[0]
    h_sa = HAPropsSI('H', 'T', t_sa + 273.15, 'P', p_oa, 'W', d_sa)

    return t_sa, d_sa, h_sa



# PUE and WUE model functions
def pue_wue_ae_chiller(inputs):
    """
    Calculate PUE and WUE for airside economizer plus water-cooled chiller cooling.
    """
    power_it = 1
    t_oa = inputs[0]
    rh_oa = inputs[1]
    p_oa = inputs[2]
    ups_efficiency = inputs[3]
    power_distribution_loss_rate = inputs[4]
    lighting_percentage = inputs[5]
    delta_t_air = inputs[6]
    fan_pressure_crac = inputs[7]
    fan_efficiency_crac = inputs[8]
    pump_pressure_hd = inputs[9]
    pump_efficiency_hd = inputs[10]
    at_ct = inputs[11]
    chiller_load = inputs[12]
    delta_t_water = inputs[13]
    pump_pressure_cw = inputs[14]
    pump_efficiency_cw = inputs[15]
    delta_t_ct = inputs[16]
    pump_pressure_ct = inputs[17]
    pump_efficiency_ct = inputs[18]
    windage_rate = inputs[19]
    cycles_of_concentration = inputs[20]
    fan_pressure_ct = inputs[21]
    fan_efficiency_ct = inputs[22]
    sensible_heat_ratio = inputs[23]
    liquid_gas_ratio = inputs[24]
    t_up = inputs[25]
    t_lw = inputs[26]
    dp_up = inputs[27]
    dp_lw = inputs[28]
    rh_up = inputs[29]
    rh_lw = inputs[30]
    water_efficiency = 1
    cop_adjustment = inputs[31]

    # P = V̇ × ΔP / η = (ṁ/ρ) × ΔP / η
    fan_power = lambda mass_flow_rate, air_density, fan_pressure, fan_efficiency:\
        mass_flow_rate / air_density * fan_pressure / fan_efficiency / 1000

    # P = ΔP × ṁ / (ρ × η)
    pump_power = lambda mass_flow_rate, pump_pressure, pump_efficiency, liquid_density:\
        pump_pressure * mass_flow_rate / (1000 * pump_efficiency * liquid_density)

    heat_load = power_it + (power_it / ups_efficiency - power_it) + (power_it / (1 - power_distribution_loss_rate) - power_it) + (power_it * lighting_percentage)

    ae_result = airside_economizer(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa, delta_t_air)
    t_sa = ae_result[5]
    t_ra = t_sa + delta_t_air
    ae_use = ae_result[0]
    h_cd = ae_result[8] / 1000
    h_sa = ae_result[6] / 1000
    h_ra = h_sa + 1.01 * delta_t_air
    h_oa = HAPropsSI('H', 'T', t_oa + 273.15, 'P', p_oa, 'RH', rh_oa / 100) / 1000

    # Q = ṁ × (H_ra - H_sa)
    m_sa = heat_load / (h_ra - h_sa)
    m_cd = m_sa * (h_sa - h_ra) / (h_cd - h_ra)

    d_sa = ae_result[9]
    m_cd_dry = m_cd / (1 + d_sa)
    m_sa_dry = m_sa / (1 + d_sa)

    heat = ae_use * m_cd * (h_ra - h_cd)

    density_sa = 1 / HAPropsSI('Vha', 'T', t_sa + 273.15, 'W', d_sa, 'P', p_oa)

    power_fan_crac = fan_power(m_sa, density_sa, fan_pressure_crac, fan_efficiency_crac)

    hd_use = ae_result[1]
    delta_d = ae_result[4]
    hd_amount_ae = np.maximum(hd_use * m_cd_dry * delta_d, 0)

    chiller_heat_removed = heat_load - heat

    if chiller_heat_removed != 0:
        latent_heat_load = chiller_heat_removed / sensible_heat_ratio - chiller_heat_removed
        chiller_heat_removed = latent_heat_load + chiller_heat_removed
        d_ra_chiller = latent_heat_load / 2266
        hd_amount = np.maximum(d_ra_chiller, 0)
    else:
        latent_heat_load = 0
        hd_amount = 0

    power_pump_hd = pump_power(hd_amount, pump_pressure_hd, pump_efficiency_hd, 1000)

    wet_bulb_oa = HAPropsSI('Twb', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa) - 273.15

    chiller_cop = COP_GP.predict(np.array([wet_bulb_oa + at_ct, chiller_load]).reshape(1, 2))[0] * (1 + cop_adjustment)

    power_chiller = chiller_heat_removed / chiller_cop

    m_sw = chiller_heat_removed / (4.184 * delta_t_water)

    power_pump_cw = pump_power(m_sw, pump_pressure_cw, pump_efficiency_cw, 1000)

    ct_heat_removed = chiller_heat_removed + power_chiller

    ct_water_mass_flow = ct_heat_removed / (4.184 * delta_t_ct)

    power_pump_ct = pump_power(ct_water_mass_flow, pump_pressure_ct, pump_efficiency_ct, 1000)

    ct_air_mass_flow = cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, ct_heat_removed, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)[1]

    density_oa = 1 / HAPropsSI('Vha', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa)
    power_fan_ct = fan_power(ct_air_mass_flow, density_oa, fan_pressure_ct, fan_efficiency_ct)

    power_components = np.array([
        power_it,
        (power_it / ups_efficiency - power_it),
        (power_it / (1 - power_distribution_loss_rate) - power_it),
        power_it * lighting_percentage,
        power_fan_crac,
        power_pump_hd,
        power_chiller,
        power_pump_cw,
        power_pump_ct,
        power_fan_ct
    ])
    pue = np.sum(power_components) / power_it

    ct_water = cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, ct_heat_removed, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)
    water_components = np.concatenate((
        [(hd_amount + hd_amount_ae) / water_efficiency],
        np.maximum([0, 0, 0], ct_water[3:6])
    ))
    wue = np.sum(water_components) * 3600 / power_it

    return pue, wue


def pue_wue_chiller_waterside_economizer(inputs):
    """
    Calculate PUE and WUE for waterside economizer plus water-cooled chiller cooling.
    """
    power_it = 1
    t_oa = inputs[0]
    rh_oa = inputs[1]
    p_oa = inputs[2]
    ups_efficiency = inputs[3]
    power_distribution_loss_rate = inputs[4]
    lighting_percentage = inputs[5]
    delta_t_air = inputs[6]
    fan_pressure_crac = inputs[7]
    fan_efficiency_crac = inputs[8]
    pump_pressure_hd = inputs[9]
    pump_efficiency_hd = inputs[10]
    at_ct = inputs[11]
    chiller_load = inputs[12]
    delta_t_water = inputs[13]
    pump_pressure_cw = inputs[14]
    pump_efficiency_cw = inputs[15]
    delta_t_ct = inputs[16]
    pump_pressure_ct = inputs[17]
    pump_efficiency_ct = inputs[18]
    windage_rate = inputs[19]
    cycles_of_concentration = inputs[20]
    fan_pressure_ct = inputs[21]
    fan_efficiency_ct = inputs[22]
    sensible_heat_ratio = inputs[23]
    liquid_gas_ratio = inputs[24]
    t_up = inputs[25]
    t_lw = inputs[26]
    dp_up = inputs[27]
    dp_lw = inputs[28]
    rh_up = inputs[29]
    rh_lw = inputs[30]
    cop_adjustment = inputs[31]
    heat_transfer_effectiveness = inputs[32]
    at_he = inputs[33]
    pump_pressure_we = inputs[34]
    pump_efficiency_we = inputs[35]
    water_efficiency = 1

    fan_power = lambda mass_flow_rate, air_density, fan_pressure, fan_efficiency:\
        mass_flow_rate / air_density * fan_pressure / fan_efficiency / 1000
    pump_power = lambda mass_flow_rate, pump_pressure, pump_efficiency, liquid_density:\
        pump_pressure * mass_flow_rate / (1000 * pump_efficiency * liquid_density)

    wet_bulb_oa = HAPropsSI('Twb', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa) - 273.15

    heat_load = power_it + (power_it / ups_efficiency - power_it) + (power_it / (1 - power_distribution_loss_rate) - power_it) + (power_it * lighting_percentage)

    latent_heat_load = heat_load / sensible_heat_ratio - heat_load

    cs_result = chiller_system(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa, delta_t_air)
    t_sa = cs_result[0]
    d_sa = cs_result[1]
    h_sa = cs_result[2] / 1000

    t_ra = t_sa + delta_t_air

    m_sa = heat_load / (1.01 * delta_t_air)
    m_sa_dry = m_sa / (1 + d_sa)

    density_sa = 1 / HAPropsSI('Vha', 'T', t_sa + 273.15, 'W', d_sa, 'P', p_oa)

    power_fan_crac = fan_power(m_sa, density_sa, fan_pressure_crac, fan_efficiency_crac)

    h_ra = (latent_heat_load + heat_load) / m_sa + h_sa
    d_ra = HAPropsSI('W', 'H', h_ra * 1000, 'T', t_ra + 273.15, 'P', p_oa)
    d_ra_chiller = latent_heat_load / 2266

    hd_amount = np.maximum(d_ra_chiller, 0)

    power_pump_hd = pump_power(hd_amount, pump_pressure_hd, pump_efficiency_hd, 1000)

    cooling_required = heat_load + latent_heat_load

    t_sfw = t_ra - (t_ra - t_sa) / heat_transfer_effectiveness
    t_rfw = t_sfw + delta_t_water

    we_use = waterside_economizer(t_sfw, t_rfw, wet_bulb_oa, at_ct, at_he, cooling_required)[0]
    we_heat_removed = waterside_economizer(t_sfw, t_rfw, wet_bulb_oa, at_ct, at_he, cooling_required)[1]

    m_sfw = we_heat_removed / (4.2 * delta_t_water)

    power_pump_we = pump_power(m_sfw, pump_pressure_we, pump_efficiency_we, 1000)

    chiller_heat_removed = cooling_required - we_heat_removed

    m_sw = chiller_heat_removed / (4.2 * delta_t_water)

    power_pump_cw = pump_power(m_sw, pump_pressure_cw, pump_efficiency_cw, 1000)

    chiller_cop = COP_GP.predict(np.array([wet_bulb_oa + at_ct, chiller_load]).reshape(1, 2))[0] * (1 + cop_adjustment)

    power_chiller = chiller_heat_removed / chiller_cop

    ct_heat_removed = cooling_required + power_chiller
    ct_water_mass_flow = ct_heat_removed / (4.184 * delta_t_ct)
    power_pump_ct = pump_power(ct_water_mass_flow, pump_pressure_ct, pump_efficiency_ct, 1000)
    ct_air_mass_flow = cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, heat_load, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)[1]
    density_oa = 1 / HAPropsSI('Vha', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa)
    power_fan_ct = fan_power(ct_air_mass_flow, density_oa, fan_pressure_ct, fan_efficiency_ct)

    power_components = np.array([
        power_it,
        (power_it / ups_efficiency - power_it),
        (power_it / (1 - power_distribution_loss_rate) - power_it),
        power_it * lighting_percentage,
        power_fan_crac,
        power_pump_hd,
        power_pump_we,
        power_chiller,
        power_pump_cw,
        power_pump_ct,
        power_fan_ct
    ])
    pue = np.sum(power_components) / power_it

    water_components = np.concatenate((
        [hd_amount / water_efficiency],
        cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, heat_load, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)[3:6]
    ))
    wue = np.sum(water_components) * 3600 / power_it

    return pue, wue


def pue_wue_ae_immersion_chiller(inputs):
    """
    Calculate single-phase and two-phase immersion PUE/WUE with airside economizer and chiller cooling.
    """
    power_it = 1
    t_oa = inputs[0]
    rh_oa = inputs[1]
    p_oa = inputs[2]
    ups_efficiency = inputs[3]
    power_distribution_loss_rate = inputs[4]
    lighting_percentage = inputs[5]
    delta_t_air = inputs[6]
    fan_pressure_crac = inputs[7]
    fan_efficiency_crac = inputs[8]
    pump_pressure_hd = inputs[9]
    pump_efficiency_hd = inputs[10]
    at_ct = inputs[11]
    chiller_load = inputs[12]
    delta_t_water = inputs[13]
    pump_pressure_cw = inputs[14]
    pump_efficiency_cw = inputs[15]
    delta_t_ct = inputs[16]
    pump_pressure_ct = inputs[17]
    pump_efficiency_ct = inputs[18]
    windage_rate = inputs[19]
    cycles_of_concentration = inputs[20]
    fan_pressure_ct = inputs[21]
    fan_efficiency_ct = inputs[22]
    sensible_heat_ratio = inputs[23]
    liquid_gas_ratio = inputs[24]
    t_up = inputs[25]
    t_lw = inputs[26]
    dp_up = inputs[27]
    dp_lw = inputs[28]
    rh_up = inputs[29]
    rh_lw = inputs[30]
    water_efficiency = 1
    cop_adjustment = inputs[31]
    coolant_density = inputs[32]
    coolant_flow_rate = inputs[33]
    coolant_mass_flow = coolant_flow_rate * coolant_density / (1000 * 60)
    pump_pressure_cl = inputs[34]
    pump_efficiency_cl = inputs[35]

    fan_power = lambda mass_flow_rate, air_density, fan_pressure, fan_efficiency:\
        mass_flow_rate / air_density * fan_pressure / fan_efficiency / 1000
    pump_power = lambda mass_flow_rate, pump_pressure, pump_efficiency, liquid_density:\
        pump_pressure * mass_flow_rate / (1000 * pump_efficiency * liquid_density)

    heat_load = power_it + (power_it / (1 - power_distribution_loss_rate) - power_it)

    ae_result = airside_economizer(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa, delta_t_air)
    t_sa = ae_result[5]
    t_ra = t_sa + delta_t_air
    ae_use = ae_result[0]
    h_cd = ae_result[8] / 1000
    h_sa = ae_result[6] / 1000
    h_ra = h_sa + 1.01 * delta_t_air
    h_oa = HAPropsSI('H', 'T', t_oa + 273.15, 'P', p_oa, 'RH', rh_oa / 100) / 1000

    m_sa = heat_load / (h_ra - h_sa)
    m_cd = m_sa * (h_sa - h_ra) / (h_cd - h_ra)
    d_sa = ae_result[9]
    m_cd_dry = m_cd / (1 + d_sa)
    m_sa_dry = m_sa / (1 + d_sa)
    heat = ae_use * m_cd * (h_ra - h_cd)
    density_sa = 1 / HAPropsSI('Vha', 'T', t_sa + 273.15, 'W', d_sa, 'P', p_oa)

    power_fan_crac = fan_power(m_sa, density_sa, fan_pressure_crac, fan_efficiency_crac)
    hd_use = ae_result[1]
    delta_d = ae_result[4]
    hd_amount_ae = np.maximum(hd_use * m_cd_dry * delta_d, 0)

    chiller_heat_removed = heat_load - heat
    if chiller_heat_removed != 0:
        latent_heat_load = chiller_heat_removed / sensible_heat_ratio - chiller_heat_removed
        chiller_heat_removed = latent_heat_load + chiller_heat_removed
        d_ra_chiller = latent_heat_load / 2266
        hd_amount = np.maximum(d_ra_chiller, 0)
    else:
        latent_heat_load = 0
        hd_amount = 0

    power_pump_hd = pump_power(hd_amount, pump_pressure_hd, pump_efficiency_hd, 1000)
    wet_bulb_oa = HAPropsSI('Twb', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa) - 273.15
    chiller_cop = COP_GP.predict(np.array([wet_bulb_oa + at_ct, chiller_load]).reshape(1, 2))[0] * (1 + cop_adjustment)
    power_chiller = chiller_heat_removed / chiller_cop
    m_sw = chiller_heat_removed / (4.184 * delta_t_water)
    power_pump_cw = pump_power(m_sw, pump_pressure_cw, pump_efficiency_cw, 1000)

    ct_heat_removed = chiller_heat_removed + power_chiller
    ct_water_mass_flow = ct_heat_removed / (4.184 * delta_t_ct)
    power_pump_ct = pump_power(ct_water_mass_flow, pump_pressure_ct, pump_efficiency_ct, 1000)
    ct_air_mass_flow = cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, ct_heat_removed, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)[1]
    density_oa = 1 / HAPropsSI('Vha', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa)
    power_fan_ct = fan_power(ct_air_mass_flow, density_oa, fan_pressure_ct, fan_efficiency_ct)

    power_pump_cl = pump_power(coolant_mass_flow, pump_pressure_cl, pump_efficiency_cl, coolant_density)

    power_components_single = np.array([
        power_it,
        (power_it / ups_efficiency - power_it),
        (power_it / (1 - power_distribution_loss_rate) - power_it),
        power_it * lighting_percentage,
        power_pump_hd,
        power_chiller,
        power_pump_cw,
        power_pump_ct,
        power_fan_ct,
        power_pump_cl
    ])

    power_components_two_phase = np.array([
        power_it,
        (power_it / ups_efficiency - power_it),
        (power_it / (1 - power_distribution_loss_rate) - power_it),
        power_it * lighting_percentage,
        power_pump_hd,
        power_chiller,
        power_pump_cw,
        power_pump_ct,
        power_fan_ct
    ])

    pue_single = np.sum(power_components_single) / power_it
    pue_two_phase = np.sum(power_components_two_phase) / power_it

    ct_water = cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, ct_heat_removed, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)
    water_components = np.concatenate((
        [(hd_amount + hd_amount_ae) / water_efficiency],
        np.maximum([0, 0, 0], ct_water[3:6])
    ))
    wue_single = np.sum(water_components) * 3600 / power_it
    wue_two_phase = np.sum(water_components) * 3600 / power_it

    return pue_single, pue_two_phase, wue_single, wue_two_phase


def pue_wue_immersion_chiller_waterside_economizer(inputs):
    """
    Calculate single-phase and two-phase immersion PUE/WUE with waterside economizer and chiller cooling.
    """
    power_it = 1
    t_oa = inputs[0]
    rh_oa = inputs[1]
    p_oa = inputs[2]
    ups_efficiency = inputs[3]
    power_distribution_loss_rate = inputs[4]
    lighting_percentage = inputs[5]
    delta_t_air = inputs[6]
    fan_pressure_crac = inputs[7]
    fan_efficiency_crac = inputs[8]
    pump_pressure_hd = inputs[9]
    pump_efficiency_hd = inputs[10]
    at_ct = inputs[11]
    chiller_load = inputs[12]
    delta_t_water = inputs[13]
    pump_pressure_cw = inputs[14]
    pump_efficiency_cw = inputs[15]
    delta_t_ct = inputs[16]
    pump_pressure_ct = inputs[17]
    pump_efficiency_ct = inputs[18]
    windage_rate = inputs[19]
    cycles_of_concentration = inputs[20]
    fan_pressure_ct = inputs[21]
    fan_efficiency_ct = inputs[22]
    sensible_heat_ratio = inputs[23]
    liquid_gas_ratio = inputs[24]
    t_up = inputs[25]
    t_lw = inputs[26]
    dp_up = inputs[27]
    dp_lw = inputs[28]
    rh_up = inputs[29]
    rh_lw = inputs[30]
    cop_adjustment = inputs[31]
    heat_transfer_effectiveness = inputs[32]
    at_he = inputs[33]
    pump_pressure_we = inputs[34]
    pump_efficiency_we = inputs[35]
    water_efficiency = 1
    coolant_density = inputs[36]
    coolant_flow_rate = inputs[37]
    coolant_mass_flow = coolant_flow_rate * coolant_density / (1000 * 60)
    pump_pressure_cl = inputs[38]
    pump_efficiency_cl = inputs[39]

    fan_power = lambda mass_flow_rate, air_density, fan_pressure, fan_efficiency:\
        mass_flow_rate / air_density * fan_pressure / fan_efficiency / 1000
    pump_power = lambda mass_flow_rate, pump_pressure, pump_efficiency, liquid_density:\
        pump_pressure * mass_flow_rate / (1000 * pump_efficiency * liquid_density)

    wet_bulb_oa = HAPropsSI('Twb', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa) - 273.15
    heat_load = power_it + (power_it / (1 - power_distribution_loss_rate) - power_it)
    latent_heat_load = heat_load / sensible_heat_ratio - heat_load

    cs_result = chiller_system(t_up, t_lw, dp_up, dp_lw, rh_up, rh_lw, t_oa, rh_oa, p_oa, delta_t_air)
    t_sa = cs_result[0]
    d_sa = cs_result[1]
    h_sa = cs_result[2] / 1000
    t_ra = t_sa + delta_t_air
    m_sa = heat_load / (1.01 * delta_t_air)
    m_sa_dry = m_sa / (1 + d_sa)
    density_sa = 1 / HAPropsSI('Vha', 'T', t_sa + 273.15, 'W', d_sa, 'P', p_oa)

    power_fan_crac = fan_power(m_sa, density_sa, fan_pressure_crac, fan_efficiency_crac)
    h_ra = (latent_heat_load + heat_load) / m_sa + h_sa
    d_ra = HAPropsSI('W', 'H', h_ra * 1000, 'T', t_ra + 273.15, 'P', p_oa)
    d_ra_chiller = latent_heat_load / 2266
    hd_amount = np.maximum(d_ra_chiller, 0)
    power_pump_hd = pump_power(hd_amount, pump_pressure_hd, pump_efficiency_hd, 1000)

    cooling_required = heat_load + latent_heat_load
    t_sfw = t_ra - (t_ra - t_sa) / heat_transfer_effectiveness
    t_rfw = t_sfw + delta_t_water

    we_use = waterside_economizer(t_sfw, t_rfw, wet_bulb_oa, at_ct, at_he, cooling_required)[0]
    we_heat_removed = waterside_economizer(t_sfw, t_rfw, wet_bulb_oa, at_ct, at_he, cooling_required)[1]
    m_sfw = we_heat_removed / (4.2 * delta_t_water)
    power_pump_we = pump_power(m_sfw, pump_pressure_we, pump_efficiency_we, 1000)

    chiller_heat_removed = cooling_required - we_heat_removed
    m_sw = chiller_heat_removed / (4.2 * delta_t_water)
    power_pump_cw = pump_power(m_sw, pump_pressure_cw, pump_efficiency_cw, 1000)

    chiller_cop = COP_GP.predict(np.array([wet_bulb_oa + at_ct, chiller_load]).reshape(1, 2))[0] * (1 + cop_adjustment)
    power_chiller = chiller_heat_removed / chiller_cop

    ct_heat_removed = cooling_required + power_chiller
    ct_water_mass_flow = ct_heat_removed / (4.184 * delta_t_ct)
    power_pump_ct = pump_power(ct_water_mass_flow, pump_pressure_ct, pump_efficiency_ct, 1000)
    ct_air_mass_flow = cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, heat_load, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)[1]
    density_oa = 1 / HAPropsSI('Vha', 'T', t_oa + 273.15, 'RH', rh_oa / 100, 'P', p_oa)
    power_fan_ct = fan_power(ct_air_mass_flow, density_oa, fan_pressure_ct, fan_efficiency_ct)

    power_pump_cl = pump_power(coolant_mass_flow, pump_pressure_cl, pump_efficiency_cl, coolant_density)

    power_components_single = np.array([
        power_it,
        (power_it / ups_efficiency - power_it),
        (power_it / (1 - power_distribution_loss_rate) - power_it),
        power_it * lighting_percentage,
        power_pump_hd,
        power_pump_we,
        power_chiller,
        power_pump_cw,
        power_pump_ct,
        power_fan_ct,
        power_pump_cl
    ])

    power_components_two_phase = np.array([
        power_it,
        (power_it / ups_efficiency - power_it),
        (power_it / (1 - power_distribution_loss_rate) - power_it),
        power_it * lighting_percentage,
        power_pump_hd,
        power_pump_we,
        power_chiller,
        power_pump_cw,
        power_pump_ct,
        power_fan_ct
    ])

    pue_single = np.sum(power_components_single) / power_it
    pue_two_phase = np.sum(power_components_two_phase) / power_it

    ct_water = cooling_tower(t_oa, rh_oa, p_oa, at_ct, power_it, heat_load, delta_t_ct, windage_rate, cycles_of_concentration, liquid_gas_ratio)
    water_components = np.concatenate((
        [hd_amount / water_efficiency],
        ct_water[3:6]
    ))
    wue_single = np.sum(water_components) * 3600 / power_it
    wue_two_phase = np.sum(water_components) * 3600 / power_it

    return pue_single, pue_two_phase, wue_single, wue_two_phase
