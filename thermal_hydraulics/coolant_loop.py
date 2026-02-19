"""
Primary and Intermediate Coolant Loop Hydraulic Analysis
========================================================

Component-by-component pressure drop calculation for the primary FLiBe loop
and the intermediate FLiNaK loop.  Includes pump sizing and natural
circulation capability estimation for safety analysis (ULOF scenario).

Primary loop components
-----------------------
1. Core fuel channels
2. Upper plenum (form loss)
3. Lower plenum (form loss)
4. Hot leg piping
5. Heat exchanger primary side
6. Cold leg piping
7. Pump (head source)

References
----------
* Todreas & Kazimi, "Nuclear Systems I"
* ORNL-TM-3832 (MSRE hydraulic analysis)
* El-Wakil, "Nuclear Heat Transport"
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import (
    THERMAL_POWER, CORE_INLET_TEMP, CORE_OUTLET_TEMP, CORE_AVG_TEMP,
    CHANNEL_DIAMETER, CHANNEL_PITCH, OPERATING_PRESSURE,
    SECONDARY_INLET_TEMP, SECONDARY_OUTLET_TEMP,
    FUEL_SALT_FRACTION, compute_derived,
)
from thermal_hydraulics.salt_properties import (
    flibe_density, flibe_viscosity, flibe_specific_heat,
    flibe_thermal_conductivity, flibe_prandtl,
    flinak_density, flinak_viscosity, flinak_specific_heat,
    flinak_thermal_conductivity,
    friction_factor as calc_friction_factor,
)


# ==========================================================================
# Data classes
# ==========================================================================

@dataclass
class ComponentDrop:
    """Pressure drop contribution from a single loop component."""
    name: str
    delta_p: float              # Pa
    length: float = 0.0        # m (characteristic length)
    diameter: float = 0.0      # m (hydraulic diameter)
    velocity: float = 0.0      # m/s
    K_loss: float = 0.0        # form loss coefficient (if applicable)
    f_friction: float = 0.0    # Darcy friction factor (if applicable)


@dataclass
class LoopResult:
    """Complete hydraulic analysis results for a coolant loop."""

    loop_name: str
    components: List[ComponentDrop]         # component-by-component breakdown

    # Totals
    total_pressure_drop: float              # Pa
    pump_head_required: float               # m of fluid
    pump_power_hydraulic: float             # W
    pump_power_shaft: float                 # W (accounting for efficiency)
    pump_efficiency: float                  # fraction

    # Flow conditions
    mass_flow_rate: float                   # kg/s
    volumetric_flow_rate: float             # m3/s

    # Natural circulation
    natural_circ_dp: float                  # Pa (buoyancy driving head)
    natural_circ_flow: float                # kg/s (estimated NC mass flow)
    natural_circ_fraction: float            # fraction of nominal flow


# ==========================================================================
# Pressure drop helpers
# ==========================================================================

def _pipe_friction_dp(L, D, rho, v, Re):
    """Frictional pressure drop in a straight pipe section.

    Args:
        L: Pipe length, m.
        D: Inner diameter, m.
        rho: Fluid density, kg/m3.
        v: Mean velocity, m/s.
        Re: Reynolds number.

    Returns:
        (delta_p, f): Pressure drop in Pa and Darcy friction factor.
    """
    f = calc_friction_factor(Re, roughness=0.0, D=D)
    dp = f * (L / D) * 0.5 * rho * v**2
    return dp, f


def _form_loss_dp(K, rho, v):
    """Pressure drop from a form loss (elbow, expansion, contraction, etc.).

    Args:
        K: Loss coefficient.
        rho: Fluid density, kg/m3.
        v: Reference velocity, m/s.

    Returns:
        Pressure drop in Pa.
    """
    return K * 0.5 * rho * v**2


# ==========================================================================
# Primary loop analysis
# ==========================================================================

def primary_loop_analysis(design_params=None, pipe_length=5.0,
                          pipe_diameter=0.3, hx_dp_fraction=0.25,
                          pump_efficiency=0.75):
    """Perform hydraulic analysis of the primary FLiBe loop.

    Args:
        design_params: DerivedParameters from config.  Computed if None.
        pipe_length: Length of each pipe leg (hot and cold), m.
        pipe_diameter: Inner diameter of primary piping, m.
        hx_dp_fraction: HX primary-side pressure drop as fraction of core dP.
        pump_efficiency: Pump isentropic efficiency.

    Returns:
        LoopResult for the primary loop.
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    T_avg = CORE_AVG_TEMP
    rho = flibe_density(T_avg, uf4_mol_fraction=0.05)
    mu = flibe_viscosity(T_avg)
    cp = flibe_specific_heat(T_avg)
    m_dot = d.mass_flow_rate
    Q_vol = d.volumetric_flow_rate

    components = []

    # ------------------------------------------------------------------
    # 1. Core channels
    # ------------------------------------------------------------------
    v_core = d.salt_velocity
    Re_core = d.reynolds_number
    dp_core, f_core = _pipe_friction_dp(d.core_height, CHANNEL_DIAMETER,
                                         rho, v_core, Re_core)
    components.append(ComponentDrop(
        name="Core channels",
        delta_p=dp_core,
        length=d.core_height,
        diameter=CHANNEL_DIAMETER,
        velocity=v_core,
        f_friction=f_core,
    ))

    # ------------------------------------------------------------------
    # 2. Upper plenum (expansion + turning)
    # ------------------------------------------------------------------
    # Plenum velocity: roughly flow area of vessel downcomer annulus
    A_vessel = math.pi * d.vessel_inner_radius**2
    A_core = math.pi * d.core_radius**2
    A_plenum = A_vessel  # approximate
    v_plenum = Q_vol / A_plenum
    K_upper = 1.5
    dp_upper = _form_loss_dp(K_upper, rho, v_plenum)
    components.append(ComponentDrop(
        name="Upper plenum",
        delta_p=dp_upper,
        velocity=v_plenum,
        K_loss=K_upper,
    ))

    # ------------------------------------------------------------------
    # 3. Lower plenum (turning + contraction)
    # ------------------------------------------------------------------
    K_lower = 1.5
    dp_lower = _form_loss_dp(K_lower, rho, v_plenum)
    components.append(ComponentDrop(
        name="Lower plenum",
        delta_p=dp_lower,
        velocity=v_plenum,
        K_loss=K_lower,
    ))

    # ------------------------------------------------------------------
    # 4. Hot leg piping
    # ------------------------------------------------------------------
    A_pipe = math.pi / 4.0 * pipe_diameter**2
    v_pipe = Q_vol / A_pipe
    Re_pipe = rho * v_pipe * pipe_diameter / mu
    dp_hot, f_hot = _pipe_friction_dp(pipe_length, pipe_diameter,
                                       rho, v_pipe, Re_pipe)
    # Add 2 elbows (K=0.3 each)
    dp_hot += _form_loss_dp(0.6, rho, v_pipe)
    components.append(ComponentDrop(
        name="Hot leg piping",
        delta_p=dp_hot,
        length=pipe_length,
        diameter=pipe_diameter,
        velocity=v_pipe,
        f_friction=f_hot,
        K_loss=0.6,
    ))

    # ------------------------------------------------------------------
    # 5. Heat exchanger (primary side)
    # ------------------------------------------------------------------
    dp_hx = hx_dp_fraction * dp_core  # estimate as fraction of core dP
    components.append(ComponentDrop(
        name="Heat exchanger (primary)",
        delta_p=dp_hx,
    ))

    # ------------------------------------------------------------------
    # 6. Cold leg piping (same geometry as hot leg)
    # ------------------------------------------------------------------
    dp_cold = dp_hot  # symmetric layout
    components.append(ComponentDrop(
        name="Cold leg piping",
        delta_p=dp_cold,
        length=pipe_length,
        diameter=pipe_diameter,
        velocity=v_pipe,
        f_friction=f_hot,
        K_loss=0.6,
    ))

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    total_dp = sum(c.delta_p for c in components)
    pump_head = total_dp / (rho * 9.81)
    pump_power_hyd = total_dp * Q_vol
    pump_power_shaft = pump_power_hyd / pump_efficiency

    # ------------------------------------------------------------------
    # Natural circulation estimate
    # ------------------------------------------------------------------
    nc_dp, nc_flow, nc_frac = _natural_circulation_estimate(
        d, rho, mu, cp, total_dp, m_dot)

    return LoopResult(
        loop_name="Primary (FLiBe)",
        components=components,
        total_pressure_drop=total_dp,
        pump_head_required=pump_head,
        pump_power_hydraulic=pump_power_hyd,
        pump_power_shaft=pump_power_shaft,
        pump_efficiency=pump_efficiency,
        mass_flow_rate=m_dot,
        volumetric_flow_rate=Q_vol,
        natural_circ_dp=nc_dp,
        natural_circ_flow=nc_flow,
        natural_circ_fraction=nc_frac,
    )


def _natural_circulation_estimate(d, rho_avg, mu, cp, total_dp_forced, m_dot):
    """Estimate natural circulation driving head and flow.

    The buoyancy driving pressure is:
        dP_nat = rho * g * delta_H * beta * delta_T

    where delta_H is the elevation difference between the core thermal centre
    and the heat exchanger thermal centre, beta is the volumetric thermal
    expansion coefficient, and delta_T is the loop temperature difference.

    In natural circulation the driving head equals the loop resistance:
        dP_nat = dP_loss(m_dot_nc)
    Since dP ~ m_dot^2 in turbulent flow:
        m_dot_nc / m_dot_forced = sqrt(dP_nat / dP_forced)

    Args:
        d: DerivedParameters.
        rho_avg: Average salt density, kg/m3.
        mu: Dynamic viscosity, Pa-s.
        cp: Specific heat, J/(kg-K).
        total_dp_forced: Total loop pressure drop at nominal flow, Pa.
        m_dot: Nominal mass flow rate, kg/s.

    Returns:
        (dp_natural, m_dot_nc, fraction_of_nominal)
    """
    delta_T = CORE_OUTLET_TEMP - CORE_INLET_TEMP  # K

    # Volumetric thermal expansion coefficient for FLiBe
    # beta = -(1/rho)(drho/dT) ~ 0.488 / rho
    beta = 0.488 / rho_avg  # 1/K

    # Height difference: HX centre is assumed ~2 m above core centre
    # (compact marine layout)
    delta_H = 2.0  # m

    g = 9.81  # m/s2
    dp_natural = rho_avg * g * delta_H * beta * delta_T  # Pa

    # Natural circulation flow (turbulent scaling: dP ~ m_dot^2)
    if total_dp_forced > 0 and dp_natural > 0:
        flow_ratio = math.sqrt(dp_natural / total_dp_forced)
    else:
        flow_ratio = 0.0

    m_dot_nc = m_dot * flow_ratio

    return dp_natural, m_dot_nc, flow_ratio


# ==========================================================================
# Intermediate loop analysis
# ==========================================================================

def intermediate_loop_analysis(design_params=None, pipe_length=6.0,
                               pipe_diameter=0.25, pump_efficiency=0.75):
    """Perform hydraulic analysis of the intermediate FLiNaK loop.

    The intermediate loop transfers heat from the primary HX to the
    sCO2 power conversion system (or steam generator).

    Args:
        design_params: DerivedParameters from config.  Computed if None.
        pipe_length: Length of each pipe leg, m.
        pipe_diameter: Inner diameter of secondary piping, m.
        pump_efficiency: Pump isentropic efficiency.

    Returns:
        LoopResult for the intermediate loop.
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    T_avg_sec = (SECONDARY_INLET_TEMP + SECONDARY_OUTLET_TEMP) / 2.0
    rho = flinak_density(T_avg_sec)
    mu = flinak_viscosity(T_avg_sec)
    cp = flinak_specific_heat(T_avg_sec)
    k_salt = flinak_thermal_conductivity(T_avg_sec)

    delta_T_sec = SECONDARY_OUTLET_TEMP - SECONDARY_INLET_TEMP
    m_dot = THERMAL_POWER / (cp * delta_T_sec)
    Q_vol = m_dot / rho

    A_pipe = math.pi / 4.0 * pipe_diameter**2
    v_pipe = Q_vol / A_pipe
    Re_pipe = rho * v_pipe * pipe_diameter / mu

    components = []

    # ------------------------------------------------------------------
    # 1. Primary HX secondary side
    # ------------------------------------------------------------------
    # Estimate: tube side of shell-and-tube HX
    hx_tube_D = 0.015  # m (15 mm tubes)
    hx_tube_L = 2.0    # m (tube length)
    v_hx = v_pipe * (pipe_diameter / hx_tube_D)**2 / 200  # approximate
    # Simplified: use pipe velocity as reference
    Re_hx = rho * v_pipe * hx_tube_D / mu
    dp_hx, f_hx = _pipe_friction_dp(hx_tube_L, hx_tube_D, rho, v_pipe, Re_hx)
    dp_hx *= 0.5  # approximate correction for actual tube bundle
    dp_hx += _form_loss_dp(2.0, rho, v_pipe)  # inlet/outlet losses
    components.append(ComponentDrop(
        name="Primary HX (secondary side)",
        delta_p=dp_hx,
        length=hx_tube_L,
        diameter=hx_tube_D,
        velocity=v_pipe,
    ))

    # ------------------------------------------------------------------
    # 2. Hot leg piping
    # ------------------------------------------------------------------
    dp_hot, f_hot = _pipe_friction_dp(pipe_length, pipe_diameter,
                                       rho, v_pipe, Re_pipe)
    dp_hot += _form_loss_dp(0.6, rho, v_pipe)
    components.append(ComponentDrop(
        name="Hot leg piping",
        delta_p=dp_hot,
        length=pipe_length,
        diameter=pipe_diameter,
        velocity=v_pipe,
        f_friction=f_hot,
        K_loss=0.6,
    ))

    # ------------------------------------------------------------------
    # 3. sCO2 HX secondary side (intermediate-to-power-cycle)
    # ------------------------------------------------------------------
    dp_sco2 = dp_hx * 0.8  # similar order
    components.append(ComponentDrop(
        name="sCO2 HX (intermediate side)",
        delta_p=dp_sco2,
    ))

    # ------------------------------------------------------------------
    # 4. Cold leg piping
    # ------------------------------------------------------------------
    dp_cold = dp_hot  # symmetric
    components.append(ComponentDrop(
        name="Cold leg piping",
        delta_p=dp_cold,
        length=pipe_length,
        diameter=pipe_diameter,
        velocity=v_pipe,
        f_friction=f_hot,
        K_loss=0.6,
    ))

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    total_dp = sum(c.delta_p for c in components)
    pump_head = total_dp / (rho * 9.81)
    pump_power_hyd = total_dp * Q_vol
    pump_power_shaft = pump_power_hyd / pump_efficiency

    return LoopResult(
        loop_name="Intermediate (FLiNaK)",
        components=components,
        total_pressure_drop=total_dp,
        pump_head_required=pump_head,
        pump_power_hydraulic=pump_power_hyd,
        pump_power_shaft=pump_power_shaft,
        pump_efficiency=pump_efficiency,
        mass_flow_rate=m_dot,
        volumetric_flow_rate=Q_vol,
        natural_circ_dp=0.0,
        natural_circ_flow=0.0,
        natural_circ_fraction=0.0,
    )


# ==========================================================================
# Natural circulation standalone function
# ==========================================================================

def natural_circulation_flow(core_height, hx_height_offset, delta_T, T_avg):
    """Compute natural-circulation mass flow rate for safety analysis.

    Uses a simplified loop model where the buoyancy driving head is balanced
    against core friction losses.

    Args:
        core_height: Active core height, m.
        hx_height_offset: Vertical distance from core centre to HX centre, m.
        delta_T: Temperature difference across the core, K.
        T_avg: Average salt temperature in the loop, K.

    Returns:
        dict with keys:
          'driving_pressure' (Pa),
          'mass_flow_rate' (kg/s),
          'core_velocity' (m/s),
          'fraction_of_nominal' (dimensionless)
    """
    d = compute_derived()
    rho = flibe_density(T_avg, uf4_mol_fraction=0.05)
    mu = flibe_viscosity(T_avg)
    beta = 0.488 / rho  # volumetric expansion coefficient

    g = 9.81
    dp_buoyancy = rho * g * hx_height_offset * beta * delta_T

    # Balance against core friction: dp = f * (H/D) * 0.5 * rho * v^2
    # Assume turbulent Blasius friction: f = 0.316 * Re^-0.25
    # Re = rho * v * D / mu
    # dp = 0.316 * (rho*v*D/mu)^-0.25 * (H/D) * 0.5 * rho * v^2
    # Solve iteratively
    H = d.core_height
    D = CHANNEL_DIAMETER
    A_total = d.total_flow_area

    # Initial guess
    v = 0.1  # m/s
    for _ in range(50):
        Re = rho * v * D / mu
        if Re < 10:
            Re = 10  # avoid zero
        if Re < 2300:
            f = 64.0 / Re
        else:
            f = 0.316 * Re**(-0.25)
        dp_friction = f * (H / D) * 0.5 * rho * v**2
        # Add form losses (plena, piping) as 3x core
        dp_total_loop = dp_friction * 3.0

        # Newton-like update: dp_buoyancy = dp_total_loop
        # dp ~ v^1.75 (turbulent), so v_new = v * (dp_buoyancy/dp_total_loop)^(1/1.75)
        if dp_total_loop > 0:
            v = v * (dp_buoyancy / dp_total_loop)**(1.0 / 1.75)
        else:
            v = 0.01
        v = max(v, 1e-4)

    m_dot_nc = rho * v * A_total
    frac = m_dot_nc / d.mass_flow_rate if d.mass_flow_rate > 0 else 0.0

    return {
        'driving_pressure': dp_buoyancy,
        'mass_flow_rate': m_dot_nc,
        'core_velocity': v,
        'fraction_of_nominal': frac,
    }


# ==========================================================================
# Printing utilities
# ==========================================================================

def print_loop_results(res):
    """Print a formatted summary of loop hydraulic analysis.

    Args:
        res: LoopResult instance.
    """
    print("=" * 68)
    print(f"   LOOP HYDRAULIC ANALYSIS: {res.loop_name}")
    print("=" * 68)

    print(f"\n  {'Component':<30s}  {'dP [kPa]':>10s}  {'v [m/s]':>8s}  "
          f"{'K_loss':>6s}  {'f':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*8}")
    for c in res.components:
        print(f"  {c.name:<30s}  {c.delta_p/1e3:10.2f}  "
              f"{c.velocity:8.3f}  {c.K_loss:6.2f}  {c.f_friction:8.5f}")

    print(f"  {'':=<30s}  {'':=<10s}")
    print(f"  {'TOTAL':<30s}  {res.total_pressure_drop/1e3:10.2f} kPa")

    print(f"\n--- Pump Requirements ---")
    print(f"  Pump head:                {res.pump_head_required:10.2f} m")
    print(f"  Pump power (hydraulic):   {res.pump_power_hydraulic/1e3:10.2f} kW")
    print(f"  Pump power (shaft):       {res.pump_power_shaft/1e3:10.2f} kW")
    print(f"  Pump efficiency:          {res.pump_efficiency*100:10.1f} %")

    print(f"\n--- Flow Conditions ---")
    print(f"  Mass flow rate:           {res.mass_flow_rate:10.2f} kg/s")
    print(f"  Volumetric flow rate:     {res.volumetric_flow_rate*1e3:10.3f} L/s")

    if res.natural_circ_dp > 0:
        print(f"\n--- Natural Circulation (ULOF Safety) ---")
        print(f"  Buoyancy driving head:    {res.natural_circ_dp/1e3:10.3f} kPa")
        print(f"  NC mass flow:             {res.natural_circ_flow:10.2f} kg/s")
        print(f"  NC fraction of nominal:   {res.natural_circ_fraction*100:10.1f} %")
    print()


# ==========================================================================
# Main
# ==========================================================================

if __name__ == '__main__':
    d = compute_derived()

    print("\n" + "=" * 68)
    print("   40 MWth MARINE MSR - COOLANT LOOP HYDRAULIC ANALYSIS")
    print("=" * 68 + "\n")

    # Primary loop
    primary = primary_loop_analysis(d)
    print_loop_results(primary)

    # Intermediate loop
    intermediate = intermediate_loop_analysis(d)
    print_loop_results(intermediate)

    # Natural circulation standalone
    print("=" * 68)
    print("   NATURAL CIRCULATION ANALYSIS (STANDALONE)")
    print("=" * 68)
    nc = natural_circulation_flow(
        core_height=d.core_height,
        hx_height_offset=2.0,
        delta_T=CORE_OUTLET_TEMP - CORE_INLET_TEMP,
        T_avg=CORE_AVG_TEMP,
    )
    print(f"  Driving pressure:         {nc['driving_pressure']/1e3:10.3f} kPa")
    print(f"  NC mass flow:             {nc['mass_flow_rate']:10.2f} kg/s")
    print(f"  NC core velocity:         {nc['core_velocity']:10.4f} m/s")
    print(f"  Fraction of nominal:      {nc['fraction_of_nominal']*100:10.1f} %")
    print()
