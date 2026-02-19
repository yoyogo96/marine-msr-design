"""
Emergency Drain Tank Design
=============================

Sizing and subcriticality analysis for the emergency fuel salt drain tank.

The drain tank receives the entire fuel salt inventory in an emergency.
Design requirements:
  - Volume: 110% of total fuel salt inventory
  - Geometry: cylindrical, H/D = 1.5, surrounded by borated concrete
  - Subcriticality: keff < 0.95
  - Passive decay heat removal via natural convection air cooling
  - Gravity drain time estimate

References:
  - ORNL-TM-3763, "Emergency Drain System for the MSBR" (1972)
  - ORNL-4541, Section 6.3 (Drain Tank Design)
  - ANS-5.1-2014 (Decay Heat Standard)
  - Todreas & Kazimi, "Nuclear Systems" (natural convection)
"""

import os
import sys
import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from config import (
    THERMAL_POWER, DRAIN_TANK_KEFF_LIMIT,
    SIGMA_FISSION_U235, SIGMA_ABSORPTION_U235, SIGMA_ABSORPTION_U238,
    U235_ENRICHMENT, NEUTRONS_PER_FISSION,
    HASTELLOY_N, DESIGN_LIFE_YEARS,
    compute_derived,
)
from thermal_hydraulics.salt_properties import flibe_density, flibe_specific_heat


# =============================================================================
# Constants
# =============================================================================

G = 9.81                        # m/s2
STEFAN_BOLTZMANN = 5.67e-8      # W/(m2-K4)


# =============================================================================
# DrainTank Dataclass
# =============================================================================

@dataclass
class DrainTankDesign:
    """Emergency drain tank design output."""

    # --- Geometry ---
    salt_volume: float              # m3, total fuel salt inventory
    tank_volume: float              # m3, 110% of salt volume
    tank_diameter: float            # m
    tank_height: float              # m
    tank_wall_thickness: float      # m
    tank_outer_diameter: float      # m
    H_D_ratio: float                # height to diameter ratio

    # --- Masses ---
    salt_mass: float                # kg
    tank_mass: float                # kg (shell only)

    # --- Subcriticality ---
    k_inf_tank: float               # infinite multiplication factor in tank geometry
    keff_estimated: float           # effective multiplication factor (with leakage)
    B_squared: float                # m^-2, geometric buckling
    subcritical: bool               # True if keff < limit
    subcriticality_margin: float    # (limit - keff)

    # --- Decay Heat Removal ---
    Q_decay_1min: float             # W, decay heat at 1 minute
    Q_decay_1hr: float              # W, decay heat at 1 hour
    Q_decay_1day: float             # W, decay heat at 1 day
    T_tank_equilibrium: float       # K, equilibrium tank wall temperature
    h_air_natural: float            # W/(m2-K), natural convection coefficient
    A_tank_surface: float           # m2, total tank exterior surface area
    cooling_adequate: bool          # True if T_eq < salt boiling

    # --- Drain Time ---
    drain_orifice_diameter: float   # m
    drain_time: float               # s, estimated time to drain
    drain_pipe_length: float        # m, assumed pipe length
    head_height: float              # m, elevation head from vessel to tank


# =============================================================================
# Tank Geometry
# =============================================================================

def size_drain_tank(salt_volume, H_D_ratio=1.5, margin=1.10):
    """Size the drain tank for the given salt inventory.

    Args:
        salt_volume: Total fuel salt volume (m3)
        H_D_ratio: Height to diameter ratio
        margin: Volume margin factor (1.10 = 110%)

    Returns:
        tuple: (tank_volume, diameter, height)
    """
    V_tank = salt_volume * margin

    # V = pi/4 * D^2 * H = pi/4 * D^2 * (H/D * D) = pi/4 * H/D * D^3
    D = (4.0 * V_tank / (math.pi * H_D_ratio))**(1.0 / 3.0)
    H = H_D_ratio * D

    return V_tank, D, H


# =============================================================================
# Subcriticality Analysis
# =============================================================================

def estimate_keff_drain_tank(D, H, salt_volume, salt_mass, design_params):
    """Estimate keff for fuel salt in drain tank geometry.

    The drain tank has NO moderator (no graphite), so the spectrum is
    predominantly fast/epithermal. k_inf is much lower than in the moderated
    core.

    Simplified approach:
      k_inf = nu * Sigma_f / Sigma_a  (homogeneous, no moderator)
      keff = k_inf / (1 + B^2 * M^2)

    Without graphite moderator, the thermal utilization is very poor and
    k_inf drops well below 1.0 for reasonable enrichments.

    With borated concrete surrounding the tank, additional absorption
    further reduces keff.

    Args:
        D: Tank diameter (m)
        H: Tank height (m)
        salt_volume: Salt volume in tank (m3)
        salt_mass: Salt mass (kg)
        design_params: DerivedParameters

    Returns:
        tuple: (k_inf, keff, B_squared)
    """
    # Number densities in fuel salt (approximate)
    # Molecular weight of FLiBe+UF4: ~50 g/mol average
    MW_avg = 50.0  # g/mol (rough)
    N_A = 6.022e23  # atoms/mol

    rho_salt = salt_mass / salt_volume  # kg/m3
    rho_salt_gcc = rho_salt / 1000.0  # g/cm3

    # Number density of salt molecules
    N_salt = rho_salt_gcc * N_A / MW_avg  # molecules/cm3

    # UF4 mole fraction
    x_UF4 = config.UF4_MOLE_FRACTION

    # Number density of uranium atoms
    N_U = N_salt * x_UF4  # U atoms/cm3
    N_U235 = N_U * U235_ENRICHMENT
    N_U238 = N_U * (1.0 - U235_ENRICHMENT)

    # In the drain tank: NO moderator -> fast/epithermal spectrum
    # Use spectrum-averaged cross sections (these are thermal; in a fast spectrum
    # the fission cross section is MUCH lower, ~1-2 barns vs 430 barns thermal)
    # This is the key reason the drain tank is subcritical.

    # Fast spectrum cross-sections (approximate):
    sigma_f_235_fast = 1.2e-24    # cm2 (~1.2 barns, fast fission)
    sigma_a_235_fast = 1.7e-24    # cm2 (~1.7 barns, fast absorption)
    sigma_a_238_fast = 0.3e-24    # cm2 (~0.3 barns, fast absorption in U-238)

    # Cross-sections for salt constituents (fast spectrum, very small)
    # Li, Be, F have negligible fast absorption
    sigma_a_salt_fast = 0.001e-24  # cm2 per salt molecule (very small)

    # Macroscopic cross-sections
    Sigma_f = N_U235 * sigma_f_235_fast  # cm^-1
    Sigma_a = (N_U235 * sigma_a_235_fast +
               N_U238 * sigma_a_238_fast +
               N_salt * sigma_a_salt_fast)  # cm^-1

    nu = NEUTRONS_PER_FISSION

    # k_infinity (no leakage)
    k_inf = nu * Sigma_f / Sigma_a if Sigma_a > 0 else 0.0

    # Geometric buckling for finite cylinder
    R = D / 2.0
    # Add extrapolation distance (~0.71 * transport MFP, ~2 cm for fast neutrons in salt)
    d_extrap = 0.02  # m (2 cm, rough)
    R_eff = R + d_extrap
    H_eff = H + 2 * d_extrap

    B_sq_radial = (2.405 / R_eff)**2   # m^-2
    B_sq_axial = (math.pi / H_eff)**2  # m^-2
    B_sq = B_sq_radial + B_sq_axial     # m^-2

    # Migration area for unmoderated salt (fast spectrum)
    # M^2 ~ L^2 + tau ~ 0.01 + 0.03 = 0.04 m^2 (rough for fast system)
    M_sq = 0.04  # m^2

    # keff with leakage
    keff = k_inf / (1.0 + B_sq * M_sq)

    # Additional correction for borated concrete absorber
    # B4C lining reduces keff by ~5-10%
    boron_correction = 0.90  # conservative 10% reduction
    keff *= boron_correction

    return k_inf, keff, B_sq


# =============================================================================
# Decay Heat Removal
# =============================================================================

def decay_heat(P0, t, T_operating=3.156e7):
    """ANS standard decay heat approximation.

    P_decay(t) = P0 * 0.066 * [t^(-0.2) - (t + T_op)^(-0.2)]

    Args:
        P0: Nominal thermal power (W)
        t: Time after shutdown (s)
        T_operating: Operating time before shutdown (s)

    Returns:
        float: Decay heat power (W)
    """
    if t <= 0:
        return P0 * 0.066  # initial value
    return P0 * 0.066 * (t**(-0.2) - (t + T_operating)**(-0.2))


def equilibrium_temperature(Q_decay, A_surface, T_air=313.15, h_air=10.0):
    """Find equilibrium tank wall temperature for passive air cooling.

    Q_decay = h_air * A * (T_wall - T_air)
    T_wall = T_air + Q_decay / (h_air * A)

    For natural convection on a vertical cylinder:
    h_air ~ 5-15 W/(m2-K) depending on temperature difference

    Args:
        Q_decay: Decay heat power (W)
        A_surface: Tank exterior surface area (m2)
        T_air: Ambient air temperature (K), default 40 C
        h_air: Natural convection coefficient (W/(m2-K))

    Returns:
        float: Equilibrium wall temperature (K)
    """
    return T_air + Q_decay / (h_air * A_surface)


def natural_convection_h(T_wall, T_air, H):
    """Estimate natural convection coefficient for vertical cylinder.

    Uses Churchill-Chu correlation for natural convection on vertical plate
    (applicable to cylinders when D >> boundary layer thickness).

    Nu_L = {0.825 + 0.387 * Ra_L^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2

    For air at atmospheric pressure.

    Args:
        T_wall: Wall temperature (K)
        T_air: Air temperature (K)
        H: Height (m)

    Returns:
        float: h in W/(m2-K)
    """
    T_film = (T_wall + T_air) / 2.0
    T_C = T_film - 273.15

    # Air properties at film temperature (approximate)
    rho_air = 1.225 * 293.15 / T_film  # kg/m3 (ideal gas correction)
    mu_air = 1.8e-5 * (T_film / 293.15)**0.7  # Pa-s
    k_air = 0.026 * (T_film / 293.15)**0.7  # W/(m-K)
    cp_air = 1005.0  # J/(kg-K)
    beta_air = 1.0 / T_film  # 1/K (ideal gas)
    nu_air = mu_air / rho_air  # m2/s
    alpha_air = k_air / (rho_air * cp_air)  # m2/s
    Pr = nu_air / alpha_air

    dT = abs(T_wall - T_air)
    if dT < 0.1:
        return 5.0  # minimum

    Ra = G * beta_air * dT * H**3 / (nu_air * alpha_air)

    # Churchill-Chu correlation
    Nu = (0.825 + 0.387 * Ra**(1.0 / 6.0) /
          (1.0 + (0.492 / Pr)**(9.0 / 16.0))**(8.0 / 27.0))**2

    h = Nu * k_air / H
    return max(h, 5.0)  # minimum 5 W/(m2-K)


# =============================================================================
# Drain Time Estimate
# =============================================================================

def estimate_drain_time(V_salt, d_orifice, head_height, L_pipe=5.0):
    """Estimate gravity drain time using orifice flow equation.

    Torricelli's theorem with losses:
    Q = Cd * A_orifice * sqrt(2*g*h)

    Time to drain (constant head approximation):
    t_drain = V / Q

    More accurate: for falling head, t = V / (Cd * A * sqrt(2*g)) * 2 * sqrt(h_0) / sqrt(g)
    ... simplified to: t ~ 2 * V / (Cd * A * sqrt(2*g*h))

    Args:
        V_salt: Salt volume to drain (m3)
        d_orifice: Drain orifice diameter (m)
        head_height: Elevation difference vessel to tank (m)
        L_pipe: Drain pipe length (m)

    Returns:
        float: Estimated drain time (s)
    """
    Cd = 0.62  # Discharge coefficient for sharp-edged orifice
    A_orifice = math.pi / 4.0 * d_orifice**2

    # Account for pipe friction losses (reduce effective head by ~30%)
    h_eff = head_height * 0.70

    # Average flow rate (using average head = h/2 for falling level)
    Q_avg = Cd * A_orifice * math.sqrt(2.0 * G * h_eff / 2.0)

    # Drain time
    t_drain = V_salt / Q_avg if Q_avg > 0 else float('inf')

    return t_drain


# =============================================================================
# Main Design Function
# =============================================================================

def design_drain_tank(design_params=None):
    """Design the emergency drain tank.

    Args:
        design_params: DerivedParameters from config (computed if None)

    Returns:
        DrainTankDesign dataclass
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    # --- Salt inventory ---
    salt_volume = d.fuel_salt_volume_total
    T_avg = config.CORE_AVG_TEMP
    rho_salt = flibe_density(T_avg, uf4_mol_fraction=0.05)
    salt_mass = salt_volume * rho_salt

    # --- Tank geometry ---
    H_D = 1.5
    V_tank, D_tank, H_tank = size_drain_tank(salt_volume, H_D)

    # Tank wall thickness (ASME, simplified: same as vessel)
    t_wall = max(d.vessel_wall_thickness, 0.015)  # at least 15 mm

    D_outer = D_tank + 2 * t_wall

    # Tank mass (cylindrical shell + flat heads for simplicity)
    rho_h = HASTELLOY_N['density']
    V_shell = math.pi * D_tank * t_wall * H_tank  # side wall
    V_heads = 2 * math.pi / 4.0 * D_tank**2 * t_wall  # two flat heads
    tank_mass = (V_shell + V_heads) * rho_h

    # --- Subcriticality analysis ---
    k_inf, keff, B_sq = estimate_keff_drain_tank(D_tank, H_tank, V_tank, salt_mass, d)
    subcritical = keff < DRAIN_TANK_KEFF_LIMIT
    margin = DRAIN_TANK_KEFF_LIMIT - keff

    # --- Decay heat removal ---
    Q_1min = decay_heat(THERMAL_POWER, 60.0)
    Q_1hr = decay_heat(THERMAL_POWER, 3600.0)
    Q_1day = decay_heat(THERMAL_POWER, 86400.0)

    # Tank surface area
    A_side = math.pi * D_outer * H_tank
    A_ends = 2 * math.pi / 4.0 * D_outer**2
    A_total = A_side + A_ends

    # Find equilibrium temperature at 1 hour (when drain is complete)
    T_air = 313.15  # 40 C ambient in engine room

    # Iterate for natural convection h
    T_wall_guess = 600 + 273.15  # initial guess
    for _ in range(20):
        h_air = natural_convection_h(T_wall_guess, T_air, H_tank)
        T_wall_new = equilibrium_temperature(Q_1hr, A_total, T_air, h_air)
        if abs(T_wall_new - T_wall_guess) < 1.0:
            break
        T_wall_guess = 0.5 * (T_wall_guess + T_wall_new)

    T_eq = T_wall_guess
    cooling_ok = T_eq < config.MAX_FUEL_SALT_TEMP

    # --- Drain time ---
    d_orifice = 0.15  # m (6 inch drain line)
    head_height = 5.0  # m (vessel above drain tank)
    L_pipe = 6.0  # m
    t_drain = estimate_drain_time(salt_volume, d_orifice, head_height, L_pipe)

    return DrainTankDesign(
        salt_volume=salt_volume,
        tank_volume=V_tank,
        tank_diameter=D_tank,
        tank_height=H_tank,
        tank_wall_thickness=t_wall,
        tank_outer_diameter=D_outer,
        H_D_ratio=H_D,
        salt_mass=salt_mass,
        tank_mass=tank_mass,
        k_inf_tank=k_inf,
        keff_estimated=keff,
        B_squared=B_sq,
        subcritical=subcritical,
        subcriticality_margin=margin,
        Q_decay_1min=Q_1min,
        Q_decay_1hr=Q_1hr,
        Q_decay_1day=Q_1day,
        T_tank_equilibrium=T_eq,
        h_air_natural=h_air,
        A_tank_surface=A_total,
        cooling_adequate=cooling_ok,
        drain_orifice_diameter=d_orifice,
        drain_time=t_drain,
        drain_pipe_length=L_pipe,
        head_height=head_height,
    )


# =============================================================================
# Printing
# =============================================================================

def print_drain_tank(dt):
    """Print formatted drain tank design summary.

    Args:
        dt: DrainTankDesign dataclass instance
    """
    print("=" * 72)
    print("   EMERGENCY DRAIN TANK DESIGN")
    print("=" * 72)

    print("\n--- Salt Inventory ---")
    print(f"  Total fuel salt volume:     {dt.salt_volume:10.3f} m3")
    print(f"  Total fuel salt mass:       {dt.salt_mass:10.1f} kg ({dt.salt_mass / 1e3:.2f} tonnes)")

    print("\n--- Tank Geometry ---")
    print(f"  Tank volume (110%):         {dt.tank_volume:10.3f} m3")
    print(f"  Diameter:                   {dt.tank_diameter:10.3f} m")
    print(f"  Height:                     {dt.tank_height:10.3f} m")
    print(f"  H/D ratio:                  {dt.H_D_ratio:10.2f}")
    print(f"  Wall thickness:             {dt.tank_wall_thickness * 1e3:10.1f} mm")
    print(f"  Outer diameter:             {dt.tank_outer_diameter:10.3f} m")
    print(f"  Surface area:               {dt.A_tank_surface:10.2f} m2")

    print("\n--- Mass ---")
    print(f"  Tank shell mass:            {dt.tank_mass:10.1f} kg")

    print("\n--- Subcriticality Analysis ---")
    print(f"  k_infinity (no moderator):  {dt.k_inf_tank:10.4f}")
    print(f"  keff estimated:             {dt.keff_estimated:10.4f}")
    print(f"  Geometric buckling B^2:     {dt.B_squared:10.2f} m^-2")
    print(f"  keff limit:                 {DRAIN_TANK_KEFF_LIMIT:10.2f}")
    print(f"  Margin:                     {dt.subcriticality_margin:10.4f}")
    print(f"  Status:                     {'SUBCRITICAL - OK' if dt.subcritical else 'NOT SUBCRITICAL - REDESIGN!'}")

    print("\n--- Decay Heat Removal ---")
    print(f"  Decay heat at 1 min:        {dt.Q_decay_1min / 1e3:10.1f} kW ({dt.Q_decay_1min / THERMAL_POWER * 100:.2f}% of P0)")
    print(f"  Decay heat at 1 hour:       {dt.Q_decay_1hr / 1e3:10.1f} kW ({dt.Q_decay_1hr / THERMAL_POWER * 100:.2f}% of P0)")
    print(f"  Decay heat at 1 day:        {dt.Q_decay_1day / 1e3:10.1f} kW ({dt.Q_decay_1day / THERMAL_POWER * 100:.2f}% of P0)")
    print(f"  Natural convection h:       {dt.h_air_natural:10.2f} W/(m2-K)")
    print(f"  Equilibrium T (at 1 hr):    {dt.T_tank_equilibrium - 273.15:10.1f} C")
    print(f"  Salt boiling point:         {config.MAX_FUEL_SALT_TEMP - 273.15:10.1f} C")
    print(f"  Cooling adequate:           {'YES' if dt.cooling_adequate else 'NO - NEEDS ACTIVE COOLING'}")

    print("\n--- Drain System ---")
    print(f"  Drain orifice diameter:     {dt.drain_orifice_diameter * 1e3:10.1f} mm")
    print(f"  Elevation head:             {dt.head_height:10.1f} m")
    print(f"  Drain pipe length:          {dt.drain_pipe_length:10.1f} m")
    print(f"  Estimated drain time:       {dt.drain_time:10.1f} s ({dt.drain_time / 60:.1f} min)")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()
    dt = design_drain_tank(d)
    print_drain_tank(dt)

    # --- Decay heat curve ---
    print("\n--- Decay Heat vs Time ---")
    times = [1, 10, 60, 600, 3600, 86400, 604800]
    labels = ["1 s", "10 s", "1 min", "10 min", "1 hr", "1 day", "1 week"]

    print(f"  {'Time':>10s}  {'Q_decay [kW]':>12s}  {'% of P0':>8s}")
    print(f"  {'-' * 10}  {'-' * 12}  {'-' * 8}")
    for lbl, t in zip(labels, times):
        Q = decay_heat(THERMAL_POWER, t)
        pct = Q / THERMAL_POWER * 100
        print(f"  {lbl:>10s}  {Q / 1e3:12.1f}  {pct:7.2f}%")
