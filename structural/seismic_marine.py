"""
Ship Motion and Seismic Loads Analysis
========================================

Quasi-static acceleration loads on the reactor system from ship motions
per DNV GL rules for nuclear installations on ships.

Load cases:
  1. Operating + roll + heave (normal sea state)
  2. Operating + seismic (in-port, 0.2g horizontal)
  3. Operating + collision (1g longitudinal)

Also includes:
  - Free surface correction for liquid salt sloshing
  - Support bolt/mount sizing (simplified)
  - Weight and center of gravity estimate

References:
  - DNV GL Rules for Classification of Ships
  - DNV-OS-C502: Offshore Concrete Structures (seismic)
  - ABS Guide for Building and Classing Nuclear Ships
  - IAEA NS-G-1.6: Seismic Design
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
    THERMAL_POWER, HASTELLOY_N, GRAPHITE,
    ROLL_ANGLE_MAX, PITCH_ANGLE_MAX, HEAVE_ACCELERATION,
    ROLL_PERIOD, PITCH_PERIOD,
    SHIP_DISPLACEMENT, compute_derived,
)
from thermal_hydraulics.salt_properties import flibe_density


G = 9.81  # m/s2


# =============================================================================
# Load Case Results
# =============================================================================

@dataclass
class ShipMotionLoads:
    """Results of ship motion / seismic load analysis."""

    # --- Component weights ---
    vessel_mass: float              # kg, vessel (shell + heads)
    salt_mass: float                # kg, fuel salt in vessel
    graphite_mass: float            # kg, graphite moderator
    internals_mass: float           # kg, estimated internals
    total_mass: float               # kg, total reactor assembly
    weight: float                   # N, total weight

    # --- Center of gravity ---
    cg_height: float                # m, CG above vessel bottom

    # --- Acceleration loads (per load case) ---
    # Load case 1: Sea state (roll + heave)
    a_transverse_sea: float         # m/s2
    a_longitudinal_sea: float       # m/s2
    a_vertical_sea: float           # m/s2
    a_combined_sea: float           # m/s2

    # Load case 2: In-port seismic
    a_horizontal_seismic: float     # m/s2
    a_vertical_seismic: float       # m/s2
    a_combined_seismic: float       # m/s2

    # Load case 3: Collision
    a_longitudinal_collision: float # m/s2
    a_combined_collision: float     # m/s2

    # --- Forces on supports (worst case) ---
    F_lateral_max: float            # N, maximum lateral force
    F_vertical_max: float           # N, maximum vertical force (up or down)
    F_longitudinal_max: float       # N, maximum longitudinal force
    overturning_moment_max: float   # N-m, max overturning moment at base

    # --- Free surface effect ---
    GG_prime: float                 # m, metacentric height correction
    free_surface_moment: float      # N-m, free surface moment

    # --- Support sizing ---
    n_bolts: int                    # number of foundation bolts
    bolt_diameter: float            # m, minimum bolt diameter
    bolt_force_max: float           # N, maximum force per bolt
    bolt_stress: float              # Pa, bolt stress at max force
    bolt_safety_factor: float       # SF against yield

    # --- Design load envelope ---
    design_lateral_g: float         # g, design lateral acceleration
    design_vertical_g: float        # g, design vertical acceleration
    design_longitudinal_g: float    # g, design longitudinal acceleration


# =============================================================================
# Acceleration Calculations
# =============================================================================

def roll_acceleration(roll_angle_deg, roll_period, R_arm=5.0):
    """Transverse acceleration from ship roll.

    a_trans = g*sin(phi) + R*(2*pi/T)^2 * sin(phi)

    The first term is the gravity component; the second is centripetal
    acceleration at distance R from the roll axis.

    Args:
        roll_angle_deg: Maximum roll angle (degrees)
        roll_period: Roll period (s)
        R_arm: Distance from roll axis to reactor CG (m)

    Returns:
        float: Transverse acceleration in m/s2
    """
    phi = math.radians(roll_angle_deg)
    omega = 2.0 * math.pi / roll_period

    a_gravity = G * math.sin(phi)
    a_centripetal = R_arm * omega**2 * math.sin(phi)

    return a_gravity + a_centripetal


def pitch_acceleration(pitch_angle_deg, pitch_period=None):
    """Longitudinal acceleration from ship pitch.

    a_long = g * sin(theta)

    Args:
        pitch_angle_deg: Maximum pitch angle (degrees)
        pitch_period: Pitch period (s), not used for quasi-static

    Returns:
        float: Longitudinal acceleration in m/s2
    """
    theta = math.radians(pitch_angle_deg)
    return G * math.sin(theta)


def heave_acceleration_value(heave_g):
    """Vertical acceleration from heave motion.

    Args:
        heave_g: Heave acceleration as fraction of g

    Returns:
        float: Vertical acceleration in m/s2 (added to gravity)
    """
    return heave_g * G


def combined_acceleration(a_trans, a_long, a_vert):
    """SRSS combination of acceleration components.

    a_combined = sqrt(a_trans^2 + a_long^2 + a_vert^2)

    Args:
        a_trans: Transverse acceleration (m/s2)
        a_long: Longitudinal acceleration (m/s2)
        a_vert: Vertical acceleration (m/s2)

    Returns:
        float: Combined acceleration in m/s2
    """
    return math.sqrt(a_trans**2 + a_long**2 + a_vert**2)


# =============================================================================
# Free Surface Correction
# =============================================================================

def free_surface_correction(rho_salt, R_core, total_mass):
    """Free surface correction for liquid salt sloshing.

    GG' = rho_salt * I_fs / m_total
    where I_fs = pi * R^4 / 4 (second moment of circular free surface)

    This reduces the effective metacentric height of the ship, potentially
    affecting stability. For a reactor with a sealed vessel, this is
    conservative (the free surface is constrained).

    Args:
        rho_salt: Salt density (kg/m3)
        R_core: Core radius (m) - approximation for free surface radius
        total_mass: Total ship displacement (kg)

    Returns:
        tuple: (GG_prime in m, free_surface_moment in N-m)
    """
    I_fs = math.pi * R_core**4 / 4.0
    GG_prime = rho_salt * I_fs / total_mass

    # Free surface moment
    fs_moment = rho_salt * G * I_fs

    return GG_prime, fs_moment


# =============================================================================
# Support Bolt Sizing
# =============================================================================

def size_support_bolts(F_lateral, F_vertical, overturning_moment,
                       n_bolts=12, bolt_yield=517e6, safety_factor=3.0):
    """Size foundation bolts for reactor support structure.

    Args:
        F_lateral: Maximum lateral force (N)
        F_vertical: Maximum vertical force - can be tension (N)
        overturning_moment: Maximum overturning moment at base (N-m)
        n_bolts: Number of foundation bolts
        bolt_yield: Bolt material yield stress (Pa). Default: SA-193 B7 at 500C
        safety_factor: Required safety factor

    Returns:
        tuple: (bolt_diameter_m, max_bolt_force_N, bolt_stress_Pa)
    """
    # Bolt circle radius (assume bolts on circle at vessel flange)
    # Approximate: R_bolt ~ vessel outer radius + 0.1 m
    R_bolt_circle = 0.8  # m (approximate)

    # Maximum bolt tension from overturning moment
    # Assuming bolts equally spaced on circle:
    # F_bolt_moment = M / (n_bolts/2 * R_bolt)  (simplified, conservative)
    F_bolt_moment = overturning_moment / (n_bolts / 2.0 * R_bolt_circle)

    # Shear per bolt from lateral force
    F_bolt_shear = F_lateral / n_bolts

    # Tension per bolt from vertical uplift
    F_bolt_tension = max(0, F_vertical / n_bolts)

    # Combined bolt force (tension + moment contribution)
    F_bolt_max = F_bolt_moment + F_bolt_tension

    # Combined with shear using interaction formula: (tau/0.6Sy)^2 + (sigma/Sy)^2 <= 1
    # For sizing, use the max tension force and find required area
    A_required = F_bolt_max * safety_factor / bolt_yield

    # Bolt diameter from required area (root area ~ 0.7 * gross area for standard threads)
    A_gross = A_required / 0.7
    d_bolt = math.sqrt(4.0 * A_gross / math.pi)

    # Actual stress
    A_actual = math.pi / 4.0 * d_bolt**2 * 0.7  # root area
    bolt_stress = F_bolt_max / A_actual if A_actual > 0 else 0

    return d_bolt, F_bolt_max, bolt_stress


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_ship_loads(design_params=None):
    """Perform ship motion and seismic load analysis.

    Args:
        design_params: DerivedParameters from config (computed if None)

    Returns:
        ShipMotionLoads dataclass
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    # --- Component masses ---
    # Vessel mass (from vessel_design, approximate here)
    rho_h = HASTELLOY_N['density']
    R_i = d.vessel_inner_radius
    t_wall = d.vessel_wall_thickness
    R_o = R_i + t_wall
    V_shell = math.pi * (R_o**2 - R_i**2) * d.vessel_height
    V_heads = (4.0 / 3.0) * math.pi * ((R_i + t_wall)**3 - R_i**3)
    vessel_mass = (V_shell + V_heads) * rho_h

    # Salt mass
    T_avg = config.CORE_AVG_TEMP
    rho_salt = flibe_density(T_avg, uf4_mol_fraction=0.05)
    salt_mass = d.fuel_salt_volume_core * rho_salt

    # Graphite mass
    graphite_mass = d.graphite_mass

    # Internals (control rods, supports, instrument thimbles, ~10% of vessel)
    internals_mass = 0.10 * vessel_mass

    total_mass = vessel_mass + salt_mass + graphite_mass + internals_mass
    weight = total_mass * G

    # --- Center of gravity ---
    # Approximate CG at geometric center of vessel
    cg_height = d.vessel_height / 2.0

    # Distance from roll axis to reactor CG
    # Assume reactor at ship centerline, roll axis at waterline
    # CG of reactor is ~8 m above waterline in engine room (estimated)
    R_arm_roll = 8.0  # m from roll axis to reactor CG

    # =================================================================
    # LOAD CASE 1: Sea state (roll + pitch + heave)
    # =================================================================
    a_trans_sea = roll_acceleration(ROLL_ANGLE_MAX, ROLL_PERIOD, R_arm_roll)
    a_long_sea = pitch_acceleration(PITCH_ANGLE_MAX)
    a_vert_sea = heave_acceleration_value(HEAVE_ACCELERATION)
    a_comb_sea = combined_acceleration(a_trans_sea, a_long_sea, a_vert_sea)

    # =================================================================
    # LOAD CASE 2: In-port seismic (0.2g horizontal, 0.13g vertical)
    # =================================================================
    a_horiz_seismic = 0.2 * G
    a_vert_seismic = 0.13 * G  # 2/3 of horizontal
    a_comb_seismic = combined_acceleration(a_horiz_seismic, 0, a_vert_seismic)

    # =================================================================
    # LOAD CASE 3: Collision (1g longitudinal deceleration)
    # =================================================================
    a_long_collision = 1.0 * G
    a_comb_collision = combined_acceleration(0, a_long_collision, 0)

    # =================================================================
    # Force envelopes (take max across load cases)
    # =================================================================
    F_lateral_max = total_mass * max(a_trans_sea, a_horiz_seismic)
    F_vertical_max = total_mass * (G + max(a_vert_sea, a_vert_seismic))
    F_longitudinal_max = total_mass * max(a_long_sea, a_long_collision)

    # Overturning moment at base (lateral force x CG height)
    M_overturn_sea = total_mass * a_trans_sea * cg_height
    M_overturn_seismic = total_mass * a_horiz_seismic * cg_height
    M_overturn_collision = total_mass * a_long_collision * cg_height
    M_overturn_max = max(M_overturn_sea, M_overturn_seismic, M_overturn_collision)

    # =================================================================
    # Free surface correction
    # =================================================================
    ship_total_mass = SHIP_DISPLACEMENT * 1000.0  # tonnes -> kg
    GG_prime, fs_moment = free_surface_correction(rho_salt, d.core_radius,
                                                   ship_total_mass)

    # =================================================================
    # Support bolt sizing
    # =================================================================
    n_bolts = 12
    d_bolt, F_bolt_max, bolt_stress = size_support_bolts(
        F_lateral_max, F_vertical_max, M_overturn_max, n_bolts
    )

    bolt_yield = 517e6  # Pa
    bolt_sf = bolt_yield / bolt_stress if bolt_stress > 0 else float('inf')

    # Design envelope in g's
    design_lat_g = max(a_trans_sea, a_horiz_seismic) / G
    design_vert_g = (G + max(a_vert_sea, a_vert_seismic)) / G
    design_long_g = max(a_long_sea, a_long_collision) / G

    return ShipMotionLoads(
        vessel_mass=vessel_mass,
        salt_mass=salt_mass,
        graphite_mass=graphite_mass,
        internals_mass=internals_mass,
        total_mass=total_mass,
        weight=weight,
        cg_height=cg_height,
        a_transverse_sea=a_trans_sea,
        a_longitudinal_sea=a_long_sea,
        a_vertical_sea=a_vert_sea,
        a_combined_sea=a_comb_sea,
        a_horizontal_seismic=a_horiz_seismic,
        a_vertical_seismic=a_vert_seismic,
        a_combined_seismic=a_comb_seismic,
        a_longitudinal_collision=a_long_collision,
        a_combined_collision=a_comb_collision,
        F_lateral_max=F_lateral_max,
        F_vertical_max=F_vertical_max,
        F_longitudinal_max=F_longitudinal_max,
        overturning_moment_max=M_overturn_max,
        GG_prime=GG_prime,
        free_surface_moment=fs_moment,
        n_bolts=n_bolts,
        bolt_diameter=d_bolt,
        bolt_force_max=F_bolt_max,
        bolt_stress=bolt_stress,
        bolt_safety_factor=bolt_sf,
        design_lateral_g=design_lat_g,
        design_vertical_g=design_vert_g,
        design_longitudinal_g=design_long_g,
    )


# =============================================================================
# Printing
# =============================================================================

def print_ship_loads(s):
    """Print formatted ship motion and seismic load results.

    Args:
        s: ShipMotionLoads dataclass instance
    """
    print("=" * 72)
    print("   SHIP MOTION & SEISMIC LOADS ANALYSIS")
    print("=" * 72)

    print("\n--- Component Masses ---")
    print(f"  Vessel (shell + heads):     {s.vessel_mass:10.1f} kg")
    print(f"  Fuel salt (core):           {s.salt_mass:10.1f} kg")
    print(f"  Graphite moderator:         {s.graphite_mass:10.1f} kg")
    print(f"  Internals (est.):           {s.internals_mass:10.1f} kg")
    print(f"  Total reactor assembly:     {s.total_mass:10.1f} kg ({s.total_mass / 1e3:.2f} tonnes)")
    print(f"  Total weight:               {s.weight / 1e3:10.1f} kN")
    print(f"  CG height above base:       {s.cg_height:10.3f} m")

    print("\n--- Load Case 1: Sea State (Roll + Pitch + Heave) ---")
    print(f"  Roll angle:                 {ROLL_ANGLE_MAX:10.1f} deg")
    print(f"  Roll period:                {ROLL_PERIOD:10.1f} s")
    print(f"  Pitch angle:                {PITCH_ANGLE_MAX:10.1f} deg")
    print(f"  Heave:                      {HEAVE_ACCELERATION:10.2f} g")
    print(f"  Transverse acceleration:    {s.a_transverse_sea:10.2f} m/s2 ({s.a_transverse_sea / G:.3f} g)")
    print(f"  Longitudinal acceleration:  {s.a_longitudinal_sea:10.2f} m/s2 ({s.a_longitudinal_sea / G:.3f} g)")
    print(f"  Vertical acceleration:      {s.a_vertical_sea:10.2f} m/s2 ({s.a_vertical_sea / G:.3f} g)")
    print(f"  Combined (SRSS):            {s.a_combined_sea:10.2f} m/s2 ({s.a_combined_sea / G:.3f} g)")

    print("\n--- Load Case 2: In-Port Seismic ---")
    print(f"  Horizontal acceleration:    {s.a_horizontal_seismic:10.2f} m/s2 ({s.a_horizontal_seismic / G:.3f} g)")
    print(f"  Vertical acceleration:      {s.a_vertical_seismic:10.2f} m/s2 ({s.a_vertical_seismic / G:.3f} g)")
    print(f"  Combined:                   {s.a_combined_seismic:10.2f} m/s2 ({s.a_combined_seismic / G:.3f} g)")

    print("\n--- Load Case 3: Collision ---")
    print(f"  Longitudinal deceleration:  {s.a_longitudinal_collision:10.2f} m/s2 ({s.a_longitudinal_collision / G:.3f} g)")
    print(f"  Combined:                   {s.a_combined_collision:10.2f} m/s2 ({s.a_combined_collision / G:.3f} g)")

    print("\n--- Design Load Envelope ---")
    print(f"  Lateral (max):              {s.design_lateral_g:10.3f} g")
    print(f"  Vertical (max incl. 1g):    {s.design_vertical_g:10.3f} g")
    print(f"  Longitudinal (max):         {s.design_longitudinal_g:10.3f} g")

    print("\n--- Forces on Supports ---")
    print(f"  Max lateral force:          {s.F_lateral_max / 1e3:10.1f} kN")
    print(f"  Max vertical force:         {s.F_vertical_max / 1e3:10.1f} kN")
    print(f"  Max longitudinal force:     {s.F_longitudinal_max / 1e3:10.1f} kN")
    print(f"  Max overturning moment:     {s.overturning_moment_max / 1e3:10.1f} kN-m")

    print("\n--- Free Surface Correction ---")
    print(f"  Metacentric correction GG': {s.GG_prime * 1e3:10.3f} mm")
    print(f"  Free surface moment:        {s.free_surface_moment:10.1f} N-m")
    status = "NEGLIGIBLE" if s.GG_prime < 0.01 else "SIGNIFICANT"
    print(f"  Assessment:                 {status}")

    print("\n--- Foundation Bolt Sizing ---")
    print(f"  Number of bolts:            {s.n_bolts:10d}")
    print(f"  Min bolt diameter:          {s.bolt_diameter * 1e3:10.1f} mm")
    print(f"  Max bolt force:             {s.bolt_force_max / 1e3:10.2f} kN")
    print(f"  Bolt stress:                {s.bolt_stress / 1e6:10.1f} MPa")
    print(f"  Safety factor:              {s.bolt_safety_factor:10.2f}"
          f"  {'OK' if s.bolt_safety_factor > 2.0 else 'INSUFFICIENT'}")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()
    loads = analyze_ship_loads(d)
    print_ship_loads(loads)

    # Summary comparison
    print("\n--- Load Case Comparison ---")
    print(f"  {'Load Case':<25s}  {'a_combined [g]':>14s}  {'Governing':>10s}")
    print(f"  {'-' * 25}  {'-' * 14}  {'-' * 10}")

    cases = [
        ("Sea state (roll+heave)", loads.a_combined_sea / G),
        ("In-port seismic", loads.a_combined_seismic / G),
        ("Collision (1g)", loads.a_combined_collision / G),
    ]
    max_case = max(cases, key=lambda x: x[1])
    for name, val in cases:
        gov = "  <---" if val == max_case[1] else ""
        print(f"  {name:<25s}  {val:14.3f}{gov}")
