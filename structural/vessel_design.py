"""
Reactor Vessel Design per ASME Section III, Division 5
=======================================================

Wall thickness sizing for the 40 MWth marine MSR reactor vessel.

Material: Hastelloy-N (UNS N10003)
Design code: ASME BPVC Section III, Division 5 (High Temperature Reactors)

Components:
  - Cylindrical shell
  - Hemispherical heads (upper and lower)
  - Nozzle reinforcement (simplified area replacement)
  - Total vessel weight estimate

References:
  - ASME BPVC Section III, Division 5
  - ORNL-TM-5920, "Design Considerations for Reactor Vessels" (Haubenreich, 1977)
  - ORNL-4541, Appendix C (vessel design for MSBR)
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
    OPERATING_PRESSURE, HASTELLOY_N, DESIGN_LIFE_YEARS,
    CORE_OUTLET_TEMP, compute_derived,
)


# =============================================================================
# Design Constants
# =============================================================================

JOINT_EFFICIENCY = 0.85         # Welded joint efficiency (Type I, spot examined)
CORROSION_RATE = HASTELLOY_N['corrosion_rate']  # 25 um/yr in FLiBe
DESIGN_PRESSURE = 1.5 * OPERATING_PRESSURE      # Pa (1.5x safety factor)
DESIGN_TEMP = CORE_OUTLET_TEMP + 25.0           # K (hot spot allowance)

# Nozzle parameters
NOZZLE_SIZES = {
    'primary_outlet': 0.30,     # m, ID (hot leg)
    'primary_inlet': 0.30,      # m, ID (cold leg)
    'drain': 0.15,              # m, ID (emergency drain)
    'instrumentation': 0.05,    # m, ID (instrument ports, x4)
}


# =============================================================================
# VesselDesign Dataclass
# =============================================================================

@dataclass
class VesselDesign:
    """Complete reactor vessel design output."""

    # --- Shell ---
    shell_inner_radius: float       # m
    shell_thickness_pressure: float # m, pressure-only thickness
    shell_thickness_corrosion: float  # m, corrosion allowance
    shell_thickness_total: float    # m, total shell wall thickness
    shell_outer_radius: float       # m

    # --- Heads ---
    head_type: str                  # 'hemispherical'
    head_thickness_pressure: float  # m, pressure-only thickness
    head_thickness_total: float     # m, total head thickness
    head_inner_radius: float        # m (same as shell inner radius)

    # --- Nozzles ---
    nozzle_reinforcement: dict      # {name: {'ID': m, 'thickness': m, 'reinf_area': m2}}

    # --- Vessel Dimensions ---
    vessel_height: float            # m, overall height (shell + heads)
    shell_height: float             # m, cylindrical section only
    head_height: float              # m, single head height

    # --- Design Basis ---
    design_pressure: float          # Pa
    design_temperature: float       # K
    allowable_stress: float         # Pa
    joint_efficiency: float         # dimensionless
    corrosion_allowance: float      # m

    # --- Weight ---
    shell_weight: float             # kg
    head_weight: float              # kg (both heads)
    nozzle_weight: float            # kg (all nozzles, approximate)
    total_weight: float             # kg (dry, empty)

    # --- Safety Margins ---
    pressure_safety_factor: float   # actual vs required thickness ratio
    max_hoop_stress: float          # Pa, at inner wall
    allowable_stress_ratio: float   # hoop_stress / allowable_stress


# =============================================================================
# Thickness Calculations
# =============================================================================

def cylindrical_shell_thickness(P, R, S, E):
    """Minimum wall thickness for internal pressure in cylindrical shell.

    ASME formula: t = P*R / (S*E - 0.6*P)

    Args:
        P: Design pressure (Pa)
        R: Inner radius (m)
        S: Allowable stress (Pa)
        E: Joint efficiency

    Returns:
        float: Required thickness in m
    """
    return P * R / (S * E - 0.6 * P)


def hemispherical_head_thickness(P, R, S, E):
    """Minimum wall thickness for hemispherical head under internal pressure.

    ASME formula: t = P*R / (2*S*E - 0.2*P)

    Args:
        P: Design pressure (Pa)
        R: Inner radius (m)
        S: Allowable stress (Pa)
        E: Joint efficiency

    Returns:
        float: Required thickness in m
    """
    return P * R / (2.0 * S * E - 0.2 * P)


def nozzle_reinforcement_area(d_nozzle, t_vessel, t_nozzle, t_required):
    """Calculate required reinforcement area for nozzle opening.

    Area replacement method (ASME Section VIII / Section III):
    The metal removed by the opening must be compensated by excess
    metal in the nozzle wall and surrounding shell.

    Args:
        d_nozzle: Nozzle opening diameter (m)
        t_vessel: Actual vessel wall thickness (m)
        t_nozzle: Nozzle wall thickness (m)
        t_required: Required vessel thickness without opening (m)

    Returns:
        tuple: (area_removed, area_available, area_reinforcement_needed) in m2
    """
    # Area removed by the opening
    A_removed = d_nozzle * t_required

    # Area available from excess vessel wall (within 1 diameter of opening)
    A_excess_vessel = (t_vessel - t_required) * d_nozzle

    # Area available from nozzle wall projecting inward and outward
    # Assume nozzle projects 2.5 * t_nozzle each direction
    h_nozzle = 2.5 * t_nozzle
    A_nozzle = 2 * h_nozzle * (t_nozzle - t_required * d_nozzle / (d_nozzle + 2 * t_nozzle))
    A_nozzle = max(A_nozzle, 0.0)

    A_available = A_excess_vessel + A_nozzle
    A_reinforcement = max(0.0, A_removed - A_available)

    return A_removed, A_available, A_reinforcement


# =============================================================================
# Main Design Function
# =============================================================================

def design_vessel(design_params=None):
    """Design the reactor pressure vessel.

    Args:
        design_params: DerivedParameters from config (computed if None)

    Returns:
        VesselDesign dataclass
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    # --- Design basis ---
    P = DESIGN_PRESSURE
    S = HASTELLOY_N['allowable_stress_design']  # Pa (55 MPa)
    E = JOINT_EFFICIENCY
    R_inner = d.vessel_inner_radius  # m

    # --- Corrosion allowance ---
    corrosion_allowance = CORROSION_RATE * DESIGN_LIFE_YEARS  # m
    # 25e-6 m/yr * 20 yr = 0.5 mm

    # --- Shell thickness ---
    t_shell_pressure = cylindrical_shell_thickness(P, R_inner, S, E)
    t_shell_total = t_shell_pressure + corrosion_allowance

    # Minimum practical thickness (manufacturing & handling)
    t_shell_min = 0.015  # m (15 mm minimum)
    t_shell_total = max(t_shell_total, t_shell_min)

    R_outer = R_inner + t_shell_total

    # --- Head thickness ---
    t_head_pressure = hemispherical_head_thickness(P, R_inner, S, E)
    t_head_total = t_head_pressure + corrosion_allowance
    t_head_total = max(t_head_total, t_shell_min)

    # Head height (hemisphere inner)
    h_head = R_inner  # hemisphere height = radius

    # --- Shell height ---
    # Core + upper plenum + lower plenum
    shell_height = d.vessel_height  # from config (core_height + 1.0 m plena)

    # Total vessel height
    vessel_height = shell_height + 2 * h_head

    # --- Nozzle reinforcement ---
    nozzle_data = {}
    for name, d_nozzle in NOZZLE_SIZES.items():
        # Nozzle wall thickness: same schedule as vessel (simplified)
        t_nozzle = max(t_shell_total * 0.8, 0.010)  # 80% of vessel or 10 mm min

        A_removed, A_avail, A_reinf = nozzle_reinforcement_area(
            d_nozzle, t_shell_total, t_nozzle, t_shell_pressure + corrosion_allowance
        )

        nozzle_data[name] = {
            'ID': d_nozzle,
            'thickness': t_nozzle,
            'area_removed': A_removed,
            'area_available': A_avail,
            'reinforcement_needed': A_reinf,
        }

    # --- Weight estimates ---
    rho = HASTELLOY_N['density']

    # Shell weight (cylindrical, uniform thickness)
    V_shell = math.pi * ((R_outer)**2 - R_inner**2) * shell_height
    shell_weight = V_shell * rho

    # Head weight (two hemispheres = one sphere shell)
    R_head_outer = R_inner + t_head_total
    V_heads = (4.0 / 3.0) * math.pi * (R_head_outer**3 - R_inner**3)
    head_weight = V_heads * rho

    # Nozzle weight (approximate: short cylinders)
    nozzle_weight = 0.0
    for name, ndata in nozzle_data.items():
        d_n = ndata['ID']
        t_n = ndata['thickness']
        L_n = 0.3  # m, assumed nozzle length
        r_n_inner = d_n / 2.0
        r_n_outer = r_n_inner + t_n
        V_n = math.pi * (r_n_outer**2 - r_n_inner**2) * L_n
        n_count = 4 if name == 'instrumentation' else 1
        nozzle_weight += V_n * rho * n_count

    total_weight = shell_weight + head_weight + nozzle_weight

    # --- Stress analysis (simple) ---
    # Hoop stress at inner wall: sigma_h = P * (R_o^2 + R_i^2) / (R_o^2 - R_i^2)
    # (Lame equation for thick-walled cylinder)
    sigma_hoop = P * (R_outer**2 + R_inner**2) / (R_outer**2 - R_inner**2)
    stress_ratio = sigma_hoop / S
    safety_factor = t_shell_total / (t_shell_pressure + corrosion_allowance)

    return VesselDesign(
        shell_inner_radius=R_inner,
        shell_thickness_pressure=t_shell_pressure,
        shell_thickness_corrosion=corrosion_allowance,
        shell_thickness_total=t_shell_total,
        shell_outer_radius=R_outer,
        head_type='hemispherical',
        head_thickness_pressure=t_head_pressure,
        head_thickness_total=t_head_total,
        head_inner_radius=R_inner,
        nozzle_reinforcement=nozzle_data,
        vessel_height=vessel_height,
        shell_height=shell_height,
        head_height=h_head,
        design_pressure=P,
        design_temperature=DESIGN_TEMP,
        allowable_stress=S,
        joint_efficiency=E,
        corrosion_allowance=corrosion_allowance,
        shell_weight=shell_weight,
        head_weight=head_weight,
        nozzle_weight=nozzle_weight,
        total_weight=total_weight,
        pressure_safety_factor=safety_factor,
        max_hoop_stress=sigma_hoop,
        allowable_stress_ratio=stress_ratio,
    )


# =============================================================================
# Printing
# =============================================================================

def print_vessel_design(v):
    """Print formatted reactor vessel design summary.

    Args:
        v: VesselDesign dataclass instance
    """
    print("=" * 72)
    print("   REACTOR VESSEL DESIGN - ASME Section III, Division 5")
    print("=" * 72)

    print("\n--- Design Basis ---")
    print(f"  Design pressure:            {v.design_pressure / 1e6:10.3f} MPa")
    print(f"  Operating pressure:         {OPERATING_PRESSURE / 1e6:10.3f} MPa")
    print(f"  Design temperature:         {v.design_temperature - 273.15:10.1f} C")
    print(f"  Allowable stress:           {v.allowable_stress / 1e6:10.1f} MPa")
    print(f"  Joint efficiency:           {v.joint_efficiency:10.2f}")
    print(f"  Corrosion allowance:        {v.corrosion_allowance * 1e3:10.2f} mm")
    print(f"  Design life:                {DESIGN_LIFE_YEARS:10d} years")

    print("\n--- Cylindrical Shell ---")
    print(f"  Inner radius:               {v.shell_inner_radius:10.3f} m")
    print(f"  Inner diameter:             {2 * v.shell_inner_radius:10.3f} m")
    print(f"  Thickness (pressure):       {v.shell_thickness_pressure * 1e3:10.2f} mm")
    print(f"  Thickness (corrosion):      {v.shell_thickness_corrosion * 1e3:10.2f} mm")
    print(f"  Thickness (total):          {v.shell_thickness_total * 1e3:10.2f} mm")
    print(f"  Outer radius:               {v.shell_outer_radius:10.3f} m")
    print(f"  Shell height:               {v.shell_height:10.3f} m")

    print("\n--- Hemispherical Heads ---")
    print(f"  Thickness (pressure):       {v.head_thickness_pressure * 1e3:10.2f} mm")
    print(f"  Thickness (total):          {v.head_thickness_total * 1e3:10.2f} mm")
    print(f"  Head height (each):         {v.head_height:10.3f} m")

    print("\n--- Nozzle Reinforcement ---")
    for name, ndata in v.nozzle_reinforcement.items():
        print(f"  {name}:")
        print(f"    ID:                       {ndata['ID'] * 1e3:10.1f} mm")
        print(f"    Wall thickness:           {ndata['thickness'] * 1e3:10.1f} mm")
        print(f"    Area removed:             {ndata['area_removed'] * 1e4:10.2f} cm2")
        print(f"    Area available:           {ndata['area_available'] * 1e4:10.2f} cm2")
        reinforcement_status = "OK" if ndata['reinforcement_needed'] <= 0 else "NEEDS PAD"
        print(f"    Reinforcement:            {reinforcement_status}")

    print("\n--- Overall Dimensions ---")
    print(f"  Total vessel height:        {v.vessel_height:10.3f} m")
    print(f"  Outer diameter:             {2 * v.shell_outer_radius:10.3f} m")

    print("\n--- Weight ---")
    print(f"  Shell:                      {v.shell_weight:10.1f} kg")
    print(f"  Heads (both):               {v.head_weight:10.1f} kg")
    print(f"  Nozzles:                    {v.nozzle_weight:10.1f} kg")
    print(f"  Total (dry, empty):         {v.total_weight:10.1f} kg ({v.total_weight / 1e3:.2f} tonnes)")

    print("\n--- Stress Check ---")
    print(f"  Max hoop stress:            {v.max_hoop_stress / 1e6:10.3f} MPa")
    print(f"  Allowable stress:           {v.allowable_stress / 1e6:10.1f} MPa")
    print(f"  Stress ratio:               {v.allowable_stress_ratio:10.4f} (< 1.0 OK)")
    print(f"  Safety factor:              {v.pressure_safety_factor:10.3f}")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()
    v = design_vessel(d)
    print_vessel_design(v)
