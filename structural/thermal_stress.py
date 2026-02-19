"""
Thermal Stress Analysis for Thick-Walled Cylindrical Vessel
=============================================================

Calculates thermal stresses in the reactor vessel wall from the radial
temperature gradient between inner (salt-wetted) and outer (insulated/cooled)
surfaces.

Method: Goodier's solution for steady-state thermal stresses in a
thick-walled elastic cylinder with radial temperature gradient.

Stress components:
  - Tangential (hoop) stress: sigma_theta
  - Axial stress: sigma_z
  - Radial stress: sigma_r
  - Combined stress intensity per ASME: S_I = max(|s1-s2|, |s2-s3|, |s3-s1|)

Limits:
  - Primary + secondary stress: < 3*S_m
  - Creep-rupture: < 83 MPa at 700 C / 10,000 hr (Hastelloy-N)

References:
  - Timoshenko & Goodier, "Theory of Elasticity"
  - ASME BPVC Section III, Div. 5, HBB-T-1300 (Elastic Analysis)
  - ORNL-TM-5920
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
    HASTELLOY_N, CORE_OUTLET_TEMP, CORE_INLET_TEMP,
    compute_derived,
)


# =============================================================================
# Material Properties
# =============================================================================

ALPHA = HASTELLOY_N['thermal_expansion']     # 1/K (12.3e-6)
E_MOD = HASTELLOY_N['elastic_modulus']       # Pa (219 GPa)
NU = HASTELLOY_N['poisson_ratio']            # 0.32
CREEP_RUPTURE_LIMIT = HASTELLOY_N['creep_rupture_700C_10khr']  # 83 MPa


# =============================================================================
# ThermalStressResult Dataclass
# =============================================================================

@dataclass
class ThermalStressResult:
    """Results of thermal stress analysis through the vessel wall."""

    # --- Geometry ---
    r_inner: float              # m, inner radius
    r_outer: float              # m, outer radius
    wall_thickness: float       # m

    # --- Temperature ---
    T_inner: float              # K, inner wall temperature
    T_outer: float              # K, outer wall temperature
    delta_T: float              # K, temperature difference across wall

    # --- Radial profiles ---
    r: np.ndarray               # m, radial positions
    T_profile: np.ndarray       # K, temperature at each r
    sigma_theta: np.ndarray     # Pa, tangential (hoop) stress
    sigma_z: np.ndarray         # Pa, axial stress
    sigma_r: np.ndarray         # Pa, radial stress
    stress_intensity: np.ndarray  # Pa, ASME stress intensity

    # --- Peak values ---
    max_sigma_theta: float      # Pa
    max_sigma_z: float          # Pa
    max_sigma_r: float          # Pa
    max_stress_intensity: float # Pa
    max_SI_location: str        # 'inner' or 'outer'

    # --- Safety margins ---
    creep_rupture_limit: float  # Pa (83 MPa)
    creep_safety_factor: float  # creep_limit / max_stress_intensity
    asme_3Sm_limit: float       # Pa (3 * allowable stress)
    asme_safety_factor: float   # 3Sm / max_stress_intensity


# =============================================================================
# Temperature Distribution
# =============================================================================

def temperature_profile(r, r_i, r_o, T_i, T_o):
    """Steady-state temperature distribution in cylindrical wall.

    T(r) = T_i + (T_o - T_i) * ln(r/r_i) / ln(r_o/r_i)

    Args:
        r: Radial position(s) (m), scalar or array
        r_i: Inner radius (m)
        r_o: Outer radius (m)
        T_i: Inner wall temperature (K)
        T_o: Outer wall temperature (K)

    Returns:
        Temperature(s) at r in K
    """
    return T_i + (T_o - T_i) * np.log(r / r_i) / np.log(r_o / r_i)


# =============================================================================
# Thermal Stress Components (Goodier)
# =============================================================================

def _thermal_stress_integrals(r, r_i, r_o, T_inner, T_outer):
    """Compute temperature integrals needed for thermal stress.

    For T(r) = T_i + (T_o - T_i) * ln(r/a) / ln(b/a):

    Integral_a_to_r(T * r' * dr') and Integral_a_to_b(T * r' * dr')

    These are used in the Boley & Weiner / Timoshenko stress formulas.

    Args:
        r: Radial position(s) (m)
        r_i, r_o: Inner and outer radii
        T_inner, T_outer: Wall temperatures (K)

    Returns:
        tuple: (integral_to_r, integral_to_b) evaluated analytically
    """
    a, b = r_i, r_o
    DT = T_outer - T_inner
    ln_ba = np.log(b / a)

    # integral_a_to_r of T(r') * r' dr'
    # = integral of [T_i + DT*ln(r'/a)/ln(b/a)] * r' dr'
    # = T_i * (r^2 - a^2)/2 + DT/ln(b/a) * integral_a_to_r[ r' * ln(r'/a) dr' ]
    # where integral r'*ln(r'/a) dr' = r'^2/2 * ln(r'/a) - (r'^2 - a^2)/4
    #   evaluated from a to r:
    #   = r^2/2 * ln(r/a) - (r^2 - a^2)/4

    int_to_r = (T_inner * (r**2 - a**2) / 2.0
                + DT / ln_ba * (r**2 / 2.0 * np.log(r / a) - (r**2 - a**2) / 4.0))

    int_to_b = (T_inner * (b**2 - a**2) / 2.0
                + DT / ln_ba * (b**2 / 2.0 * np.log(b / a) - (b**2 - a**2) / 4.0))
    # Simplifies: DT/ln_ba * (b^2/2 * ln_ba - (b^2-a^2)/4)
    #           = DT * b^2/2 - DT*(b^2-a^2)/(4*ln_ba)
    int_to_b = (T_inner * (b**2 - a**2) / 2.0
                + DT * b**2 / 2.0 - DT * (b**2 - a**2) / (4.0 * ln_ba))

    return int_to_r, int_to_b


def thermal_stress_theta(r, r_i, r_o, delta_T, alpha, E, nu):
    """Tangential (hoop) thermal stress in thick-walled cylinder.

    Uses Boley & Weiner formulation (equivalent to Timoshenko):

    sigma_theta = alpha*E/(1-nu) * { 1/(b^2-a^2) * [Int_a^b + (a^2/r^2+1)*Int_a^r/(?)...]  }
    ... simplified to the direct formula:

    sigma_theta = alpha*E/(1-nu) * {
        [a^2 + r^2] / [r^2 * (b^2-a^2)] * Int_a^b
        - [1/r^2] * Int_a^r
        - T(r)  }  +  alpha*E*T_mean/(1-nu)
    ... this is getting complicated. Use the simplest verified form.

    For a thick-walled cylinder with temperature T(r), free inner and outer
    surfaces, and free ends (generalized plane strain), from Boresi & Schmidt:

    sigma_r     = alpha*E/(1-nu) * [ 1/(b^2-a^2) * I_ab - 1/r^2 * I_ar ]
    sigma_theta = alpha*E/(1-nu) * [ 1/(b^2-a^2) * I_ab + 1/r^2 * I_ar - T(r) + T_mean ]
    sigma_z     = alpha*E/(1-nu) * [ 2/(b^2-a^2) * I_ab - T(r) + T_mean ]

    where I_ar = integral from a to r of T(r')*r' dr' / (r^2 - comes from integration)
    ... No, the standard form uses:

    I_r = (2/(r^2 - a^2)) * integral_a^r T(r') r' dr'   ... nope.

    Let me just use the VERIFIED simple form from Ugural & Fenster,
    "Advanced Mechanics of Materials and Applied Elasticity":

    sigma_r = alpha*E/(1-nu) * {
        (1-a^2/r^2)/(b^2-a^2) * integral_a^b T(r')r'dr'
        - 1/r^2 * integral_a^r T(r')r'dr'  }

    sigma_theta = alpha*E/(1-nu) * {
        (1+a^2/r^2)/(b^2-a^2) * integral_a^b T(r')r'dr'
        + 1/r^2 * integral_a^r T(r')r'dr'
        - T(r)  }

    sigma_z = alpha*E/(1-nu) * {
        2/(b^2-a^2) * integral_a^b T(r')r'dr'
        - T(r)  }    (for free ends: add constant so net Fz = 0)

    But these formulas need T measured from a stress-free reference.
    If the cylinder is stress-free at some reference temperature T_ref,
    then replace T(r) with [T(r) - T_ref].

    For thermal stress from the temperature GRADIENT only (not absolute T),
    we can set T_ref = T_mean (mean temperature across the wall).

    Args:
        r: Radial position(s) (m)
        r_i, r_o: Inner and outer radii (m)
        delta_T: T_outer - T_inner (K). Negative if inner is hotter.
        alpha: Thermal expansion coefficient (1/K)
        E: Elastic modulus (Pa)
        nu: Poisson's ratio

    Returns:
        Tangential stress in Pa
    """
    a, b = r_i, r_o
    ln_ba = np.log(b / a)
    T_inner = 0.0  # Use relative temperatures (stress-free at T=0)
    T_outer = delta_T
    K = alpha * E / (1.0 - nu)

    # Temperature at r (relative)
    T_r = delta_T * np.log(r / a) / ln_ba

    # Integrals
    int_to_r = (delta_T / ln_ba * (r**2 / 2.0 * np.log(r / a) - (r**2 - a**2) / 4.0))
    int_to_b = (delta_T / ln_ba * (b**2 / 2.0 * ln_ba - (b**2 - a**2) / 4.0))
    # Simplify int_to_b: = delta_T * b^2/2 - delta_T*(b^2-a^2)/(4*ln_ba)
    int_to_b = delta_T * b**2 / 2.0 - delta_T * (b**2 - a**2) / (4.0 * ln_ba)

    sigma = K * (
        (1.0 + a**2 / r**2) / (b**2 - a**2) * int_to_b
        + 1.0 / r**2 * int_to_r
        - T_r
    )

    return sigma


def thermal_stress_radial(r, r_i, r_o, delta_T, alpha, E, nu):
    """Radial thermal stress in thick-walled cylinder.

    sigma_r = alpha*E/(1-nu) * {
        (1-a^2/r^2)/(b^2-a^2) * Int_ab - 1/r^2 * Int_ar }

    Verified: sigma_r = 0 at r=a and r=b (traction-free surfaces).

    Args:
        r: Radial position(s) (m)
        r_i, r_o: Inner and outer radii (m)
        delta_T: T_outer - T_inner (K)
        alpha, E, nu: Material properties

    Returns:
        Radial stress in Pa
    """
    a, b = r_i, r_o
    ln_ba = np.log(b / a)
    K = alpha * E / (1.0 - nu)

    int_to_r = (delta_T / ln_ba * (r**2 / 2.0 * np.log(r / a) - (r**2 - a**2) / 4.0))
    int_to_b = delta_T * b**2 / 2.0 - delta_T * (b**2 - a**2) / (4.0 * ln_ba)

    sigma = K * (
        (1.0 - a**2 / r**2) / (b**2 - a**2) * int_to_b
        - 1.0 / r**2 * int_to_r
    )

    return sigma


def thermal_stress_axial(r, r_i, r_o, delta_T, alpha, E, nu):
    """Axial thermal stress in thick-walled cylinder (free ends).

    sigma_z = alpha*E/(1-nu) * {
        2/(b^2-a^2) * Int_ab - T(r) }

    For free ends, a constant is added so net axial force = 0.
    This is automatically satisfied by the integral formulation.

    Args:
        r: Radial position(s) (m)
        r_i, r_o: Inner and outer radii (m)
        delta_T: T_outer - T_inner (K)
        alpha, E, nu: Material properties

    Returns:
        Axial stress in Pa
    """
    a, b = r_i, r_o
    ln_ba = np.log(b / a)
    K = alpha * E / (1.0 - nu)

    T_r = delta_T * np.log(r / a) / ln_ba
    int_to_b = delta_T * b**2 / 2.0 - delta_T * (b**2 - a**2) / (4.0 * ln_ba)

    sigma = K * (
        2.0 / (b**2 - a**2) * int_to_b
        - T_r
    )

    return sigma


# =============================================================================
# ASME Stress Intensity
# =============================================================================

def stress_intensity(sigma_r, sigma_theta, sigma_z):
    """ASME stress intensity (Tresca criterion).

    S_I = max(|sigma_1 - sigma_2|, |sigma_2 - sigma_3|, |sigma_3 - sigma_1|)

    For the principal stresses in cylindrical coordinates (sigma_r, sigma_theta, sigma_z).

    Args:
        sigma_r: Radial stress (Pa), scalar or array
        sigma_theta: Tangential stress (Pa)
        sigma_z: Axial stress (Pa)

    Returns:
        Stress intensity in Pa
    """
    d1 = np.abs(sigma_theta - sigma_r)
    d2 = np.abs(sigma_r - sigma_z)
    d3 = np.abs(sigma_z - sigma_theta)

    return np.maximum(np.maximum(d1, d2), d3)


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_thermal_stress(r_inner=None, r_outer=None, T_inner=None, T_outer=None,
                           design_params=None, n_points=50):
    """Perform thermal stress analysis through the vessel wall.

    Args:
        r_inner: Inner radius (m). Uses vessel design if None.
        r_outer: Outer radius (m). Uses vessel design if None.
        T_inner: Inner wall temperature (K). Default: core outlet temp.
        T_outer: Outer wall temperature (K). Default: T_inner - 50 K.
        design_params: DerivedParameters (computed if None)
        n_points: Number of radial points for profile

    Returns:
        ThermalStressResult dataclass
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    if r_inner is None:
        r_inner = d.vessel_inner_radius
    if r_outer is None:
        r_outer = d.vessel_outer_radius
    if T_inner is None:
        T_inner = CORE_OUTLET_TEMP  # Inner wall near hot salt
    if T_outer is None:
        # Outer wall is cooler (insulated, some heat loss to air gap)
        T_outer = T_inner - 50.0  # K, rough estimate of dT through wall

    delta_T = T_outer - T_inner  # negative if inner is hotter (compressive at inner)
    wall_thickness = r_outer - r_inner

    # Radial positions
    r = np.linspace(r_inner, r_outer, n_points)

    # Temperature profile
    T_prof = temperature_profile(r, r_inner, r_outer, T_inner, T_outer)

    # Stress components
    sig_theta = thermal_stress_theta(r, r_inner, r_outer, delta_T, ALPHA, E_MOD, NU)
    sig_r = thermal_stress_radial(r, r_inner, r_outer, delta_T, ALPHA, E_MOD, NU)
    sig_z = thermal_stress_axial(r, r_inner, r_outer, delta_T, ALPHA, E_MOD, NU)

    # Stress intensity
    SI = stress_intensity(sig_r, sig_theta, sig_z)

    # Peak values
    max_sigma_theta = float(np.max(np.abs(sig_theta)))
    max_sigma_r = float(np.max(np.abs(sig_r)))
    max_sigma_z = float(np.max(np.abs(sig_z)))
    max_SI = float(np.max(SI))

    # Location of max stress intensity
    idx_max = int(np.argmax(SI))
    if r[idx_max] < (r_inner + r_outer) / 2.0:
        max_loc = 'inner'
    else:
        max_loc = 'outer'

    # Safety margins
    Sm = HASTELLOY_N['allowable_stress_design']
    three_Sm = 3.0 * Sm
    creep_sf = CREEP_RUPTURE_LIMIT / max_SI if max_SI > 0 else float('inf')
    asme_sf = three_Sm / max_SI if max_SI > 0 else float('inf')

    return ThermalStressResult(
        r_inner=r_inner,
        r_outer=r_outer,
        wall_thickness=wall_thickness,
        T_inner=T_inner,
        T_outer=T_outer,
        delta_T=delta_T,
        r=r,
        T_profile=T_prof,
        sigma_theta=sig_theta,
        sigma_z=sig_z,
        sigma_r=sig_r,
        stress_intensity=SI,
        max_sigma_theta=max_sigma_theta,
        max_sigma_z=max_sigma_z,
        max_sigma_r=max_sigma_r,
        max_stress_intensity=max_SI,
        max_SI_location=max_loc,
        creep_rupture_limit=CREEP_RUPTURE_LIMIT,
        creep_safety_factor=creep_sf,
        asme_3Sm_limit=three_Sm,
        asme_safety_factor=asme_sf,
    )


# =============================================================================
# Printing
# =============================================================================

def print_thermal_stress(result):
    """Print formatted thermal stress analysis results.

    Args:
        result: ThermalStressResult dataclass instance
    """
    r = result
    print("=" * 72)
    print("   THERMAL STRESS ANALYSIS - Thick-Walled Cylindrical Vessel")
    print("=" * 72)

    print("\n--- Geometry ---")
    print(f"  Inner radius:               {r.r_inner:10.3f} m")
    print(f"  Outer radius:               {r.r_outer:10.3f} m")
    print(f"  Wall thickness:             {r.wall_thickness * 1e3:10.2f} mm")

    print("\n--- Temperature ---")
    print(f"  Inner wall temp:            {r.T_inner - 273.15:10.1f} C ({r.T_inner:.1f} K)")
    print(f"  Outer wall temp:            {r.T_outer - 273.15:10.1f} C ({r.T_outer:.1f} K)")
    print(f"  Delta-T across wall:        {r.delta_T:10.1f} K")

    print("\n--- Material Properties (Hastelloy-N) ---")
    print(f"  Thermal expansion coeff:    {ALPHA * 1e6:10.2f} x10^-6 1/K")
    print(f"  Elastic modulus:            {E_MOD / 1e9:10.1f} GPa")
    print(f"  Poisson's ratio:            {NU:10.3f}")

    print("\n--- Peak Stresses ---")
    print(f"  Max |sigma_theta|:          {r.max_sigma_theta / 1e6:10.2f} MPa")
    print(f"  Max |sigma_z|:              {r.max_sigma_z / 1e6:10.2f} MPa")
    print(f"  Max |sigma_r|:              {r.max_sigma_r / 1e6:10.2f} MPa")
    print(f"  Max stress intensity (SI):  {r.max_stress_intensity / 1e6:10.2f} MPa")
    print(f"  Location of max SI:         {r.max_SI_location:>10s} wall")

    print("\n--- Safety Margins ---")
    print(f"  Creep-rupture limit:        {r.creep_rupture_limit / 1e6:10.1f} MPa (700 C, 10 khr)")
    print(f"  Creep safety factor:        {r.creep_safety_factor:10.2f}"
          f"  {'OK' if r.creep_safety_factor > 1.5 else 'WARNING'}")
    print(f"  ASME 3*Sm limit:            {r.asme_3Sm_limit / 1e6:10.1f} MPa")
    print(f"  ASME safety factor:         {r.asme_safety_factor:10.2f}"
          f"  {'OK' if r.asme_safety_factor > 1.0 else 'FAIL'}")

    # Print radial profile at selected points
    print("\n--- Radial Stress Profile ---")
    n = len(r.r)
    indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    print(f"  {'r [mm]':>8s}  {'T [C]':>8s}  {'sig_th [MPa]':>12s}  "
          f"{'sig_z [MPa]':>11s}  {'sig_r [MPa]':>11s}  {'SI [MPa]':>9s}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 12}  {'-' * 11}  {'-' * 11}  {'-' * 9}")
    for i in indices:
        print(f"  {r.r[i] * 1e3:8.2f}  {r.T_profile[i] - 273.15:8.1f}  "
              f"{r.sigma_theta[i] / 1e6:12.3f}  "
              f"{r.sigma_z[i] / 1e6:11.3f}  "
              f"{r.sigma_r[i] / 1e6:11.3f}  "
              f"{r.stress_intensity[i] / 1e6:9.3f}")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()

    print("CASE 1: Normal operating conditions")
    print("  Inner wall at core outlet temperature (700 C)")
    print("  Outer wall at 650 C (50 K gradient through wall)")
    result1 = analyze_thermal_stress(design_params=d)
    print_thermal_stress(result1)

    print("\n\nCASE 2: Transient - Cold shock (emergency cooldown)")
    print("  Inner wall at 700 C, outer wall at 500 C (200 K gradient)")
    result2 = analyze_thermal_stress(
        T_inner=700 + 273.15,
        T_outer=500 + 273.15,
        design_params=d,
    )
    print_thermal_stress(result2)

    print("\n\nCASE 3: Startup - gradual heatup")
    print("  Inner wall at 600 C, outer wall at 550 C (50 K gradient)")
    result3 = analyze_thermal_stress(
        T_inner=600 + 273.15,
        T_outer=550 + 273.15,
        design_params=d,
    )
    print_thermal_stress(result3)
