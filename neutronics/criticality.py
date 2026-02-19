"""
Criticality Analysis for Graphite-Moderated Marine MSR
======================================================

Provides two methods for keff estimation:

1. **Modified Four-Factor Formula** (analytical):
   keff = eta * f * p * epsilon * P_NL
   with corrections for MSR-specific physics:
   - Self-shielded resonance integral (Dancoff + Wigner)
   - Graphite reflector savings for non-leakage probability
   - Dilute fuel (5 mol% UF4) treated correctly

2. **Direct k-infinity from cross-sections** (more reliable):
   k_inf = nu*Sigma_f / Sigma_a  (from homogenized 1-group data)
   keff = k_inf * P_NL

The direct method is more reliable for MSRs because the 1-group
cross-sections already account for the thermal spectrum characteristics,
while the four-factor formula was developed for solid-fuel heterogeneous
reactors and requires careful adaptation for liquid-fueled systems.

Critical enrichment search uses bisection on the direct keff formula.

Sources:
  - Duderstadt & Hamilton, "Nuclear Reactor Analysis", Ch. 3-4
  - Lamarsh, "Introduction to Nuclear Engineering", Ch. 4
  - ORNL-4541: MSBR conceptual design data
  - Haubenreich & Engel, "Experience with MSRE" (1970)
  - Dresner, "Resonance Absorption in Nuclear Reactors" (1960)
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    THERMAL_POWER, CORE_AVG_TEMP,
    U235_ENRICHMENT, ENRICHMENT_MIN, ENRICHMENT_MAX,
    GRAPHITE_VOLUME_FRACTION, FUEL_SALT_FRACTION,
    UF4_MOLE_FRACTION,
    NEUTRONS_PER_FISSION,
    SIGMA_FISSION_U235, SIGMA_ABSORPTION_U235, SIGMA_ABSORPTION_U238,
    SIGMA_SCATTER_GRAPHITE, SIGMA_ABSORPTION_GRAPHITE,
    RESONANCE_INTEGRAL_U238,
    MIGRATION_LENGTH_SQUARED, FERMI_AGE,
    GRAPHITE, CHANNEL_DIAMETER, CHANNEL_PITCH,
)
from neutronics.cross_sections import (
    compute_homogenized_cross_sections,
    compute_number_densities,
    compute_graphite_number_density,
)
from neutronics.core_geometry import design_core


# ==============================================================================
# PHYSICS CONSTANTS
# ==============================================================================

# Average logarithmic energy decrement for graphite (C-12)
XI_GRAPHITE = 0.158

# Fast fission factor for thermal MSR (small, mostly thermal fissions)
EPSILON_FAST_FISSION = 1.02


def _reflector_savings(D_core, geometry):
    """Compute reflector savings for a graphite-reflected core.

    The reflector savings delta increases the effective dimensions of
    the core beyond its physical boundaries, reducing the geometric
    buckling and thus reducing neutron leakage.

    For a finite graphite reflector of thickness t:
        delta = L_refl * tanh(t / L_refl)

    where L_refl = sqrt(D_refl / Sigma_a_refl) is the diffusion length
    in the graphite reflector.

    Args:
        D_core: Core diffusion coefficient in m
        geometry: CoreGeometry instance

    Returns:
        tuple: (delta_reflector in m, L_refl in m)
    """
    N_C = compute_graphite_number_density()
    D_refl = 1.0 / (3.0 * N_C * SIGMA_SCATTER_GRAPHITE)
    Sigma_a_refl = N_C * SIGMA_ABSORPTION_GRAPHITE
    L_refl = math.sqrt(D_refl / Sigma_a_refl) if Sigma_a_refl > 0 else 1.0
    t_refl = geometry.reflector_thickness

    delta_reflector = L_refl * math.tanh(t_refl / L_refl)
    return delta_reflector, L_refl


def _non_leakage_probability(geometry, D_core, Sigma_a_total):
    """Compute non-leakage probability P_NL for the finite core.

    P_NL = 1 / (1 + B^2 * M^2)

    where:
      B^2 = (2.405/R_ext)^2 + (pi/H_ext)^2  (geometric buckling)
      M^2 = L^2 + tau                          (migration area)
      L^2 = D / Sigma_a                        (thermal diffusion length^2)
      tau = Fermi age                           (slowing-down area)

    Includes reflector savings in the effective dimensions.

    Args:
        geometry: CoreGeometry instance
        D_core: Core diffusion coefficient in m
        Sigma_a_total: Total macroscopic absorption in 1/m

    Returns:
        dict: P_NL, B2, B2_radial, B2_axial, M2, L2, tau,
              delta_reflector, R_ext, H_ext, L_refl
    """
    delta, L_refl = _reflector_savings(D_core, geometry)

    R_ext = geometry.core_radius + delta
    H_ext = geometry.core_height + 2.0 * delta

    B2_radial = (2.405 / R_ext)**2
    B2_axial = (math.pi / H_ext)**2
    B2 = B2_radial + B2_axial

    L2 = D_core / Sigma_a_total
    tau = FERMI_AGE
    M2 = L2 + tau

    P_NL = 1.0 / (1.0 + B2 * M2)

    return {
        'P_NL': P_NL,
        'B2': B2,
        'B2_radial': B2_radial,
        'B2_axial': B2_axial,
        'M2': M2,
        'L2': L2,
        'tau': tau,
        'delta_reflector': delta,
        'R_ext': R_ext,
        'H_ext': H_ext,
        'L_refl': L_refl,
    }


def four_factor(enrichment=None, geometry=None, T=None,
                graphite_fraction=None, uf4_fraction=None):
    """Compute keff using both four-factor and direct methods.

    The "direct" method uses the ratio of 1-group cross-sections:
        k_inf = nu*Sigma_f / Sigma_a
        keff = k_inf * P_NL

    This is the PRIMARY result for MSR analysis.

    The four-factor decomposition is also computed for physical insight:
        eta = nu * Sigma_f_fuel / Sigma_a_fuel  (reproduction)
        f = Sigma_a_fuel / Sigma_a_total         (thermal utilization)
        p = resonance escape probability          (from self-shielded integral)
        epsilon = 1.02                            (fast fission factor)

    NOTE: For the MSR, the 1-group cross-sections are already spectrum-
    averaged for the thermalized graphite-moderated spectrum. The
    "resonance escape probability" is effectively built into the 1-group
    Sigma_a values. Therefore k_inf_direct = nu*Sigma_f/Sigma_a is the
    more physically meaningful k_inf, while the four-factor decomposition
    (eta*f) captures the same physics and p*epsilon ~ 1 for the already-
    thermalized cross-sections.

    Args:
        enrichment: U-235 enrichment (weight fraction). Default from config.
        geometry: CoreGeometry instance. Default: design_core().
        T: Temperature in K. Default from config.
        graphite_fraction: Graphite volume fraction. Default from config.
        uf4_fraction: UF4 mole fraction. Default from config.

    Returns:
        dict: Four factors, leakage parameters, keff, and all intermediates.
    """
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if geometry is None:
        geometry = design_core()
    if T is None:
        T = CORE_AVG_TEMP
    if graphite_fraction is None:
        graphite_fraction = GRAPHITE_VOLUME_FRACTION
    if uf4_fraction is None:
        uf4_fraction = UF4_MOLE_FRACTION

    salt_fraction = 1.0 - graphite_fraction

    # Get homogenized cross-sections
    xs = compute_homogenized_cross_sections(
        enrichment=enrichment,
        graphite_fraction=graphite_fraction,
        uf4_fraction=uf4_fraction,
        T=T,
    )

    nd = xs['number_densities']
    N_C = compute_graphite_number_density()

    # ==================================================================
    # DIRECT METHOD (primary, more reliable for MSR)
    # ==================================================================
    # k_inf from 1-group homogenized cross-sections
    k_inf_direct = xs['nu_sigma_f'] / xs['sigma_a']

    # Non-leakage probability
    leak = _non_leakage_probability(geometry, xs['D'], xs['sigma_a'])
    P_NL = leak['P_NL']

    # keff from direct method
    keff = k_inf_direct * P_NL

    # ==================================================================
    # FOUR-FACTOR DECOMPOSITION (for physical insight)
    # ==================================================================
    # eta: Reproduction factor (neutrons per absorption in fuel)
    Sigma_f_fuel = nd['N_U235'] * SIGMA_FISSION_U235
    Sigma_a_fuel_unmixed = nd['N_U235'] * SIGMA_ABSORPTION_U235 + nd['N_U238'] * SIGMA_ABSORPTION_U238
    eta = NEUTRONS_PER_FISSION * Sigma_f_fuel / Sigma_a_fuel_unmixed

    # f: Thermal utilization (fraction absorbed in fuel)
    Sigma_a_fuel_hom = salt_fraction * Sigma_a_fuel_unmixed
    f = Sigma_a_fuel_hom / xs['sigma_a']

    # Note: eta * f = nu*Sigma_f / Sigma_a = k_inf_direct (by construction)
    # So we define p_effective and epsilon such that k_inf = eta*f*p*eps
    # For spectrum-averaged 1-group data: p_eff * eps = 1.0
    # But we compute the actual four-factor p for educational purposes

    # Resonance escape probability (informational)
    Sigma_s_mod = graphite_fraction * N_C * SIGMA_SCATTER_GRAPHITE
    N_238_hom = salt_fraction * nd['N_U238']

    # Self-shielded resonance integral
    from neutronics.cross_sections import (
        SIGMA_TR_LI7, SIGMA_TR_BE, SIGMA_TR_F, SIGMA_TR_U235, SIGMA_TR_U238
    )
    sigma_pot_num = (nd['N_Li7'] * SIGMA_TR_LI7 + nd['N_Be'] * SIGMA_TR_BE +
                     nd['N_F'] * SIGMA_TR_F + nd['N_U235'] * SIGMA_TR_U235 +
                     nd['N_U238'] * SIGMA_TR_U238)
    N_238 = nd['N_U238']
    sigma_pot = sigma_pot_num / N_238 if N_238 > 0 else 1e-24

    r_fuel = geometry.channel_diameter / 2.0
    a_bell = 1.16

    Sigma_s_graphite = N_C * SIGMA_SCATTER_GRAPHITE
    mfp_mod = 1.0 / Sigma_s_graphite
    d_surface = geometry.channel_pitch - geometry.channel_diameter
    C_dancoff = math.exp(-d_surface / mfp_mod)

    sigma_esc = a_bell * (1.0 - C_dancoff) / (N_238 * r_fuel) if N_238 > 0 else 1e-24
    sigma_0 = sigma_pot + sigma_esc
    I_inf = RESONANCE_INTEGRAL_U238
    I_eff = I_inf * math.sqrt(sigma_0 / (sigma_0 + I_inf))

    resonance_exponent = N_238_hom * I_eff / (XI_GRAPHITE * Sigma_s_mod)
    p_four_factor = math.exp(-resonance_exponent)

    # epsilon: fast fission factor
    epsilon = EPSILON_FAST_FISSION

    # Four-factor k_inf (for comparison)
    k_inf_four = eta * f * p_four_factor * epsilon

    return {
        # PRIMARY RESULTS (direct method)
        'keff': keff,
        'k_inf': k_inf_direct,
        'P_NL': P_NL,
        'enrichment': enrichment,

        # Four-factor decomposition
        'eta': eta,
        'f': f,
        'p': p_four_factor,
        'epsilon': epsilon,
        'k_inf_four_factor': k_inf_four,

        # Leakage details
        'B2': leak['B2'],
        'B2_radial': leak['B2_radial'],
        'B2_axial': leak['B2_axial'],
        'M2': leak['M2'],
        'L2': leak['L2'],
        'tau': leak['tau'],
        'D': xs['D'],
        'delta_reflector': leak['delta_reflector'],
        'R_ext': leak['R_ext'],
        'H_ext': leak['H_ext'],
        'L_refl': leak['L_refl'],

        # Resonance self-shielding
        'I_eff': I_eff,
        'I_eff_barns': I_eff / 1e-28,
        'sigma_0': sigma_0,
        'sigma_0_barns': sigma_0 / 1e-28,
        'C_dancoff': C_dancoff,
        'resonance_exponent': resonance_exponent,

        # Cross-sections
        'Sigma_f_fuel': Sigma_f_fuel,
        'Sigma_a_fuel': Sigma_a_fuel_unmixed,
        'Sigma_a_total': xs['sigma_a'],
        'Sigma_s_mod': Sigma_s_mod,
        'N_238_hom': N_238_hom,
        'xs': xs,
    }


def find_critical_enrichment(geometry=None, T=None, target_keff=1.0,
                              e_min=None, e_max=None, tol=1e-5,
                              max_iter=100, graphite_fraction=None,
                              uf4_fraction=None):
    """Find the critical enrichment using bisection search.

    Searches for the U-235 enrichment that gives keff = target_keff
    using the direct k_inf method (nu*Sigma_f / Sigma_a * P_NL).

    Args:
        geometry: CoreGeometry instance. Default: design_core().
        T: Temperature in K. Default from config.
        target_keff: Target keff value (default 1.0 for exact criticality)
        e_min: Minimum enrichment search bound. Default from config.
        e_max: Maximum enrichment search bound. Default from config.
        tol: Convergence tolerance on keff. Default 1e-5.
        max_iter: Maximum bisection iterations. Default 100.
        graphite_fraction: Volume fraction of graphite.
        uf4_fraction: UF4 mole fraction.

    Returns:
        dict:
            - enrichment_critical: Critical enrichment (weight fraction)
            - keff: keff at critical enrichment
            - iterations: Number of bisection iterations
            - converged: Boolean
            - four_factor_result: Full result at critical enrichment
    """
    if geometry is None:
        geometry = design_core()
    if T is None:
        T = CORE_AVG_TEMP
    if e_min is None:
        e_min = ENRICHMENT_MIN
    if e_max is None:
        e_max = ENRICHMENT_MAX

    # Verify bracket
    result_min = four_factor(enrichment=e_min, geometry=geometry, T=T,
                             graphite_fraction=graphite_fraction,
                             uf4_fraction=uf4_fraction)
    result_max = four_factor(enrichment=e_max, geometry=geometry, T=T,
                             graphite_fraction=graphite_fraction,
                             uf4_fraction=uf4_fraction)

    if result_min['keff'] > target_keff:
        print(f"  NOTE: keff({e_min*100:.1f}%) = {result_min['keff']:.4f} > {target_keff}")
        print(f"  Lower bound already supercritical.")
        return {
            'enrichment_critical': e_min,
            'keff': result_min['keff'],
            'iterations': 0,
            'converged': False,
            'four_factor_result': result_min,
        }

    if result_max['keff'] < target_keff:
        print(f"  NOTE: keff({e_max*100:.1f}%) = {result_max['keff']:.4f} < {target_keff}")
        print(f"  Upper bound still subcritical. Extending search to 50%...")
        e_max = 0.50
        result_max = four_factor(enrichment=e_max, geometry=geometry, T=T,
                                 graphite_fraction=graphite_fraction,
                                 uf4_fraction=uf4_fraction)
        if result_max['keff'] < target_keff:
            print(f"  WARNING: Even at {e_max*100:.0f}%, keff = {result_max['keff']:.4f} < {target_keff}")
            return {
                'enrichment_critical': e_max,
                'keff': result_max['keff'],
                'iterations': 0,
                'converged': False,
                'four_factor_result': result_max,
            }

    # Bisection search
    e_lo = e_min
    e_hi = e_max
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        e_mid = (e_lo + e_hi) / 2.0
        result_mid = four_factor(enrichment=e_mid, geometry=geometry, T=T,
                                 graphite_fraction=graphite_fraction,
                                 uf4_fraction=uf4_fraction)

        k_mid = result_mid['keff']
        n_iter = iteration + 1

        if abs(k_mid - target_keff) < tol:
            converged = True
            break

        if k_mid < target_keff:
            e_lo = e_mid
        else:
            e_hi = e_mid

    e_final = (e_lo + e_hi) / 2.0
    result_final = four_factor(enrichment=e_final, geometry=geometry, T=T,
                               graphite_fraction=graphite_fraction,
                               uf4_fraction=uf4_fraction)

    return {
        'enrichment_critical': e_final,
        'keff': result_final['keff'],
        'iterations': n_iter,
        'converged': converged,
        'four_factor_result': result_final,
    }


def print_neutron_balance(results=None):
    """Print a formatted neutron balance and criticality analysis.

    Shows multiplication factors from both the direct method and the
    four-factor decomposition, along with absorption, leakage, and
    resonance self-shielding details.

    Args:
        results: Dict from four_factor(). Computed with defaults if None.
    """
    if results is None:
        results = four_factor()

    xs = results['xs']

    print("=" * 72)
    print("  CRITICALITY ANALYSIS - NEUTRON BALANCE")
    print("=" * 72)

    print(f"\n  Enrichment:  {results['enrichment']*100:.2f} % U-235")
    print(f"  Temperature: {CORE_AVG_TEMP - 273.15:.0f} C")

    print(f"\n  --- Multiplication Factors (Direct Method) ---")
    print(f"    k_inf (nu*Sf/Sa):     {results['k_inf']:10.4f}")
    print(f"    P_NL (non-leakage):   {results['P_NL']:10.4f}")
    print(f"    keff = k_inf * P_NL:  {results['keff']:10.4f}")
    rho_pcm = (results['keff'] - 1.0) / results['keff'] * 1e5
    print(f"    Reactivity rho:       {rho_pcm:10.1f} pcm")

    print(f"\n  --- Four-Factor Decomposition (for insight) ---")
    print(f"    eta (reproduction):   {results['eta']:10.4f}")
    print(f"    f   (thermal util.):  {results['f']:10.4f}")
    print(f"    eta*f = k_inf:        {results['eta']*results['f']:10.4f}")
    print(f"    p   (res. escape):    {results['p']:10.4f}  (informational)")
    print(f"    eps (fast fission):    {results['epsilon']:10.4f}  (informational)")
    print(f"    k_inf (4-factor):     {results['k_inf_four_factor']:10.4f}  (for comparison)")

    print(f"\n  --- Leakage ---")
    print(f"    B^2 (geometric):      {results['B2']:10.4f} 1/m^2")
    print(f"      B^2_radial:         {results['B2_radial']:10.4f} 1/m^2")
    print(f"      B^2_axial:          {results['B2_axial']:10.4f} 1/m^2")
    print(f"    M^2 (migration):      {results['M2']:10.6f} m^2")
    print(f"      L^2 (diffusion):    {results['L2']:10.6f} m^2")
    print(f"      tau (Fermi age):    {results['tau']:10.6f} m^2")
    print(f"    Leakage fraction:     {1.0-results['P_NL']:10.4f}  "
          f"({(1.0-results['P_NL'])*100:.1f}%)")
    print(f"    Reflector savings:    {results['delta_reflector']*100:10.2f} cm")
    print(f"    R_ext (effective):    {results['R_ext']:10.4f} m")
    print(f"    H_ext (effective):    {results['H_ext']:10.4f} m")
    print(f"    L_refl (graphite):    {results['L_refl']:10.4f} m")

    print(f"\n  --- Absorption Breakdown ---")
    Sigma_a_total = results['Sigma_a_total']
    sf = xs['salt_fraction']
    gf = xs['graphite_fraction']
    print(f"    Total Sigma_a:        {Sigma_a_total:10.4f} 1/m")
    components = [
        ("U-235", sf * xs['Sigma_a_U235']),
        ("U-238", sf * xs['Sigma_a_U238']),
        ("Li-7",  sf * xs['Sigma_a_Li7']),
        ("Be",    sf * xs['Sigma_a_Be']),
        ("F",     sf * xs['Sigma_a_F']),
        ("Graphite", gf * xs['Sigma_a_C']),
    ]
    for name, sigma in components:
        frac = sigma / Sigma_a_total * 100
        print(f"    {name:12s}          {sigma:10.4f} 1/m  ({frac:5.1f}%)")

    print(f"\n  --- Resonance Self-Shielding (informational) ---")
    print(f"    I_eff:                {results['I_eff_barns']:10.1f} barns")
    print(f"    I_inf:                {RESONANCE_INTEGRAL_U238/1e-28:10.1f} barns")
    print(f"    Shielding factor:     {results['I_eff']/RESONANCE_INTEGRAL_U238:10.4f}")
    print(f"    sigma_0 (dilution):   {results['sigma_0_barns']:10.1f} barns")
    print(f"    Dancoff correction:   {results['C_dancoff']:10.4f}")

    print()


if __name__ == '__main__':
    # --- Criticality analysis at nominal enrichment ---
    print("Criticality analysis at nominal enrichment (12%)...\n")
    result = four_factor()
    print_neutron_balance(result)

    # --- Enrichment search ---
    print("\n" + "=" * 72)
    print("  CRITICAL ENRICHMENT SEARCH")
    print("=" * 72)

    geom = design_core()
    crit = find_critical_enrichment(geometry=geom)

    print(f"\n  Critical Enrichment:  {crit['enrichment_critical']*100:.3f} %")
    print(f"  keff at critical:     {crit['keff']:.6f}")
    print(f"  Converged:            {'Yes' if crit['converged'] else 'No'}")
    print(f"  Iterations:           {crit['iterations']}")

    # Print the neutron balance at critical enrichment
    print(f"\n  --- Neutron balance at critical enrichment ---")
    print_neutron_balance(crit['four_factor_result'])

    # --- Enrichment sensitivity table ---
    print("\n  --- keff vs. Enrichment ---")
    print(f"  {'Enrich':>8s}  {'k_inf':>8s}  {'P_NL':>8s}  {'keff':>8s}  "
          f"{'eta':>8s}  {'f':>8s}  {'rho(pcm)':>10s}")
    for e in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.19]:
        r = four_factor(enrichment=e, geometry=geom)
        rho = (r['keff'] - 1.0) / r['keff'] * 1e5
        print(f"  {e*100:6.1f} %  {r['k_inf']:8.4f}  {r['P_NL']:8.4f}  {r['keff']:8.4f}  "
              f"{r['eta']:8.4f}  {r['f']:8.4f}  {rho:10.0f}")
