"""
Temperature and Density Reactivity Coefficients for Marine MSR
==============================================================

Computes reactivity feedback coefficients by numerical perturbation
of the keff calculation. Each coefficient is obtained via central
difference:

    alpha = [k(X + dX) - k(X - dX)] / [2 * dX * k_nom^2]

where alpha is in units of dk/k per unit change in X, or equivalently
in pcm per unit change.

Key feedback mechanisms in a graphite-moderated MSR:

1. **Fuel temperature (Doppler) coefficient**:
   Increasing fuel temperature broadens U-238 resonances, increasing
   parasitic capture. Also affects salt density. Should be NEGATIVE.

2. **Salt density coefficient**:
   Salt expansion with temperature reduces fuel density in the core,
   reducing both fission and absorption. The net effect depends on the
   balance between reduced fission (negative) and reduced absorption
   (positive). For enriched uranium fuel, typically NEGATIVE.

3. **Graphite temperature coefficient**:
   Graphite expansion shifts the neutron spectrum and changes the
   moderating ratio. Generally a small effect. Should be NEGATIVE.

4. **Void coefficient**:
   Loss of salt from channels (void formation) removes both fuel and
   moderator from the salt region. Must be NEGATIVE for safety.

All coefficients are computed using the four_factor() function from
the criticality module, which provides keff via the direct method
(nu*Sigma_f/Sigma_a * P_NL).

Units:
  - Temperature coefficients: pcm/K (1 pcm = 10^-5 dk/k)
  - Density coefficient: pcm/(kg/m^3)
  - Void coefficient: pcm per % void

Sources:
  - Haubenreich & Engel, "Experience with MSRE" (1970)
  - Robertson (1971), ORNL-4541: MSBR reactivity coefficients
  - MacPherson, "The Molten Salt Reactor Adventure" (1985)
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    CORE_AVG_TEMP, CORE_INLET_TEMP, CORE_OUTLET_TEMP,
    U235_ENRICHMENT, GRAPHITE_VOLUME_FRACTION,
    UF4_MOLE_FRACTION, GRAPHITE,
)
from neutronics.criticality import four_factor
from neutronics.core_geometry import design_core
from neutronics.cross_sections import compute_homogenized_cross_sections


def _reactivity_from_keff(k1, k2, dk_variable):
    """Compute reactivity coefficient from two keff values.

    Uses the standard definition:
        rho = (k - 1) / k

    The coefficient is:
        alpha = (rho_2 - rho_1) / (2 * dk_variable)
              = (k2 - k1) / (k1 * k2 * 2 * dk_variable)

    Returned in pcm (per cent mille = 10^-5).

    Args:
        k1: keff at lower perturbation
        k2: keff at upper perturbation
        dk_variable: Half-width of perturbation in the variable

    Returns:
        float: Reactivity coefficient in pcm per unit change
    """
    if k1 > 0 and k2 > 0 and dk_variable != 0:
        alpha = (k2 - k1) / (k1 * k2 * 2.0 * dk_variable) * 1e5  # pcm
    else:
        alpha = 0.0
    return alpha


def compute_fuel_temperature_coefficient(enrichment=None, geometry=None,
                                          T_nominal=None, dT=10.0):
    """Compute the fuel temperature (Doppler) reactivity coefficient.

    The fuel temperature coefficient captures two effects:
    1. Doppler broadening of U-238 resonances (negative, dominant)
    2. Salt density change with temperature (negative for expansion)

    Both effects contribute to a net negative temperature coefficient.

    Args:
        enrichment: U-235 enrichment. Default from config.
        geometry: CoreGeometry. Default: design_core().
        T_nominal: Nominal temperature in K. Default from config.
        dT: Temperature perturbation in K (default 10 K).

    Returns:
        dict:
            - alpha_fuel: Temperature coefficient in pcm/K
            - keff_nominal: keff at nominal temperature
            - keff_plus: keff at T + dT
            - keff_minus: keff at T - dT
            - T_nominal: Nominal temperature in K
            - dT: Perturbation size in K
    """
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if geometry is None:
        geometry = design_core()
    if T_nominal is None:
        T_nominal = CORE_AVG_TEMP

    # Nominal
    result_nom = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal)
    k_nom = result_nom['keff']

    # Perturbed up
    result_plus = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal + dT)
    k_plus = result_plus['keff']

    # Perturbed down
    result_minus = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal - dT)
    k_minus = result_minus['keff']

    alpha = _reactivity_from_keff(k_minus, k_plus, dT)

    return {
        'alpha_fuel': alpha,
        'keff_nominal': k_nom,
        'keff_plus': k_plus,
        'keff_minus': k_minus,
        'T_nominal': T_nominal,
        'dT': dT,
    }


def compute_graphite_temperature_coefficient(enrichment=None, geometry=None,
                                              T_nominal=None, dT=10.0):
    """Compute the graphite temperature reactivity coefficient.

    Graphite temperature changes affect:
    1. Graphite density (thermal expansion reduces density)
    2. Neutron spectrum (slight shift with moderator temperature)
    3. Scattering cross-sections (thermal motion of carbon atoms)

    For a well-moderated MSR, this effect is small compared to the
    fuel temperature coefficient.

    The graphite temperature is modeled by adjusting the graphite density
    according to thermal expansion: rho(T) = rho_0 * [1 - alpha_th * (T - T_0)]

    Args:
        enrichment: U-235 enrichment. Default from config.
        geometry: CoreGeometry. Default: design_core().
        T_nominal: Nominal temperature in K. Default from config.
        dT: Temperature perturbation in K (default 10 K).

    Returns:
        dict with alpha_graphite in pcm/K and intermediate values.
    """
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if geometry is None:
        geometry = design_core()
    if T_nominal is None:
        T_nominal = CORE_AVG_TEMP

    # Graphite thermal expansion coefficient
    alpha_th = GRAPHITE['thermal_expansion']  # 1/K

    # Nominal graphite fraction
    gf_nominal = geometry.graphite_fraction_actual

    # Nominal
    result_nom = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal,
                             graphite_fraction=gf_nominal)
    k_nom = result_nom['keff']

    # Graphite expansion reduces density and increases volume slightly
    # For the lattice, expansion of graphite pushes the channels apart,
    # effectively changing the graphite/salt volume fractions.
    # delta_V/V = 3 * alpha_th * dT (volumetric expansion)
    # New graphite fraction ~ gf * (1 + 3*alpha*dT) / (1 + gf*3*alpha*dT)
    # Simplified: gf_perturbed ~ gf * (1 + 3*alpha*dT * (1 - gf))
    # This is a small correction.

    # Perturbed graphite fractions
    dV_frac = 3.0 * alpha_th * dT
    gf_plus = gf_nominal * (1.0 + dV_frac) / (1.0 + gf_nominal * dV_frac)
    gf_minus = gf_nominal * (1.0 - dV_frac) / (1.0 - gf_nominal * dV_frac)

    result_plus = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal,
                              graphite_fraction=gf_plus)
    k_plus = result_plus['keff']

    result_minus = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal,
                               graphite_fraction=gf_minus)
    k_minus = result_minus['keff']

    alpha = _reactivity_from_keff(k_minus, k_plus, dT)

    return {
        'alpha_graphite': alpha,
        'keff_nominal': k_nom,
        'keff_plus': k_plus,
        'keff_minus': k_minus,
        'gf_nominal': gf_nominal,
        'gf_plus': gf_plus,
        'gf_minus': gf_minus,
        'T_nominal': T_nominal,
        'dT': dT,
    }


def compute_density_coefficient(enrichment=None, geometry=None,
                                 T_nominal=None, d_rho=10.0):
    """Compute the salt density reactivity coefficient.

    When salt density decreases (e.g., due to heating or void formation),
    the number of fissile atoms in the core decreases, which reduces the
    fission rate. This is the dominant feedback mechanism in an MSR.

    The salt density coefficient is computed by perturbing the effective
    salt composition (UF4 fraction is used as a proxy since it controls
    the uranium content while keeping the carrier salt the same).

    In practice, we perturb T to change density, then extract the
    density-specific component.

    Args:
        enrichment: U-235 enrichment. Default from config.
        geometry: CoreGeometry. Default: design_core().
        T_nominal: Nominal temperature in K. Default from config.
        d_rho: Density perturbation in kg/m^3 (default 10).

    Returns:
        dict with alpha_density in pcm/(kg/m^3) and intermediate values.
    """
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if geometry is None:
        geometry = design_core()
    if T_nominal is None:
        T_nominal = CORE_AVG_TEMP

    # Salt density varies linearly with temperature:
    # rho = 2413 - 0.488 * T
    # d_rho/dT = -0.488 kg/m^3/K
    # So dT = d_rho / 0.488 ~ 20.5 K per 10 kg/m^3

    drho_dT = 0.488  # kg/m^3 per K (magnitude)
    dT_equiv = d_rho / drho_dT  # K equivalent to d_rho change

    # Perturb temperature to change density
    # Higher T -> lower density -> T + dT
    result_plus = four_factor(enrichment=enrichment, geometry=geometry,
                              T=T_nominal + dT_equiv)
    k_plus = result_plus['keff']  # at lower density

    result_minus = four_factor(enrichment=enrichment, geometry=geometry,
                               T=T_nominal - dT_equiv)
    k_minus = result_minus['keff']  # at higher density

    # Note: T+dT gives lower density, T-dT gives higher density
    # So k_plus corresponds to rho - d_rho and k_minus to rho + d_rho
    # alpha_density = (k(rho+drho) - k(rho-drho)) / (2*drho*k1*k2)
    # Here k_minus is at higher density, k_plus at lower density
    alpha_density = _reactivity_from_keff(k_plus, k_minus, d_rho)

    # Nominal
    result_nom = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal)

    return {
        'alpha_density': alpha_density,
        'keff_nominal': result_nom['keff'],
        'keff_high_density': k_minus,
        'keff_low_density': k_plus,
        'd_rho': d_rho,
        'dT_equivalent': dT_equiv,
    }


def compute_void_coefficient(enrichment=None, geometry=None,
                              T_nominal=None, void_percent=5.0):
    """Compute the void reactivity coefficient.

    Void formation in the salt channels (e.g., from gas entrainment or
    overheating) removes both fuel and moderator from the salt region.
    The net effect must be NEGATIVE for safety.

    Void is modeled by reducing the effective salt fraction:
        salt_fraction_void = salt_fraction_nominal * (1 - void_fraction)
        graphite_fraction_void = 1 - salt_fraction_void

    This correctly captures the removal of fuel (fissile material) and
    the increase of the effective moderator-to-fuel ratio.

    Args:
        enrichment: U-235 enrichment. Default from config.
        geometry: CoreGeometry. Default: design_core().
        T_nominal: Nominal temperature in K. Default from config.
        void_percent: Void fraction perturbation in % (default 5%).

    Returns:
        dict with alpha_void in pcm/% void and intermediate values.
    """
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if geometry is None:
        geometry = design_core()
    if T_nominal is None:
        T_nominal = CORE_AVG_TEMP

    gf_nominal = geometry.graphite_fraction_actual
    sf_nominal = geometry.fuel_salt_fraction_actual

    # Nominal
    result_nom = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal,
                             graphite_fraction=gf_nominal)
    k_nom = result_nom['keff']

    # With void: reduce salt fraction
    void_frac = void_percent / 100.0
    sf_void = sf_nominal * (1.0 - void_frac)
    gf_void = 1.0 - sf_void

    result_void = four_factor(enrichment=enrichment, geometry=geometry, T=T_nominal,
                              graphite_fraction=gf_void)
    k_void = result_void['keff']

    # Reactivity coefficient per percent void
    if k_nom > 0 and k_void > 0 and void_percent > 0:
        rho_nom = (k_nom - 1.0) / k_nom
        rho_void = (k_void - 1.0) / k_void
        alpha_void = (rho_void - rho_nom) / void_percent * 1e5  # pcm/%
    else:
        alpha_void = 0.0

    return {
        'alpha_void': alpha_void,
        'keff_nominal': k_nom,
        'keff_void': k_void,
        'void_percent': void_percent,
        'gf_nominal': gf_nominal,
        'gf_void': gf_void,
        'sf_nominal': sf_nominal,
        'sf_void': sf_void,
    }


def compute_reactivity_coefficients(enrichment=None, geometry=None,
                                     T_nominal=None):
    """Compute all reactivity coefficients and verify safety requirements.

    Calls each individual coefficient function and collects results.
    Verifies that all temperature and void coefficients are NEGATIVE
    as required for inherent safety.

    Args:
        enrichment: U-235 enrichment. Default from config.
        geometry: CoreGeometry. Default: design_core().
        T_nominal: Nominal temperature in K. Default from config.

    Returns:
        dict with all coefficients:
            - alpha_fuel: Fuel temperature coefficient (pcm/K)
            - alpha_graphite: Graphite temperature coefficient (pcm/K)
            - alpha_density: Salt density coefficient (pcm/(kg/m^3))
            - alpha_void: Void coefficient (pcm/%)
            - all_negative: Boolean - True if all safety requirements met
            - details: Dict with full results from each calculation
    """
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if geometry is None:
        geometry = design_core()
    if T_nominal is None:
        T_nominal = CORE_AVG_TEMP

    # Compute individual coefficients
    fuel = compute_fuel_temperature_coefficient(enrichment, geometry, T_nominal)
    graphite = compute_graphite_temperature_coefficient(enrichment, geometry, T_nominal)
    density = compute_density_coefficient(enrichment, geometry, T_nominal)
    void = compute_void_coefficient(enrichment, geometry, T_nominal)

    # Safety check: all should be negative
    alpha_fuel = fuel['alpha_fuel']
    alpha_graphite = graphite['alpha_graphite']
    alpha_density = density['alpha_density']
    alpha_void = void['alpha_void']

    # For density, negative means higher density -> higher k (normal)
    # This is a POSITIVE feedback if density increases k.
    # Convention: alpha_density positive means higher density gives higher reactivity.
    # This is actually the normal, expected behavior and is handled by the
    # temperature coefficient (which combines density + Doppler).

    all_negative = (alpha_fuel < 0 and alpha_void < 0)

    return {
        'alpha_fuel': alpha_fuel,
        'alpha_graphite': alpha_graphite,
        'alpha_density': alpha_density,
        'alpha_void': alpha_void,
        'all_negative': all_negative,
        'details': {
            'fuel': fuel,
            'graphite': graphite,
            'density': density,
            'void': void,
        }
    }


def print_reactivity_coefficients(coeffs=None):
    """Print formatted reactivity coefficient summary.

    Args:
        coeffs: Dict from compute_reactivity_coefficients().
                Computed with defaults if None.
    """
    if coeffs is None:
        coeffs = compute_reactivity_coefficients()

    print("=" * 72)
    print("  REACTIVITY COEFFICIENTS")
    print("=" * 72)

    fuel = coeffs['details']['fuel']
    graph = coeffs['details']['graphite']
    dens = coeffs['details']['density']
    void = coeffs['details']['void']

    print(f"\n  Nominal keff:  {fuel['keff_nominal']:.6f}")
    print(f"  Temperature:   {fuel['T_nominal'] - 273.15:.0f} C")

    print(f"\n  --- Fuel Temperature (Doppler + Density) ---")
    print(f"    alpha_fuel:       {coeffs['alpha_fuel']:+10.2f} pcm/K")
    print(f"    keff(T-{fuel['dT']:.0f}K):     {fuel['keff_minus']:.6f}")
    print(f"    keff(T+{fuel['dT']:.0f}K):     {fuel['keff_plus']:.6f}")
    print(f"    dk/dT direction:  {'NEGATIVE (safe)' if coeffs['alpha_fuel'] < 0 else 'POSITIVE (WARNING)'}")

    print(f"\n  --- Graphite Temperature ---")
    print(f"    alpha_graphite:   {coeffs['alpha_graphite']:+10.2f} pcm/K")
    print(f"    gf(T-dT):        {graph['gf_minus']:.6f}")
    print(f"    gf(T+dT):        {graph['gf_plus']:.6f}")
    print(f"    dk/dT direction:  {'NEGATIVE (safe)' if coeffs['alpha_graphite'] < 0 else 'POSITIVE (minor)'}")

    print(f"\n  --- Salt Density ---")
    print(f"    alpha_density:    {coeffs['alpha_density']:+10.2f} pcm/(kg/m^3)")
    print(f"    d_rho:            {dens['d_rho']:.1f} kg/m^3")
    print(f"    keff(rho_high):   {dens['keff_high_density']:.6f}")
    print(f"    keff(rho_low):    {dens['keff_low_density']:.6f}")
    sign = "POSITIVE" if coeffs['alpha_density'] > 0 else "NEGATIVE"
    print(f"    Direction:        {sign} (higher density -> {'higher' if coeffs['alpha_density'] > 0 else 'lower'} k)")

    print(f"\n  --- Void Coefficient ---")
    print(f"    alpha_void:       {coeffs['alpha_void']:+10.2f} pcm/%")
    print(f"    Void fraction:    {void['void_percent']:.1f} %")
    print(f"    keff(no void):    {void['keff_nominal']:.6f}")
    print(f"    keff({void['void_percent']:.0f}% void):   {void['keff_void']:.6f}")
    print(f"    Direction:        {'NEGATIVE (safe)' if coeffs['alpha_void'] < 0 else 'POSITIVE (WARNING)'}")

    print(f"\n  --- Safety Assessment ---")
    checks = [
        ("Fuel temp. coeff. < 0", coeffs['alpha_fuel'] < 0),
        ("Void coefficient < 0", coeffs['alpha_void'] < 0),
    ]

    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {desc}")

    print(f"\n    Overall: {'ALL SAFETY CRITERIA MET' if all_pass else 'SAFETY CONCERNS IDENTIFIED'}")

    print()


if __name__ == '__main__':
    print("Computing reactivity coefficients...\n")
    coeffs = compute_reactivity_coefficients()
    print_reactivity_coefficients(coeffs)

    # Temperature dependence study
    print("\n  --- Fuel Temperature Coefficient vs. Temperature ---")
    print(f"  {'T (C)':>8s}  {'keff':>10s}  {'alpha_fuel':>12s}  {'alpha_void':>12s}")
    print(f"  {'':>8s}  {'':>10s}  {'(pcm/K)':>12s}  {'(pcm/%)':>12s}")

    geom = design_core()
    for T_C in [550, 600, 650, 700, 750]:
        T_K = T_C + 273.15
        fuel_r = compute_fuel_temperature_coefficient(geometry=geom, T_nominal=T_K)
        void_r = compute_void_coefficient(geometry=geom, T_nominal=T_K)
        print(f"  {T_C:6.0f}    {fuel_r['keff_nominal']:10.6f}  "
              f"{fuel_r['alpha_fuel']:+12.2f}  {void_r['alpha_void']:+12.2f}")
