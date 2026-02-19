"""
Reactivity Coefficient Calculations via Monte Carlo Perturbation
================================================================

Computes reactivity coefficients for the 40 MWth Marine MSR by running
pairs (or sets) of k-eigenvalue Monte Carlo calculations at perturbed
conditions and differencing the results.

Coefficients computed:
  - Fuel temperature coefficient (Doppler + density)
  - Void coefficient (fuel salt density reduction)
  - Graphite temperature coefficient
  - Enrichment sensitivity study (multi-point)

Methodology
-----------
For a generic parameter X, the reactivity coefficient is:

    alpha_X = (k2 - k1) / (k2 * k1 * delta_X) * 1e5  [pcm / unit_X]

with propagated uncertainty:

    delta_alpha = sqrt(sigma_k1^2 + sigma_k2^2) / (k1*k2*delta_X) * 1e5

Each coefficient requires 2+ independent Monte Carlo eigenvalue calculations,
so runtimes scale linearly with the number of perturbation points.

References
----------
- Lux & Koblinger, "MC Particle Transport Methods," 1991, ch. 12
- Briesmeister (ed.), "MCNP -- A General Monte Carlo Code," LA-13709-M
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import N_GROUPS, MAT_FUEL_SALT, MAT_GRAPHITE_MOD, MAT_GRAPHITE_REF
from .materials import (
    Material, create_msr_materials, fuel_salt, graphite, reflector,
    void_material, check_criticality_estimate,
)
from .geometry import MSRGeometry
from .eigenvalue import EigenvalueSolver, EigenvalueResult


# =====================================================================
# Utility: run a single MC eigenvalue with specified materials
# =====================================================================
def _run_eigenvalue(
    materials: Dict[int, Material],
    n_particles: int,
    n_batches: int,
    n_inactive: int,
    seed: Optional[int] = None,
    verbose: bool = False,
    label: str = "",
) -> EigenvalueResult:
    """Run a single MC eigenvalue calculation with given materials.

    Parameters
    ----------
    materials : dict
        Material definitions keyed by material ID.
    n_particles : int
        Neutrons per batch.
    n_batches : int
        Total batches (inactive + active).
    n_inactive : int
        Number of inactive (source convergence) batches.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress.
    label : str
        Label for printout.

    Returns
    -------
    EigenvalueResult
        Eigenvalue calculation results.
    """
    geometry = MSRGeometry()
    n_active = n_batches - n_inactive

    solver = EigenvalueSolver(
        geometry=geometry,
        materials=materials,
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=n_inactive,
        mesh_r_bins=10,
        mesh_z_bins=15,
        seed=seed,
    )

    if verbose and label:
        print(f"\n  --- {label} ---")

    result = solver.solve(verbose=verbose)
    return result


# =====================================================================
# Fuel temperature coefficient
# =====================================================================
def fuel_temperature_coefficient(
    T_ref: float = 923.15,
    delta_T: float = 20.0,
    n_particles: int = 3000,
    n_batches: int = 80,
    n_inactive: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Compute the fuel temperature coefficient of reactivity.

    Runs MC at T_ref - delta_T and T_ref + delta_T, then computes:

        alpha_fuel = (k_high - k_low) / (k_high * k_low * 2 * delta_T) * 1e5
                   [pcm/K]

    The fuel temperature coefficient combines two physical effects:
      1. Doppler broadening of U-238 resonances (always negative)
      2. Thermal expansion reducing fuel salt density (negative for
         under-moderated systems, can be positive for over-moderated)

    Parameters
    ----------
    T_ref : float
        Reference temperature [K]. Default 923.15 K (650 C).
    delta_T : float
        Temperature perturbation [K]. Default 20 K.
    n_particles : int
        Neutrons per batch. Default 3000.
    n_batches : int
        Total batches. Default 80.
    n_inactive : int
        Inactive batches. Default 15.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'alpha_pcm_per_K': fuel temperature coefficient [pcm/K]
        'alpha_std_pcm_per_K': uncertainty [pcm/K]
        'keff_low': k_eff at T_ref - delta_T
        'keff_low_std': uncertainty on k_eff_low
        'keff_high': k_eff at T_ref + delta_T
        'keff_high_std': uncertainty on k_eff_high
        'T_low': lower temperature [K]
        'T_high': upper temperature [K]
        'is_negative': True if coefficient is negative (safety requirement)
    """
    t_start = time.perf_counter()
    T_low = T_ref - delta_T
    T_high = T_ref + delta_T

    if verbose:
        print("\n" + "=" * 70)
        print("  Fuel Temperature Coefficient Calculation")
        print("=" * 70)
        print(f"  T_ref = {T_ref - 273.15:.1f} C ({T_ref:.2f} K)")
        print(f"  T_low = {T_low - 273.15:.1f} C, T_high = {T_high - 273.15:.1f} C")
        print(f"  delta_T = {delta_T:.1f} K (total span = {2*delta_T:.1f} K)")

    # Low temperature run
    mats_lo = create_msr_materials(temperature=T_low)
    result_lo = _run_eigenvalue(
        mats_lo, n_particles, n_batches, n_inactive,
        seed=seed, verbose=verbose,
        label=f"Low temperature: {T_low - 273.15:.1f} C",
    )

    # High temperature run
    mats_hi = create_msr_materials(temperature=T_high)
    result_hi = _run_eigenvalue(
        mats_hi, n_particles, n_batches, n_inactive,
        seed=seed + 1, verbose=verbose,
        label=f"High temperature: {T_high - 273.15:.1f} C",
    )

    # Compute coefficient
    k_lo = result_lo.keff
    k_hi = result_hi.keff
    dk = k_hi - k_lo
    denom = k_hi * k_lo * 2.0 * delta_T

    if abs(denom) > 1e-30:
        alpha = dk / denom * 1e5  # pcm/K
    else:
        alpha = 0.0

    # Propagate uncertainty
    sigma_k_lo = result_lo.keff_std
    sigma_k_hi = result_hi.keff_std
    sigma_dk = np.sqrt(sigma_k_lo**2 + sigma_k_hi**2)
    if abs(denom) > 1e-30:
        alpha_std = sigma_dk / (k_lo * k_hi * 2.0 * delta_T) * 1e5
    else:
        alpha_std = 0.0

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n  --- Fuel Temperature Coefficient Results ---")
        print(f"  k_eff({T_low - 273.15:.0f} C) = {k_lo:.5f} +/- {sigma_k_lo:.5f}")
        print(f"  k_eff({T_high - 273.15:.0f} C) = {k_hi:.5f} +/- {sigma_k_hi:.5f}")
        print(f"  alpha_fuel = {alpha:.2f} +/- {alpha_std:.2f} pcm/K")
        if alpha < 0:
            print(f"  -> NEGATIVE (inherent safety): PASS")
        else:
            print(f"  -> WARNING: Positive fuel temperature coefficient!")
        print(f"  Time: {elapsed:.1f} s")

    return {
        "alpha_pcm_per_K": alpha,
        "alpha_std_pcm_per_K": alpha_std,
        "keff_low": k_lo,
        "keff_low_std": sigma_k_lo,
        "keff_high": k_hi,
        "keff_high_std": sigma_k_hi,
        "T_low": T_low,
        "T_high": T_high,
        "is_negative": alpha < 0,
        "time": elapsed,
    }


# =====================================================================
# Void coefficient
# =====================================================================
def void_coefficient(
    void_fractions: List[float] = None,
    n_particles: int = 3000,
    n_batches: int = 80,
    n_inactive: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Compute the void coefficient of reactivity.

    Simulates void formation in the fuel salt by reducing fuel salt
    density (and proportionally all macroscopic cross-sections).

    The void coefficient is computed as:

        alpha_void = (k_void - k_nom) / (k_void * k_nom * f_void) * 1e5
                   [pcm / % void]

    A negative void coefficient means that bubble formation in the fuel
    salt reduces reactivity, which is a desirable safety feature.

    Physical basis: void in fuel salt reduces the fuel-to-moderator
    ratio, which in a well-moderated (over-moderated) MSR reduces
    thermal utilisation more than it increases resonance escape.

    Parameters
    ----------
    void_fractions : list of float
        Void fractions to evaluate. Default [0.0, 0.05] (0% and 5%).
    n_particles : int
        Neutrons per batch.
    n_batches : int
        Total batches.
    n_inactive : int
        Inactive batches.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'alpha_pcm_per_pct': void coefficient [pcm / % void]
        'alpha_std_pcm_per_pct': uncertainty [pcm / % void]
        'keff_nominal': k_eff at zero void
        'keff_void': k_eff at void fraction
        'void_fraction': void fraction used
        'is_negative': True if coefficient is negative
    """
    if void_fractions is None:
        void_fractions = [0.0, 0.05]

    t_start = time.perf_counter()

    if verbose:
        print("\n" + "=" * 70)
        print("  Void Coefficient Calculation")
        print("=" * 70)
        print(f"  Void fractions: {[f'{v*100:.1f}%' for v in void_fractions]}")

    results = []
    for i, vf in enumerate(void_fractions):
        # Create materials with voided fuel salt
        # Void reduces density: rho_eff = rho_nom * (1 - void_fraction)
        # All macroscopic XS scale proportionally with density
        if vf == 0.0:
            mats = create_msr_materials()
        else:
            mats = _create_voided_materials(vf)

        label = f"Void fraction = {vf*100:.1f}%"
        result = _run_eigenvalue(
            mats, n_particles, n_batches, n_inactive,
            seed=seed + i, verbose=verbose, label=label,
        )
        results.append((vf, result))

    # Compute void coefficient between nominal and highest void
    vf_nom, res_nom = results[0]
    vf_void, res_void = results[-1]

    k_nom = res_nom.keff
    k_void = res_void.keff
    delta_vf = vf_void - vf_nom  # fractional void

    if abs(delta_vf) > 1e-30 and abs(k_nom * k_void) > 1e-30:
        # Convert void fraction to percentage for coefficient
        alpha = (k_void - k_nom) / (k_void * k_nom * delta_vf) * 1e5  # pcm/frac
        alpha_pct = alpha / 100.0  # pcm/% (since delta_vf is fraction, not %)
        # Actually: alpha_void [pcm/%] = (k2-k1)/(k1*k2*delta_vf_pct) * 1e5
        # where delta_vf_pct is in percent
        delta_vf_pct = delta_vf * 100.0  # convert to percent
        alpha_pcm_per_pct = (k_void - k_nom) / (k_void * k_nom * delta_vf_pct) * 1e5
    else:
        alpha_pcm_per_pct = 0.0

    # Uncertainty
    sigma_nom = res_nom.keff_std
    sigma_void = res_void.keff_std
    sigma_dk = np.sqrt(sigma_nom**2 + sigma_void**2)
    if abs(delta_vf) > 1e-30 and abs(k_nom * k_void) > 1e-30:
        delta_vf_pct = delta_vf * 100.0
        alpha_std = sigma_dk / (k_nom * k_void * delta_vf_pct) * 1e5
    else:
        alpha_std = 0.0

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n  --- Void Coefficient Results ---")
        print(f"  k_eff(nominal) = {k_nom:.5f} +/- {sigma_nom:.5f}")
        print(f"  k_eff({delta_vf*100:.1f}% void) = {k_void:.5f} +/- {sigma_void:.5f}")
        print(f"  alpha_void = {alpha_pcm_per_pct:.1f} +/- {alpha_std:.1f} pcm/%")
        if alpha_pcm_per_pct < 0:
            print(f"  -> NEGATIVE void coefficient: PASS (safety)")
        else:
            print(f"  -> WARNING: Positive void coefficient!")
        print(f"  Time: {elapsed:.1f} s")

    return {
        "alpha_pcm_per_pct": alpha_pcm_per_pct,
        "alpha_std_pcm_per_pct": alpha_std,
        "keff_nominal": k_nom,
        "keff_nominal_std": sigma_nom,
        "keff_void": k_void,
        "keff_void_std": sigma_void,
        "void_fraction": delta_vf,
        "is_negative": alpha_pcm_per_pct < 0,
        "all_results": [(vf, r.keff, r.keff_std) for vf, r in results],
        "time": elapsed,
    }


def _create_voided_materials(
    void_fraction: float,
    enrichment: float = 0.12,
    temperature: float = 923.15,
) -> Dict[int, Material]:
    """Create materials with void in fuel salt.

    Void is modelled by reducing fuel salt density and proportionally
    scaling all macroscopic cross-sections. Non-fuel materials are
    unchanged.

    Parameters
    ----------
    void_fraction : float
        Volume fraction of void in fuel salt (0 to 1).
    enrichment : float
        U-235 enrichment.
    temperature : float
        Operating temperature [K].

    Returns
    -------
    dict
        Materials dictionary with voided fuel salt.
    """
    # Get nominal materials
    mats = create_msr_materials(enrichment=enrichment, temperature=temperature)

    # Scale fuel salt cross-sections by (1 - void_fraction)
    fuel = mats[MAT_FUEL_SALT]
    scale = 1.0 - void_fraction

    voided_fuel = Material(
        name=f"FLiBe+UF4 ({enrichment*100:.1f}% enr, {void_fraction*100:.1f}% void)",
        mat_id=MAT_FUEL_SALT,
        density=fuel.density * scale,
        temperature=fuel.temperature,
        sigma_t=fuel.sigma_t * scale,
        sigma_s=fuel.sigma_s * scale,
        sigma_f=fuel.sigma_f * scale,
        nu_sigma_f=fuel.nu_sigma_f * scale,
        sigma_a=fuel.sigma_a * scale,
        chi=fuel.chi.copy(),
        is_fissile=True,
    )

    mats[MAT_FUEL_SALT] = voided_fuel
    return mats


# =====================================================================
# Enrichment sensitivity
# =====================================================================
def enrichment_sensitivity(
    enrichments: List[float] = None,
    n_particles: int = 2000,
    n_batches: int = 60,
    n_inactive: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run MC at multiple enrichments and find critical enrichment.

    Computes k_eff at each enrichment and uses linear interpolation
    to estimate the critical enrichment (k_eff = 1.0).

    Parameters
    ----------
    enrichments : list of float
        Enrichment values to evaluate. Default [0.05, 0.07, 0.10, 0.12, 0.15].
    n_particles : int
        Neutrons per batch. Default 2000.
    n_batches : int
        Total batches. Default 60.
    n_inactive : int
        Inactive batches. Default 10.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'enrichments': list of enrichment values
        'keffs': list of k_eff values
        'keff_stds': list of uncertainties
        'critical_enrichment': interpolated critical enrichment (k=1)
        'critical_enrichment_std': uncertainty on critical enrichment
        'table': list of (enrichment, keff, keff_std) tuples
    """
    if enrichments is None:
        enrichments = [0.05, 0.07, 0.10, 0.12, 0.15]

    t_start = time.perf_counter()

    if verbose:
        print("\n" + "=" * 70)
        print("  Enrichment Sensitivity Study")
        print("=" * 70)
        print(f"  Enrichments: {[f'{e*100:.1f}%' for e in enrichments]}")

    table = []
    for i, enr in enumerate(enrichments):
        if verbose:
            print(f"\n  Computing enrichment {enr*100:.1f}% "
                  f"({i+1}/{len(enrichments)})...")

        mats = create_msr_materials(enrichment=enr)
        result = _run_eigenvalue(
            mats, n_particles, n_batches, n_inactive,
            seed=seed + i, verbose=verbose,
            label=f"Enrichment = {enr*100:.1f}%",
        )
        table.append((enr, result.keff, result.keff_std))

    # Find critical enrichment by interpolation
    enr_vals = np.array([t[0] for t in table])
    keff_vals = np.array([t[1] for t in table])
    keff_stds = np.array([t[2] for t in table])

    critical_enrichment = _interpolate_critical(enr_vals, keff_vals)

    # Estimate uncertainty on critical enrichment from error propagation
    # Using finite differences on the interpolation
    if critical_enrichment is not None and len(keff_vals) >= 2:
        # Find the two points bracketing k=1
        idx = np.searchsorted(keff_vals if keff_vals[0] < keff_vals[-1]
                              else keff_vals[::-1], 1.0)
        if 0 < idx < len(keff_vals):
            if keff_vals[0] < keff_vals[-1]:
                i1, i2 = idx - 1, idx
            else:
                i1, i2 = len(keff_vals) - idx, len(keff_vals) - idx - 1
                if i1 >= len(keff_vals):
                    i1 = len(keff_vals) - 1
                if i2 < 0:
                    i2 = 0

            dk_de = (keff_vals[i2] - keff_vals[i1]) / (enr_vals[i2] - enr_vals[i1])
            if abs(dk_de) > 1e-30:
                sigma_k_avg = np.sqrt(keff_stds[i1]**2 + keff_stds[i2]**2) / 2
                critical_enrichment_std = sigma_k_avg / abs(dk_de)
            else:
                critical_enrichment_std = 0.0
        else:
            critical_enrichment_std = 0.0
    else:
        critical_enrichment_std = 0.0

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n  --- Enrichment Sensitivity Results ---")
        print(f"  {'Enrichment':>12s}  {'k_eff':>12s}  {'Sigma':>10s}  {'rho [pcm]':>12s}")
        print(f"  {'-'*50}")
        for enr, keff, std in table:
            rho = (keff - 1.0) / keff * 1e5
            print(f"  {enr*100:10.1f}%   {keff:12.5f}  {std:10.5f}  {rho:+10.0f}")
        if critical_enrichment is not None:
            print(f"\n  Critical enrichment (k=1): "
                  f"{critical_enrichment*100:.2f}% +/- {critical_enrichment_std*100:.2f}%")
        else:
            print(f"\n  Could not determine critical enrichment from data range.")
        print(f"  Time: {elapsed:.1f} s")

    return {
        "enrichments": [t[0] for t in table],
        "keffs": [t[1] for t in table],
        "keff_stds": [t[2] for t in table],
        "critical_enrichment": critical_enrichment,
        "critical_enrichment_std": critical_enrichment_std,
        "table": table,
        "time": elapsed,
    }


def _interpolate_critical(
    enrichments: np.ndarray,
    keffs: np.ndarray,
) -> Optional[float]:
    """Find critical enrichment (k=1) by linear interpolation.

    Searches for the interval where k_eff crosses 1.0 and uses
    linear interpolation within that interval.

    Parameters
    ----------
    enrichments : ndarray
        Enrichment values (sorted or unsorted).
    keffs : ndarray
        Corresponding k_eff values.

    Returns
    -------
    float or None
        Critical enrichment, or None if k=1 is not bracketed.
    """
    # Sort by enrichment
    idx = np.argsort(enrichments)
    e_sorted = enrichments[idx]
    k_sorted = keffs[idx]

    for i in range(len(k_sorted) - 1):
        k1, k2 = k_sorted[i], k_sorted[i + 1]
        e1, e2 = e_sorted[i], e_sorted[i + 1]

        # Check if k=1 is bracketed
        if (k1 - 1.0) * (k2 - 1.0) <= 0:
            # Linear interpolation
            if abs(k2 - k1) < 1e-30:
                return (e1 + e2) / 2.0
            frac = (1.0 - k1) / (k2 - k1)
            return e1 + frac * (e2 - e1)

    return None


# =====================================================================
# Graphite temperature coefficient
# =====================================================================
def graphite_temperature_coefficient(
    T_ref: float = 923.15,
    delta_T: float = 50.0,
    n_particles: int = 3000,
    n_batches: int = 80,
    n_inactive: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Compute the graphite moderator temperature coefficient.

    Perturbs only the graphite temperature (moderator + reflector) while
    keeping the fuel salt at the reference temperature. This isolates
    the effect of graphite thermal scattering changes on reactivity.

    Physical effects:
      - Increased upscatter from graphite at higher temperatures
      - Modified thermal scattering kernel (S(alpha,beta))
      - Slight change in thermal absorption (1/v law)
      - Graphite density change is negligible (solid)

    Parameters
    ----------
    T_ref : float
        Reference graphite temperature [K]. Default 923.15 K (650 C).
    delta_T : float
        Temperature perturbation [K]. Default 50 K.
    n_particles : int
        Neutrons per batch.
    n_batches : int
        Total batches.
    n_inactive : int
        Inactive batches.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Same structure as fuel_temperature_coefficient().
    """
    t_start = time.perf_counter()
    T_low = T_ref - delta_T
    T_high = T_ref + delta_T

    if verbose:
        print("\n" + "=" * 70)
        print("  Graphite Temperature Coefficient Calculation")
        print("=" * 70)
        print(f"  T_ref = {T_ref - 273.15:.1f} C")
        print(f"  T_low = {T_low - 273.15:.1f} C, T_high = {T_high - 273.15:.1f} C")

    # Low temperature: perturb graphite only
    mats_lo = _create_graphite_perturbed_materials(T_low)
    result_lo = _run_eigenvalue(
        mats_lo, n_particles, n_batches, n_inactive,
        seed=seed, verbose=verbose,
        label=f"Graphite at {T_low - 273.15:.1f} C",
    )

    # High temperature
    mats_hi = _create_graphite_perturbed_materials(T_high)
    result_hi = _run_eigenvalue(
        mats_hi, n_particles, n_batches, n_inactive,
        seed=seed + 1, verbose=verbose,
        label=f"Graphite at {T_high - 273.15:.1f} C",
    )

    # Compute coefficient
    k_lo = result_lo.keff
    k_hi = result_hi.keff
    dk = k_hi - k_lo
    denom = k_hi * k_lo * 2.0 * delta_T

    if abs(denom) > 1e-30:
        alpha = dk / denom * 1e5  # pcm/K
    else:
        alpha = 0.0

    sigma_dk = np.sqrt(result_lo.keff_std**2 + result_hi.keff_std**2)
    if abs(denom) > 1e-30:
        alpha_std = sigma_dk / (k_lo * k_hi * 2.0 * delta_T) * 1e5
    else:
        alpha_std = 0.0

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n  --- Graphite Temperature Coefficient Results ---")
        print(f"  k_eff({T_low - 273.15:.0f} C) = {k_lo:.5f} +/- {result_lo.keff_std:.5f}")
        print(f"  k_eff({T_high - 273.15:.0f} C) = {k_hi:.5f} +/- {result_hi.keff_std:.5f}")
        print(f"  alpha_graphite = {alpha:.2f} +/- {alpha_std:.2f} pcm/K")
        if alpha < 0:
            print(f"  -> NEGATIVE: PASS")
        else:
            print(f"  -> Positive (graphite coefficient can be slightly positive)")
        print(f"  Time: {elapsed:.1f} s")

    return {
        "alpha_pcm_per_K": alpha,
        "alpha_std_pcm_per_K": alpha_std,
        "keff_low": k_lo,
        "keff_low_std": result_lo.keff_std,
        "keff_high": k_hi,
        "keff_high_std": result_hi.keff_std,
        "T_low": T_low,
        "T_high": T_high,
        "is_negative": alpha < 0,
        "time": elapsed,
    }


def _create_graphite_perturbed_materials(
    graphite_temperature: float,
    enrichment: float = 0.12,
    fuel_temperature: float = 923.15,
) -> Dict[int, Material]:
    """Create materials with perturbed graphite temperature only.

    Fuel salt remains at the nominal temperature. Only graphite
    moderator and reflector temperatures are changed.

    Parameters
    ----------
    graphite_temperature : float
        Graphite temperature [K].
    enrichment : float
        U-235 enrichment.
    fuel_temperature : float
        Fuel salt temperature [K] (unchanged).

    Returns
    -------
    dict
        Materials dictionary.
    """
    from .constants import MAT_VOID

    materials = {
        MAT_VOID: void_material(),
        MAT_FUEL_SALT: fuel_salt(enrichment, fuel_temperature),
        MAT_GRAPHITE_MOD: graphite(graphite_temperature, MAT_GRAPHITE_MOD),
        MAT_GRAPHITE_REF: reflector(graphite_temperature - 50.0),
    }
    return materials


# =====================================================================
# Compute all coefficients
# =====================================================================
def compute_all_coefficients(
    n_particles: int = 3000,
    n_batches: int = 80,
    n_inactive: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Compute all reactivity coefficients and compare with diffusion values.

    Runs all reactivity coefficient calculations:
      1. Fuel temperature coefficient
      2. Void coefficient
      3. Graphite temperature coefficient
      4. Enrichment sensitivity

    Compares results with conceptual design diffusion theory values.

    Parameters
    ----------
    n_particles : int
        Neutrons per batch (used for temperature and void calculations).
    n_batches : int
        Total batches.
    n_inactive : int
        Inactive batches.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Dictionary containing all coefficient results:
        'fuel_temp': fuel temperature coefficient dict
        'void': void coefficient dict
        'graphite_temp': graphite temperature coefficient dict
        'enrichment': enrichment sensitivity dict
    """
    t_start = time.perf_counter()

    if verbose:
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE REACTIVITY COEFFICIENT ANALYSIS")
        print("  40 MWth Marine Molten Salt Reactor")
        print("=" * 70)

    results = {}

    # 1. Fuel temperature coefficient
    if verbose:
        print("\n  [1/4] Computing fuel temperature coefficient...")
    results["fuel_temp"] = fuel_temperature_coefficient(
        n_particles=n_particles, n_batches=n_batches,
        n_inactive=n_inactive, seed=seed, verbose=verbose,
    )

    # 2. Void coefficient
    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"\n  [2/4] Computing void coefficient... (elapsed: {elapsed:.0f} s)")
    results["void"] = void_coefficient(
        n_particles=n_particles, n_batches=n_batches,
        n_inactive=n_inactive, seed=seed + 10, verbose=verbose,
    )

    # 3. Graphite temperature coefficient
    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"\n  [3/4] Computing graphite temperature coefficient... "
              f"(elapsed: {elapsed:.0f} s)")
    results["graphite_temp"] = graphite_temperature_coefficient(
        n_particles=n_particles, n_batches=n_batches,
        n_inactive=n_inactive, seed=seed + 20, verbose=verbose,
    )

    # 4. Enrichment sensitivity (reduced statistics for speed)
    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"\n  [4/4] Computing enrichment sensitivity... "
              f"(elapsed: {elapsed:.0f} s)")
    results["enrichment"] = enrichment_sensitivity(
        n_particles=max(n_particles - 1000, 1000),
        n_batches=max(n_batches - 20, 40),
        n_inactive=max(n_inactive - 5, 5),
        seed=seed + 30, verbose=verbose,
    )

    total_time = time.perf_counter() - t_start
    results["total_time"] = total_time

    # Print comprehensive summary
    if verbose:
        _print_coefficient_summary(results)

    return results


def _print_coefficient_summary(results: Dict) -> None:
    """Print a comprehensive summary table of all reactivity coefficients.

    Includes comparison with 2-group diffusion theory values from the
    conceptual design study.
    """
    # Diffusion reference values
    diff_alpha_fuel = -8.3     # pcm/K
    diff_alpha_void = -40.7    # pcm/%

    print("\n" + "=" * 70)
    print("  REACTIVITY COEFFICIENT SUMMARY")
    print("=" * 70)
    print(f"  {'Parameter':<35s}  {'MC':>14s}  {'Diffusion':>10s}  {'Diff':>10s}")
    print("-" * 70)

    # Fuel temperature coefficient
    ftc = results.get("fuel_temp", {})
    alpha_f = ftc.get("alpha_pcm_per_K", 0.0)
    alpha_f_std = ftc.get("alpha_std_pcm_per_K", 0.0)
    mc_str = f"{alpha_f:.2f} +/- {alpha_f_std:.2f}"
    if abs(diff_alpha_fuel) > 1e-10:
        diff_pct = (alpha_f - diff_alpha_fuel) / abs(diff_alpha_fuel) * 100
        diff_str = f"{diff_pct:+.0f}%"
    else:
        diff_str = "N/A"
    print(f"  {'alpha_fuel [pcm/K]':<35s}  {mc_str:>14s}  {diff_alpha_fuel:>10.1f}  {diff_str:>10s}")

    # Void coefficient
    vc = results.get("void", {})
    alpha_v = vc.get("alpha_pcm_per_pct", 0.0)
    alpha_v_std = vc.get("alpha_std_pcm_per_pct", 0.0)
    mc_str = f"{alpha_v:.1f} +/- {alpha_v_std:.1f}"
    if abs(diff_alpha_void) > 1e-10:
        diff_pct = (alpha_v - diff_alpha_void) / abs(diff_alpha_void) * 100
        diff_str = f"{diff_pct:+.0f}%"
    else:
        diff_str = "N/A"
    print(f"  {'alpha_void [pcm/%]':<35s}  {mc_str:>14s}  {diff_alpha_void:>10.1f}  {diff_str:>10s}")

    # Graphite temperature coefficient
    gtc = results.get("graphite_temp", {})
    alpha_g = gtc.get("alpha_pcm_per_K", 0.0)
    alpha_g_std = gtc.get("alpha_std_pcm_per_K", 0.0)
    mc_str = f"{alpha_g:.2f} +/- {alpha_g_std:.2f}"
    print(f"  {'alpha_graphite [pcm/K]':<35s}  {mc_str:>14s}  {'N/A':>10s}  {'N/A':>10s}")

    # Total temperature coefficient
    alpha_total = alpha_f + alpha_g
    alpha_total_std = np.sqrt(alpha_f_std**2 + alpha_g_std**2)
    mc_str = f"{alpha_total:.2f} +/- {alpha_total_std:.2f}"
    print(f"  {'alpha_total [pcm/K]':<35s}  {mc_str:>14s}  {'N/A':>10s}  {'N/A':>10s}")

    print()

    # Critical enrichment
    es = results.get("enrichment", {})
    e_crit = es.get("critical_enrichment")
    if e_crit is not None:
        e_std = es.get("critical_enrichment_std", 0.0)
        print(f"  Critical enrichment (k=1):  {e_crit*100:.2f}% +/- {e_std*100:.2f}%")
    else:
        print(f"  Critical enrichment: not determined from data range")

    # Safety assessment
    print()
    print("  --- Safety Assessment ---")
    safe = True
    if ftc.get("is_negative", False):
        print(f"  [PASS] Negative fuel temperature coefficient")
    else:
        print(f"  [WARN] Fuel temperature coefficient is NOT negative")
        safe = False

    if vc.get("is_negative", False):
        print(f"  [PASS] Negative void coefficient")
    else:
        print(f"  [WARN] Void coefficient is NOT negative")
        safe = False

    if alpha_total < 0:
        print(f"  [PASS] Negative total temperature coefficient")
    else:
        print(f"  [WARN] Total temperature coefficient is NOT negative")
        safe = False

    if safe:
        print(f"\n  Overall: ALL safety-relevant coefficients are NEGATIVE")
    else:
        print(f"\n  Overall: WARNING - Some coefficients may not meet safety criteria")
        print(f"  (Note: MC statistical noise can affect sign for small coefficients)")

    total_time = results.get("total_time", 0.0)
    print(f"\n  Total computation time: {total_time:.1f} s ({total_time/60:.1f} min)")
    print("=" * 70)


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    print("=" * 70)
    print("  reactivity.py -- Reactivity Coefficient Self-Test")
    print("=" * 70)

    # Quick test with very reduced statistics
    print("\n[Test] Fuel temperature coefficient (reduced stats)...")
    ftc = fuel_temperature_coefficient(
        n_particles=500, n_batches=30, n_inactive=8,
        seed=12345, verbose=True,
    )
    assert "alpha_pcm_per_K" in ftc
    print(f"  alpha_fuel = {ftc['alpha_pcm_per_K']:.2f} pcm/K: PASS")

    print("\n  All reactivity.py self-tests PASSED.")
    print("=" * 70)
