"""
Post-Processing and Analysis for Monte Carlo Eigenvalue Results
===============================================================

Provides comprehensive analysis of k-eigenvalue Monte Carlo results
from the 40 MWth Marine MSR simulation.  Includes:

  - Comparison with 1-group diffusion theory results
  - Spectral analysis (fast/thermal flux ratios, disadvantage factors)
  - Power distribution analysis (peaking, profiles)
  - Four-factor formula extraction from MC tallies
  - Reaction rate and neutron balance analysis
  - Report generation

All functions accept an ``EigenvalueResult`` object (from eigenvalue.py)
and the materials dictionary as their primary inputs.

Usage
-----
>>> from mc_neutronics.eigenvalue import quick_eigenvalue
>>> from mc_neutronics.materials import create_msr_materials
>>> from mc_neutronics.analysis import generate_report
>>>
>>> result = quick_eigenvalue()
>>> mats = create_msr_materials()
>>> generate_report(result, mats, filename='mc_results.txt')
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .constants import (
    N_GROUPS, CHI, NU, ENERGY_PER_FISSION, ENERGY_PER_FISSION_EV,
    GROUP_NAMES, BETA_TOTAL,
    MAT_FUEL_SALT, MAT_GRAPHITE_MOD, MAT_GRAPHITE_REF, MAT_VOID,
    MATERIAL_NAMES,
)
from .materials import Material, create_msr_materials, check_criticality_estimate

if TYPE_CHECKING:
    from .eigenvalue import EigenvalueResult


# =====================================================================
# Diffusion comparison
# =====================================================================
def compare_with_diffusion(
    mc_result,
    diffusion_keff: float = 1.1457,
    diffusion_leakage: float = 0.025,
    diffusion_axial_peaking: float = 1.48,
    diffusion_radial_peaking: float = 1.40,
) -> Dict:
    """Compare Monte Carlo results with 1-group diffusion theory predictions.

    The diffusion values are from the conceptual design study using
    2-group diffusion theory in cylindrical (R, Z) geometry.  Expected
    differences arise because Monte Carlo captures:
      - True angular flux (no P1/diffusion approximation)
      - Energy self-shielding and resonance effects
      - Exact geometry (hex lattice vs. homogenised cylinder)
      - Streaming in channels

    Parameters
    ----------
    mc_result : EigenvalueResult
        Results from the MC eigenvalue calculation.
    diffusion_keff : float
        k_eff from diffusion calculation (default 1.1457).
    diffusion_leakage : float
        Leakage fraction from diffusion.
    diffusion_axial_peaking : float
        Axial power peaking from diffusion.
    diffusion_radial_peaking : float
        Radial power peaking from diffusion.

    Returns
    -------
    dict
        Comparison results with keys: 'keff_mc', 'keff_diff',
        'keff_delta', 'keff_delta_pcm', etc.
    """
    dk = mc_result.keff - diffusion_keff
    dk_pcm = dk * 1e5

    comparison = {
        "keff_mc": mc_result.keff,
        "keff_mc_std": mc_result.keff_std,
        "keff_diffusion": diffusion_keff,
        "keff_delta": dk,
        "keff_delta_pcm": dk_pcm,
        "leakage_mc": mc_result.leakage_fraction,
        "leakage_diffusion": diffusion_leakage,
        "axial_peaking_mc": mc_result.axial_peaking,
        "axial_peaking_diffusion": diffusion_axial_peaking,
        "radial_peaking_mc": mc_result.radial_peaking,
        "radial_peaking_diffusion": diffusion_radial_peaking,
    }

    # Print formatted comparison
    print()
    print("=" * 70)
    print("  Monte Carlo vs. Diffusion Theory Comparison")
    print("=" * 70)
    print(f"  {'Parameter':30s}  {'MC':>12s}  {'Diffusion':>12s}  {'Delta':>12s}")
    print("-" * 70)

    # keff
    mc_str = f"{mc_result.keff:.5f} +/- {mc_result.keff_std:.5f}"
    print(f"  {'k_eff':30s}  {mc_str:>25s}  {diffusion_keff:12.5f}  "
          f"{dk_pcm:+9.0f} pcm")

    # Leakage
    dl = mc_result.leakage_fraction - diffusion_leakage
    print(f"  {'Leakage fraction':30s}  {mc_result.leakage_fraction:12.4f}  "
          f"{diffusion_leakage:12.4f}  {dl:+12.4f}")

    # Peaking
    dap = mc_result.axial_peaking - diffusion_axial_peaking
    drp = mc_result.radial_peaking - diffusion_radial_peaking
    print(f"  {'Axial peaking':30s}  {mc_result.axial_peaking:12.3f}  "
          f"{diffusion_axial_peaking:12.3f}  {dap:+12.3f}")
    print(f"  {'Radial peaking':30s}  {mc_result.radial_peaking:12.3f}  "
          f"{diffusion_radial_peaking:12.3f}  {drp:+12.3f}")

    print()
    print("  Expected differences:")
    print("    - MC k_eff typically 200-500 pcm lower than diffusion due to")
    print("      transport corrections (streaming, angular flux effects)")
    print("    - MC leakage typically higher (diffusion underestimates leakage)")
    print("    - MC peaking factors closer to reality (no cosine/Bessel assumption)")
    print("=" * 70)

    return comparison


# =====================================================================
# Spectral analysis
# =====================================================================
def print_flux_spectrum(
    mc_result,
    materials: Optional[Dict[int, Material]] = None,
) -> Dict:
    """Analyse the fast/thermal flux spectrum from MC results.

    Extracts the spatial distribution of the fast-to-thermal flux ratio,
    the thermal utilisation factor, and spectral indices across the
    core, moderator, and reflector regions.

    Parameters
    ----------
    mc_result : EigenvalueResult
        MC eigenvalue results.
    materials : dict, optional
        Material definitions. If None, creates default materials.

    Returns
    -------
    dict
        Spectral analysis data.
    """
    if materials is None:
        materials = create_msr_materials()

    flux_data = mc_result.flux_data
    flux_mean = flux_data["mean"]  # [nr, nz, ng]
    r_edges = mc_result.mesh_r_edges
    z_edges = mc_result.mesh_z_edges
    nr = len(r_edges) - 1
    nz = len(z_edges) - 1

    # Compute r and z midpoints
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    # Spatially averaged fast and thermal flux
    fast_flux_total = np.sum(flux_mean[:, :, 0])
    therm_flux_total = np.sum(flux_mean[:, :, 1])

    if therm_flux_total > 0:
        global_ft_ratio = fast_flux_total / therm_flux_total
    else:
        global_ft_ratio = float("inf")

    # Regional flux ratios (approximate from mesh geometry)
    # Need to know which bins correspond to fuel/graphite/reflector
    # Use the geometry to classify bins by dominant material
    from .geometry import MSRGeometry
    geom = MSRGeometry()

    # For each (r, z) bin, determine dominant material
    fuel_flux = np.zeros(N_GROUPS)
    mod_flux = np.zeros(N_GROUPS)
    ref_flux = np.zeros(N_GROUPS)

    for ir in range(nr):
        r = r_mid[ir]
        for iz in range(nz):
            z = z_mid[iz]
            # Classify bin by radial position
            if r < geom.core_radius:
                # In core: split between fuel and moderator by volume fraction
                f_frac = geom.fuel_fraction
                g_frac = geom.graphite_fraction
                for g in range(N_GROUPS):
                    fuel_flux[g] += flux_mean[ir, iz, g] * f_frac
                    mod_flux[g] += flux_mean[ir, iz, g] * g_frac
            elif r < geom.outer_radius:
                for g in range(N_GROUPS):
                    ref_flux[g] += flux_mean[ir, iz, g]

    result = {
        "global_fast_thermal_ratio": global_ft_ratio,
        "fuel_fast_flux": fuel_flux[0],
        "fuel_thermal_flux": fuel_flux[1],
        "mod_fast_flux": mod_flux[0],
        "mod_thermal_flux": mod_flux[1],
        "ref_fast_flux": ref_flux[0],
        "ref_thermal_flux": ref_flux[1],
    }

    # Compute ratios
    for region, f_arr in [("fuel", fuel_flux), ("mod", mod_flux),
                           ("ref", ref_flux)]:
        if f_arr[1] > 0:
            result[f"{region}_fast_thermal_ratio"] = f_arr[0] / f_arr[1]
        else:
            result[f"{region}_fast_thermal_ratio"] = float("inf")

    # Thermal utilisation factor: f = Sigma_a_fuel * phi_fuel /
    #                                  (sum over all regions of Sigma_a * phi)
    fuel_mat = materials.get(MAT_FUEL_SALT)
    mod_mat = materials.get(MAT_GRAPHITE_MOD)
    ref_mat = materials.get(MAT_GRAPHITE_REF)

    if fuel_mat and mod_mat and ref_mat:
        # Thermal group (index 1)
        fuel_abs = float(fuel_mat.sigma_a[1]) * fuel_flux[1]
        mod_abs = float(mod_mat.sigma_a[1]) * mod_flux[1]
        ref_abs = float(ref_mat.sigma_a[1]) * ref_flux[1]
        total_abs = fuel_abs + mod_abs + ref_abs

        if total_abs > 0:
            thermal_utilisation = fuel_abs / total_abs
        else:
            thermal_utilisation = 0.0
        result["thermal_utilisation_f"] = thermal_utilisation
    else:
        result["thermal_utilisation_f"] = 0.0

    # Disadvantage factor: phi_thermal_mod / phi_thermal_fuel
    if fuel_flux[1] > 0:
        result["disadvantage_factor"] = mod_flux[1] / fuel_flux[1]
    else:
        result["disadvantage_factor"] = float("inf")

    # Print
    print()
    print("=" * 70)
    print("  Flux Spectrum Analysis")
    print("=" * 70)
    print(f"  Global fast/thermal flux ratio: {global_ft_ratio:.3f}")
    print()
    print(f"  {'Region':20s}  {'Fast flux':>12s}  {'Therm flux':>12s}  {'F/T ratio':>12s}")
    print("-" * 62)
    for region, label in [("fuel", "Fuel salt"),
                           ("mod", "Graphite mod"),
                           ("ref", "Graphite refl")]:
        ff = result[f"{region}_fast_flux"]
        tf = result[f"{region}_thermal_flux"]
        ratio = result[f"{region}_fast_thermal_ratio"]
        if ratio == float("inf"):
            ratio_str = "inf"
        else:
            ratio_str = f"{ratio:.3f}"
        print(f"  {label:20s}  {ff:12.4e}  {tf:12.4e}  {ratio_str:>12s}")

    print()
    print(f"  Thermal utilisation factor (f): {result['thermal_utilisation_f']:.4f}")
    print(f"  Disadvantage factor (mod/fuel): {result['disadvantage_factor']:.3f}")
    print("=" * 70)

    return result


# =====================================================================
# Power distribution analysis
# =====================================================================
def print_power_distribution(mc_result) -> Dict:
    """Analyse and print the power distribution from MC results.

    Computes axial and radial power profiles, peaking factors, and
    compares with analytical cosine/Bessel shapes.

    Parameters
    ----------
    mc_result : EigenvalueResult
        MC eigenvalue results.

    Returns
    -------
    dict
        Power distribution analysis data.
    """
    peaking = mc_result.peaking_factors
    r_edges = mc_result.mesh_r_edges
    z_edges = mc_result.mesh_z_edges

    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    radial_profile = peaking.get("radial_profile", np.array([]))
    axial_profile = peaking.get("axial_profile", np.array([]))

    result = {
        "axial_peaking": peaking.get("axial", 1.0),
        "radial_peaking": peaking.get("radial", 1.0),
        "total_peaking": peaking.get("total", 1.0),
        "radial_profile": radial_profile,
        "axial_profile": axial_profile,
        "r_midpoints": r_mid,
        "z_midpoints": z_mid,
    }

    # Analytical comparison: cosine axial, J0 Bessel radial
    from .geometry import MSRGeometry
    geom = MSRGeometry()
    H = geom.core_height  # cm
    R = geom.core_radius  # cm

    # Analytical cosine axial peaking for bare cylinder
    # P(z) ~ cos(pi*z/H_e) where H_e = H + 2*delta (extrapolation)
    # Bare cylinder peaking = pi/2 ~ 1.571
    # Reflected cylinder peaking ~ 1.2-1.5 depending on reflector savings
    analytical_axial_peaking = np.pi / 2.0  # 1.571 for bare
    result["analytical_axial_peaking_bare"] = analytical_axial_peaking

    # Analytical J0 Bessel radial peaking for bare cylinder
    # P(r) ~ J0(2.405*r/R_e) where R_e = R + delta
    # Peak/avg = 2.405^2 / (2 * J1(2.405) * 2.405) ~ 2.32 for bare
    # With reflector, radial peaking ~ 1.3-1.8
    analytical_radial_peaking = 2.32  # bare cylinder
    result["analytical_radial_peaking_bare"] = analytical_radial_peaking

    # Print
    print()
    print("=" * 70)
    print("  Power Distribution Analysis")
    print("=" * 70)
    print(f"  {'Peaking factor':30s}  {'MC':>10s}  {'Bare cyl.':>10s}")
    print("-" * 55)
    print(f"  {'Axial':30s}  {peaking.get('axial', 0):10.3f}  "
          f"{analytical_axial_peaking:10.3f}")
    print(f"  {'Radial':30s}  {peaking.get('radial', 0):10.3f}  "
          f"{analytical_radial_peaking:10.3f}")
    print(f"  {'Total (3D)':30s}  {peaking.get('total', 0):10.3f}  "
          f"{analytical_axial_peaking * analytical_radial_peaking:10.3f}")
    print()
    print("  Note: Bare cylinder values are upper bounds.  The graphite")
    print("  reflector flattens both axial and radial profiles.")

    # Print axial profile
    if len(axial_profile) > 0 and np.max(axial_profile) > 0:
        axial_norm = axial_profile / np.max(axial_profile)
        print()
        print("  Axial Power Profile (normalised to peak):")
        n_print = min(len(z_mid), 15)
        step = max(1, len(z_mid) // n_print)
        print(f"    {'z [cm]':>10s}  {'P(z)/P_max':>10s}  {'bar':30s}")
        for i in range(0, len(z_mid), step):
            bar = "#" * int(axial_norm[i] * 30)
            print(f"    {z_mid[i]:10.1f}  {axial_norm[i]:10.3f}  {bar}")

    # Print radial profile
    if len(radial_profile) > 0 and np.max(radial_profile) > 0:
        radial_norm = radial_profile / np.max(radial_profile)
        print()
        print("  Radial Power Profile (normalised to peak):")
        n_print = min(len(r_mid), 12)
        step = max(1, len(r_mid) // n_print)
        print(f"    {'r [cm]':>10s}  {'P(r)/P_max':>10s}  {'bar':30s}")
        for i in range(0, len(r_mid), step):
            bar = "#" * int(radial_norm[i] * 30)
            print(f"    {r_mid[i]:10.1f}  {radial_norm[i]:10.3f}  {bar}")

    print("=" * 70)
    return result


# =====================================================================
# Spectral indices
# =====================================================================
def compute_spectral_indices(
    mc_result,
    materials: Optional[Dict[int, Material]] = None,
) -> Dict:
    """Compute spectral indices from MC flux results.

    Parameters
    ----------
    mc_result : EigenvalueResult
        MC eigenvalue results.
    materials : dict, optional
        Material definitions.

    Returns
    -------
    dict
        Spectral indices:
        - 'thermal_to_fast_ratio': global phi_th / phi_fast
        - 'spectral_hardness': Sigma_f_fast*phi_fast / (Sigma_f_th*phi_th)
        - 'disadvantage_factor': phi_th_graphite / phi_th_fuel (approx)
        - 'mean_eta': eta = nu * Sigma_f / Sigma_a (fuel thermal)
    """
    if materials is None:
        materials = create_msr_materials()

    flux_mean = mc_result.flux_data["mean"]  # [nr, nz, ng]

    # Global thermal-to-fast ratio
    fast_total = np.sum(flux_mean[:, :, 0])
    therm_total = np.sum(flux_mean[:, :, 1])
    if fast_total > 0:
        tf_ratio = therm_total / fast_total
    else:
        tf_ratio = 0.0

    # Spectral hardness: ratio of fast fission to thermal fission rates
    fuel = materials.get(MAT_FUEL_SALT)
    if fuel:
        sigma_f_fast = float(fuel.sigma_f[0])
        sigma_f_therm = float(fuel.sigma_f[1])
        if sigma_f_therm * therm_total > 0:
            spectral_hardness = (
                sigma_f_fast * fast_total
                / (sigma_f_therm * therm_total)
            )
        else:
            spectral_hardness = float("inf")

        # Eta = nu * Sigma_f / Sigma_a (thermal group in fuel)
        if float(fuel.sigma_a[1]) > 0:
            eta_thermal = float(fuel.nu_sigma_f[1]) / float(fuel.sigma_a[1])
        else:
            eta_thermal = 0.0
    else:
        spectral_hardness = 0.0
        eta_thermal = 0.0

    result = {
        "thermal_to_fast_ratio": tf_ratio,
        "spectral_hardness": spectral_hardness,
        "mean_eta_thermal": eta_thermal,
    }

    # Print
    print()
    print("=" * 70)
    print("  Spectral Indices")
    print("=" * 70)
    print(f"  Thermal-to-fast flux ratio:  {tf_ratio:.4f}")
    print(f"  Spectral hardness (Rf/Rth):  {spectral_hardness:.4f}")
    print(f"  Eta (thermal, fuel):         {eta_thermal:.4f}")
    print(f"  (eta = nu*Sigma_f / Sigma_a in fuel thermal group)")
    print("=" * 70)

    return result


# =====================================================================
# Reaction rate analysis
# =====================================================================
def compute_reaction_rates(
    mc_result,
    materials: Optional[Dict[int, Material]] = None,
) -> Dict:
    """Compute reaction rates per material from MC flux.

    Uses the mesh flux and material cross-sections to estimate
    spatially-integrated reaction rates:
        R_x = integral( Sigma_x(r) * phi(r) ) dV

    For the mesh tally, this is approximated as:
        R_x = sum_bins( Sigma_x * phi_bin * V_bin )

    Parameters
    ----------
    mc_result : EigenvalueResult
        MC eigenvalue results.
    materials : dict, optional
        Material definitions.

    Returns
    -------
    dict
        Reaction rates per material and total neutron balance.
    """
    if materials is None:
        materials = create_msr_materials()

    flux_mean = mc_result.flux_data["mean"]  # [nr, nz, ng]
    r_edges = mc_result.mesh_r_edges
    z_edges = mc_result.mesh_z_edges
    nr = len(r_edges) - 1
    nz = len(z_edges) - 1

    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    from .geometry import MSRGeometry
    geom = MSRGeometry()

    # Compute bin volumes
    volumes = np.zeros((nr, nz))
    for ir in range(nr):
        r_lo = r_edges[ir]
        r_hi = r_edges[ir + 1]
        area = np.pi * (r_hi**2 - r_lo**2)
        for iz in range(nz):
            dz = z_edges[iz + 1] - z_edges[iz]
            volumes[ir, iz] = area * dz

    # Classify bins and compute rates
    rates = {}
    for mat_id, mat in materials.items():
        if mat_id == MAT_VOID:
            continue
        label = MATERIAL_NAMES.get(mat_id, f"Mat {mat_id}")
        rates[label] = {
            "fission": np.zeros(N_GROUPS),
            "absorption": np.zeros(N_GROUPS),
            "scattering": np.zeros(N_GROUPS),
            "production": np.zeros(N_GROUPS),
        }

    # Accumulate over mesh bins
    for ir in range(nr):
        r = r_mid[ir]
        for iz in range(nz):
            z = z_mid[iz]
            vol = volumes[ir, iz]

            # Determine material region
            if r < geom.core_radius:
                # Core region: weighted by fuel/graphite fraction
                fuel = materials.get(MAT_FUEL_SALT)
                mod = materials.get(MAT_GRAPHITE_MOD)
                fuel_label = MATERIAL_NAMES[MAT_FUEL_SALT]
                mod_label = MATERIAL_NAMES[MAT_GRAPHITE_MOD]

                for g in range(N_GROUPS):
                    phi = flux_mean[ir, iz, g] * vol  # flux * volume

                    if fuel and fuel_label in rates:
                        f_frac = geom.fuel_fraction
                        rates[fuel_label]["fission"][g] += (
                            float(fuel.sigma_f[g]) * phi * f_frac
                        )
                        rates[fuel_label]["absorption"][g] += (
                            float(fuel.sigma_a[g]) * phi * f_frac
                        )
                        rates[fuel_label]["scattering"][g] += (
                            fuel.total_scatter_from(g) * phi * f_frac
                        )
                        rates[fuel_label]["production"][g] += (
                            float(fuel.nu_sigma_f[g]) * phi * f_frac
                        )

                    if mod and mod_label in rates:
                        g_frac = geom.graphite_fraction
                        rates[mod_label]["absorption"][g] += (
                            float(mod.sigma_a[g]) * phi * g_frac
                        )
                        rates[mod_label]["scattering"][g] += (
                            mod.total_scatter_from(g) * phi * g_frac
                        )

            elif r < geom.outer_radius:
                # Reflector
                ref = materials.get(MAT_GRAPHITE_REF)
                ref_label = MATERIAL_NAMES[MAT_GRAPHITE_REF]
                if ref and ref_label in rates:
                    for g in range(N_GROUPS):
                        phi = flux_mean[ir, iz, g] * vol
                        rates[ref_label]["absorption"][g] += (
                            float(ref.sigma_a[g]) * phi
                        )
                        rates[ref_label]["scattering"][g] += (
                            ref.total_scatter_from(g) * phi
                        )

    # Total neutron balance
    total_production = sum(
        np.sum(r["production"]) for r in rates.values()
    )
    total_absorption = sum(
        np.sum(r["absorption"]) for r in rates.values()
    )
    total_fission = sum(
        np.sum(r["fission"]) for r in rates.values()
    )

    # Leakage from MC
    leakage_frac = mc_result.leakage_fraction

    balance = {
        "total_production": total_production,
        "total_absorption": total_absorption,
        "total_fission": total_fission,
        "leakage_fraction": leakage_frac,
        "rates_by_material": rates,
    }

    # Print
    print()
    print("=" * 70)
    print("  Reaction Rate Analysis")
    print("=" * 70)

    for label, r in rates.items():
        total_abs = np.sum(r["absorption"])
        total_fis = np.sum(r["fission"])
        total_scat = np.sum(r["scattering"])
        total_prod = np.sum(r["production"])
        print(f"\n  {label}:")
        print(f"    {'Reaction':15s}  {'Fast':>12s}  {'Thermal':>12s}  {'Total':>12s}")
        print(f"    {'Absorption':15s}  {r['absorption'][0]:12.4e}  "
              f"{r['absorption'][1]:12.4e}  {total_abs:12.4e}")
        print(f"    {'Fission':15s}  {r['fission'][0]:12.4e}  "
              f"{r['fission'][1]:12.4e}  {total_fis:12.4e}")
        print(f"    {'Scattering':15s}  {r['scattering'][0]:12.4e}  "
              f"{r['scattering'][1]:12.4e}  {total_scat:12.4e}")
        print(f"    {'Production':15s}  {r['production'][0]:12.4e}  "
              f"{r['production'][1]:12.4e}  {total_prod:12.4e}")

    print(f"\n  Neutron Balance:")
    print(f"    Total production: {total_production:.4e}")
    print(f"    Total absorption: {total_absorption:.4e}")
    if total_absorption > 0:
        print(f"    k_inf (prod/abs): {total_production / total_absorption:.5f}")
    print(f"    Leakage fraction: {leakage_frac:.4f}")
    print("=" * 70)

    return balance


# =====================================================================
# Four-factor formula from MC
# =====================================================================
def compute_four_factor_from_mc(
    mc_result,
    materials: Optional[Dict[int, Material]] = None,
) -> Dict:
    """Extract the four-factor formula components from MC results.

    The four-factor formula for an infinite lattice:
        k_inf = eta * f * p * epsilon

    Where:
      - eta: reproduction factor = nu * Sigma_f_fuel / Sigma_a_fuel (thermal)
      - f: thermal utilisation = Sigma_a_fuel * phi_fuel /
                                  (sum_i Sigma_a_i * phi_i)  (thermal)
      - p: resonance escape probability (estimated from fast/thermal balance)
      - epsilon: fast fission factor = total fissions / thermal fissions

    For the actual reactor:
        k_eff = k_inf * P_NL

    where P_NL = 1 - leakage_fraction is the non-leakage probability.

    Parameters
    ----------
    mc_result : EigenvalueResult
        MC eigenvalue results.
    materials : dict, optional
        Material definitions.

    Returns
    -------
    dict
        Four-factor components: eta, f, p, epsilon, k_inf, P_NL, k_eff_4factor.
    """
    if materials is None:
        materials = create_msr_materials()

    flux_mean = mc_result.flux_data["mean"]
    r_edges = mc_result.mesh_r_edges
    z_edges = mc_result.mesh_z_edges
    nr = len(r_edges) - 1
    nz = len(z_edges) - 1
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    from .geometry import MSRGeometry
    geom = MSRGeometry()

    fuel = materials.get(MAT_FUEL_SALT)
    mod = materials.get(MAT_GRAPHITE_MOD)
    ref = materials.get(MAT_GRAPHITE_REF)

    if not fuel:
        print("  ERROR: No fuel material found.")
        return {}

    # Compute volume-weighted fluxes per region per group
    fuel_phi = np.zeros(N_GROUPS)
    mod_phi = np.zeros(N_GROUPS)
    ref_phi = np.zeros(N_GROUPS)

    volumes = np.zeros((nr, nz))
    for ir in range(nr):
        r_lo = r_edges[ir]
        r_hi = r_edges[ir + 1]
        area = np.pi * (r_hi**2 - r_lo**2)
        for iz in range(nz):
            dz = z_edges[iz + 1] - z_edges[iz]
            volumes[ir, iz] = area * dz

    for ir in range(nr):
        r = r_mid[ir]
        for iz in range(nz):
            vol = volumes[ir, iz]
            for g in range(N_GROUPS):
                phi_vol = flux_mean[ir, iz, g] * vol
                if r < geom.core_radius:
                    fuel_phi[g] += phi_vol * geom.fuel_fraction
                    mod_phi[g] += phi_vol * geom.graphite_fraction
                elif r < geom.outer_radius:
                    ref_phi[g] += phi_vol

    # --- Eta: reproduction factor ---
    # eta = nu * Sigma_f_fuel / Sigma_a_fuel (thermal group)
    sigma_a_fuel_th = float(fuel.sigma_a[1])
    nu_sigma_f_fuel_th = float(fuel.nu_sigma_f[1])
    if sigma_a_fuel_th > 0:
        eta = nu_sigma_f_fuel_th / sigma_a_fuel_th
    else:
        eta = 0.0

    # --- f: thermal utilisation ---
    # f = Sigma_a_fuel * phi_fuel / sum(Sigma_a_i * phi_i)  [thermal]
    abs_fuel_th = sigma_a_fuel_th * fuel_phi[1]
    abs_mod_th = float(mod.sigma_a[1]) * mod_phi[1] if mod else 0.0
    abs_ref_th = float(ref.sigma_a[1]) * ref_phi[1] if ref else 0.0
    abs_total_th = abs_fuel_th + abs_mod_th + abs_ref_th

    if abs_total_th > 0:
        f = abs_fuel_th / abs_total_th
    else:
        f = 0.0

    # --- p: resonance escape probability ---
    # Estimated from the balance: neutrons entering thermal group
    # vs. neutrons removed in the fast group.
    # Source to fast group = chi[0] * total_production
    # Removal from fast group = absorption_fast + downscatter_fast
    # p ~ 1 - (fast absorption / fast removal)
    #
    # More practical: p = (thermal absorption rate) / (total absorption rate
    #   that would occur if all slowed to thermal)
    #
    # Simplest estimate: from cross-section ratios
    sigma_a_fuel_fast = float(fuel.sigma_a[0])
    sigma_s_12_fuel = float(fuel.sigma_s[0, 1])  # downscatter
    sigma_rem_fast_fuel = sigma_a_fuel_fast + sigma_s_12_fuel
    if sigma_rem_fast_fuel > 0:
        p_fuel = sigma_s_12_fuel / sigma_rem_fast_fuel
    else:
        p_fuel = 1.0

    # Better estimate using actual flux-weighted rates
    fast_abs_total = (
        sigma_a_fuel_fast * fuel_phi[0]
        + (float(mod.sigma_a[0]) * mod_phi[0] if mod else 0.0)
        + (float(ref.sigma_a[0]) * ref_phi[0] if ref else 0.0)
    )

    total_source = np.sum([
        float(fuel.nu_sigma_f[g]) * fuel_phi[g]
        for g in range(N_GROUPS)
    ])

    # p ~ 1 - (fast absorption / total source)
    if total_source > 0:
        p = 1.0 - fast_abs_total / total_source
    else:
        p = p_fuel
    p = max(0.0, min(1.0, p))  # clamp to [0, 1]

    # --- epsilon: fast fission factor ---
    # epsilon = total fission rate / thermal fission rate
    fission_fast = float(fuel.sigma_f[0]) * fuel_phi[0]
    fission_thermal = float(fuel.sigma_f[1]) * fuel_phi[1]
    total_fission = fission_fast + fission_thermal

    if fission_thermal > 0:
        epsilon = total_fission / fission_thermal
    else:
        epsilon = 1.0

    # --- k_inf and k_eff ---
    k_inf = eta * f * p * epsilon
    P_NL = 1.0 - mc_result.leakage_fraction
    k_eff_4factor = k_inf * P_NL

    result = {
        "eta": eta,
        "f": f,
        "p": p,
        "epsilon": epsilon,
        "k_inf": k_inf,
        "P_NL": P_NL,
        "k_eff_4factor": k_eff_4factor,
        "k_eff_mc": mc_result.keff,
    }

    # Print
    print()
    print("=" * 70)
    print("  Four-Factor Formula Analysis")
    print("=" * 70)
    print(f"  k_inf = eta * f * p * epsilon")
    print(f"        = {eta:.4f} * {f:.4f} * {p:.4f} * {epsilon:.4f}")
    print(f"        = {k_inf:.5f}")
    print()
    print(f"  {'eta (reproduction factor)':35s} = {eta:.4f}")
    print(f"  {'f (thermal utilisation)':35s} = {f:.4f}")
    print(f"  {'p (resonance escape prob)':35s} = {p:.4f}")
    print(f"  {'epsilon (fast fission factor)':35s} = {epsilon:.4f}")
    print()
    print(f"  k_eff = k_inf * P_NL")
    print(f"        = {k_inf:.5f} * {P_NL:.4f}")
    print(f"        = {k_eff_4factor:.5f}")
    print()
    print(f"  MC k_eff    = {mc_result.keff:.5f} +/- {mc_result.keff_std:.5f}")
    print(f"  4-factor    = {k_eff_4factor:.5f}")
    delta = (mc_result.keff - k_eff_4factor) * 1e5
    print(f"  Difference  = {delta:+.0f} pcm")
    print()
    print("  Note: Discrepancy is expected because the 4-factor decomposition")
    print("  uses simplified groupings of MC flux data, while k_eff_MC is the")
    print("  direct stochastic estimate from the power iteration ratio.")
    print("=" * 70)

    return result


# =====================================================================
# Report generation
# =====================================================================
def generate_report(
    mc_result,
    materials: Optional[Dict[int, Material]] = None,
    filename: str = "mc_results.txt",
) -> str:
    """Generate a comprehensive results report and save to file.

    Calls all analysis functions and writes the combined output to
    a text file.

    Parameters
    ----------
    mc_result : EigenvalueResult
        MC eigenvalue results.
    materials : dict, optional
        Material definitions.
    filename : str
        Output filename (default 'mc_results.txt').

    Returns
    -------
    str
        Path to the generated report file.
    """
    if materials is None:
        materials = create_msr_materials()

    lines = []

    def capture(func, *args, **kwargs):
        """Capture printed output from a function."""
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        return buffer.getvalue()

    # Header
    lines.append("=" * 70)
    lines.append("  40 MWth Marine MSR - Monte Carlo Neutronics Report")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append(mc_result.summary())
    lines.append("")

    # Shannon entropy convergence
    lines.append("=" * 70)
    lines.append("  Shannon Entropy Convergence")
    lines.append("=" * 70)
    if len(mc_result.entropy_history) > 0:
        n_inactive = mc_result.n_inactive
        for i, H in enumerate(mc_result.entropy_history):
            status = "active" if i >= n_inactive else "inactive"
            lines.append(f"  Batch {i+1:4d} ({status:8s}): H = {H:.4f}")
        early_H = np.mean(mc_result.entropy_history[:max(5, n_inactive // 2)])
        late_H = np.mean(mc_result.entropy_history[-5:])
        lines.append(f"\n  Early mean H: {early_H:.4f}")
        lines.append(f"  Late mean H:  {late_H:.4f}")
        lines.append(f"  Relative change: "
                     f"{abs(late_H - early_H) / max(early_H, 1e-10) * 100:.1f}%")
    lines.append("")

    # keff history
    lines.append("=" * 70)
    lines.append("  k_eff History")
    lines.append("=" * 70)
    n_inactive = mc_result.n_inactive
    active_keffs = mc_result.keff_history[n_inactive:]
    if len(active_keffs) > 0:
        cum_mean = []
        cum_std = []
        for i in range(1, len(active_keffs) + 1):
            subset = active_keffs[:i]
            cm = np.mean(subset)
            cs = np.std(subset, ddof=1) / np.sqrt(i) if i >= 2 else 0.0
            cum_mean.append(cm)
            cum_std.append(cs)

        for i in range(len(active_keffs)):
            batch_num = n_inactive + i + 1
            lines.append(
                f"  Batch {batch_num:4d}: k = {active_keffs[i]:.5f}, "
                f"cumulative = {cum_mean[i]:.5f} +/- {cum_std[i]:.5f}"
            )
    lines.append("")

    # Diffusion comparison
    lines.append(capture(compare_with_diffusion, mc_result))

    # Flux spectrum
    lines.append(capture(print_flux_spectrum, mc_result, materials))

    # Power distribution
    lines.append(capture(print_power_distribution, mc_result))

    # Spectral indices
    lines.append(capture(compute_spectral_indices, mc_result, materials))

    # Reaction rates
    lines.append(capture(compute_reaction_rates, mc_result, materials))

    # Four-factor formula
    lines.append(capture(compute_four_factor_from_mc, mc_result, materials))

    # Material summary
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Material Cross-Section Summary")
    lines.append("=" * 70)
    for mat_id, mat in sorted(materials.items()):
        if mat_id == MAT_VOID:
            continue
        lines.append(f"\n{mat.summary()}")
        k_inf = check_criticality_estimate(mat)
        if k_inf is not None:
            lines.append(f"  k_inf (2-group diffusion est): {k_inf:.4f}")
    lines.append("")

    # Write to file
    report_text = "\n".join(lines)

    filepath = os.path.abspath(filename)
    with open(filepath, "w") as fh:
        fh.write(report_text)
    print(f"\n  Report saved to: {filepath}")
    print(f"  Report size: {len(report_text):,} characters, "
          f"{report_text.count(chr(10)):,} lines")

    return filepath


# =====================================================================
# Convenience: quick eigenvalue + full analysis
# =====================================================================
def quick_eigenvalue_with_analysis(
    n_particles: int = 2000,
    n_batches: int = 50,
    n_inactive: int = 10,
    seed: int = 42,
    report_file: Optional[str] = None,
):
    """Run a quick eigenvalue calculation and perform full analysis.

    Convenience function that creates geometry, materials, runs the MC
    calculation, and then calls all analysis functions.

    Parameters
    ----------
    n_particles : int
        Neutrons per batch.
    n_batches : int
        Total batches.
    n_inactive : int
        Inactive batches.
    seed : int
        Random seed.
    report_file : str, optional
        If provided, save report to this file.

    Returns
    -------
    EigenvalueResult
        MC calculation results.
    """
    from .eigenvalue import quick_eigenvalue

    materials = create_msr_materials()

    # Run MC
    result = quick_eigenvalue(
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=n_inactive,
        seed=seed,
        verbose=True,
    )

    # Analysis
    print("\n" + "=" * 70)
    print("  POST-PROCESSING ANALYSIS")
    print("=" * 70)

    compare_with_diffusion(result)
    print_flux_spectrum(result, materials)
    print_power_distribution(result)
    compute_spectral_indices(result, materials)
    compute_reaction_rates(result, materials)
    compute_four_factor_from_mc(result, materials)

    if report_file:
        generate_report(result, materials, filename=report_file)

    return result


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  analysis.py -- Analysis Module Self-Test")
    print("=" * 70)

    # Run a quick eigenvalue calculation
    result = quick_eigenvalue_with_analysis(
        n_particles=500,
        n_batches=25,
        n_inactive=5,
        seed=12345,
        report_file=None,
    )

    # Verify key outputs
    print(f"\n--- Verification ---")
    print(f"  k_eff = {result.keff:.5f} +/- {result.keff_std:.5f}")
    assert 0.5 < result.keff < 2.0, f"k_eff out of range: {result.keff}"
    print(f"  k_eff in plausible range: PASS")

    assert result.axial_peaking >= 1.0, "Axial peaking should be >= 1"
    print(f"  Axial peaking >= 1.0: PASS")

    assert result.leakage_fraction >= 0.0, "Leakage should be non-negative"
    print(f"  Leakage non-negative: PASS")

    print(f"\n  All analysis.py self-tests PASSED.")
    print("=" * 70)
