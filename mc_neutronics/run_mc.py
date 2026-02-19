#!/usr/bin/env python3
"""
Master Monte Carlo Neutron Transport Analysis Script
=====================================================

Complete MC nuclear analysis suite for the 40 MWth Marine Molten Salt Reactor.
Orchestrates all computation modules to produce a comprehensive neutronics
characterisation:

  1. Reference eigenvalue calculation (production quality)
  2. Post-processing analysis (flux spectrum, power, four-factor, etc.)
  3. Reactivity coefficient calculations
  4. Enrichment sensitivity study
  5. Report generation

Usage
-----
From the project root directory:

    # Full analysis (production quality, ~30-60 min)
    python3 mc_neutronics/run_mc.py

    # Quick analysis (reduced statistics, ~1-2 min)
    python3 mc_neutronics/run_mc.py --quick

    # Coefficients only (skip reference eigenvalue)
    python3 mc_neutronics/run_mc.py --coefficients

The script automatically handles sys.path so it can be run from any
directory within the project tree.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# sys.path setup: allow running from project root or mc_neutronics/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from mc_neutronics.constants import (
    N_GROUPS, GROUP_NAMES, CHI, NU,
    ENERGY_PER_FISSION, ENERGY_PER_FISSION_EV,
    MAT_FUEL_SALT, MAT_GRAPHITE_MOD, MAT_GRAPHITE_REF,
    MATERIAL_NAMES, BETA_TOTAL, PROMPT_NEUTRON_LIFETIME,
)
from mc_neutronics.materials import (
    create_msr_materials, fuel_salt, check_criticality_estimate,
)
from mc_neutronics.geometry import MSRGeometry
from mc_neutronics.eigenvalue import EigenvalueSolver, EigenvalueResult
from mc_neutronics.analysis import (
    compare_with_diffusion,
    print_flux_spectrum,
    print_power_distribution,
    compute_spectral_indices,
    compute_reaction_rates,
    compute_four_factor_from_mc,
    generate_report,
)
from mc_neutronics.reactivity import (
    fuel_temperature_coefficient,
    void_coefficient,
    graphite_temperature_coefficient,
    enrichment_sensitivity,
    compute_all_coefficients,
)


# =====================================================================
# Banner
# =====================================================================
BANNER = """
================================================================================
  MONTE CARLO NEUTRON TRANSPORT ANALYSIS
  40 MWth Marine Molten Salt Reactor - Basic Design
================================================================================
  Code:           Custom 2-Group MC Neutron Transport
  Energy groups:  2 (Fast > 0.625 eV, Thermal < 0.625 eV)
  Geometry:       3D hexagonal lattice, cylindrical core + reflector
  Fuel salt:      FLiBe + 5 mol% UF4, 12% HALEU
  Moderator:      IG-110 nuclear graphite
  Temperature:    650 C (923.15 K) nominal
================================================================================
"""

DIVIDER = "=" * 80


# =====================================================================
# Design parameters summary
# =====================================================================
def _print_design_parameters():
    """Print a summary of the reactor design parameters."""
    geom = MSRGeometry()
    mats = create_msr_materials()
    fuel = mats[MAT_FUEL_SALT]

    print()
    print(DIVIDER)
    print("  DESIGN PARAMETERS")
    print(DIVIDER)
    print(f"  Thermal power:           40 MWth")
    print(f"  Core radius:             {geom.core_radius:.2f} cm ({geom.core_radius/100:.4f} m)")
    print(f"  Core height:             {geom.core_height:.2f} cm ({geom.core_height/100:.4f} m)")
    print(f"  Core H/D ratio:          {geom.core_height / (2*geom.core_radius):.2f}")
    print(f"  Fuel channels:           {geom.n_channels}")
    print(f"  Channel radius:          {geom.channel_radius:.2f} cm")
    print(f"  Channel pitch:           {geom.channel_pitch:.2f} cm")
    print(f"  Fuel volume fraction:    {geom.fuel_fraction:.4f} ({geom.fuel_fraction*100:.1f}%)")
    print(f"  Reflector thickness:     {geom.reflector_thickness:.2f} cm")
    print(f"  Fuel salt density:       {fuel.density:.1f} kg/m^3")
    print(f"  Fuel salt temperature:   {fuel.temperature - 273.15:.0f} C")
    print(f"  Enrichment:              12.0% U-235 (HALEU)")

    # k-infinity estimates
    k_inf = check_criticality_estimate(fuel)
    if k_inf is not None:
        print(f"  k-inf (2-grp diffusion): {k_inf:.4f}")

    # Volume summary
    fuel_vol = geom.get_fuel_volume()
    mod_vol = geom.get_moderator_volume()
    ref_vol = geom.get_reflector_volume()
    print(f"  Fuel volume:             {fuel_vol:.0f} cm^3 ({fuel_vol/1e6:.4f} m^3)")
    print(f"  Moderator volume:        {mod_vol:.0f} cm^3 ({mod_vol/1e6:.4f} m^3)")
    print(f"  Reflector volume:        {ref_vol:.0f} cm^3 ({ref_vol/1e6:.4f} m^3)")
    print(DIVIDER)


# =====================================================================
# Full production analysis
# =====================================================================
def run_full_analysis(seed: int = 42) -> Dict:
    """Complete MC nuclear analysis for 40 MWth Marine MSR.

    Runs:
      1. Reference eigenvalue (5000 particles, 150 batches, 30 inactive)
      2. Post-processing analysis (flux, power, four-factor, reaction rates)
      3. Reactivity coefficients (fuel temp, void, graphite temp)
      4. Enrichment sensitivity (5 enrichment points)
      5. Report generation

    Parameters
    ----------
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    dict
        All results: 'reference', 'analysis', 'coefficients', 'enrichment'.
    """
    t_start = time.perf_counter()

    print(BANNER)
    _print_design_parameters()

    all_results = {}
    materials = create_msr_materials()

    # =================================================================
    # 1. Reference eigenvalue calculation (production quality)
    # =================================================================
    print()
    print(DIVIDER)
    print("  PHASE 1: REFERENCE EIGENVALUE CALCULATION")
    print(DIVIDER)
    print(f"  Parameters: 5000 particles, 150 batches, 30 inactive")
    print(f"  Total histories: {5000 * 150:,}")

    geometry = MSRGeometry()
    solver = EigenvalueSolver(
        geometry=geometry,
        materials=materials,
        n_particles=5000,
        n_batches=150,
        n_inactive=30,
        mesh_r_bins=25,
        mesh_z_bins=40,
        seed=seed,
    )
    ref_result = solver.solve(verbose=True)
    all_results["reference"] = ref_result

    t_phase1 = time.perf_counter() - t_start

    # Print reference results
    print()
    print(DIVIDER)
    print("  REFERENCE EIGENVALUE RESULTS")
    print(DIVIDER)
    ci_lo, ci_hi = ref_result.keff_ci_95
    reactivity_pcm = (ref_result.keff - 1.0) / ref_result.keff * 1e5
    print(f"  k_eff (MC):              {ref_result.keff:.5f} +/- {ref_result.keff_std:.5f}")
    print(f"  95% CI:                  [{ci_lo:.5f}, {ci_hi:.5f}]")
    print(f"  Reactivity:              {reactivity_pcm:+.0f} pcm")
    print(f"  k_eff (diffusion):       1.1457")
    dk_pcm = (ref_result.keff - 1.1457) * 1e5
    print(f"  MC - Diffusion:          {dk_pcm:+.0f} pcm")
    print(f"  Leakage fraction:        {ref_result.leakage_fraction:.4f} "
          f"({ref_result.leakage_fraction*100:.2f}%)")
    print(f"  Non-leakage prob:        {1.0 - ref_result.leakage_fraction:.4f}")
    print(f"  Axial peaking:           {ref_result.axial_peaking:.3f}")
    print(f"  Radial peaking:          {ref_result.radial_peaking:.3f}")
    print(f"  Total peaking:           {ref_result.total_peaking:.3f}")
    print(f"  Phase 1 time:            {t_phase1:.1f} s ({t_phase1/60:.1f} min)")
    print(DIVIDER)

    # =================================================================
    # 2. Post-processing analysis
    # =================================================================
    print()
    print(DIVIDER)
    print("  PHASE 2: POST-PROCESSING ANALYSIS")
    print(DIVIDER)

    # Diffusion comparison
    comparison = compare_with_diffusion(ref_result)
    all_results["diffusion_comparison"] = comparison

    # Flux spectrum
    spectrum = print_flux_spectrum(ref_result, materials)
    all_results["spectrum"] = spectrum

    # Power distribution
    power = print_power_distribution(ref_result)
    all_results["power"] = power

    # Four-factor decomposition
    four_factor = compute_four_factor_from_mc(ref_result, materials)
    all_results["four_factor"] = four_factor

    # Reaction rates and neutron balance
    reaction_rates = compute_reaction_rates(ref_result, materials)
    all_results["reaction_rates"] = reaction_rates

    # Spectral indices
    spectral = compute_spectral_indices(ref_result, materials)
    all_results["spectral_indices"] = spectral

    t_phase2 = time.perf_counter() - t_start

    # =================================================================
    # 3. Reactivity coefficients
    # =================================================================
    print()
    print(DIVIDER)
    print("  PHASE 3: REACTIVITY COEFFICIENTS")
    print(DIVIDER)

    coeff_results = {}

    # Fuel temperature coefficient
    print(f"\n  Computing fuel temperature coefficient...")
    coeff_results["fuel_temp"] = fuel_temperature_coefficient(
        n_particles=3000, n_batches=80, n_inactive=15,
        seed=seed + 100, verbose=True,
    )

    # Void coefficient
    t_elapsed = time.perf_counter() - t_start
    print(f"\n  Computing void coefficient... (total elapsed: {t_elapsed:.0f} s)")
    coeff_results["void"] = void_coefficient(
        n_particles=3000, n_batches=80, n_inactive=15,
        seed=seed + 200, verbose=True,
    )

    # Graphite temperature coefficient
    t_elapsed = time.perf_counter() - t_start
    print(f"\n  Computing graphite temperature coefficient... "
          f"(total elapsed: {t_elapsed:.0f} s)")
    coeff_results["graphite_temp"] = graphite_temperature_coefficient(
        n_particles=3000, n_batches=80, n_inactive=15,
        seed=seed + 300, verbose=True,
    )

    all_results["coefficients"] = coeff_results

    t_phase3 = time.perf_counter() - t_start

    # =================================================================
    # 4. Enrichment sensitivity
    # =================================================================
    print()
    print(DIVIDER)
    print("  PHASE 4: ENRICHMENT SENSITIVITY")
    print(DIVIDER)

    enrich = enrichment_sensitivity(
        enrichments=[0.05, 0.07, 0.10, 0.12, 0.15],
        n_particles=2000, n_batches=60, n_inactive=10,
        seed=seed + 400, verbose=True,
    )
    all_results["enrichment"] = enrich

    t_phase4 = time.perf_counter() - t_start

    # =================================================================
    # 5. Save results and generate reports
    # =================================================================
    # 5. Save results and generate reports
    # =================================================================
    total_time = time.perf_counter() - t_start
    all_results["total_time"] = total_time

    print()
    print(DIVIDER)
    print("  PHASE 5: REPORT GENERATION")
    print(DIVIDER)

    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Text report
    report_path = os.path.join(results_dir, "mc_analysis_results.txt")
    _save_text_report(all_results, report_path)
    print(f"  Text report: {report_path}")

    # Markdown summary
    md_path = os.path.join(results_dir, "mc_summary.md")
    _save_markdown_summary(all_results, md_path)
    print(f"  Markdown summary: {md_path}")

    # =================================================================
    # 6. Comprehensive summary
    # =================================================================
    _print_comprehensive_summary(all_results)

    return all_results


# =====================================================================
# Quick analysis (reduced statistics for testing)
# =====================================================================
def run_quick_analysis(seed: int = 42) -> Dict:
    """Quick MC analysis for testing with reduced statistics.

    Runs a fast eigenvalue calculation and basic analysis.
    Skips reactivity coefficients to save time.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results dictionary.

    Runtime
    -------
    Approximately 1-2 minutes on a modern laptop.
    """
    t_start = time.perf_counter()

    print(BANNER)
    print("  *** QUICK MODE: Reduced statistics for testing ***")
    print()
    _print_design_parameters()

    all_results = {}
    materials = create_msr_materials()

    # =================================================================
    # 1. Quick eigenvalue calculation
    # =================================================================
    print()
    print(DIVIDER)
    print("  EIGENVALUE CALCULATION (Quick Mode)")
    print(DIVIDER)
    print(f"  Parameters: 2000 particles, 50 batches, 10 inactive")
    print(f"  Total histories: {2000 * 50:,}")

    geometry = MSRGeometry()
    solver = EigenvalueSolver(
        geometry=geometry,
        materials=materials,
        n_particles=2000,
        n_batches=50,
        n_inactive=10,
        mesh_r_bins=15,
        mesh_z_bins=20,
        seed=seed,
    )
    ref_result = solver.solve(verbose=True)
    all_results["reference"] = ref_result

    t_phase1 = time.perf_counter() - t_start

    # =================================================================
    # 2. Post-processing analysis
    # =================================================================
    print()
    print(DIVIDER)
    print("  POST-PROCESSING ANALYSIS")
    print(DIVIDER)

    comparison = compare_with_diffusion(ref_result)
    all_results["diffusion_comparison"] = comparison

    spectrum = print_flux_spectrum(ref_result, materials)
    all_results["spectrum"] = spectrum

    power = print_power_distribution(ref_result)
    all_results["power"] = power

    four_factor = compute_four_factor_from_mc(ref_result, materials)
    all_results["four_factor"] = four_factor

    reaction_rates = compute_reaction_rates(ref_result, materials)
    all_results["reaction_rates"] = reaction_rates

    spectral = compute_spectral_indices(ref_result, materials)
    all_results["spectral_indices"] = spectral

    # =================================================================
    # 3. Save results
    # =================================================================
    total_time = time.perf_counter() - t_start
    all_results["total_time"] = total_time

    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    report_path = os.path.join(results_dir, "mc_analysis_results.txt")
    _save_text_report(all_results, report_path)
    print(f"\n  Text report: {report_path}")

    md_path = os.path.join(results_dir, "mc_summary.md")
    _save_markdown_summary(all_results, md_path)
    print(f"  Markdown summary: {md_path}")

    # =================================================================
    # 4. Summary
    # =================================================================
    _print_quick_summary(all_results)

    return all_results


# =====================================================================
# Report generation helpers
# =====================================================================
def _save_text_report(results: Dict, filepath: str) -> None:
    """Save comprehensive text report to file."""
    import io

    lines = []
    lines.append(DIVIDER)
    lines.append("  40 MWth Marine MSR - Monte Carlo Neutronics Analysis Report")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(DIVIDER)
    lines.append("")

    ref = results.get("reference")
    if ref is not None:
        lines.append(ref.summary())
        lines.append("")

        # Neutron balance
        nb = ref.neutron_balance
        lines.append("--- Neutron Balance ---")
        lines.append(f"  Leakage:       {nb.get('leakage', 0):.4e}")
        lines.append(f"  Absorption:    {nb.get('absorption', 0):.4e}")
        lines.append(f"  Fission:       {nb.get('fission', 0):.4e}")
        lines.append(f"  Leakage frac:  {nb.get('leakage_fraction', 0):.4f}")
        lines.append("")

    # Diffusion comparison
    comp = results.get("diffusion_comparison", {})
    if comp:
        lines.append("--- MC vs. Diffusion Comparison ---")
        lines.append(f"  k_eff MC:        {comp.get('keff_mc', 0):.5f} +/- "
                     f"{comp.get('keff_mc_std', 0):.5f}")
        lines.append(f"  k_eff diffusion: {comp.get('keff_diffusion', 0):.5f}")
        lines.append(f"  Delta:           {comp.get('keff_delta_pcm', 0):+.0f} pcm")
        lines.append("")

    # Four-factor
    ff = results.get("four_factor", {})
    if ff:
        lines.append("--- Four-Factor Formula ---")
        lines.append(f"  eta     = {ff.get('eta', 0):.4f}")
        lines.append(f"  f       = {ff.get('f', 0):.4f}")
        lines.append(f"  p       = {ff.get('p', 0):.4f}")
        lines.append(f"  epsilon = {ff.get('epsilon', 0):.4f}")
        lines.append(f"  k_inf   = {ff.get('k_inf', 0):.5f}")
        lines.append(f"  P_NL    = {ff.get('P_NL', 0):.4f}")
        lines.append(f"  k_eff   = {ff.get('k_eff_4factor', 0):.5f}")
        lines.append("")

    # Reactivity coefficients
    coeffs = results.get("coefficients", {})
    if coeffs:
        lines.append("--- Reactivity Coefficients ---")
        ftc = coeffs.get("fuel_temp", {})
        if ftc:
            lines.append(f"  alpha_fuel:      {ftc.get('alpha_pcm_per_K', 0):.2f} "
                         f"+/- {ftc.get('alpha_std_pcm_per_K', 0):.2f} pcm/K")
        vc = coeffs.get("void", {})
        if vc:
            lines.append(f"  alpha_void:      {vc.get('alpha_pcm_per_pct', 0):.1f} "
                         f"+/- {vc.get('alpha_std_pcm_per_pct', 0):.1f} pcm/%")
        gtc = coeffs.get("graphite_temp", {})
        if gtc:
            lines.append(f"  alpha_graphite:  {gtc.get('alpha_pcm_per_K', 0):.2f} "
                         f"+/- {gtc.get('alpha_std_pcm_per_K', 0):.2f} pcm/K")
        lines.append("")

    # Enrichment sensitivity
    enrich = results.get("enrichment", {})
    if enrich:
        lines.append("--- Enrichment Sensitivity ---")
        table = enrich.get("table", [])
        lines.append(f"  {'Enrichment':>12s}  {'k_eff':>12s}  {'sigma':>10s}  {'rho [pcm]':>12s}")
        for e, k, s in table:
            rho = (k - 1.0) / k * 1e5 if k > 0 else 0
            lines.append(f"  {e*100:10.1f}%   {k:12.5f}  {s:10.5f}  {rho:+10.0f}")
        e_crit = enrich.get("critical_enrichment")
        if e_crit is not None:
            e_std = enrich.get("critical_enrichment_std", 0)
            lines.append(f"\n  Critical enrichment: {e_crit*100:.2f}% +/- {e_std*100:.2f}%")
        lines.append("")

    # Timing
    total_time = results.get("total_time", 0)
    lines.append(f"  Total analysis time: {total_time:.1f} s ({total_time/60:.1f} min)")
    lines.append(DIVIDER)

    report_text = "\n".join(lines)
    with open(filepath, "w") as f:
        f.write(report_text)


def _save_markdown_summary(results: Dict, filepath: str) -> None:
    """Save a markdown summary of MC results."""
    lines = []
    lines.append("# Monte Carlo Neutronics Analysis")
    lines.append("## 40 MWth Marine Molten Salt Reactor")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    ref = results.get("reference")
    if ref is not None:
        lines.append("## Reference Eigenvalue")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| k_eff (MC) | {ref.keff:.5f} +/- {ref.keff_std:.5f} |")
        lines.append(f"| k_eff (diffusion) | 1.1457 |")
        dk_pcm = (ref.keff - 1.1457) * 1e5
        lines.append(f"| MC - Diffusion | {dk_pcm:+.0f} pcm |")
        rho = (ref.keff - 1.0) / ref.keff * 1e5
        lines.append(f"| Reactivity | {rho:+.0f} pcm |")
        lines.append(f"| Leakage fraction | {ref.leakage_fraction:.4f} |")
        lines.append(f"| Axial peaking | {ref.axial_peaking:.3f} |")
        lines.append(f"| Radial peaking | {ref.radial_peaking:.3f} |")
        lines.append(f"| Total peaking | {ref.total_peaking:.3f} |")
        lines.append(f"| Batches | {ref.n_batches} ({ref.n_inactive} inactive + {ref.n_active} active) |")
        lines.append(f"| Particles/batch | {ref.n_particles:,} |")
        lines.append(f"| Wall time | {ref.total_time:.1f} s |")
        lines.append("")

    # Four-factor
    ff = results.get("four_factor", {})
    if ff:
        lines.append("## Four-Factor Formula")
        lines.append("")
        lines.append("| Factor | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| eta (reproduction) | {ff.get('eta', 0):.4f} |")
        lines.append(f"| f (thermal utilisation) | {ff.get('f', 0):.4f} |")
        lines.append(f"| p (resonance escape) | {ff.get('p', 0):.4f} |")
        lines.append(f"| epsilon (fast fission) | {ff.get('epsilon', 0):.4f} |")
        lines.append(f"| k_inf | {ff.get('k_inf', 0):.5f} |")
        lines.append(f"| P_NL | {ff.get('P_NL', 0):.4f} |")
        lines.append(f"| k_eff (4-factor) | {ff.get('k_eff_4factor', 0):.5f} |")
        lines.append("")

    # Reactivity coefficients
    coeffs = results.get("coefficients", {})
    if coeffs:
        lines.append("## Reactivity Coefficients")
        lines.append("")
        lines.append("| Coefficient | MC Value | Diffusion | Unit |")
        lines.append("|------------|----------|-----------|------|")

        ftc = coeffs.get("fuel_temp", {})
        if ftc:
            a = ftc.get("alpha_pcm_per_K", 0)
            s = ftc.get("alpha_std_pcm_per_K", 0)
            lines.append(f"| alpha_fuel | {a:.2f} +/- {s:.2f} | -8.3 | pcm/K |")

        vc = coeffs.get("void", {})
        if vc:
            a = vc.get("alpha_pcm_per_pct", 0)
            s = vc.get("alpha_std_pcm_per_pct", 0)
            lines.append(f"| alpha_void | {a:.1f} +/- {s:.1f} | -40.7 | pcm/% |")

        gtc = coeffs.get("graphite_temp", {})
        if gtc:
            a = gtc.get("alpha_pcm_per_K", 0)
            s = gtc.get("alpha_std_pcm_per_K", 0)
            lines.append(f"| alpha_graphite | {a:.2f} +/- {s:.2f} | N/A | pcm/K |")

        lines.append("")

    # Enrichment
    enrich = results.get("enrichment", {})
    if enrich:
        lines.append("## Enrichment Sensitivity")
        lines.append("")
        lines.append("| Enrichment (%) | k_eff | sigma | Reactivity (pcm) |")
        lines.append("|----------------|-------|-------|-------------------|")
        for e, k, s in enrich.get("table", []):
            rho = (k - 1.0) / k * 1e5 if k > 0 else 0
            lines.append(f"| {e*100:.1f} | {k:.5f} | {s:.5f} | {rho:+.0f} |")
        e_crit = enrich.get("critical_enrichment")
        if e_crit is not None:
            e_std = enrich.get("critical_enrichment_std", 0)
            lines.append(f"\nCritical enrichment: **{e_crit*100:.2f}%** +/- {e_std*100:.2f}%")
        lines.append("")

    total_time = results.get("total_time", 0)
    lines.append(f"---")
    lines.append(f"Total analysis time: {total_time:.1f} s ({total_time/60:.1f} min)")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


# =====================================================================
# Summary printout helpers
# =====================================================================
def _print_comprehensive_summary(results: Dict) -> None:
    """Print the final comprehensive summary table."""
    ref = results.get("reference")
    coeffs = results.get("coefficients", {})
    enrich = results.get("enrichment", {})
    total_time = results.get("total_time", 0)

    print()
    print(DIVIDER)
    print("  COMPREHENSIVE SUMMARY")
    print(DIVIDER)

    if ref is not None:
        ci_lo, ci_hi = ref.keff_ci_95
        rho = (ref.keff - 1.0) / ref.keff * 1e5

        print()
        print("  --- Reference Eigenvalue ---")
        print(f"  k_eff (MC):         {ref.keff:.5f} +/- {ref.keff_std:.5f}")
        print(f"  k_eff (diffusion):  1.1457")
        dk_pcm = (ref.keff - 1.1457) * 1e5
        print(f"  Difference:         {dk_pcm:+.0f} pcm")
        print(f"  Reactivity:         {rho:+.0f} pcm")
        print(f"  Leakage:            {ref.leakage_fraction*100:.2f}%")

    print()
    print("  --- Power Distribution ---")
    if ref is not None:
        print(f"  Axial peaking:      {ref.axial_peaking:.3f}")
        print(f"  Radial peaking:     {ref.radial_peaking:.3f}")
        print(f"  Total peaking:      {ref.total_peaking:.3f}")

    if coeffs:
        print()
        print("  --- Reactivity Coefficients ---")
        print(f"  {'Parameter':<24s}  {'MC (pcm/unit)':>16s}  {'Diffusion':>10s}  {'Difference':>12s}")
        print(f"  {'-'*66}")

        ftc = coeffs.get("fuel_temp", {})
        a = ftc.get("alpha_pcm_per_K", 0)
        s = ftc.get("alpha_std_pcm_per_K", 0)
        diff_val = -8.3
        if abs(diff_val) > 0:
            diff_pct = (a - diff_val) / abs(diff_val) * 100
            diff_str = f"{diff_pct:+.0f}%"
        else:
            diff_str = "N/A"
        print(f"  {'alpha_fuel':<24s}  {a:7.2f} +/- {s:5.2f}  {diff_val:>10.1f}  {diff_str:>12s}")

        vc = coeffs.get("void", {})
        a = vc.get("alpha_pcm_per_pct", 0)
        s = vc.get("alpha_std_pcm_per_pct", 0)
        diff_val = -40.7
        if abs(diff_val) > 0:
            diff_pct = (a - diff_val) / abs(diff_val) * 100
            diff_str = f"{diff_pct:+.0f}%"
        else:
            diff_str = "N/A"
        print(f"  {'alpha_void':<24s}  {a:7.1f} +/- {s:5.1f}  {diff_val:>10.1f}  {diff_str:>12s}")

        gtc = coeffs.get("graphite_temp", {})
        a = gtc.get("alpha_pcm_per_K", 0)
        s = gtc.get("alpha_std_pcm_per_K", 0)
        print(f"  {'alpha_graphite':<24s}  {a:7.2f} +/- {s:5.2f}  {'N/A':>10s}  {'N/A':>12s}")

    if enrich:
        e_crit = enrich.get("critical_enrichment")
        if e_crit is not None:
            e_std = enrich.get("critical_enrichment_std", 0)
            print()
            print(f"  --- Critical Enrichment ---")
            print(f"  Critical enrichment (k=1): {e_crit*100:.2f}% +/- {e_std*100:.2f}%")

    # Safety assessment
    print()
    print("  --- Safety Assessment ---")
    safe_items = []

    ftc = coeffs.get("fuel_temp", {})
    if ftc.get("is_negative", False):
        safe_items.append(("Negative fuel temp coeff", True))
    elif ftc:
        safe_items.append(("Negative fuel temp coeff", False))

    vc = coeffs.get("void", {})
    if vc.get("is_negative", False):
        safe_items.append(("Negative void coeff", True))
    elif vc:
        safe_items.append(("Negative void coeff", False))

    if ref is not None and ref.keff > 1.0:
        safe_items.append(("Supercritical (k > 1)", True))

    for item, ok in safe_items:
        status = "PASS" if ok else "WARN"
        print(f"  [{status}] {item}")

    # Timing
    print()
    print(f"  --- Timing ---")
    print(f"  Total analysis time: {total_time:.1f} s ({total_time/60:.1f} min)")
    if ref is not None:
        total_histories = ref.n_batches * ref.n_particles
        rate = total_histories / max(ref.total_time, 0.01)
        print(f"  Histories/sec (ref): {rate:,.0f}")

    print()
    print(DIVIDER)
    print("  Analysis complete.")
    print(DIVIDER)


def _print_quick_summary(results: Dict) -> None:
    """Print a compact summary for quick mode."""
    ref = results.get("reference")
    total_time = results.get("total_time", 0)

    print()
    print(DIVIDER)
    print("  QUICK ANALYSIS SUMMARY")
    print(DIVIDER)

    if ref is not None:
        rho = (ref.keff - 1.0) / ref.keff * 1e5
        dk_pcm = (ref.keff - 1.1457) * 1e5

        print(f"  k_eff (MC):         {ref.keff:.5f} +/- {ref.keff_std:.5f}")
        print(f"  k_eff (diffusion):  1.1457")
        print(f"  Difference:         {dk_pcm:+.0f} pcm")
        print(f"  Reactivity:         {rho:+.0f} pcm")
        print(f"  Leakage:            {ref.leakage_fraction*100:.2f}%")
        print(f"  Axial peaking:      {ref.axial_peaking:.3f}")
        print(f"  Radial peaking:     {ref.radial_peaking:.3f}")

    ff = results.get("four_factor", {})
    if ff:
        print()
        print(f"  Four-factor: k_inf = {ff.get('eta',0):.3f} * {ff.get('f',0):.3f} "
              f"* {ff.get('p',0):.3f} * {ff.get('epsilon',0):.3f} "
              f"= {ff.get('k_inf',0):.4f}")
        print(f"  k_eff (4-factor) = {ff.get('k_inf',0):.4f} * "
              f"{ff.get('P_NL',0):.4f} = {ff.get('k_eff_4factor',0):.5f}")

    print()
    print(f"  Total time: {total_time:.1f} s ({total_time/60:.1f} min)")
    print(f"  (Use full mode for reactivity coefficients and enrichment study)")
    print(DIVIDER)


# =====================================================================
# Coefficients-only mode
# =====================================================================
def run_coefficients_only(seed: int = 42) -> Dict:
    """Run only the reactivity coefficient calculations.

    Skips the reference eigenvalue and jumps straight to computing
    all reactivity coefficients. Useful if the reference eigenvalue
    has already been computed.

    Parameters
    ----------
    seed : int
        Base random seed.

    Returns
    -------
    dict
        Coefficient results.
    """
    t_start = time.perf_counter()

    print(BANNER)
    print("  *** COEFFICIENTS MODE: Reactivity coefficients only ***")
    print()

    results = compute_all_coefficients(
        n_particles=3000, n_batches=80, n_inactive=15,
        seed=seed, verbose=True,
    )

    total_time = time.perf_counter() - t_start
    results["total_time"] = total_time

    return results


# =====================================================================
# Main entry point
# =====================================================================
if __name__ == "__main__":
    mode = "full"

    # Parse command-line arguments
    if "--quick" in sys.argv:
        mode = "quick"
    elif "--coefficients" in sys.argv:
        mode = "coefficients"
    elif "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python3 mc_neutronics/run_mc.py [OPTIONS]")
        print()
        print("Options:")
        print("  --quick          Quick analysis (reduced stats, ~1-2 min)")
        print("  --coefficients   Reactivity coefficients only")
        print("  --help, -h       Show this help message")
        print()
        print("Default: Full production analysis (~30-60 min)")
        sys.exit(0)

    # Run
    if mode == "quick":
        results = run_quick_analysis()
    elif mode == "coefficients":
        results = run_coefficients_only()
    else:
        results = run_full_analysis()
