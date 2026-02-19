#!/usr/bin/env python3
"""
40 MWth Marine MSR Design Analysis - Master Runner
====================================================

Executes ALL analysis modules for the 40 MWth graphite-moderated marine
molten salt reactor and collects results into a comprehensive design summary.

Analysis sequence:
  [1/8] Foundation       - Core configuration and derived parameters
  [2/8] Neutronics       - Cross-sections, criticality, reactivity, burnup
  [3/8] Thermal-Hydraulics - Channel analysis, loop hydraulics, temperature map
  [4/8] Heat Exchanger   - Shell-and-tube HX sizing and performance
  [5/8] Structural       - Vessel design, thermal stress, ship motion loads
  [6/8] Shielding        - Source term, attenuation, dose map
  [7/8] Safety           - Design basis transients (ULOF, UTOP, SBO)
  [8/8] Drain Tank       - Emergency drain tank sizing and subcriticality

Usage:
    python main.py
"""

import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from config import compute_derived, print_summary
from utils.tables import (
    print_section_header, print_param_table, format_value,
    results_to_markdown,
)


# =============================================================================
# Banner
# =============================================================================

BANNER = r"""
================================================================================

     40 MWth Marine Molten Salt Reactor
     Design Analysis Package

     Salt:       FLiBe + 5 mol% UF4
     Moderator:  Graphite (hexagonal lattice)
     Structure:  Hastelloy-N
     Ship class: 6,000 TEU Panamax Container

================================================================================
"""


# =============================================================================
# Helper
# =============================================================================

def step_header(step, total, title):
    """Print a progress step header."""
    tag = f"[{step}/{total}]"
    print(f"\n{'=' * 80}")
    print(f"  {tag} {title}")
    print(f"{'=' * 80}")


def safe_get(obj, attr, default="N/A"):
    """Safely get an attribute from an object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# =============================================================================
# Module runners - each returns a results dict (or None on failure)
# =============================================================================

def run_foundation():
    """[1/8] Foundation: core configuration and derived parameters."""
    step_header(1, 8, "FOUNDATION - Core Configuration & Derived Parameters")

    d = compute_derived()
    print_summary(d)

    results = {
        "Thermal power": (config.THERMAL_POWER / 1e6, "MWth"),
        "Electrical power": (config.ELECTRICAL_POWER / 1e6, "MWe"),
        "Thermal efficiency": (config.THERMAL_EFFICIENCY * 100, "%"),
        "Core diameter": (d.core_diameter, "m"),
        "Core height": (d.core_height, "m"),
        "Core volume": (d.core_volume, "m3"),
        "Number of channels": (d.n_channels, ""),
        "Channel diameter": (config.CHANNEL_DIAMETER * 100, "cm"),
        "Channel pitch": (config.CHANNEL_PITCH * 100, "cm"),
        "Fuel salt fraction": (config.FUEL_SALT_FRACTION, ""),
        "Graphite fraction": (config.GRAPHITE_VOLUME_FRACTION, ""),
        "Mass flow rate": (d.mass_flow_rate, "kg/s"),
        "Reynolds number": (d.reynolds_number, ""),
        "Inlet temperature": (config.CORE_INLET_TEMP - 273.15, "C"),
        "Outlet temperature": (config.CORE_OUTLET_TEMP - 273.15, "C"),
        "Uranium mass": (d.uranium_mass, "kg"),
        "U-235 mass": (d.u235_mass, "kg"),
    }

    return d, results


def run_neutronics(d):
    """[2/8] Neutronics: cross-sections, criticality, reactivity, burnup."""
    step_header(2, 8, "NEUTRONICS - Cross-Sections, Criticality, Reactivity & Burnup")

    from neutronics.cross_sections import compute_homogenized_cross_sections, print_cross_sections
    from neutronics.core_geometry import design_core, print_core_geometry
    from neutronics.diffusion import solve_diffusion, get_power_profile, print_diffusion_results
    from neutronics.criticality import four_factor, find_critical_enrichment, print_neutron_balance
    from neutronics.reactivity import compute_reactivity_coefficients, print_reactivity_coefficients
    from neutronics.burnup import estimate_burnup, print_burnup_summary

    results = {}

    # --- Cross-sections ---
    print("\n--- Homogenized Cross-Sections ---")
    xs = compute_homogenized_cross_sections()
    print_cross_sections(xs)
    results["Sigma_a (1/cm)"] = (xs['sigma_a'], "1/cm")
    results["Sigma_f (1/cm)"] = (xs['sigma_f'], "1/cm")
    results["nu*Sigma_f (1/cm)"] = (xs['nu_sigma_f'], "1/cm")
    results["Diffusion coeff D (cm)"] = (xs['D'], "cm")

    # --- Core geometry ---
    print("\n--- Core Geometry ---")
    geom = design_core()
    print_core_geometry(geom)
    results["Core diameter (geometry)"] = (geom.core_diameter, "m")
    results["Core height (geometry)"] = (geom.core_height, "m")
    results["Power density"] = (geom.power_density / 1e6, "MW/m3")
    results["Graphite mass"] = (geom.graphite_mass, "kg")

    # --- Diffusion solver ---
    print("\n--- 1-Group Diffusion Eigenvalue ---")
    diff = solve_diffusion(
        core_radius=geom.core_radius,
        core_height=geom.core_height,
        D=xs['D'],
        sigma_a=xs['sigma_a'],
        nu_sigma_f=xs['nu_sigma_f'],
    )
    profiles = get_power_profile(diff)
    print_diffusion_results(diff, profiles)
    results["k_eff (diffusion)"] = (diff['keff'], "")
    results["Axial peaking factor"] = (profiles['axial_peaking'], "")
    results["Radial peaking factor"] = (profiles['radial_peaking'], "")

    # --- Four-factor criticality ---
    print("\n--- Four-Factor Criticality ---")
    ff = four_factor(geometry=geom)
    print_neutron_balance(ff)
    results["k_eff (four-factor)"] = (ff['keff'], "")
    results["k_inf"] = (ff['k_inf'], "")
    results["Non-leakage probability"] = (ff['P_NL'], "")

    # --- Critical enrichment search ---
    print("\n--- Critical Enrichment Search ---")
    ce = find_critical_enrichment(geometry=geom)
    results["Critical enrichment"] = (ce['enrichment_critical'] * 100, "%")
    results["k_eff at critical enrich."] = (ce['keff'], "")

    # --- Reactivity coefficients ---
    print("\n--- Reactivity Coefficients ---")
    rc = compute_reactivity_coefficients(geometry=geom)
    print_reactivity_coefficients(rc)
    results["alpha_fuel"] = (rc['alpha_fuel'], "pcm/K")
    results["alpha_graphite"] = (rc['alpha_graphite'], "pcm/K")
    results["alpha_density"] = (rc['alpha_density'], "pcm/(kg/m3)")
    results["All negative feedback"] = ("Yes" if rc['all_negative'] else "NO", "")

    # --- Burnup analysis ---
    print("\n--- Burnup and Fuel Utilization ---")
    bu = estimate_burnup(geometry=geom)
    print_burnup_summary(bu)
    results["Burnup"] = (bu['burnup_MWd_per_kg'], "MWd/kgU")
    results["Core lifetime (EFPD)"] = (bu['core_lifetime_EFPD'], "EFPD")
    results["Core lifetime (years)"] = (bu['core_lifetime_years'], "yr")

    return results


def run_thermal_hydraulics(d):
    """[3/8] Thermal-Hydraulics: channel, loops, temperature map, limits."""
    step_header(3, 8, "THERMAL-HYDRAULICS - Channel Analysis, Loops & Temperature Map")

    from thermal_hydraulics.channel_analysis import (
        run_nominal_channel_analysis, print_channel_results,
    )
    from thermal_hydraulics.coolant_loop import (
        primary_loop_analysis, intermediate_loop_analysis, print_loop_results,
    )
    from thermal_hydraulics.temperature import (
        compute_temperature_map, check_thermal_limits,
        print_temperature_summary, print_thermal_limits,
    )

    results = {}

    # --- Channel analysis ---
    print("\n--- Single-Channel Analysis ---")
    ch = run_nominal_channel_analysis()
    print_channel_results(ch)
    results["Peak salt temp (nominal)"] = (ch.peak_salt_temp - 273.15, "C")
    results["Peak salt temp (hot ch.)"] = (ch.peak_salt_temp_hot - 273.15, "C")
    results["Peak graphite temp"] = (ch.peak_graphite_temp - 273.15, "C")
    results["Peak graphite (hot ch.)"] = (ch.peak_graphite_temp_hot - 273.15, "C")
    results["Channel Re"] = (ch.Re, "")
    results["Channel pressure drop"] = (ch.pressure_drop / 1e3, "kPa")

    # --- Primary loop ---
    print("\n--- Primary Loop Hydraulics ---")
    pl = primary_loop_analysis(design_params=d)
    print_loop_results(pl)
    results["Primary dP total"] = (pl.total_pressure_drop / 1e3, "kPa")
    results["Pump power (primary)"] = (pl.pump_power_shaft / 1e3, "kW")
    results["Natural circ fraction"] = (pl.natural_circ_fraction * 100, "%")

    # --- Intermediate loop ---
    print("\n--- Intermediate Loop Hydraulics ---")
    il = intermediate_loop_analysis(design_params=d)
    print_loop_results(il)
    results["Intermediate dP total"] = (il.total_pressure_drop / 1e3, "kPa")
    results["Pump power (intermediate)"] = (il.pump_power_shaft / 1e3, "kW")

    # --- Temperature map ---
    print("\n--- System Temperature Map ---")
    tmap = compute_temperature_map(channel_result=ch, loop_result=pl, design_params=d)
    print_temperature_summary(tmap)

    results["Peak salt (system)"] = (tmap.peak_salt - 273.15, "C")
    results["Peak graphite (system)"] = (tmap.peak_graphite - 273.15, "C")
    results["Peak vessel wall"] = (tmap.peak_vessel - 273.15, "C")

    # --- Thermal limits check ---
    print("\n--- Thermal Limits Check ---")
    tlim = check_thermal_limits(tmap)
    print_thermal_limits(tlim)
    results["All thermal limits met"] = ("Yes" if tlim.all_passed else "NO", "")

    return results


def run_heat_exchanger(d):
    """[4/8] Heat Exchanger: shell-and-tube sizing and off-design performance."""
    step_header(4, 8, "HEAT EXCHANGER - Shell-and-Tube Design & Performance")

    from heat_exchanger.design import design_heat_exchanger, print_hx_design
    from heat_exchanger.performance import (
        part_load_curve, fouling_sensitivity,
        print_part_load_results, print_fouling_results,
    )

    results = {}

    # --- Design ---
    print("\n--- Heat Exchanger Design ---")
    hx = design_heat_exchanger(design_params=d)
    print_hx_design(hx)
    results["HX duty"] = (hx.Q / 1e6, "MW")
    results["LMTD"] = (hx.LMTD, "K")
    results["U overall"] = (hx.U_overall, "W/(m2.K)")
    results["HX area"] = (hx.A_required, "m2")
    results["Number of tubes"] = (hx.N_tubes, "")
    results["Tube length"] = (hx.tube_length, "m")
    results["Shell diameter"] = (hx.shell_diameter, "m")
    results["Effectiveness"] = (hx.effectiveness * 100, "%")
    results["HX total mass"] = (hx.total_mass, "kg")
    results["Tube-side dP"] = (hx.dp_tube / 1e3, "kPa")
    results["Shell-side dP"] = (hx.dp_shell / 1e3, "kPa")

    # --- Part-load ---
    print("\n--- Part-Load Performance ---")
    pl = part_load_curve(hx)
    print_part_load_results(pl)

    # --- Fouling sensitivity ---
    print("\n--- Fouling Sensitivity ---")
    fs = fouling_sensitivity(hx)
    print_fouling_results(fs)

    return results


def run_structural(d):
    """[5/8] Structural: vessel design, thermal stress, ship motion loads."""
    step_header(5, 8, "STRUCTURAL - Vessel Design, Thermal Stress & Ship Loads")

    from structural.vessel_design import design_vessel, print_vessel_design
    from structural.thermal_stress import analyze_thermal_stress, print_thermal_stress
    from structural.seismic_marine import analyze_ship_loads, print_ship_loads

    results = {}

    # --- Vessel design ---
    print("\n--- Reactor Vessel Design (ASME Sec III Div 5) ---")
    vessel = design_vessel(design_params=d)
    print_vessel_design(vessel)
    results["Shell thickness (total)"] = (vessel.shell_thickness_total * 100, "cm")
    results["Vessel outer radius"] = (vessel.shell_outer_radius, "m")
    results["Vessel height"] = (vessel.vessel_height, "m")
    results["Design pressure"] = (vessel.design_pressure / 1e6, "MPa")
    results["Allowable stress"] = (vessel.allowable_stress / 1e6, "MPa")
    results["Pressure safety factor"] = (vessel.pressure_safety_factor, "")
    results["Vessel total weight"] = (vessel.total_weight, "kg")

    # --- Thermal stress ---
    print("\n--- Vessel Thermal Stress (Goodier) ---")
    ts = analyze_thermal_stress(design_params=d)
    print_thermal_stress(ts)
    results["Max stress intensity"] = (ts.max_stress_intensity / 1e6, "MPa")
    results["ASME 3Sm limit"] = (ts.asme_3Sm_limit / 1e6, "MPa")
    results["ASME safety factor"] = (ts.asme_safety_factor, "")
    results["Creep rupture limit"] = (ts.creep_rupture_limit / 1e6, "MPa")
    results["Creep safety factor"] = (ts.creep_safety_factor, "")

    # --- Ship motion loads ---
    print("\n--- Ship Motion & Seismic Loads (DNV GL) ---")
    ship = analyze_ship_loads(design_params=d)
    print_ship_loads(ship)
    results["Total reactor mass"] = (ship.total_mass, "kg")
    results["Max lateral force"] = (ship.F_lateral_max / 1e3, "kN")
    results["Max vertical force"] = (ship.F_vertical_max / 1e3, "kN")
    results["Overturning moment"] = (ship.overturning_moment_max / 1e3, "kN.m")
    results["Bolt safety factor"] = (ship.bolt_safety_factor, "")

    return results


def run_shielding(d):
    """[6/8] Shielding: source term, attenuation, dose rate map."""
    step_header(6, 8, "SHIELDING - Source Term, Attenuation & Dose Map")

    from shielding.source_term import compute_source_term, print_source_term
    from shielding.attenuation import (
        default_shield_layers, compute_attenuation, print_attenuation,
    )
    from shielding.dose_rate import (
        compute_dose_map, print_dose_map, print_dose_breakdown,
    )

    results = {}

    # --- Source term ---
    print("\n--- Radiation Source Term ---")
    st = compute_source_term(design_params=d)
    print_source_term(st)
    results["Neutron source"] = (st.S_neutron, "n/s")
    results["Gamma source (total)"] = (st.S_gamma_total, "photons/s")
    results["Gamma power (total)"] = (st.P_gamma_total / 1e3, "kW")

    # --- Shield layers & attenuation ---
    print("\n--- Biological Shield Attenuation ---")
    layers = default_shield_layers(d)
    att = compute_attenuation(layers, st.S_neutron, st.S_gamma_total)
    print_attenuation(att)
    results["Neutron dose rate (5m)"] = (att.neutron_dose_rate * 1e3, "uSv/hr")
    results["Gamma dose rate (5m)"] = (att.gamma_dose_rate * 1e3, "uSv/hr")
    results["Total dose rate (5m)"] = (att.total_dose_rate * 1e3, "uSv/hr")

    # --- Dose map ---
    print("\n--- Dose Rate Map (All Ship Locations) ---")
    dm = compute_dose_map(shield_layers=layers, design_params=d)
    print_dose_map(dm)
    print_dose_breakdown(dm)
    results["Max dose rate"] = (dm.max_dose_rate * 1e3, "uSv/hr")
    results["Max annual dose"] = (dm.max_annual_dose, "mSv/yr")
    results["All within dose limits"] = ("Yes" if dm.all_within_limits else "NO", "")

    return results


def run_safety_transients(d):
    """[7/8] Safety: design basis transients (ULOF, UTOP, SBO)."""
    step_header(7, 8, "SAFETY - Design Basis Accident Transients")

    from safety.transients import (
        simulate_ulof, simulate_utop, simulate_sbo,
        print_transient_summary,
    )

    results = {}

    # --- ULOF ---
    print("\n--- ULOF: Unprotected Loss of Flow ---")
    ulof = simulate_ulof(design_params=d)
    print_transient_summary(ulof)
    results["ULOF peak power"] = (ulof.peak_power, "x nominal")
    results["ULOF peak fuel temp"] = (ulof.peak_T_fuel - 273.15, "C")
    results["ULOF margin to boiling"] = (ulof.margin_to_boiling, "K")

    # --- UTOP ---
    print("\n--- UTOP: Unprotected Transient Overpower ---")
    utop = simulate_utop(design_params=d)
    print_transient_summary(utop)
    results["UTOP peak power"] = (utop.peak_power, "x nominal")
    results["UTOP peak fuel temp"] = (utop.peak_T_fuel - 273.15, "C")
    results["UTOP margin to boiling"] = (utop.margin_to_boiling, "K")

    # --- SBO ---
    print("\n--- SBO: Station Blackout ---")
    sbo = simulate_sbo(design_params=d)
    print_transient_summary(sbo)
    results["SBO peak fuel temp"] = (sbo.peak_T_fuel - 273.15, "C")
    results["SBO min fuel temp"] = (sbo.min_T_fuel - 273.15, "C")
    results["SBO margin to freezing"] = (sbo.margin_to_freezing, "K")

    return results


def run_drain_tank(d):
    """[8/8] Emergency drain tank sizing and subcriticality."""
    step_header(8, 8, "DRAIN TANK - Emergency Drain Tank Design")

    from safety.drain_tank import design_drain_tank, print_drain_tank

    results = {}

    dt = design_drain_tank(design_params=d)
    print_drain_tank(dt)

    results["Salt volume"] = (dt.salt_volume, "m3")
    results["Tank volume"] = (dt.tank_volume, "m3")
    results["Tank diameter"] = (dt.tank_diameter, "m")
    results["Tank height"] = (dt.tank_height, "m")
    results["k_eff (drain tank)"] = (dt.keff_estimated, "")
    results["Subcritical"] = ("Yes" if dt.subcritical else "NO", "")
    results["Subcriticality margin"] = (dt.subcriticality_margin, "dk")
    results["Decay heat @ 1 hr"] = (dt.Q_decay_1hr / 1e3, "kW")
    results["Tank equilibrium temp"] = (dt.T_tank_equilibrium - 273.15, "C")
    results["Cooling adequate"] = ("Yes" if dt.cooling_adequate else "NO", "")
    results["Drain time"] = (dt.drain_time, "s")

    return results


# =============================================================================
# Design summary printer
# =============================================================================

def print_design_summary(all_results):
    """Print the comprehensive design summary table."""
    width = 80
    print("\n" + "=" * width)
    print("  COMPREHENSIVE DESIGN SUMMARY")
    print("  40 MWth Graphite-Moderated Marine Molten Salt Reactor")
    print("=" * width)

    for section_name, section_results in all_results.items():
        if section_results is None:
            print(f"\n  --- {section_name} --- [FAILED - see errors above]")
            continue

        print(f"\n  --- {section_name} ---")
        for key, val in section_results.items():
            if isinstance(val, tuple) and len(val) == 2:
                value, unit = val
                if isinstance(value, float):
                    print(f"    {key:<40s}  {format_value(value):>14s}  {unit}")
                elif isinstance(value, int):
                    print(f"    {key:<40s}  {value:>14d}  {unit}")
                else:
                    print(f"    {key:<40s}  {str(value):>14s}  {unit}")
            elif isinstance(val, tuple) and len(val) == 1:
                # String-only result (e.g., "Yes"/"NO")
                print(f"    {key:<40s}  {val[0]:>14s}")
            else:
                print(f"    {key:<40s}  {str(val):>14s}")

    print("\n" + "=" * width)


# =============================================================================
# Save results
# =============================================================================

def save_results(all_results, output_dir):
    """Save results to results/ directory."""

    os.makedirs(output_dir, exist_ok=True)

    # Build markdown-compatible dict for results_to_markdown
    # Normalize single-element tuples to (value, "") for compatibility
    md_results = {}
    for section_name, section_results in all_results.items():
        if section_results is None:
            md_results[section_name] = "Analysis FAILED - see console output for errors."
            continue
        normalized = {}
        for key, val in section_results.items():
            if isinstance(val, tuple) and len(val) == 1:
                normalized[key] = (val[0], "")
            else:
                normalized[key] = val
        md_results[section_name] = normalized

    # Save summary markdown
    md_path = os.path.join(output_dir, "summary.md")
    results_to_markdown(md_results, md_path)

    # Save plain-text summary
    txt_path = os.path.join(output_dir, "summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  40 MWth Marine MSR - Design Analysis Summary\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for section_name, section_results in all_results.items():
            f.write(f"\n{'=' * 60}\n")
            f.write(f"  {section_name}\n")
            f.write(f"{'=' * 60}\n\n")

            if section_results is None:
                f.write("  *** ANALYSIS FAILED ***\n\n")
                continue

            for key, val in section_results.items():
                if isinstance(val, tuple) and len(val) == 2:
                    value, unit = val
                    if isinstance(value, float):
                        f.write(f"  {key:<40s}  {format_value(value):>14s}  {unit}\n")
                    elif isinstance(value, int):
                        f.write(f"  {key:<40s}  {value:>14d}  {unit}\n")
                    else:
                        f.write(f"  {key:<40s}  {str(value):>14s}  {unit}\n")
                elif isinstance(val, tuple) and len(val) == 1:
                    f.write(f"  {key:<40s}  {val[0]:>14s}\n")
                else:
                    f.write(f"  {key:<40s}  {str(val):>14s}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\n  Plain-text summary saved: {txt_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Execute the complete MSR design analysis."""
    print(BANNER)
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python {sys.version.split()[0]}, NumPy {np.__version__}")
    print()

    t_start = time.time()

    all_results = {}
    errors = []
    d = None  # DerivedParameters, set by foundation step

    # ---- [1/8] Foundation ----
    try:
        d, foundation_results = run_foundation()
        all_results["1. Foundation"] = foundation_results
    except Exception as e:
        print(f"\n  *** FOUNDATION FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Foundation", str(e)))
        all_results["1. Foundation"] = None
        # Cannot continue without derived parameters
        print("\n  FATAL: Cannot proceed without foundation parameters.")
        return

    # ---- [2/8] Neutronics ----
    try:
        all_results["2. Neutronics"] = run_neutronics(d)
    except Exception as e:
        print(f"\n  *** NEUTRONICS FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Neutronics", str(e)))
        all_results["2. Neutronics"] = None

    # ---- [3/8] Thermal-Hydraulics ----
    try:
        all_results["3. Thermal-Hydraulics"] = run_thermal_hydraulics(d)
    except Exception as e:
        print(f"\n  *** THERMAL-HYDRAULICS FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Thermal-Hydraulics", str(e)))
        all_results["3. Thermal-Hydraulics"] = None

    # ---- [4/8] Heat Exchanger ----
    try:
        all_results["4. Heat Exchanger"] = run_heat_exchanger(d)
    except Exception as e:
        print(f"\n  *** HEAT EXCHANGER FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Heat Exchanger", str(e)))
        all_results["4. Heat Exchanger"] = None

    # ---- [5/8] Structural ----
    try:
        all_results["5. Structural"] = run_structural(d)
    except Exception as e:
        print(f"\n  *** STRUCTURAL FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Structural", str(e)))
        all_results["5. Structural"] = None

    # ---- [6/8] Shielding ----
    try:
        all_results["6. Shielding"] = run_shielding(d)
    except Exception as e:
        print(f"\n  *** SHIELDING FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Shielding", str(e)))
        all_results["6. Shielding"] = None

    # ---- [7/8] Safety Transients ----
    try:
        all_results["7. Safety Transients"] = run_safety_transients(d)
    except Exception as e:
        print(f"\n  *** SAFETY TRANSIENTS FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Safety Transients", str(e)))
        all_results["7. Safety Transients"] = None

    # ---- [8/8] Drain Tank ----
    try:
        all_results["8. Drain Tank"] = run_drain_tank(d)
    except Exception as e:
        print(f"\n  *** DRAIN TANK FAILED: {e} ***")
        traceback.print_exc()
        errors.append(("Drain Tank", str(e)))
        all_results["8. Drain Tank"] = None

    # ---- Summary ----
    t_elapsed = time.time() - t_start

    print_design_summary(all_results)

    # ---- Save ----
    output_dir = os.path.join(PROJECT_ROOT, "results")
    save_results(all_results, output_dir)

    # ---- Final status ----
    n_ok = sum(1 for v in all_results.values() if v is not None)
    n_total = len(all_results)

    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Modules passed: {n_ok}/{n_total}")
    print(f"  Elapsed time:   {t_elapsed:.1f} s")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for mod, err in errors:
            print(f"    - {mod}: {err}")
    else:
        print(f"  Status:         ALL MODULES PASSED")
    print(f"\n  Results saved to: {output_dir}/")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
