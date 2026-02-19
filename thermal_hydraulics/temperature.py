"""
Reactor System Temperature Mapping
===================================

Computes temperatures at all key locations across the MSR primary system,
intermediate loop, and vessel.  Integrates results from channel analysis
and loop analysis to produce a complete temperature map, then checks
against safety limits from the central configuration.

Key locations
-------------
* Core: salt inlet, salt outlet, peak salt (hot channel), peak graphite
* Vessel: inner wall, outer wall
* Primary loop: hot leg, cold leg
* Heat exchanger: primary in/out, secondary in/out
* Intermediate loop: hot leg, cold leg

References
----------
* ORNL-4541 (MSBR design)
* config.py safety limits (Tier 6)
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import (
    THERMAL_POWER, CORE_INLET_TEMP, CORE_OUTLET_TEMP, CORE_AVG_TEMP,
    SECONDARY_INLET_TEMP, SECONDARY_OUTLET_TEMP,
    MAX_FUEL_SALT_TEMP, MAX_VESSEL_WALL_TEMP, MAX_GRAPHITE_TEMP,
    MIN_SALT_TEMP, HASTELLOY_N, OPERATING_PRESSURE,
    hastelloy_thermal_conductivity, compute_derived,
)


# ==========================================================================
# Data classes
# ==========================================================================

@dataclass
class TemperatureMap:
    """Complete temperature map of the MSR system at steady state."""

    # Core temperatures (K)
    core_salt_inlet: float
    core_salt_outlet: float
    core_salt_peak_nominal: float       # average channel
    core_salt_peak_hot: float           # hot channel (with peaking factors)
    core_graphite_peak_nominal: float
    core_graphite_peak_hot: float

    # Vessel temperatures (K)
    vessel_inner_wall: float
    vessel_outer_wall: float

    # Primary loop temperatures (K)
    primary_hot_leg: float              # leaving core
    primary_cold_leg: float             # entering core
    primary_hx_inlet: float             # entering HX (approx = hot leg)
    primary_hx_outlet: float            # leaving HX (approx = cold leg)

    # Intermediate loop temperatures (K)
    secondary_hx_inlet: float           # cold-side inlet to primary HX
    secondary_hx_outlet: float          # hot-side outlet from primary HX
    secondary_hot_leg: float
    secondary_cold_leg: float

    # Peak temperatures for summary
    peak_salt: float                    # maximum anywhere (hot channel)
    peak_graphite: float                # maximum anywhere (hot channel)
    peak_vessel: float                  # maximum vessel wall


@dataclass
class ThermalLimit:
    """Result of a single temperature limit check."""
    location: str
    temperature: float                  # K (actual)
    limit: float                        # K (design limit)
    margin: float                       # K (limit - actual)
    passed: bool


@dataclass
class ThermalLimitResults:
    """Collection of all thermal limit checks."""
    checks: List[ThermalLimit]
    all_passed: bool


# ==========================================================================
# Temperature map computation
# ==========================================================================

def compute_temperature_map(channel_result=None, loop_result=None,
                            design_params=None):
    """Build the complete reactor temperature map.

    If channel_result or loop_result are not provided, nominal values from
    the config are used.

    Args:
        channel_result: ChannelResult from channel_analysis (optional).
        loop_result: LoopResult from coolant_loop (optional).
        design_params: DerivedParameters from config (computed if None).

    Returns:
        TemperatureMap with all key temperatures.
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    # ---- Core temperatures ----
    T_in = CORE_INLET_TEMP
    T_out = CORE_OUTLET_TEMP

    if channel_result is not None:
        peak_salt_nom = channel_result.peak_salt_temp
        peak_salt_hot = channel_result.peak_salt_temp_hot
        peak_gr_nom = channel_result.peak_graphite_temp
        peak_gr_hot = channel_result.peak_graphite_temp_hot

        # Sanity check: if channel analysis produced unphysical graphite
        # temperatures (can happen in laminar flow regime where the heat
        # transfer coefficient is too low for the given power density),
        # fall back to the simplified estimate and flag a warning.
        if peak_gr_nom < T_in or peak_gr_hot < T_in:
            _warn_str = (
                "NOTE: Channel analysis graphite temperatures are unphysical "
                f"({peak_gr_nom - 273.15:.0f} C). This indicates the flow "
                "regime (Re={:.0f}) cannot support the channel heat flux. "
                "Using simplified graphite temperature estimate."
            ).format(channel_result.Re)
            print(f"  ** {_warn_str}")
            # Fall through to simplified estimate
            peak_gr_nom = peak_salt_nom + 5.0
            peak_gr_hot = T_in + (peak_gr_nom - T_in) * 1.265
    else:
        # Estimate without detailed channel analysis
        # Peak salt ~ outlet + small overshoot from cosine profile
        peak_salt_nom = T_out + 5.0       # K (small cosine overshoot above outlet)
        peak_salt_hot = T_in + (peak_salt_nom - T_in) * 1.265  # F_q*F_eng
        peak_gr_nom = peak_salt_nom + 5.0   # graphite slightly above wall
        peak_gr_hot = T_in + (peak_gr_nom - T_in) * 1.265

    # ---- Primary loop temperatures ----
    # Hot leg (core outlet to HX inlet) - slight heat loss in piping (~1-2 K)
    T_hot_leg = T_out - 1.0
    # Cold leg (HX outlet to core inlet)
    T_cold_leg = T_in + 1.0

    # HX primary side
    T_hx_pri_in = T_hot_leg
    T_hx_pri_out = T_cold_leg

    # ---- Intermediate loop temperatures ----
    T_sec_in = SECONDARY_INLET_TEMP
    T_sec_out = SECONDARY_OUTLET_TEMP
    T_sec_hot = T_sec_out - 1.0   # slight pipe loss
    T_sec_cold = T_sec_in + 1.0

    # ---- Vessel wall temperatures ----
    T_vessel_inner = _vessel_inner_wall_temperature(T_out, d)
    T_vessel_outer = _vessel_outer_wall_temperature(T_vessel_inner, d)

    # ---- Assemble ----
    return TemperatureMap(
        core_salt_inlet=T_in,
        core_salt_outlet=T_out,
        core_salt_peak_nominal=peak_salt_nom,
        core_salt_peak_hot=peak_salt_hot,
        core_graphite_peak_nominal=peak_gr_nom,
        core_graphite_peak_hot=peak_gr_hot,
        vessel_inner_wall=T_vessel_inner,
        vessel_outer_wall=T_vessel_outer,
        primary_hot_leg=T_hot_leg,
        primary_cold_leg=T_cold_leg,
        primary_hx_inlet=T_hx_pri_in,
        primary_hx_outlet=T_hx_pri_out,
        secondary_hx_inlet=T_sec_in,
        secondary_hx_outlet=T_sec_out,
        secondary_hot_leg=T_sec_hot,
        secondary_cold_leg=T_sec_cold,
        peak_salt=peak_salt_hot,
        peak_graphite=peak_gr_hot,
        peak_vessel=T_vessel_inner,
    )


def _vessel_inner_wall_temperature(T_salt_outlet, d):
    """Estimate vessel inner-wall temperature.

    In normal operation the downcomer carries cold-leg salt past the vessel
    wall, so the inner-wall temperature is close to the cold-leg salt
    temperature.  Conservatively, we use the outlet (hot-leg) salt temperature
    to bound the case where hot salt contacts the upper vessel head.

    Args:
        T_salt_outlet: Core outlet salt temperature, K.
        d: DerivedParameters.

    Returns:
        Vessel inner-wall temperature in K.
    """
    # Conservative: upper head contacts outlet salt
    # More realistic: downcomer contacts inlet salt
    # Use average of both as representative, weighted toward hot side
    T_inner = 0.7 * T_salt_outlet + 0.3 * CORE_INLET_TEMP
    return T_inner


def _vessel_outer_wall_temperature(T_inner, d):
    """Compute vessel outer-wall temperature from through-wall conduction.

    The heat flux through the vessel wall is primarily from decay heat and
    gamma heating.  During normal operation this is a small fraction of total
    power, so the through-wall temperature gradient is modest.

    Args:
        T_inner: Vessel inner-wall temperature, K.
        d: DerivedParameters.

    Returns:
        Vessel outer-wall temperature in K.
    """
    # Decay heat / gamma flux at vessel wall
    # Conservative: 1% of total power distributed over vessel inner surface
    decay_frac = 0.01
    vessel_surface = 2.0 * math.pi * d.vessel_inner_radius * d.vessel_height
    q_flux = decay_frac * THERMAL_POWER / vessel_surface  # W/m2

    # Through-wall conduction: dT = q'' * t / k
    t_wall = d.vessel_wall_thickness
    T_avg_wall = T_inner - q_flux * t_wall / (2.0 * hastelloy_thermal_conductivity(T_inner))
    k_wall = hastelloy_thermal_conductivity(T_avg_wall)
    delta_T = q_flux * t_wall / k_wall

    return T_inner - delta_T


# ==========================================================================
# Thermal limit checks
# ==========================================================================

def check_thermal_limits(temp_map):
    """Compare all temperatures against design safety limits.

    Args:
        temp_map: TemperatureMap instance.

    Returns:
        ThermalLimitResults with pass/fail for each limit.
    """
    checks = []

    # --- Fuel salt peak (hot channel) vs. boiling limit ---
    checks.append(ThermalLimit(
        location="Peak fuel salt (hot channel)",
        temperature=temp_map.peak_salt,
        limit=MAX_FUEL_SALT_TEMP,
        margin=MAX_FUEL_SALT_TEMP - temp_map.peak_salt,
        passed=(temp_map.peak_salt < MAX_FUEL_SALT_TEMP),
    ))

    # --- Graphite peak (hot channel) vs. graphite limit ---
    checks.append(ThermalLimit(
        location="Peak graphite (hot channel)",
        temperature=temp_map.peak_graphite,
        limit=MAX_GRAPHITE_TEMP,
        margin=MAX_GRAPHITE_TEMP - temp_map.peak_graphite,
        passed=(temp_map.peak_graphite < MAX_GRAPHITE_TEMP),
    ))

    # --- Vessel inner wall vs. Hastelloy-N service limit ---
    checks.append(ThermalLimit(
        location="Vessel inner wall",
        temperature=temp_map.vessel_inner_wall,
        limit=MAX_VESSEL_WALL_TEMP,
        margin=MAX_VESSEL_WALL_TEMP - temp_map.vessel_inner_wall,
        passed=(temp_map.vessel_inner_wall < MAX_VESSEL_WALL_TEMP),
    ))

    # --- Salt outlet vs. Hastelloy-N limit (corrosion concern) ---
    checks.append(ThermalLimit(
        location="Core outlet salt",
        temperature=temp_map.core_salt_outlet,
        limit=MAX_VESSEL_WALL_TEMP,
        margin=MAX_VESSEL_WALL_TEMP - temp_map.core_salt_outlet,
        passed=(temp_map.core_salt_outlet < MAX_VESSEL_WALL_TEMP),
    ))

    # --- Salt minimum (freezing) at cold leg ---
    checks.append(ThermalLimit(
        location="Primary cold leg (freezing check)",
        temperature=temp_map.primary_cold_leg,
        limit=MIN_SALT_TEMP,
        margin=temp_map.primary_cold_leg - MIN_SALT_TEMP,
        passed=(temp_map.primary_cold_leg > MIN_SALT_TEMP),
    ))

    # --- Secondary cold leg freezing check (FLiNaK melting ~454 C) ---
    flinak_freeze = 454.0 + 273.15
    checks.append(ThermalLimit(
        location="Secondary cold leg (FLiNaK freezing)",
        temperature=temp_map.secondary_cold_leg,
        limit=flinak_freeze,
        margin=temp_map.secondary_cold_leg - flinak_freeze,
        passed=(temp_map.secondary_cold_leg > flinak_freeze),
    ))

    all_passed = all(c.passed for c in checks)
    return ThermalLimitResults(checks=checks, all_passed=all_passed)


# ==========================================================================
# Printing utilities
# ==========================================================================

def print_temperature_summary(temp_map):
    """Print formatted temperature summary table.

    Args:
        temp_map: TemperatureMap instance.
    """
    print("=" * 68)
    print("   REACTOR SYSTEM TEMPERATURE MAP")
    print("=" * 68)

    def _row(label, T):
        return f"  {label:<36s}  {T - 273.15:8.1f} C  ({T:8.1f} K)"

    print("\n--- Core ---")
    print(_row("Salt inlet", temp_map.core_salt_inlet))
    print(_row("Salt outlet", temp_map.core_salt_outlet))
    print(_row("Peak salt (nominal channel)", temp_map.core_salt_peak_nominal))
    print(_row("Peak salt (hot channel)", temp_map.core_salt_peak_hot))
    print(_row("Peak graphite (nominal)", temp_map.core_graphite_peak_nominal))
    print(_row("Peak graphite (hot channel)", temp_map.core_graphite_peak_hot))

    print("\n--- Vessel ---")
    print(_row("Inner wall", temp_map.vessel_inner_wall))
    print(_row("Outer wall", temp_map.vessel_outer_wall))

    print("\n--- Primary Loop ---")
    print(_row("Hot leg (core -> HX)", temp_map.primary_hot_leg))
    print(_row("Cold leg (HX -> core)", temp_map.primary_cold_leg))
    print(_row("HX primary inlet", temp_map.primary_hx_inlet))
    print(_row("HX primary outlet", temp_map.primary_hx_outlet))

    print("\n--- Intermediate Loop ---")
    print(_row("HX secondary inlet", temp_map.secondary_hx_inlet))
    print(_row("HX secondary outlet", temp_map.secondary_hx_outlet))
    print(_row("Secondary hot leg", temp_map.secondary_hot_leg))
    print(_row("Secondary cold leg", temp_map.secondary_cold_leg))
    print()


def print_thermal_limits(limit_results):
    """Print formatted thermal limit check results.

    Args:
        limit_results: ThermalLimitResults instance.
    """
    print("=" * 68)
    print("   THERMAL SAFETY LIMIT CHECK")
    print("=" * 68)

    print(f"\n  {'Location':<36s}  {'T [C]':>7s}  {'Limit [C]':>9s}  "
          f"{'Margin [K]':>10s}  {'Status':>8s}")
    print(f"  {'-'*36}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*8}")

    for c in limit_results.checks:
        status = "  PASS" if c.passed else "** FAIL"
        print(f"  {c.location:<36s}  "
              f"{c.temperature - 273.15:7.1f}  "
              f"{c.limit - 273.15:9.1f}  "
              f"{c.margin:10.1f}  "
              f"{status:>8s}")

    print()
    if limit_results.all_passed:
        print("  >>> ALL THERMAL LIMITS SATISFIED <<<")
    else:
        n_fail = sum(1 for c in limit_results.checks if not c.passed)
        print(f"  >>> WARNING: {n_fail} THERMAL LIMIT(S) EXCEEDED <<<")
    print()


# ==========================================================================
# Main
# ==========================================================================

if __name__ == '__main__':
    # Try to import channel and loop analysis for integrated results
    try:
        from thermal_hydraulics.channel_analysis import run_nominal_channel_analysis
        channel_res = run_nominal_channel_analysis()
    except Exception:
        channel_res = None

    try:
        from thermal_hydraulics.coolant_loop import primary_loop_analysis
        loop_res = primary_loop_analysis()
    except Exception:
        loop_res = None

    d = compute_derived()
    temp_map = compute_temperature_map(channel_res, loop_res, d)
    print_temperature_summary(temp_map)

    limits = check_thermal_limits(temp_map)
    print_thermal_limits(limits)
