"""
Dose Rate Calculation at Key Ship Locations
=============================================

Point-kernel dose rate calculations at regulatory and operational
locations throughout the ship.

Locations:
  1. Reactor compartment boundary (5 m from core center)
  2. Control room (10 m)
  3. Crew quarters (15 m)
  4. Weather deck above reactor (8 m vertical)

Dose limits:
  - Occupational: 20 mSv/yr (ICRP 103)
  - Public: 1 mSv/yr
  - Compartment boundary: target < 10 uSv/hr

References:
  - ICRP Publication 103 (2007)
  - 10 CFR 20 (NRC Radiation Protection Standards)
  - IMO Resolution A.491(XII) (Nuclear Ships Code)
  - IAEA SSR-2/1 (Safety of Nuclear Power Plants)
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
    OCCUPATIONAL_DOSE_LIMIT, PUBLIC_DOSE_LIMIT,
    EMERGENCY_DOSE_LIMIT, compute_derived,
)
from shielding.source_term import compute_source_term, decay_gamma_source_vs_time
from shielding.attenuation import (
    default_shield_layers, compute_attenuation,
    neutron_flux_to_dose, gamma_flux_to_dose,
)


# =============================================================================
# Ship Location Definitions
# =============================================================================

@dataclass
class ShipLocation:
    """Definition of a dose assessment location on the ship."""

    name: str
    distance_from_core: float   # m
    occupancy_hours_per_year: float  # hr/yr
    dose_limit_mSv_yr: float   # mSv/yr
    category: str               # 'occupational', 'public', 'restricted'
    description: str = ""


# Standard assessment locations
ASSESSMENT_LOCATIONS = [
    ShipLocation(
        name="Reactor compartment boundary",
        distance_from_core=5.0,
        occupancy_hours_per_year=500,
        dose_limit_mSv_yr=20.0,
        category="restricted",
        description="Boundary of reactor shielded compartment",
    ),
    ShipLocation(
        name="Control room",
        distance_from_core=10.0,
        occupancy_hours_per_year=2000,
        dose_limit_mSv_yr=5.0,
        category="occupational",
        description="Main engine control room",
    ),
    ShipLocation(
        name="Crew quarters",
        distance_from_core=15.0,
        occupancy_hours_per_year=4000,
        dose_limit_mSv_yr=1.0,
        category="public",
        description="Crew living spaces",
    ),
    ShipLocation(
        name="Weather deck above reactor",
        distance_from_core=8.0,
        occupancy_hours_per_year=1000,
        dose_limit_mSv_yr=5.0,
        category="occupational",
        description="Weather deck directly above reactor compartment",
    ),
    ShipLocation(
        name="Engine room access",
        distance_from_core=6.0,
        occupancy_hours_per_year=1000,
        dose_limit_mSv_yr=20.0,
        category="occupational",
        description="Engineering access corridors",
    ),
]


# =============================================================================
# Dose Assessment Results
# =============================================================================

@dataclass
class LocationDose:
    """Dose assessment for a single location."""

    location: ShipLocation
    neutron_dose_rate: float     # mSv/hr
    gamma_dose_rate: float       # mSv/hr
    total_dose_rate: float       # mSv/hr
    annual_dose: float           # mSv/yr
    dose_limit: float            # mSv/yr
    margin: float               # fraction remaining to limit
    meets_limit: bool


@dataclass
class DoseMap:
    """Complete dose map for all assessment locations."""

    locations: list             # List of LocationDose
    shield_description: str     # Shield configuration summary
    max_dose_rate: float        # mSv/hr (worst location)
    max_annual_dose: float      # mSv/yr (worst location)
    all_within_limits: bool     # True if all locations pass


# =============================================================================
# Dose Calculations
# =============================================================================

def compute_dose_at_location(location, shield_layers, S_neutron, S_gamma):
    """Compute dose rate and annual dose at a ship location.

    Args:
        location: ShipLocation definition
        shield_layers: List of ShieldLayer objects
        S_neutron: Neutron source rate (n/s)
        S_gamma: Gamma source rate (photons/s)

    Returns:
        LocationDose dataclass
    """
    R_cm = location.distance_from_core * 100.0  # m -> cm

    result = compute_attenuation(shield_layers, S_neutron, S_gamma, R_ref_cm=R_cm)

    annual_dose = result.total_dose_rate * location.occupancy_hours_per_year
    margin = 1.0 - annual_dose / location.dose_limit_mSv_yr if location.dose_limit_mSv_yr > 0 else 0.0
    meets = annual_dose <= location.dose_limit_mSv_yr

    return LocationDose(
        location=location,
        neutron_dose_rate=result.neutron_dose_rate,
        gamma_dose_rate=result.gamma_dose_rate,
        total_dose_rate=result.total_dose_rate,
        annual_dose=annual_dose,
        dose_limit=location.dose_limit_mSv_yr,
        margin=margin,
        meets_limit=meets,
    )


def compute_dose_map(shield_layers=None, locations=None, design_params=None):
    """Compute dose map for all assessment locations.

    Args:
        shield_layers: Shield configuration (default from default_shield_layers)
        locations: List of ShipLocation (default: ASSESSMENT_LOCATIONS)
        design_params: DerivedParameters (computed if None)

    Returns:
        DoseMap dataclass
    """
    if design_params is None:
        design_params = compute_derived()
    if shield_layers is None:
        shield_layers = default_shield_layers(design_params)
    if locations is None:
        locations = ASSESSMENT_LOCATIONS

    st = compute_source_term(design_params)

    results = []
    for loc in locations:
        ld = compute_dose_at_location(loc, shield_layers, st.S_neutron, st.S_gamma_total)
        results.append(ld)

    # Shield description
    shield_desc = " | ".join(f"{l.name}: {l.thickness:.0f} cm" for l in shield_layers)

    max_dose_rate = max(r.total_dose_rate for r in results)
    max_annual = max(r.annual_dose for r in results)
    all_ok = all(r.meets_limit for r in results)

    return DoseMap(
        locations=results,
        shield_description=shield_desc,
        max_dose_rate=max_dose_rate,
        max_annual_dose=max_annual,
        all_within_limits=all_ok,
    )


def compute_shutdown_dose_map(shield_layers=None, locations=None,
                               t_after_shutdown=3600.0, design_params=None):
    """Compute dose map at a given time after reactor shutdown.

    After shutdown, only decay gammas contribute (no fission neutrons).

    Args:
        shield_layers: Shield configuration
        locations: Assessment locations
        t_after_shutdown: Time after shutdown (s)
        design_params: DerivedParameters

    Returns:
        DoseMap dataclass
    """
    if design_params is None:
        design_params = compute_derived()
    if shield_layers is None:
        shield_layers = default_shield_layers(design_params)
    if locations is None:
        locations = ASSESSMENT_LOCATIONS

    # Decay gamma source at t_after_shutdown
    t_arr, P_gamma_arr, S_gamma_arr = decay_gamma_source_vs_time(
        config.THERMAL_POWER, np.array([t_after_shutdown])
    )
    S_gamma_decay = float(S_gamma_arr[0])

    # No fission neutrons after shutdown
    S_neutron = 0.0

    results = []
    for loc in locations:
        ld = compute_dose_at_location(loc, shield_layers, S_neutron, S_gamma_decay)
        results.append(ld)

    shield_desc = f"Shutdown t={t_after_shutdown:.0f}s, decay gammas only"
    max_dose_rate = max(r.total_dose_rate for r in results)
    max_annual = max(r.annual_dose for r in results)
    all_ok = all(r.meets_limit for r in results)

    return DoseMap(
        locations=results,
        shield_description=shield_desc,
        max_dose_rate=max_dose_rate,
        max_annual_dose=max_annual,
        all_within_limits=all_ok,
    )


# =============================================================================
# Printing
# =============================================================================

def print_dose_map(dm):
    """Print formatted dose map table.

    Args:
        dm: DoseMap dataclass
    """
    print("=" * 90)
    print("   DOSE RATE MAP - 40 MWth Marine MSR")
    print("=" * 90)

    print(f"\n  Shield: {dm.shield_description}")

    print(f"\n  {'Location':<30s}  {'Dist [m]':>8s}  {'Dose rate':>10s}  "
          f"{'Annual':>10s}  {'Limit':>10s}  {'Margin':>8s}  {'Status':>8s}")
    print(f"  {'':>30s}  {'':>8s}  {'[uSv/hr]':>10s}  "
          f"{'[mSv/yr]':>10s}  {'[mSv/yr]':>10s}  {'[%]':>8s}  {'':>8s}")
    print(f"  {'-' * 30}  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    for ld in dm.locations:
        status = "OK" if ld.meets_limit else "FAIL"
        margin_pct = ld.margin * 100
        print(f"  {ld.location.name:<30s}  "
              f"{ld.location.distance_from_core:8.1f}  "
              f"{ld.total_dose_rate * 1e3:10.3f}  "
              f"{ld.annual_dose:10.3f}  "
              f"{ld.dose_limit:10.1f}  "
              f"{margin_pct:7.1f}%  "
              f"{'  ' + status:>8s}")

    print(f"\n  Overall status: {'ALL LIMITS MET' if dm.all_within_limits else 'SOME LIMITS EXCEEDED'}")
    print(f"  Max dose rate:  {dm.max_dose_rate * 1e3:.3f} uSv/hr")
    print(f"  Max annual:     {dm.max_annual_dose:.3f} mSv/yr")

    print("\n" + "=" * 90)


def print_dose_breakdown(dm):
    """Print neutron/gamma breakdown for each location.

    Args:
        dm: DoseMap dataclass
    """
    print("=" * 80)
    print("   DOSE RATE BREAKDOWN (Neutron / Gamma)")
    print("=" * 80)

    print(f"\n  {'Location':<30s}  {'Neutron':>12s}  {'Gamma':>12s}  "
          f"{'Total':>12s}  {'n/g ratio':>9s}")
    print(f"  {'':>30s}  {'[uSv/hr]':>12s}  {'[uSv/hr]':>12s}  "
          f"{'[uSv/hr]':>12s}  {'':>9s}")
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 9}")

    for ld in dm.locations:
        n_dose = ld.neutron_dose_rate * 1e3
        g_dose = ld.gamma_dose_rate * 1e3
        total = ld.total_dose_rate * 1e3
        ratio = n_dose / g_dose if g_dose > 0 else float('inf')
        print(f"  {ld.location.name:<30s}  "
              f"{n_dose:12.3f}  "
              f"{g_dose:12.3f}  "
              f"{total:12.3f}  "
              f"{ratio:9.2f}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()

    # --- Normal operation dose map ---
    print("NORMAL OPERATION:")
    dm = compute_dose_map(design_params=d)
    print_dose_map(dm)
    print_dose_breakdown(dm)

    # --- Post-shutdown dose map ---
    print("\nPOST-SHUTDOWN (1 hour after scram):")
    dm_sd = compute_shutdown_dose_map(t_after_shutdown=3600.0, design_params=d)
    print_dose_map(dm_sd)

    # --- Dose vs time after shutdown ---
    print("\n--- Dose Rate at Compartment Boundary vs Time After Shutdown ---")
    times = [10, 60, 600, 3600, 86400, 604800]
    labels = ["10 s", "1 min", "10 min", "1 hr", "1 day", "1 week"]

    shield_layers = default_shield_layers(d)
    comp_boundary = ASSESSMENT_LOCATIONS[0]

    t_arr, P_gamma_arr, S_gamma_arr = decay_gamma_source_vs_time(
        config.THERMAL_POWER, np.array(times, dtype=float)
    )

    print(f"  {'Time':>10s}  {'Dose rate [uSv/hr]':>18s}  {'Relative to op.':>16s}")
    print(f"  {'-' * 10}  {'-' * 18}  {'-' * 16}")

    # Operating dose rate at boundary for reference
    st = compute_source_term(d)
    res_op = compute_attenuation(shield_layers, st.S_neutron, st.S_gamma_total,
                                  R_ref_cm=comp_boundary.distance_from_core * 100)
    dose_op = res_op.total_dose_rate

    for lbl, S_g in zip(labels, S_gamma_arr):
        res = compute_attenuation(shield_layers, 0.0, float(S_g),
                                   R_ref_cm=comp_boundary.distance_from_core * 100)
        ratio = res.total_dose_rate / dose_op if dose_op > 0 else 0
        print(f"  {lbl:>10s}  {res.total_dose_rate * 1e3:18.4f}  {ratio * 100:15.2f}%")

    print(f"\n  Operating dose rate at boundary: {dose_op * 1e3:.4f} uSv/hr")
