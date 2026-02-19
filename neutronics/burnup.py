"""
Simplified Fuel Depletion and Core Lifetime Estimate
====================================================

Computes the rate of U-235 consumption by fission and estimates the
core lifetime as the time until keff drops below 1.0.

Model:
  - All power comes from U-235 thermal fission
  - U-235 depletion follows exponential decay:
      N_235(t) = N_235(0) * exp(-sigma_f * phi_avg * t)
  - Average neutron flux is derived from power:
      phi_avg = P / (E_f * Sigma_f * V_core)
  - Core lifetime = time until keff < 1.0

Burnup is expressed in:
  - MWd/kg (megawatt-days per kilogram of initial heavy metal)
  - EFPD (effective full-power days)
  - Calendar years (at the design capacity factor)

Simplifications and caveats:
  - No fission product buildup (Xe-135, Sm-149 poisoning)
  - No breeding from U-238 captures (Pu-239 production)
  - No online reprocessing or fuel addition
  - Single-group flux (no spectral effects during depletion)
  - Constant power operation

For a real MSR with online processing, core lifetime would be
significantly longer due to:
  - Continuous removal of fission product poisons
  - Online fuel salt processing and makeup
  - Plutonium breeding from U-238

Sources:
  - Duderstadt & Hamilton, "Nuclear Reactor Analysis", Ch. 10
  - ORNL-4541: MSBR fuel cycle analysis
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    THERMAL_POWER, CORE_AVG_TEMP,
    U235_ENRICHMENT, UF4_MOLE_FRACTION,
    ENERGY_PER_FISSION, SIGMA_FISSION_U235,
    GRAPHITE_VOLUME_FRACTION,
    DESIGN_LIFE_YEARS, CAPACITY_FACTOR,
)
from neutronics.cross_sections import compute_homogenized_cross_sections, compute_number_densities
from neutronics.core_geometry import design_core
from neutronics.criticality import four_factor


# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================
AVOGADRO = 6.02214076e23     # 1/mol
MW_U235 = 235.04             # g/mol
SECONDS_PER_DAY = 86400.0    # s/day
DAYS_PER_YEAR = 365.25       # days/year


def u235_consumption_rate(thermal_power=None):
    """Compute the U-235 mass consumption rate from fission.

    Every fission of U-235 releases ~200 MeV and consumes one U-235 atom.
    The consumption rate is:
        dm/dt = P / E_f * M_235 / N_A

    where:
        P = thermal power (W)
        E_f = energy per fission (J)
        M_235 = atomic mass of U-235 (g/mol)
        N_A = Avogadro's number

    Args:
        thermal_power: Thermal power in W. Default from config.

    Returns:
        dict:
            - kg_per_day: U-235 consumption in kg/day
            - kg_per_year: U-235 consumption in kg/year
            - fission_rate: Fissions per second
            - atoms_per_day: U-235 atoms consumed per day
    """
    if thermal_power is None:
        thermal_power = THERMAL_POWER

    # Fission rate = P / E_f
    fission_rate = thermal_power / ENERGY_PER_FISSION  # fissions/s

    # Mass consumption rate
    atoms_per_day = fission_rate * SECONDS_PER_DAY
    grams_per_day = atoms_per_day * MW_U235 / AVOGADRO
    kg_per_day = grams_per_day / 1000.0
    kg_per_year = kg_per_day * DAYS_PER_YEAR

    return {
        'kg_per_day': kg_per_day,
        'kg_per_year': kg_per_year,
        'fission_rate': fission_rate,
        'atoms_per_day': atoms_per_day,
    }


def estimate_burnup(enrichment=None, geometry=None, thermal_power=None,
                     T=None, capacity_factor=None):
    """Estimate fuel burnup and core lifetime.

    Tracks U-235 depletion over time and determines when keff drops
    below 1.0 (end of cycle). The burnup is then computed from the
    total energy produced divided by initial heavy metal mass.

    Algorithm:
      1. Compute initial U-235 mass from salt composition
      2. Compute average flux from power and cross-sections
      3. Step through time, depleting U-235 exponentially
      4. At each step, recompute keff with reduced enrichment
      5. Core lifetime = time when keff < 1.0

    Args:
        enrichment: Initial U-235 enrichment. Default from config.
        geometry: CoreGeometry. Default: design_core().
        thermal_power: Thermal power in W. Default from config.
        T: Temperature in K. Default from config.
        capacity_factor: Plant capacity factor. Default from config.

    Returns:
        dict:
            - burnup_MWd_per_kg: Burnup in MWd per kg initial heavy metal
            - core_lifetime_EFPD: Core lifetime in effective full-power days
            - core_lifetime_years: Core lifetime in calendar years
            - consumption: U-235 consumption rate details
            - u235_mass_initial: Initial U-235 mass in kg
            - u_mass_initial: Initial total U mass in kg (heavy metal)
            - depletion_history: List of (time_days, enrichment, keff) tuples
    """
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if geometry is None:
        geometry = design_core()
    if thermal_power is None:
        thermal_power = THERMAL_POWER
    if T is None:
        T = CORE_AVG_TEMP
    if capacity_factor is None:
        capacity_factor = CAPACITY_FACTOR

    # --- Initial conditions ---
    # Number densities in salt
    nd = compute_number_densities(enrichment, T)
    N_U235_0 = nd['N_U235']  # 1/m^3 in salt
    N_U_total = nd['N_U_total']  # 1/m^3

    # Total salt volume (core + external)
    V_salt_total = geometry.fuel_salt_volume_total  # m^3

    # Initial masses
    u235_atoms_total = N_U235_0 * V_salt_total
    u_atoms_total = N_U_total * V_salt_total

    u235_mass_initial = u235_atoms_total * MW_U235 / AVOGADRO / 1000.0  # kg
    u_mass_initial = u_atoms_total * 238.03 / AVOGADRO / 1000.0  # kg (approx, using avg MW)

    # --- U-235 consumption rate ---
    consumption = u235_consumption_rate(thermal_power)

    # --- Average neutron flux ---
    # phi_avg = P / (E_f * Sigma_f * V_core)
    xs = compute_homogenized_cross_sections(enrichment=enrichment, T=T)
    V_core = geometry.core_volume
    Sigma_f = xs['sigma_f']  # 1/m

    if Sigma_f > 0 and V_core > 0:
        phi_avg = thermal_power / (ENERGY_PER_FISSION * Sigma_f * V_core)
    else:
        phi_avg = 0.0

    # --- Time-stepping depletion ---
    # dN_235/dt = -sigma_f * phi * N_235
    # N_235(t) = N_235(0) * exp(-sigma_f * phi * t)
    # Enrichment(t) = N_235(t) / N_U_total(t)
    # (assuming N_U_total doesn't change significantly)

    dt_days = 30.0  # Time step: 30 days
    dt_seconds = dt_days * SECONDS_PER_DAY

    depletion_history = []
    time_days = 0.0
    current_enrichment = enrichment
    core_lifetime_EFPD = 0.0
    lifetime_found = False

    # Depletion constant
    lambda_depl = consumption['kg_per_day'] / u235_mass_initial  # 1/day

    max_time_days = 20 * DAYS_PER_YEAR  # Max 20 years

    # Track U-235 mass
    u235_mass_current = u235_mass_initial

    while time_days < max_time_days:
        # Current enrichment from remaining U-235
        # enrichment ~ u235_mass / u_mass_initial (weight fraction, approximately)
        current_enrichment = u235_mass_current / u_mass_initial

        # Compute keff at current enrichment
        result = four_factor(enrichment=current_enrichment, geometry=geometry, T=T)
        k_current = result['keff']

        depletion_history.append((time_days, current_enrichment, k_current))

        if k_current < 1.0 and not lifetime_found:
            core_lifetime_EFPD = time_days
            lifetime_found = True
            break

        # Deplete U-235 for next step
        u235_mass_current -= consumption['kg_per_day'] * dt_days

        if u235_mass_current <= 0:
            u235_mass_current = 0.001  # Prevent going to zero
            core_lifetime_EFPD = time_days
            lifetime_found = True
            break

        time_days += dt_days

    if not lifetime_found:
        core_lifetime_EFPD = max_time_days

    # Calendar years at capacity factor
    core_lifetime_years = core_lifetime_EFPD / DAYS_PER_YEAR / capacity_factor

    # Burnup: total energy / initial heavy metal
    total_energy_MWd = thermal_power / 1e6 * core_lifetime_EFPD  # MW * days
    burnup_MWd_per_kg = total_energy_MWd / u_mass_initial  # MWd/kg

    return {
        'burnup_MWd_per_kg': burnup_MWd_per_kg,
        'core_lifetime_EFPD': core_lifetime_EFPD,
        'core_lifetime_years': core_lifetime_years,
        'consumption': consumption,
        'u235_mass_initial': u235_mass_initial,
        'u_mass_initial': u_mass_initial,
        'phi_avg': phi_avg,
        'enrichment_initial': enrichment,
        'enrichment_final': current_enrichment,
        'depletion_history': depletion_history,
        'capacity_factor': capacity_factor,
    }


def print_burnup_summary(result=None):
    """Print formatted burnup and core lifetime summary.

    Args:
        result: Dict from estimate_burnup(). Computed with defaults if None.
    """
    if result is None:
        result = estimate_burnup()

    print("=" * 72)
    print("  FUEL DEPLETION AND CORE LIFETIME ESTIMATE")
    print("=" * 72)

    print(f"\n  --- Initial Fuel Loading ---")
    print(f"    Enrichment (initial):     {result['enrichment_initial']*100:10.2f} %")
    print(f"    U-235 mass:               {result['u235_mass_initial']:10.2f} kg")
    print(f"    Total U mass (HM):        {result['u_mass_initial']:10.2f} kg")
    print(f"    Average flux:             {result['phi_avg']:10.3e} n/(m^2-s)")

    cons = result['consumption']
    print(f"\n  --- U-235 Consumption Rate ---")
    print(f"    Fission rate:             {cons['fission_rate']:10.3e} fissions/s")
    print(f"    U-235 consumed:           {cons['kg_per_day']:10.4f} kg/day")
    print(f"    U-235 consumed:           {cons['kg_per_year']:10.2f} kg/year")

    print(f"\n  --- Core Lifetime ---")
    print(f"    Lifetime (EFPD):          {result['core_lifetime_EFPD']:10.0f} days")
    print(f"    Lifetime (calendar):      {result['core_lifetime_years']:10.1f} years")
    print(f"    Capacity factor:          {result['capacity_factor']*100:10.1f} %")
    print(f"    Enrichment (final):       {result['enrichment_final']*100:10.2f} %")

    print(f"\n  --- Burnup ---")
    print(f"    Burnup:                   {result['burnup_MWd_per_kg']:10.1f} MWd/kgHM")
    print(f"    Burnup:                   {result['burnup_MWd_per_kg']*1000:10.0f} MWd/tHM")

    # Depletion history table
    history = result['depletion_history']
    print(f"\n  --- Depletion History ---")
    print(f"  {'Time (days)':>12s}  {'Time (yr)':>10s}  {'Enrichment':>12s}  {'keff':>10s}")

    # Print every Nth entry for readability
    n_entries = len(history)
    if n_entries <= 20:
        indices = range(n_entries)
    else:
        step = max(1, n_entries // 15)
        indices = list(range(0, n_entries, step))
        if n_entries - 1 not in indices:
            indices.append(n_entries - 1)

    for i in indices:
        t, e, k = history[i]
        print(f"  {t:12.0f}  {t/DAYS_PER_YEAR:10.2f}  {e*100:10.2f} %  {k:10.4f}")

    # Comparison with design life
    print(f"\n  --- Design Life Comparison ---")
    print(f"    Design life target:       {DESIGN_LIFE_YEARS:10d} years")
    cal_life = result['core_lifetime_years']
    if cal_life >= DESIGN_LIFE_YEARS:
        print(f"    Status:                   MEETS design life requirement")
        print(f"    Margin:                   {cal_life - DESIGN_LIFE_YEARS:10.1f} years")
    else:
        n_reloads = math.ceil(DESIGN_LIFE_YEARS / cal_life) - 1
        print(f"    Status:                   DOES NOT meet design life (need reprocessing)")
        print(f"    Reloads needed:           {n_reloads:10d} over {DESIGN_LIFE_YEARS} years")
        print(f"    Note: MSR online reprocessing can extend this significantly")

    print()


if __name__ == '__main__':
    print("Estimating fuel burnup and core lifetime...\n")
    result = estimate_burnup()
    print_burnup_summary(result)
