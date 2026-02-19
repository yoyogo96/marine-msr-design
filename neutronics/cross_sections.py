"""
Effective 1-Group Cross-Section Computation for Graphite-Moderated FLiBe+UF4 Lattice
=====================================================================================

Computes homogenized macroscopic cross-sections for the 40 MWth Marine MSR.

The lattice consists of hexagonal graphite blocks with cylindrical fuel salt
channels. Homogenization is performed via volume-weighted mixing of the salt
and graphite regions:

    Sigma_hom = f_salt * Sigma_salt + f_graphite * Sigma_graphite

where f_salt and f_graphite are the volume fractions.

Salt region contains: Li-7, Be, F, U-235, U-238  (from FLiBe + UF4)
Graphite region contains: C-12

Number densities are computed from molecular composition, salt density, and
Avogadro's number. Temperature dependence enters through:
  - Salt density rho(T)
  - Doppler broadening of U-238 resonances (Westcott g-factor correction)

All cross-sections are 1-group spectrum-averaged values appropriate for a
well-thermalized graphite-moderated MSR.

Units:
  - Microscopic cross-sections: m^2 (input as barns * 1e-28)
  - Macroscopic cross-sections: 1/m
  - Number densities: 1/m^3

Sources:
  - ENDF/B-VIII.0 evaluated nuclear data library
  - ORNL-4541 (MSBR conceptual design)
  - Duderstadt & Hamilton, "Nuclear Reactor Analysis"
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    LIF_MOLE_FRACTION, BEF2_MOLE_FRACTION, UF4_MOLE_FRACTION,
    LI7_ENRICHMENT,
    SIGMA_FISSION_U235, SIGMA_ABSORPTION_U235, SIGMA_ABSORPTION_U238,
    SIGMA_ABSORPTION_LI7, SIGMA_ABSORPTION_BE, SIGMA_ABSORPTION_F,
    SIGMA_SCATTER_GRAPHITE, SIGMA_ABSORPTION_GRAPHITE,
    NEUTRONS_PER_FISSION,
    GRAPHITE, GRAPHITE_VOLUME_FRACTION, FUEL_SALT_FRACTION,
    CORE_AVG_TEMP, U235_ENRICHMENT,
)
from thermal_hydraulics.salt_properties import flibe_density


# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================
AVOGADRO = 6.02214076e23       # 1/mol
BARN = 1e-28                   # m^2

# Molecular weights (g/mol)
MW_LIF = 25.94                 # LiF (using Li-7)
MW_BEF2 = 47.01               # BeF2
MW_UF4 = 314.02               # UF4 (avg U mass)
MW_U235 = 235.04               # U-235
MW_U238 = 238.05               # U-238
MW_LI7 = 7.016                 # Li-7
MW_BE = 9.012                  # Be-9
MW_F = 18.998                  # F-19
MW_C = 12.011                  # C-12 (graphite)

# Transport cross-section estimates (barns -> m^2)
SIGMA_TR_U235 = 15.0e-28      # m^2 (transport)
SIGMA_TR_U238 = 15.0e-28      # m^2
SIGMA_TR_LI7 = 1.4e-28        # m^2
SIGMA_TR_BE = 6.1e-28          # m^2
SIGMA_TR_F = 3.6e-28           # m^2
SIGMA_TR_C = 4.7e-28           # m^2 (same as scatter for graphite, low absorption)
SIGMA_SCATTER_C = 4.7e-28      # m^2


def _salt_molecular_density(T, uf4_mol_fraction=UF4_MOLE_FRACTION):
    """Compute molecular number density of the FLiBe+UF4 salt mixture.

    The salt is treated as a mixture of LiF, BeF2, and UF4 molecules.
    Total molecular density = rho * N_A / MW_avg

    Args:
        T: Temperature in K
        uf4_mol_fraction: UF4 mole fraction (default from config)

    Returns:
        float: Total molecular number density in 1/m^3
    """
    rho = flibe_density(T, uf4_mol_fraction)  # kg/m^3

    # Average molecular weight of the salt mixture (g/mol)
    # Renormalize mole fractions to ensure they sum to 1
    x_lif = LIF_MOLE_FRACTION
    x_bef2 = BEF2_MOLE_FRACTION
    x_uf4 = uf4_mol_fraction
    x_total = x_lif + x_bef2 + x_uf4
    x_lif /= x_total
    x_bef2 /= x_total
    x_uf4 /= x_total

    MW_avg = x_lif * MW_LIF + x_bef2 * MW_BEF2 + x_uf4 * MW_UF4  # g/mol

    # molecular density: (kg/m^3) * (1/mol) / (g/mol) * (1000 g/kg)
    N_mol = rho * AVOGADRO / (MW_avg * 1e-3)  # 1/m^3
    return N_mol, x_lif, x_bef2, x_uf4, MW_avg


def compute_number_densities(enrichment, T, uf4_mol_fraction=UF4_MOLE_FRACTION):
    """Compute individual atom number densities for salt constituents.

    From the molecular density of the salt and the stoichiometry of each
    molecule (LiF has 1 Li + 1 F, BeF2 has 1 Be + 2 F, UF4 has 1 U + 4 F),
    we extract individual atom densities.

    Args:
        enrichment: U-235 weight fraction in uranium (0 to 1)
        T: Temperature in K
        uf4_mol_fraction: UF4 mole fraction

    Returns:
        dict: Atom number densities in 1/m^3 for each isotope
    """
    N_mol, x_lif, x_bef2, x_uf4, MW_avg = _salt_molecular_density(T, uf4_mol_fraction)

    # Atom densities from stoichiometry
    # LiF: 1 Li + 1 F per molecule
    N_Li = N_mol * x_lif * 1.0
    N_F_from_LiF = N_mol * x_lif * 1.0

    # BeF2: 1 Be + 2 F per molecule
    N_Be = N_mol * x_bef2 * 1.0
    N_F_from_BeF2 = N_mol * x_bef2 * 2.0

    # UF4: 1 U + 4 F per molecule
    N_U = N_mol * x_uf4 * 1.0
    N_F_from_UF4 = N_mol * x_uf4 * 4.0

    # Total fluorine
    N_F = N_F_from_LiF + N_F_from_BeF2 + N_F_from_UF4

    # Li is enriched in Li-7, so effectively all Li is Li-7
    N_Li7 = N_Li * LI7_ENRICHMENT

    # U-235 / U-238 split by enrichment (weight fraction)
    # n_235 / n_total = (e / M_235) / (e/M_235 + (1-e)/M_238)
    atom_frac_235 = (enrichment / MW_U235) / (
        enrichment / MW_U235 + (1.0 - enrichment) / MW_U238
    )
    N_U235 = N_U * atom_frac_235
    N_U238 = N_U * (1.0 - atom_frac_235)

    return {
        'N_Li7': N_Li7,
        'N_Be': N_Be,
        'N_F': N_F,
        'N_U235': N_U235,
        'N_U238': N_U238,
        'N_U_total': N_U,
        'N_mol': N_mol,
        'MW_avg': MW_avg,
    }


def compute_graphite_number_density():
    """Compute carbon atom number density in nuclear graphite.

    Returns:
        float: C-12 number density in 1/m^3
    """
    rho_C = GRAPHITE['density']  # kg/m^3
    N_C = rho_C * AVOGADRO / (MW_C * 1e-3)  # 1/m^3
    return N_C


def _doppler_correction(T, T_ref=293.15):
    """Simple Doppler broadening correction factor for U-238 absorption.

    The effective resonance absorption cross-section of U-238 increases
    with temperature due to Doppler broadening. A square-root
    dependence is used as a first approximation:

        sigma_a(T) = sigma_a(T_ref) * sqrt(T / T_ref)

    This gives approximately +0.4% per 100 K increase, consistent with
    ENDF-based calculations for graphite-moderated spectra.

    Args:
        T: Actual temperature in K
        T_ref: Reference temperature in K (default 293.15 K = 20 C)

    Returns:
        float: Multiplicative correction factor (> 1 for T > T_ref)
    """
    return math.sqrt(T / T_ref)


def compute_homogenized_cross_sections(enrichment=None, graphite_fraction=None,
                                        uf4_fraction=None, T=None):
    """Compute effective 1-group homogenized macroscopic cross-sections.

    Performs volume-weighted homogenization of salt and graphite regions
    in the hexagonal channel lattice.

    Args:
        enrichment: U-235 enrichment (weight fraction). Default from config.
        graphite_fraction: Volume fraction of graphite. Default from config.
        uf4_fraction: UF4 mole fraction in salt. Default from config.
        T: Temperature in K. Default from config.

    Returns:
        dict: Macroscopic cross-sections and nuclear parameters:
            - sigma_a: Total macroscopic absorption cross-section (1/m)
            - sigma_f: Macroscopic fission cross-section (1/m)
            - nu_sigma_f: nu * macroscopic fission (1/m)
            - sigma_tr: Macroscopic transport cross-section (1/m)
            - D: Diffusion coefficient (m)
            - sigma_s: Macroscopic scattering cross-section (1/m)
            - sigma_a_fuel: Absorption in fuel isotopes only (1/m)
            - sigma_a_salt: Total absorption in salt region (1/m)
            - sigma_a_graphite: Absorption in graphite (1/m)
            - number_densities: dict of atom number densities
    """
    # Apply defaults
    if enrichment is None:
        enrichment = U235_ENRICHMENT
    if graphite_fraction is None:
        graphite_fraction = GRAPHITE_VOLUME_FRACTION
    if uf4_fraction is None:
        uf4_fraction = UF4_MOLE_FRACTION
    if T is None:
        T = CORE_AVG_TEMP

    salt_fraction = 1.0 - graphite_fraction

    # --- Compute number densities ---
    nd = compute_number_densities(enrichment, T, uf4_fraction)
    N_C = compute_graphite_number_density()

    # --- Doppler correction for U-238 ---
    doppler = _doppler_correction(T)

    # --- Macroscopic cross-sections in the SALT region ---
    # Absorption
    Sigma_a_U235 = nd['N_U235'] * SIGMA_ABSORPTION_U235
    Sigma_a_U238 = nd['N_U238'] * SIGMA_ABSORPTION_U238 * doppler
    Sigma_a_Li7 = nd['N_Li7'] * SIGMA_ABSORPTION_LI7
    Sigma_a_Be = nd['N_Be'] * SIGMA_ABSORPTION_BE
    Sigma_a_F = nd['N_F'] * SIGMA_ABSORPTION_F

    Sigma_a_salt = Sigma_a_U235 + Sigma_a_U238 + Sigma_a_Li7 + Sigma_a_Be + Sigma_a_F
    Sigma_a_fuel = Sigma_a_U235 + Sigma_a_U238  # Fuel isotopes only

    # Fission (only U-235 for thermal spectrum)
    Sigma_f_salt = nd['N_U235'] * SIGMA_FISSION_U235
    nu_Sigma_f_salt = NEUTRONS_PER_FISSION * Sigma_f_salt

    # Transport (salt)
    Sigma_tr_salt = (nd['N_U235'] * SIGMA_TR_U235 +
                     nd['N_U238'] * SIGMA_TR_U238 +
                     nd['N_Li7'] * SIGMA_TR_LI7 +
                     nd['N_Be'] * SIGMA_TR_BE +
                     nd['N_F'] * SIGMA_TR_F)

    # Scattering in salt (transport - absorption, approximate)
    Sigma_s_salt = max(Sigma_tr_salt - Sigma_a_salt, 0.0)

    # --- Macroscopic cross-sections in the GRAPHITE region ---
    Sigma_a_C = N_C * SIGMA_ABSORPTION_GRAPHITE
    Sigma_tr_C = N_C * SIGMA_TR_C
    Sigma_s_C = N_C * SIGMA_SCATTER_C

    # --- Homogenized macroscopic cross-sections (volume-weighted) ---
    sigma_a = salt_fraction * Sigma_a_salt + graphite_fraction * Sigma_a_C
    sigma_f = salt_fraction * Sigma_f_salt  # Fission only in salt
    nu_sigma_f = salt_fraction * nu_Sigma_f_salt
    sigma_tr = salt_fraction * Sigma_tr_salt + graphite_fraction * Sigma_tr_C
    sigma_s = salt_fraction * Sigma_s_salt + graphite_fraction * Sigma_s_C

    # Diffusion coefficient: D = 1 / (3 * Sigma_tr)
    D = 1.0 / (3.0 * sigma_tr)

    # Also provide salt-only and graphite-only absorption for four-factor formula
    sigma_a_salt_hom = salt_fraction * Sigma_a_salt
    sigma_a_graphite_hom = graphite_fraction * Sigma_a_C
    sigma_a_fuel_hom = salt_fraction * Sigma_a_fuel

    return {
        'sigma_a': sigma_a,
        'sigma_f': sigma_f,
        'nu_sigma_f': nu_sigma_f,
        'sigma_tr': sigma_tr,
        'D': D,
        'sigma_s': sigma_s,
        'sigma_a_fuel': sigma_a_fuel_hom,
        'sigma_a_salt': sigma_a_salt_hom,
        'sigma_a_graphite': sigma_a_graphite_hom,
        'number_densities': nd,
        # Component breakdowns
        'Sigma_a_U235': Sigma_a_U235,
        'Sigma_a_U238': Sigma_a_U238,
        'Sigma_a_Li7': Sigma_a_Li7,
        'Sigma_a_Be': Sigma_a_Be,
        'Sigma_a_F': Sigma_a_F,
        'Sigma_a_C': Sigma_a_C,
        'Sigma_f_U235': Sigma_f_salt,
        'salt_fraction': salt_fraction,
        'graphite_fraction': graphite_fraction,
    }


def print_cross_sections(xs=None):
    """Print formatted cross-section summary.

    Args:
        xs: Cross-section dict from compute_homogenized_cross_sections().
            Computed with defaults if not provided.
    """
    if xs is None:
        xs = compute_homogenized_cross_sections()

    nd = xs['number_densities']

    print("=" * 72)
    print("  HOMOGENIZED 1-GROUP CROSS-SECTIONS")
    print("=" * 72)

    print(f"\n  --- Number Densities (in salt region) ---")
    print(f"    N(U-235):    {nd['N_U235']:.4e}  1/m^3")
    print(f"    N(U-238):    {nd['N_U238']:.4e}  1/m^3")
    print(f"    N(Li-7):     {nd['N_Li7']:.4e}  1/m^3")
    print(f"    N(Be):       {nd['N_Be']:.4e}  1/m^3")
    print(f"    N(F):        {nd['N_F']:.4e}  1/m^3")

    print(f"\n  --- Component Absorption (macroscopic, salt region) ---")
    print(f"    Sigma_a(U-235):  {xs['Sigma_a_U235']:12.4f}  1/m")
    print(f"    Sigma_a(U-238):  {xs['Sigma_a_U238']:12.4f}  1/m")
    print(f"    Sigma_a(Li-7):   {xs['Sigma_a_Li7']:12.6f}  1/m")
    print(f"    Sigma_a(Be):     {xs['Sigma_a_Be']:12.6f}  1/m")
    print(f"    Sigma_a(F):      {xs['Sigma_a_F']:12.6f}  1/m")
    print(f"    Sigma_a(C):      {xs['Sigma_a_C']:12.6f}  1/m")

    print(f"\n  --- Homogenized Cross-Sections ---")
    print(f"    Volume fractions: salt={xs['salt_fraction']:.3f}, "
          f"graphite={xs['graphite_fraction']:.3f}")
    print(f"    Sigma_a:      {xs['sigma_a']:12.4f}  1/m")
    print(f"    Sigma_f:      {xs['sigma_f']:12.4f}  1/m")
    print(f"    nu*Sigma_f:   {xs['nu_sigma_f']:12.4f}  1/m")
    print(f"    Sigma_tr:     {xs['sigma_tr']:12.4f}  1/m")
    print(f"    Sigma_s:      {xs['sigma_s']:12.4f}  1/m")
    print(f"    D:            {xs['D']:12.6f}  m")
    print(f"    k_inf (est):  {xs['nu_sigma_f']/xs['sigma_a']:12.4f}")

    print()


if __name__ == '__main__':
    print("Computing cross-sections with default config parameters...\n")
    xs = compute_homogenized_cross_sections()
    print_cross_sections(xs)

    # Parametric study: enrichment sensitivity
    print("\n  --- Enrichment Sensitivity ---")
    print(f"  {'Enrichment':>12s}  {'sigma_a':>10s}  {'sigma_f':>10s}  {'nu*sigma_f':>10s}  {'k_inf':>10s}")
    for e in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        xs_e = compute_homogenized_cross_sections(enrichment=e)
        k_inf = xs_e['nu_sigma_f'] / xs_e['sigma_a']
        print(f"  {e*100:10.1f} %  {xs_e['sigma_a']:10.4f}  {xs_e['sigma_f']:10.4f}  "
              f"{xs_e['nu_sigma_f']:10.4f}  {k_inf:10.4f}")
