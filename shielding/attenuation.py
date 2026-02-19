"""
Multi-Layer Shield Attenuation Analysis
=========================================

Removal cross-section method (Goldstein approximation) for computing
neutron and gamma attenuation through the multi-layer biological shield.

Shield configuration (inside to outside):
  1. Reactor vessel wall (Hastelloy-N, ~2 cm)
  2. Primary shield (steel + B4C composite, ~30-50 cm)
  3. Biological shield (heavy concrete / baryte, ~100-150 cm)

Method:
  Neutrons:  Phi(d) = S/(4*pi*R^2) * exp(-Sigma_R * d) * B_n(d)
  Gammas:    Phi(d) = S/(4*pi*R^2) * exp(-mu * d) * B_gamma(d)

References:
  - Goldstein, "Fundamental Aspects of Reactor Shielding"
  - Shultis & Faw, "Radiation Shielding"
  - Lamarsh & Baratta, Ch. 10
  - ANS-6.4.3 (removal cross-sections)
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from config import (
    HASTELLOY_N, CONCRETE, B4C, STEEL,
    OCCUPATIONAL_DOSE_LIMIT, PUBLIC_DOSE_LIMIT,
    compute_derived,
)
from shielding.source_term import compute_source_term


# =============================================================================
# Material Shielding Properties
# =============================================================================

# Removal cross-sections for fast neutrons (cm^-1)
# Source: Shultis & Faw, Lamarsh
REMOVAL_XS = {
    'water':           0.103,   # cm^-1
    'iron':            0.168,   # cm^-1
    'steel':           0.168,   # cm^-1 (similar to iron)
    'concrete':        0.089,   # cm^-1 (regular)
    'heavy_concrete':  0.094,   # cm^-1 (baryte)
    'B4C':             0.060,   # cm^-1 (thermal absorption very high)
    'hastelloy_N':     0.155,   # cm^-1 (Ni-based alloy, ~like Ni)
    'lead':            0.118,   # cm^-1
}

# Mass attenuation coefficients for gammas at various energies (cm^2/g)
# Approximate values for 1-2 MeV range (representative of fission gammas)
GAMMA_MU_RHO = {
    # material: mu/rho at ~1.5 MeV  [cm^2/g]
    'water':           0.057,
    'iron':            0.047,
    'steel':           0.047,
    'concrete':        0.052,
    'heavy_concrete':  0.048,
    'B4C':             0.055,
    'hastelloy_N':     0.046,
    'lead':            0.052,
}

# Material densities (g/cm^3) for use with mu/rho
MATERIAL_DENSITY = {
    'water':           1.0,
    'iron':            7.87,
    'steel':           STEEL['density'] / 1000,       # kg/m3 -> g/cm3
    'concrete':        CONCRETE['density'] / 1000,
    'heavy_concrete':  CONCRETE['density_heavy'] / 1000,
    'B4C':             B4C['density'] / 1000,
    'hastelloy_N':     HASTELLOY_N['density'] / 1000,
    'lead':            11.34,
}


# =============================================================================
# Shield Layer Definition
# =============================================================================

@dataclass
class ShieldLayer:
    """Definition of a single shield layer."""

    name: str                   # descriptive name
    material: str               # key into property dictionaries
    thickness: float            # cm
    density: float = 0.0        # g/cm3 (filled from tables if 0)

    def __post_init__(self):
        if self.density <= 0:
            self.density = MATERIAL_DENSITY.get(self.material, 1.0)


@dataclass
class AttenuationResult:
    """Results of multi-layer shield attenuation calculation."""

    # --- Shield definition ---
    layers: List[ShieldLayer]
    total_thickness: float      # cm

    # --- Neutron attenuation ---
    neutron_source: float       # n/s
    neutron_flux_unshielded: float  # n/(cm2-s) at reference distance
    neutron_flux_shielded: float    # n/(cm2-s) after all layers
    neutron_attenuation_factor: float
    neutron_dose_rate: float    # mSv/hr at reference point

    # --- Gamma attenuation ---
    gamma_source: float         # photons/s
    gamma_flux_unshielded: float    # ph/(cm2-s)
    gamma_flux_shielded: float      # ph/(cm2-s)
    gamma_attenuation_factor: float
    gamma_dose_rate: float      # mSv/hr at reference point

    # --- Combined ---
    total_dose_rate: float      # mSv/hr
    annual_dose: float          # mSv/yr (2000 hr occupancy)

    # --- Per-layer breakdown ---
    layer_attenuation_n: list   # neutron attenuation per layer
    layer_attenuation_g: list   # gamma attenuation per layer

    # --- Reference distance ---
    reference_distance: float   # cm from core center


# =============================================================================
# Buildup Factors
# =============================================================================

def neutron_buildup(thickness_cm, material='concrete'):
    """Simplified neutron buildup factor.

    B_n ~ 1 + a*d for thin shields, saturates for thick shields.
    Conservative: B = 1 + 0.5 * (Sigma_R * d)^0.5

    Args:
        thickness_cm: Shield thickness (cm)
        material: Shield material key

    Returns:
        float: Buildup factor (> 1)
    """
    sigma_r = REMOVAL_XS.get(material, 0.089)
    mfp = sigma_r * thickness_cm

    if mfp < 0.1:
        return 1.0

    # Taylor form approximation
    return 1.0 + 0.5 * math.sqrt(mfp)


def gamma_buildup(thickness_cm, material='concrete', energy_MeV=1.5):
    """Simplified gamma buildup factor using Berger's formula.

    B = 1 + a * mu*d * exp(b * mu*d)
    Simplified: use Taylor approximation for moderate shields.

    Args:
        thickness_cm: Shield thickness (cm)
        material: Shield material key
        energy_MeV: Gamma energy (MeV)

    Returns:
        float: Gamma buildup factor (> 1)
    """
    mu_rho = GAMMA_MU_RHO.get(material, 0.050)
    rho = MATERIAL_DENSITY.get(material, 2.0)
    mu = mu_rho * rho  # cm^-1

    mfp = mu * thickness_cm

    if mfp < 0.1:
        return 1.0

    # Berger approximation (conservative for shielding design)
    # For concrete at 1-2 MeV: B ~ 1 + mfp (roughly linear for moderate thickness)
    # More accurate: B = 1 + 0.8*mfp for concrete, B = 1 + 0.5*mfp for steel
    if material in ('concrete', 'heavy_concrete'):
        a = 0.8
    elif material in ('iron', 'steel', 'hastelloy_N'):
        a = 0.5
    else:
        a = 0.6

    return 1.0 + a * mfp * math.exp(-0.1 * mfp)


# =============================================================================
# Flux-to-Dose Conversion
# =============================================================================

def neutron_flux_to_dose(flux_n_cm2_s):
    """Convert neutron flux to dose rate.

    Using ICRP conversion factor for fast fission neutrons (~2 MeV):
    ~3.5e-6 mSv*cm2/n (= 350 pSv*cm2/n)

    Args:
        flux_n_cm2_s: Neutron flux in n/(cm2*s)

    Returns:
        float: Dose rate in mSv/hr
    """
    # Conversion: 3.5e-6 mSv*cm2 per neutron
    dose_per_neutron = 3.5e-6  # mSv*cm2
    dose_rate_per_s = flux_n_cm2_s * dose_per_neutron  # mSv/s
    return dose_rate_per_s * 3600  # mSv/hr


def gamma_flux_to_dose(flux_g_cm2_s, energy_MeV=1.5):
    """Convert gamma flux to dose rate.

    Using ICRP conversion for gammas at ~1.5 MeV:
    ~5.0e-7 mSv*cm2/photon (= 50 pSv*cm2/ph)

    Args:
        flux_g_cm2_s: Gamma flux in photons/(cm2*s)
        energy_MeV: Average gamma energy (MeV)

    Returns:
        float: Dose rate in mSv/hr
    """
    # Energy-dependent: scale linearly with energy from 1 MeV reference
    dose_per_photon = 5.0e-7 * (energy_MeV / 1.5)  # mSv*cm2
    dose_rate_per_s = flux_g_cm2_s * dose_per_photon
    return dose_rate_per_s * 3600  # mSv/hr


# =============================================================================
# Multi-Layer Attenuation
# =============================================================================

def compute_attenuation(layers, S_neutron, S_gamma, R_ref_cm=500.0):
    """Compute neutron and gamma attenuation through multi-layer shield.

    Uses point-kernel method with removal cross-sections for neutrons
    and mass attenuation coefficients for gammas.

    Args:
        layers: List of ShieldLayer objects
        S_neutron: Total neutron source rate (n/s)
        S_gamma: Total gamma source rate (photons/s)
        R_ref_cm: Reference distance from core center (cm)

    Returns:
        AttenuationResult dataclass
    """
    total_thickness = sum(layer.thickness for layer in layers)

    # Unshielded fluxes at reference distance (inverse square)
    phi_n_unshielded = S_neutron / (4.0 * math.pi * R_ref_cm**2)
    phi_g_unshielded = S_gamma / (4.0 * math.pi * R_ref_cm**2)

    # Compute attenuation through each layer
    neutron_atten = 1.0
    gamma_atten = 1.0
    layer_atten_n = []
    layer_atten_g = []

    for layer in layers:
        mat = layer.material
        d = layer.thickness  # cm

        # Neutron removal
        sigma_r = REMOVAL_XS.get(mat, 0.089)
        B_n = neutron_buildup(d, mat)
        atten_n = math.exp(-sigma_r * d) * B_n
        layer_atten_n.append(atten_n)
        neutron_atten *= atten_n

        # Gamma attenuation
        mu_rho = GAMMA_MU_RHO.get(mat, 0.050)
        rho = MATERIAL_DENSITY.get(mat, 2.0)
        mu = mu_rho * rho
        B_g = gamma_buildup(d, mat)
        atten_g = math.exp(-mu * d) * B_g
        layer_atten_g.append(atten_g)
        gamma_atten *= atten_g

    # Shielded fluxes
    phi_n_shielded = phi_n_unshielded * neutron_atten
    phi_g_shielded = phi_g_unshielded * gamma_atten

    # Dose rates
    dose_n = neutron_flux_to_dose(phi_n_shielded)
    dose_g = gamma_flux_to_dose(phi_g_shielded)
    dose_total = dose_n + dose_g

    # Annual dose (assuming 2000 hr occupancy for workers)
    annual_dose = dose_total * 2000

    return AttenuationResult(
        layers=layers,
        total_thickness=total_thickness,
        neutron_source=S_neutron,
        neutron_flux_unshielded=phi_n_unshielded,
        neutron_flux_shielded=phi_n_shielded,
        neutron_attenuation_factor=neutron_atten,
        neutron_dose_rate=dose_n,
        gamma_source=S_gamma,
        gamma_flux_unshielded=phi_g_unshielded,
        gamma_flux_shielded=phi_g_shielded,
        gamma_attenuation_factor=gamma_atten,
        gamma_dose_rate=dose_g,
        total_dose_rate=dose_total,
        annual_dose=annual_dose,
        layer_attenuation_n=layer_atten_n,
        layer_attenuation_g=layer_atten_g,
        reference_distance=R_ref_cm,
    )


# =============================================================================
# Shield Design Iteration
# =============================================================================

def design_shield(S_neutron, S_gamma, target_dose_rate_mSv_hr=0.010,
                  R_ref_cm=500.0, max_iterations=50):
    """Iterate shield thickness to meet dose rate target.

    Starting from a nominal shield configuration, adjusts the biological
    shield (concrete) thickness to achieve the target dose rate.

    Args:
        S_neutron: Neutron source (n/s)
        S_gamma: Gamma source (photons/s)
        target_dose_rate_mSv_hr: Target dose rate (mSv/hr), default 10 uSv/hr
        R_ref_cm: Reference distance (cm)
        max_iterations: Maximum iteration count

    Returns:
        AttenuationResult for the final shield design
    """
    # Starting configuration
    vessel_wall = 2.0   # cm
    primary_shield = 40.0  # cm (steel + B4C)
    bio_shield = 100.0   # cm (heavy concrete, starting guess)

    for iteration in range(max_iterations):
        layers = [
            ShieldLayer("Vessel wall", "hastelloy_N", vessel_wall),
            ShieldLayer("Primary shield (steel)", "steel", primary_shield * 0.6),
            ShieldLayer("Primary shield (B4C)", "B4C", primary_shield * 0.4),
            ShieldLayer("Biological shield", "heavy_concrete", bio_shield),
        ]

        result = compute_attenuation(layers, S_neutron, S_gamma, R_ref_cm)

        if result.total_dose_rate <= target_dose_rate_mSv_hr:
            break

        # Scale concrete thickness: dose ~ exp(-mu*d)
        # Need factor of (current_dose / target_dose) reduction
        ratio = result.total_dose_rate / target_dose_rate_mSv_hr
        if ratio > 1.0:
            # Increase concrete: d_additional = ln(ratio) / mu_eff
            mu_eff = 0.094 * CONCRETE['density_heavy'] / 1000 * 0.048  # rough
            mu_eff = max(mu_eff, 0.01)
            # Use simpler estimate
            d_add = 10.0 * math.log(ratio)  # cm, empirical scaling
            bio_shield += min(d_add, 20.0)  # limit step size

    return result


# =============================================================================
# Default Shield Configuration
# =============================================================================

def default_shield_layers(design_params=None):
    """Return the default marine MSR shield configuration.

    Args:
        design_params: DerivedParameters (computed if None)

    Returns:
        list of ShieldLayer
    """
    if design_params is None:
        design_params = compute_derived()

    t_vessel = design_params.vessel_wall_thickness * 100  # m -> cm

    return [
        ShieldLayer("Vessel wall (Hastelloy-N)", "hastelloy_N", t_vessel),
        ShieldLayer("Steel shield", "steel", 25.0),
        ShieldLayer("B4C absorber", "B4C", 15.0),
        ShieldLayer("Biological shield (baryte concrete)", "heavy_concrete", 120.0),
    ]


# =============================================================================
# Printing
# =============================================================================

def print_attenuation(result):
    """Print formatted attenuation analysis results.

    Args:
        result: AttenuationResult dataclass
    """
    print("=" * 72)
    print("   MULTI-LAYER SHIELD ATTENUATION ANALYSIS")
    print("=" * 72)

    print(f"\n  Reference distance from core: {result.reference_distance:.0f} cm "
          f"({result.reference_distance / 100:.1f} m)")

    print("\n--- Shield Configuration ---")
    print(f"  {'Layer':<35s}  {'Material':<18s}  {'Thickness [cm]':>14s}")
    print(f"  {'-' * 35}  {'-' * 18}  {'-' * 14}")
    for layer in result.layers:
        print(f"  {layer.name:<35s}  {layer.material:<18s}  {layer.thickness:14.1f}")
    print(f"  {'TOTAL':<35s}  {'':<18s}  {result.total_thickness:14.1f}")

    print("\n--- Per-Layer Attenuation Factors ---")
    print(f"  {'Layer':<35s}  {'Neutron':>10s}  {'Gamma':>10s}")
    print(f"  {'-' * 35}  {'-' * 10}  {'-' * 10}")
    for i, layer in enumerate(result.layers):
        print(f"  {layer.name:<35s}  {result.layer_attenuation_n[i]:10.4e}  "
              f"{result.layer_attenuation_g[i]:10.4e}")

    print("\n--- Neutron Attenuation ---")
    print(f"  Source:                         {result.neutron_source:14.3e} n/s")
    print(f"  Unshielded flux:                {result.neutron_flux_unshielded:14.3e} n/(cm2-s)")
    print(f"  Shielded flux:                  {result.neutron_flux_shielded:14.3e} n/(cm2-s)")
    print(f"  Total attenuation:              {result.neutron_attenuation_factor:14.3e}")
    print(f"  Neutron dose rate:              {result.neutron_dose_rate:14.6f} mSv/hr")

    print("\n--- Gamma Attenuation ---")
    print(f"  Source:                         {result.gamma_source:14.3e} photons/s")
    print(f"  Unshielded flux:                {result.gamma_flux_unshielded:14.3e} ph/(cm2-s)")
    print(f"  Shielded flux:                  {result.gamma_flux_shielded:14.3e} ph/(cm2-s)")
    print(f"  Total attenuation:              {result.gamma_attenuation_factor:14.3e}")
    print(f"  Gamma dose rate:                {result.gamma_dose_rate:14.6f} mSv/hr")

    print("\n--- Combined Dose ---")
    print(f"  Total dose rate:                {result.total_dose_rate:14.6f} mSv/hr"
          f"  ({result.total_dose_rate * 1e3:.3f} uSv/hr)")
    print(f"  Annual dose (2000 hr):          {result.annual_dose:14.3f} mSv/yr")
    print(f"  Occupational limit:             {OCCUPATIONAL_DOSE_LIMIT * 1e3:14.1f} mSv/yr")

    status = "MEETS LIMIT" if result.annual_dose < OCCUPATIONAL_DOSE_LIMIT * 1e3 else "EXCEEDS LIMIT"
    print(f"  Status:                         {status}")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()
    st = compute_source_term(d)

    print("Using default shield configuration:")
    layers = default_shield_layers(d)
    result = compute_attenuation(layers, st.S_neutron, st.S_gamma_total, R_ref_cm=500.0)
    print_attenuation(result)

    # Try different distances
    print("\n--- Dose Rate vs Distance ---")
    distances = [500, 800, 1000, 1500]  # cm
    labels = ["5 m (compartment boundary)", "8 m (above reactor)",
              "10 m (control room)", "15 m (crew quarters)"]

    print(f"  {'Location':<30s}  {'Distance [m]':>12s}  {'Dose [uSv/hr]':>14s}  "
          f"{'Annual [mSv/yr]':>15s}")
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 14}  {'-' * 15}")

    for dist_cm, label in zip(distances, labels):
        res = compute_attenuation(layers, st.S_neutron, st.S_gamma_total,
                                  R_ref_cm=dist_cm)
        print(f"  {label:<30s}  {dist_cm / 100:12.1f}  "
              f"{res.total_dose_rate * 1e3:14.3f}  "
              f"{res.annual_dose:15.3f}")
