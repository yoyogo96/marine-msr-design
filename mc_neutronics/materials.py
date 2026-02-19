"""
Material Definitions with 2-Group Macroscopic Cross-Sections
=============================================================

Defines materials for the 40 MWth Marine MSR Monte Carlo simulation:
  1. Fuel salt (FLiBe + UF4) with enrichment and temperature dependence
  2. Graphite moderator (IG-110 nuclear grade)
  3. Graphite reflector (same material, separate instance for tallying)
  4. Void (vacuum boundary condition)

Cross-Section Convention
------------------------
All macroscopic cross-sections are in cm^-1 (nuclear engineering convention).
Scattering matrices use [from_group, to_group] indexing, so sigma_s[0,1] is
the fast-to-thermal downscatter cross-section.

2-Group Structure
-----------------
Group 1 (index 0): Fast,    E > 0.625 eV
Group 2 (index 1): Thermal, E < 0.625 eV

Reference Values
----------------
Cross-section data are approximate values for a graphite-moderated thermal
MSR spectrum, derived from:
  - ORNL-4541 (MSBR design report)
  - ORNL/TM-2005/218 (FLiBe properties)
  - ENDF/B-VIII.0 evaluations (processed through NJOY for MSR spectrum)
  - Serp-2 benchmark calculations for similar lattice configurations

Temperature Dependence
----------------------
- Doppler broadening: resonance absorption scales as sqrt(T0/T)
- Density effect: all XS scale linearly with number density (rho/rho_ref)
- Thermal scattering: upscatter term increases with temperature
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .constants import (
    N_GROUPS, CHI, NU, MAT_VOID, MAT_FUEL_SALT,
    MAT_GRAPHITE_MOD, MAT_GRAPHITE_REF,
)


# =============================================================================
# MATERIAL DATACLASS
# =============================================================================

@dataclass
class Material:
    """Homogenized material with 2-group macroscopic cross-sections.

    All cross-sections are macroscopic (cm^-1), already multiplied by
    the number density of the relevant nuclides.

    Attributes
    ----------
    name : str
        Human-readable material name.
    mat_id : int
        Unique material identifier (matches constants.MAT_*).
    density : float
        Mass density [kg/m^3].
    temperature : float
        Temperature [K] at which cross-sections are evaluated.
    sigma_t : ndarray, shape (N_GROUPS,)
        Total macroscopic cross-section [cm^-1].
    sigma_s : ndarray, shape (N_GROUPS, N_GROUPS)
        Scattering transfer matrix [cm^-1].
        sigma_s[g_from, g_to] = macroscopic scattering from group g_from
        into group g_to.
    sigma_f : ndarray, shape (N_GROUPS,)
        Fission macroscopic cross-section [cm^-1].
    nu_sigma_f : ndarray, shape (N_GROUPS,)
        nu * fission cross-section [cm^-1] (neutron production).
    sigma_a : ndarray, shape (N_GROUPS,)
        Absorption macroscopic cross-section [cm^-1].
        sigma_a = sigma_t - sum_g'(sigma_s[g, g']).
    chi : ndarray, shape (N_GROUPS,)
        Fission spectrum (fraction of fission neutrons born in each group).
    is_fissile : bool
        True if this material can undergo fission.
    """
    name: str
    mat_id: int
    density: float                               # kg/m^3
    temperature: float                           # K
    sigma_t: np.ndarray = field(repr=False)       # [g] cm^-1
    sigma_s: np.ndarray = field(repr=False)       # [g_from, g_to] cm^-1
    sigma_f: np.ndarray = field(repr=False)       # [g] cm^-1
    nu_sigma_f: np.ndarray = field(repr=False)    # [g] cm^-1
    sigma_a: np.ndarray = field(repr=False)       # [g] cm^-1
    chi: np.ndarray = field(repr=False)           # [g]
    is_fissile: bool = False

    def __post_init__(self):
        """Validate cross-section consistency."""
        assert self.sigma_t.shape == (N_GROUPS,), \
            f"sigma_t shape must be ({N_GROUPS},), got {self.sigma_t.shape}"
        assert self.sigma_s.shape == (N_GROUPS, N_GROUPS), \
            f"sigma_s shape must be ({N_GROUPS},{N_GROUPS}), got {self.sigma_s.shape}"
        assert self.sigma_f.shape == (N_GROUPS,), \
            f"sigma_f shape must be ({N_GROUPS},), got {self.sigma_f.shape}"
        assert self.nu_sigma_f.shape == (N_GROUPS,), \
            f"nu_sigma_f shape must be ({N_GROUPS},), got {self.nu_sigma_f.shape}"
        assert self.sigma_a.shape == (N_GROUPS,), \
            f"sigma_a shape must be ({N_GROUPS},), got {self.sigma_a.shape}"
        assert self.chi.shape == (N_GROUPS,), \
            f"chi shape must be ({N_GROUPS},), got {self.chi.shape}"

        # Verify sigma_t >= sigma_s_total + sigma_a in each group
        for g in range(N_GROUPS):
            sigma_s_out = np.sum(self.sigma_s[g, :])
            reconstructed = sigma_s_out + self.sigma_a[g]
            if abs(reconstructed - self.sigma_t[g]) > 1e-6:
                # Auto-correct sigma_t for consistency
                self.sigma_t[g] = reconstructed

    def total_scatter_from(self, g: int) -> float:
        """Total scattering out of group g (sum over all destination groups).

        Parameters
        ----------
        g : int
            Source energy group index.

        Returns
        -------
        float
            Total macroscopic scattering cross-section from group g [cm^-1].
        """
        return float(np.sum(self.sigma_s[g, :]))

    def scatter_probability(self, g_from: int, g_to: int) -> float:
        """Probability of scattering from g_from to g_to (given scatter event).

        Parameters
        ----------
        g_from : int
            Source energy group.
        g_to : int
            Destination energy group.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        total = self.total_scatter_from(g_from)
        if total < 1e-30:
            return 0.0
        return float(self.sigma_s[g_from, g_to] / total)

    def summary(self) -> str:
        """Return a formatted summary of this material's cross-sections."""
        lines = [
            f"Material: {self.name} (ID={self.mat_id})",
            f"  Density:     {self.density:.1f} kg/m^3",
            f"  Temperature: {self.temperature:.1f} K ({self.temperature-273.15:.1f} C)",
            f"  Fissile:     {self.is_fissile}",
            f"  {'Group':>8s}  {'Sigma_t':>10s}  {'Sigma_a':>10s}  "
            f"{'Sigma_f':>10s}  {'nuSigma_f':>10s}  {'chi':>8s}",
        ]
        for g in range(N_GROUPS):
            lines.append(
                f"  {g+1:>8d}  {self.sigma_t[g]:10.5f}  {self.sigma_a[g]:10.5f}  "
                f"{self.sigma_f[g]:10.5f}  {self.nu_sigma_f[g]:10.5f}  "
                f"{self.chi[g]:8.4f}"
            )
        lines.append(f"  Scattering matrix (from -> to) [cm^-1]:")
        for g_from in range(N_GROUPS):
            row = "    "
            for g_to in range(N_GROUPS):
                row += f"  {self.sigma_s[g_from, g_to]:10.5f}"
            lines.append(row)
        return "\n".join(lines)


# =============================================================================
# REFERENCE CONDITIONS
# =============================================================================

# Reference enrichment and temperature for base cross-sections
_REF_ENRICHMENT = 0.12      # 12% U-235
_REF_TEMPERATURE = 923.15   # K (650 C)

# FLiBe salt density at reference temperature [kg/m^3]
# Using correlation: rho = 2413.0 - 0.488 * (T_C)
_REF_SALT_DENSITY = 2413.0 - 0.488 * (923.15 - 273.15)  # ~2095.6 kg/m^3

# Graphite density [kg/m^3]
_GRAPHITE_DENSITY = 1780.0


# =============================================================================
# FUEL SALT CROSS-SECTIONS
# =============================================================================

# Base 2-group macroscopic cross-sections for FLiBe + UF4 fuel salt
# at 12% enrichment, 650 C (923.15 K)
#
# These values represent the homogenized fuel salt mixture including:
#   - 7LiF (64.5 mol%), BeF2 (30.5 mol%), UF4 (5 mol%)
#   - U-235 (12%), U-238 (88%)
#   - All constituent nuclides (Li-7, Be-9, F-19, U-235, U-238)
#
# Group 1 (Fast, >0.625 eV):
#   - Dominated by potential scattering and resonance absorption (U-238)
#   - Fission from U-235 fast fission + some U-238 fast fission
#
# Group 2 (Thermal, <0.625 eV):
#   - Large U-235 thermal fission cross-section
#   - Significant Li-6 absorption (tiny amount from Li-7 transmutation)
#   - Thermal scattering from FLiBe matrix

# --- Fast group (index 0) ---
_FUEL_SIGMA_T_FAST = 0.226         # cm^-1  total (auto-verified by Material)
_FUEL_SIGMA_A_FAST = 0.010        # cm^-1  absorption (U-238 resonance + U-235)
_FUEL_SIGMA_F_FAST = 0.002        # cm^-1  fission (U-235 + small U-238 contrib)
_FUEL_NU_SIGMA_F_FAST = 0.005     # cm^-1  nu * fission
_FUEL_SIGMA_S_11 = 0.190          # cm^-1  fast-to-fast scattering
_FUEL_SIGMA_S_12 = 0.026          # cm^-1  fast-to-thermal downscatter

# --- Thermal group (index 1) ---
_FUEL_SIGMA_T_THERM = 0.357        # cm^-1  total
_FUEL_SIGMA_A_THERM = 0.052       # cm^-1  absorption (U-235 dominant)
_FUEL_SIGMA_F_THERM = 0.040       # cm^-1  fission (U-235 thermal)
_FUEL_NU_SIGMA_F_THERM = 0.098    # cm^-1  nu * fission
_FUEL_SIGMA_S_22 = 0.270          # cm^-1  thermal-to-thermal scattering
_FUEL_SIGMA_S_21 = 0.005          # cm^-1  thermal-to-fast upscatter


def fuel_salt(enrichment: float = 0.12,
              temperature: float = 923.15) -> Material:
    """Create fuel salt material with enrichment and temperature scaling.

    The cross-sections are scaled from reference values (12% enrichment,
    650 C) using physically motivated correction factors:

    1. **Enrichment scaling**: Fission and U-235 absorption scale linearly
       with enrichment. U-238 absorption scales with (1 - enrichment).
       Total absorption is the sum of both contributions.

    2. **Temperature (Doppler) scaling**: Resonance absorption (primarily
       U-238 in the fast group) broadens with temperature following
       Sigma_a(T) = Sigma_a(T0) * sqrt(T0/T). This is the Doppler
       broadening effect that provides negative temperature feedback.

    3. **Density scaling**: All macroscopic cross-sections scale linearly
       with number density, which is proportional to mass density.
       Salt density decreases with temperature: rho = 2413 - 0.488*T_C.

    Parameters
    ----------
    enrichment : float
        U-235 mass fraction (0.05 to 0.20). Default 0.12 (12%).
    temperature : float
        Salt temperature [K]. Default 923.15 K (650 C).

    Returns
    -------
    Material
        Fuel salt material with scaled cross-sections.
    """
    # --- Enrichment scaling factors ---
    # U-235 content scales linearly with enrichment
    enrich_ratio = enrichment / _REF_ENRICHMENT
    # U-238 content scales with (1 - enrichment) / (1 - ref_enrichment)
    u238_ratio = (1.0 - enrichment) / (1.0 - _REF_ENRICHMENT)

    # --- Temperature (Doppler) scaling ---
    # Resonance absorption broadens: sqrt(T0/T) for resolved resonances
    doppler_factor = math.sqrt(_REF_TEMPERATURE / temperature)

    # --- Density scaling ---
    T_C = temperature - 273.15
    rho = 2413.0 - 0.488 * T_C  # kg/m^3
    density_ratio = rho / _REF_SALT_DENSITY

    # =========================================================================
    # Fast group (index 0)
    # =========================================================================
    # Fission: dominated by U-235, scales with enrichment
    sigma_f_fast = _FUEL_SIGMA_F_FAST * enrich_ratio * density_ratio
    nu_sigma_f_fast = _FUEL_NU_SIGMA_F_FAST * enrich_ratio * density_ratio

    # Absorption: U-235 absorption + U-238 resonance absorption
    # Split the base absorption into U-235 and U-238 contributions
    # At 12% enrichment, fast absorption is ~60% U-238 resonance, ~40% U-235
    sigma_a_u235_fast = 0.003 * enrich_ratio * density_ratio
    sigma_a_u238_fast = 0.007 * u238_ratio * density_ratio * doppler_factor
    sigma_a_fast = sigma_a_u235_fast + sigma_a_u238_fast

    # Scattering: mostly from F, Be, Li -- weakly dependent on enrichment
    sigma_s_11 = _FUEL_SIGMA_S_11 * density_ratio
    sigma_s_12 = _FUEL_SIGMA_S_12 * density_ratio

    # Total
    sigma_t_fast = sigma_a_fast + sigma_s_11 + sigma_s_12

    # =========================================================================
    # Thermal group (index 1)
    # =========================================================================
    # Fission: dominated by U-235 thermal fission
    sigma_f_therm = _FUEL_SIGMA_F_THERM * enrich_ratio * density_ratio
    nu_sigma_f_therm = _FUEL_NU_SIGMA_F_THERM * enrich_ratio * density_ratio

    # Absorption: mostly U-235, with small contributions from Li, F, Be
    # At 12% enrichment, thermal absorption is ~85% U-235, ~15% other
    sigma_a_u235_therm = 0.047 * enrich_ratio * density_ratio
    sigma_a_other_therm = 0.005 * density_ratio  # Li, F, Be (enrichment-independent)
    sigma_a_therm = sigma_a_u235_therm + sigma_a_other_therm

    # Thermal scattering: from FLiBe matrix
    sigma_s_22 = _FUEL_SIGMA_S_22 * density_ratio
    # Upscatter: increases slightly with temperature (more thermal motion)
    upscatter_factor = math.sqrt(temperature / _REF_TEMPERATURE)
    sigma_s_21 = _FUEL_SIGMA_S_21 * density_ratio * upscatter_factor

    # Total
    sigma_t_therm = sigma_a_therm + sigma_s_22 + sigma_s_21

    # =========================================================================
    # Assemble arrays
    # =========================================================================
    sigma_t = np.array([sigma_t_fast, sigma_t_therm])
    sigma_a = np.array([sigma_a_fast, sigma_a_therm])
    sigma_f = np.array([sigma_f_fast, sigma_f_therm])
    nu_sigma_f = np.array([nu_sigma_f_fast, nu_sigma_f_therm])
    sigma_s = np.array([
        [sigma_s_11, sigma_s_12],   # from fast
        [sigma_s_21, sigma_s_22],   # from thermal
    ])
    chi = np.array(CHI, dtype=float)

    return Material(
        name=f"FLiBe+UF4 ({enrichment*100:.1f}% enr, {temperature-273.15:.0f}C)",
        mat_id=MAT_FUEL_SALT,
        density=rho,
        temperature=temperature,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        sigma_f=sigma_f,
        nu_sigma_f=nu_sigma_f,
        sigma_a=sigma_a,
        chi=chi,
        is_fissile=True,
    )


# =============================================================================
# GRAPHITE CROSS-SECTIONS
# =============================================================================

# Base 2-group macroscopic cross-sections for IG-110 nuclear graphite
# at room temperature (then temperature-corrected)
#
# Graphite is an excellent moderator with high scattering and very low
# absorption. The moderating ratio (xi * Sigma_s / Sigma_a) ~ 200.
#
# Key features:
# - Strong downscatter (fast-to-thermal) due to light C-12 nucleus
# - Very low absorption (sigma_a ~ 3.5 mb for C-12 at thermal)
# - Significant upscatter at high temperatures due to thermal motion

# --- Fast group (index 0) ---
_GRAPH_SIGMA_T_FAST = 0.386       # cm^-1  total
_GRAPH_SIGMA_A_FAST = 0.0001      # cm^-1  absorption (very low)
_GRAPH_SIGMA_S_11 = 0.330         # cm^-1  fast-to-fast
_GRAPH_SIGMA_S_12 = 0.056         # cm^-1  fast-to-thermal (moderation)

# --- Thermal group (index 1) ---
_GRAPH_SIGMA_T_THERM = 0.503      # cm^-1  total
_GRAPH_SIGMA_A_THERM = 0.00023    # cm^-1  absorption (very low)
_GRAPH_SIGMA_S_22 = 0.497         # cm^-1  thermal-to-thermal
_GRAPH_SIGMA_S_21 = 0.006         # cm^-1  thermal-to-fast (upscatter)


def graphite(temperature: float = 923.15,
             mat_id: int = MAT_GRAPHITE_MOD) -> Material:
    """Create graphite material (moderator or reflector).

    Temperature effects on graphite cross-sections:
    - Thermal scattering kernel changes with temperature (S(alpha,beta))
    - Upscatter probability increases with temperature
    - Density change is negligible for solid graphite
    - Absorption has weak 1/v temperature dependence

    Parameters
    ----------
    temperature : float
        Graphite temperature [K]. Default 923.15 K (650 C).
    mat_id : int
        Material ID. Use MAT_GRAPHITE_MOD (2) for moderator,
        MAT_GRAPHITE_REF (3) for reflector.

    Returns
    -------
    Material
        Graphite material with temperature-corrected cross-sections.
    """
    name_suffix = "Moderator" if mat_id == MAT_GRAPHITE_MOD else "Reflector"

    # Temperature correction for thermal absorption (1/v law)
    # sigma_a(T) = sigma_a(T0) * sqrt(T0/T) for 1/v absorbers
    thermal_factor = math.sqrt(_REF_TEMPERATURE / temperature)

    # Upscatter increases with temperature
    upscatter_factor = math.sqrt(temperature / _REF_TEMPERATURE)

    # Fast group
    sigma_a_fast = _GRAPH_SIGMA_A_FAST  # negligible T dependence
    sigma_s_11 = _GRAPH_SIGMA_S_11
    sigma_s_12 = _GRAPH_SIGMA_S_12
    sigma_t_fast = sigma_a_fast + sigma_s_11 + sigma_s_12

    # Thermal group
    sigma_a_therm = _GRAPH_SIGMA_A_THERM * thermal_factor
    sigma_s_22 = _GRAPH_SIGMA_S_22
    sigma_s_21 = _GRAPH_SIGMA_S_21 * upscatter_factor
    sigma_t_therm = sigma_a_therm + sigma_s_22 + sigma_s_21

    sigma_t = np.array([sigma_t_fast, sigma_t_therm])
    sigma_a = np.array([sigma_a_fast, sigma_a_therm])
    sigma_f = np.zeros(N_GROUPS)
    nu_sigma_f = np.zeros(N_GROUPS)
    sigma_s = np.array([
        [sigma_s_11, sigma_s_12],
        [sigma_s_21, sigma_s_22],
    ])
    chi = np.zeros(N_GROUPS)

    return Material(
        name=f"IG-110 Graphite {name_suffix} ({temperature-273.15:.0f}C)",
        mat_id=mat_id,
        density=_GRAPHITE_DENSITY,
        temperature=temperature,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        sigma_f=sigma_f,
        nu_sigma_f=nu_sigma_f,
        sigma_a=sigma_a,
        chi=chi,
        is_fissile=False,
    )


def reflector(temperature: float = 873.15) -> Material:
    """Create graphite reflector material.

    The reflector is slightly cooler than the in-core moderator since it
    is on the periphery of the core. Default temperature is 600 C.

    Parameters
    ----------
    temperature : float
        Reflector temperature [K]. Default 873.15 K (600 C).

    Returns
    -------
    Material
        Graphite reflector material.
    """
    return graphite(temperature=temperature, mat_id=MAT_GRAPHITE_REF)


def void_material() -> Material:
    """Create void (vacuum) material with zero cross-sections.

    Used for regions outside the geometry where neutrons escape
    (leakage boundary condition).

    Returns
    -------
    Material
        Void material with all cross-sections set to zero.
    """
    return Material(
        name="Void (Vacuum)",
        mat_id=MAT_VOID,
        density=0.0,
        temperature=0.0,
        sigma_t=np.zeros(N_GROUPS),
        sigma_s=np.zeros((N_GROUPS, N_GROUPS)),
        sigma_f=np.zeros(N_GROUPS),
        nu_sigma_f=np.zeros(N_GROUPS),
        sigma_a=np.zeros(N_GROUPS),
        chi=np.zeros(N_GROUPS),
        is_fissile=False,
    )


# =============================================================================
# MATERIAL FACTORY
# =============================================================================

def create_msr_materials(enrichment: float = 0.12,
                         temperature: float = 923.15) -> Dict[int, Material]:
    """Create all materials for the MSR Monte Carlo simulation.

    This is the primary entry point for obtaining materials. Returns a
    dictionary keyed by material ID (matching constants.MAT_*).

    Parameters
    ----------
    enrichment : float
        U-235 mass fraction. Default 0.12 (12% HALEU).
    temperature : float
        Core average temperature [K]. Default 923.15 K (650 C).
        Fuel salt and moderator use this temperature.
        Reflector is set to (temperature - 50 K).

    Returns
    -------
    dict
        Mapping of material ID -> Material instance.
        Keys: MAT_VOID (0), MAT_FUEL_SALT (1),
              MAT_GRAPHITE_MOD (2), MAT_GRAPHITE_REF (3).
    """
    materials = {
        MAT_VOID: void_material(),
        MAT_FUEL_SALT: fuel_salt(enrichment, temperature),
        MAT_GRAPHITE_MOD: graphite(temperature, MAT_GRAPHITE_MOD),
        MAT_GRAPHITE_REF: reflector(temperature - 50.0),
    }
    return materials


# =============================================================================
# PHYSICS CHECKS
# =============================================================================

def check_criticality_estimate(mat: Material) -> Optional[float]:
    """Estimate k-infinity for a homogeneous mixture of this material.

    Uses the 2-group diffusion theory formula:
        k_inf = (nu_sigma_f1 * phi1 + nu_sigma_f2 * phi2) /
                (sigma_a1 * phi1 + sigma_a2 * phi2)

    where the flux ratio phi2/phi1 is estimated from the slowing-down
    balance: sigma_s(1->2) * phi1 = sigma_a2 * phi2 + sigma_s(2->1) * phi2

    Parameters
    ----------
    mat : Material
        Material to check.

    Returns
    -------
    float or None
        Estimated k-infinity, or None if material is not fissile.
    """
    if not mat.is_fissile:
        return None

    # Estimate flux ratio from slowing-down balance
    # Fast-to-thermal source = thermal removal
    # Sigma_s(1->2) * phi_1 = (Sigma_a2 + Sigma_s(2->1)) * phi_2
    removal_2 = mat.sigma_a[1] + mat.sigma_s[1, 0]
    if removal_2 < 1e-30:
        return None

    flux_ratio = mat.sigma_s[0, 1] / removal_2  # phi_2 / phi_1

    # k_inf = (nuSf1 + nuSf2 * ratio) / (Sa1 + Sa2 * ratio)
    numerator = mat.nu_sigma_f[0] + mat.nu_sigma_f[1] * flux_ratio
    denominator = mat.sigma_a[0] + mat.sigma_a[1] * flux_ratio

    if denominator < 1e-30:
        return None

    k_inf = numerator / denominator
    return k_inf


def check_material_balance(mat: Material) -> bool:
    """Verify that sigma_t = sigma_s_total + sigma_a for each group.

    Parameters
    ----------
    mat : Material
        Material to check.

    Returns
    -------
    bool
        True if balance holds within tolerance.
    """
    ok = True
    for g in range(N_GROUPS):
        sigma_s_out = np.sum(mat.sigma_s[g, :])
        balance = sigma_s_out + mat.sigma_a[g]
        diff = abs(balance - mat.sigma_t[g])
        if diff > 1e-6:
            print(f"  WARNING: Group {g+1} balance error: "
                  f"Sigma_t={mat.sigma_t[g]:.6f}, "
                  f"Sigma_s+Sigma_a={balance:.6f}, diff={diff:.2e}")
            ok = False
    return ok


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  MSR Materials - 2-Group Cross-Section Verification")
    print("=" * 70)

    # Create all materials at reference conditions
    mats = create_msr_materials(enrichment=0.12, temperature=923.15)

    for mat_id, mat in sorted(mats.items()):
        print(f"\n{mat.summary()}")

        # Check material balance
        ok = check_material_balance(mat)
        print(f"  Balance check: {'PASS' if ok else 'FAIL'}")

        # Check criticality estimate for fuel
        k_inf = check_criticality_estimate(mat)
        if k_inf is not None:
            print(f"  k-infinity estimate: {k_inf:.4f}")
            if k_inf > 1.0:
                print(f"    -> Supercritical (expected for fuel at this enrichment)")
            else:
                print(f"    -> Subcritical")

    # --- Enrichment sensitivity study ---
    print(f"\n{'='*70}")
    print(f"  Enrichment Sensitivity (at 650 C)")
    print(f"{'='*70}")
    print(f"  {'Enrichment':>12s}  {'k_inf':>8s}  {'nuSf_fast':>10s}  "
          f"{'nuSf_therm':>10s}  {'Sa_therm':>10s}")
    for enr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.19]:
        m = fuel_salt(enrichment=enr)
        k = check_criticality_estimate(m)
        print(f"  {enr*100:10.1f}%   {k:8.4f}  {m.nu_sigma_f[0]:10.5f}  "
              f"{m.nu_sigma_f[1]:10.5f}  {m.sigma_a[1]:10.5f}")

    # --- Temperature sensitivity study ---
    print(f"\n{'='*70}")
    print(f"  Temperature Sensitivity (at 12% enrichment)")
    print(f"{'='*70}")
    print(f"  {'Temp (C)':>10s}  {'k_inf':>8s}  {'Sa_fast':>10s}  "
          f"{'Sa_therm':>10s}  {'density':>10s}")
    for T_C in [550, 600, 650, 700, 750, 800]:
        T_K = T_C + 273.15
        m = fuel_salt(temperature=T_K)
        k = check_criticality_estimate(m)
        print(f"  {T_C:10.0f}  {k:8.4f}  {m.sigma_a[0]:10.5f}  "
              f"{m.sigma_a[1]:10.5f}  {m.density:10.1f}")

    print(f"\n  Temperature coefficient should be NEGATIVE (safety requirement).")
    m_low = fuel_salt(temperature=873.15)  # 600 C
    m_high = fuel_salt(temperature=973.15)  # 700 C
    k_low = check_criticality_estimate(m_low)
    k_high = check_criticality_estimate(m_high)
    dk_dT = (k_high - k_low) / (973.15 - 873.15)  # per K
    print(f"  dk/dT ~ {dk_dT*1e5:.2f} pcm/K")
    if dk_dT < 0:
        print(f"  -> NEGATIVE temperature coefficient: PASS (inherent safety)")
    else:
        print(f"  -> WARNING: Positive temperature coefficient detected!")

    print(f"\n{'='*70}")
    print(f"  All material checks complete.")
    print(f"{'='*70}")
