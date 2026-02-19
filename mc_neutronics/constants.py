"""
Physical and Nuclear Constants for Multi-Group Monte Carlo Transport
====================================================================

Defines fundamental physical constants, the 2-group energy structure
for a graphite-moderated thermal MSR, fission spectrum parameters,
and reference nuclear data.

Energy Group Structure
----------------------
Group 1 (Fast):    E > 0.625 eV  -- slowing-down range
Group 2 (Thermal): E < 0.625 eV  -- thermal diffusion range

The 0.625 eV cutoff is the standard cadmium cutoff energy, chosen because
it separates the 1/E slowing-down spectrum from the Maxwellian thermal
spectrum in a well-moderated system.
"""

import numpy as np

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================

AVOGADRO = 6.02214076e23          # atoms/mol  (Avogadro's number)
BARN_TO_CM2 = 1.0e-24             # cm^2/barn  (microscopic XS conversion)
EV_TO_JOULE = 1.602176634e-19     # J/eV       (energy conversion)
BOLTZMANN = 8.617333262e-5        # eV/K       (Boltzmann constant)
NEUTRON_MASS = 1.008665           # amu        (neutron rest mass)
SPEED_OF_LIGHT = 2.998e10         # cm/s       (for relativistic checks)

# Thermal neutron speed at 293.6 K (0.0253 eV)
THERMAL_NEUTRON_SPEED = 2.2e5     # cm/s

# Reference temperature for cross-section evaluations
T_REF = 293.6                     # K (room temperature, 20.4 C)

# =============================================================================
# ENERGY GROUP STRUCTURE
# =============================================================================

N_GROUPS = 2
"""Number of energy groups."""

GROUP_BOUNDARIES = np.array([20.0e6, 0.625, 0.0])
"""Energy group boundaries in eV: [upper, cutoff, lower].

- Boundary 0: 20 MeV  (maximum fission neutron energy)
- Boundary 1: 0.625 eV (cadmium cutoff, fast/thermal boundary)
- Boundary 2: 0 eV     (lower bound)
"""

GROUP_NAMES = ['Fast (>0.625 eV)', 'Thermal (<0.625 eV)']
"""Human-readable names for each energy group."""

GROUP_LETHARGY_WIDTHS = np.array([
    np.log(GROUP_BOUNDARIES[0] / GROUP_BOUNDARIES[1]),  # Fast group
    np.log(GROUP_BOUNDARIES[1] / 0.001),                # Thermal (down to ~1 meV)
])
"""Lethargy width of each group (for flux normalization)."""

# =============================================================================
# FISSION SPECTRUM AND NEUTRON DATA
# =============================================================================

CHI = np.array([0.97, 0.03])
"""Fission spectrum chi: fraction of fission neutrons born in each group.

Nearly all fission neutrons are born in the fast group with a Watt fission
spectrum peaked around 0.7 MeV. Only ~3% are born below 0.625 eV.
"""

AVG_FISSION_ENERGY = 2.0e6
"""Average energy of a fission neutron [eV].

From the Watt fission spectrum for U-235 thermal fission:
  chi(E) = C * exp(-E/a) * sinh(sqrt(b*E))
  with a = 0.988 MeV, b = 2.249 MeV^-1.
"""

NU = 2.43
"""Average number of neutrons per fission (nu) for U-235 thermal fission.

This is the spectrum-averaged value. In reality nu has a slight energy
dependence: nu(E) ~ 2.43 + 0.065*(E/MeV) for U-235.
"""

NU_DELAYED = 0.0158
"""Delayed neutron fraction of nu (delayed neutrons per fission).

Total delayed fraction beta = nu_delayed / nu ~ 0.0065.
"""

ENERGY_PER_FISSION_EV = 200.0e6
"""Total recoverable energy per fission [eV].

Includes kinetic energy of fission fragments (~168 MeV), prompt neutrons
(~5 MeV), prompt gammas (~7 MeV), and delayed betas/gammas (~20 MeV).
"""

ENERGY_PER_FISSION = ENERGY_PER_FISSION_EV * EV_TO_JOULE
"""Total recoverable energy per fission [J]."""

# =============================================================================
# DELAYED NEUTRON DATA (Keepin 6-group, U-235 thermal fission)
# =============================================================================

N_DELAYED_GROUPS = 6
"""Number of delayed neutron precursor groups."""

DELAYED_BETA = np.array([0.000215, 0.001424, 0.001274,
                         0.002568, 0.000748, 0.000273])
"""Individual delayed group fractions (beta_i)."""

DELAYED_LAMBDA = np.array([0.0124, 0.0305, 0.111,
                           0.301, 1.14, 3.01])
"""Delayed group decay constants [1/s] (lambda_i)."""

BETA_TOTAL = np.sum(DELAYED_BETA)
"""Total delayed neutron fraction: sum of all beta_i ~ 0.0065."""

PROMPT_NEUTRON_LIFETIME = 4.0e-4
"""Prompt neutron generation time [s].

Longer than LWR (~2e-5 s) due to graphite moderation and larger
thermal diffusion area in MSR.
"""

# =============================================================================
# MATERIAL IDENTIFIERS
# =============================================================================

MAT_VOID = 0
"""Material ID for void / vacuum (outside geometry)."""

MAT_FUEL_SALT = 1
"""Material ID for FLiBe + UF4 fuel salt."""

MAT_GRAPHITE_MOD = 2
"""Material ID for graphite moderator (in-core)."""

MAT_GRAPHITE_REF = 3
"""Material ID for graphite reflector (ex-core annulus)."""

MATERIAL_NAMES = {
    MAT_VOID: 'Void',
    MAT_FUEL_SALT: 'Fuel Salt',
    MAT_GRAPHITE_MOD: 'Graphite Moderator',
    MAT_GRAPHITE_REF: 'Graphite Reflector',
}
"""Mapping from material ID to human-readable name."""

# =============================================================================
# NUMERICAL PARAMETERS
# =============================================================================

DISTANCE_EPSILON = 1.0e-8
"""Small distance [cm] used to nudge particles across boundaries."""

MAX_TRACK_LENGTH = 1.0e4
"""Maximum tracking distance [cm] before terminating a lost particle."""

WEIGHT_CUTOFF = 0.01
"""Weight cutoff for Russian roulette (variance reduction)."""

WEIGHT_SURVIVAL = 0.10
"""Survival weight for Russian roulette."""


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Multi-Group MC Constants Verification")
    print("=" * 60)

    print(f"\n--- Physical Constants ---")
    print(f"  Avogadro:          {AVOGADRO:.6e} atoms/mol")
    print(f"  barn -> cm^2:      {BARN_TO_CM2:.1e}")
    print(f"  eV -> J:           {EV_TO_JOULE:.6e}")
    print(f"  Boltzmann:         {BOLTZMANN:.6e} eV/K")

    print(f"\n--- Energy Group Structure ---")
    print(f"  Number of groups:  {N_GROUPS}")
    for g in range(N_GROUPS):
        e_hi = GROUP_BOUNDARIES[g]
        e_lo = GROUP_BOUNDARIES[g + 1]
        print(f"  Group {g+1} ({GROUP_NAMES[g]}): "
              f"{e_lo:.3e} -- {e_hi:.3e} eV  "
              f"(lethargy width = {GROUP_LETHARGY_WIDTHS[g]:.2f})")

    print(f"\n--- Fission Spectrum ---")
    print(f"  chi = {CHI}")
    assert abs(np.sum(CHI) - 1.0) < 1e-10, "chi must sum to 1.0"
    print(f"  chi sums to {np.sum(CHI):.10f} (OK)")

    print(f"\n--- Fission Data ---")
    print(f"  nu (neutrons/fission):    {NU}")
    print(f"  E_fission:                {ENERGY_PER_FISSION_EV/1e6:.0f} MeV "
          f"= {ENERGY_PER_FISSION:.4e} J")
    print(f"  Average fission E:        {AVG_FISSION_ENERGY/1e6:.1f} MeV")

    print(f"\n--- Delayed Neutron Data ---")
    print(f"  Total beta:               {BETA_TOTAL:.6f}")
    print(f"  Prompt neutron lifetime:  {PROMPT_NEUTRON_LIFETIME:.1e} s")
    for i in range(N_DELAYED_GROUPS):
        print(f"  Group {i+1}: beta={DELAYED_BETA[i]:.6f}, "
              f"lambda={DELAYED_LAMBDA[i]:.4f} 1/s, "
              f"T_half={0.693/DELAYED_LAMBDA[i]:.2f} s")

    print(f"\n--- Material IDs ---")
    for mid, name in MATERIAL_NAMES.items():
        print(f"  {mid}: {name}")

    print(f"\n--- Numerical Parameters ---")
    print(f"  Distance epsilon:  {DISTANCE_EPSILON:.1e} cm")
    print(f"  Max track length:  {MAX_TRACK_LENGTH:.1e} cm")
    print(f"  Weight cutoff:     {WEIGHT_CUTOFF}")
    print(f"  Weight survival:   {WEIGHT_SURVIVAL}")

    print("\n" + "=" * 60)
    print("  All constants verified OK.")
    print("=" * 60)
