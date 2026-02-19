"""
Neutron and Gamma Radiation Source Term
========================================

Computes the radiation source strength for the 40 MWth marine MSR.

Sources:
  - Prompt fission neutrons
  - Prompt fission gammas (~7 MeV/fission)
  - Fission product decay gammas (burnup-dependent, parametric)
  - Activated structural gammas (simplified estimate)
  - Post-shutdown decay gamma source vs. time

References:
  - Lamarsh & Baratta, "Introduction to Nuclear Engineering"
  - Shultis & Faw, "Radiation Shielding"
  - ANS-5.1-2014 (Decay Heat Standard)
  - ORNL-4541 (MSBR Source Terms)
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
    THERMAL_POWER, ENERGY_PER_FISSION, NEUTRONS_PER_FISSION,
    compute_derived,
)


# =============================================================================
# Constants
# =============================================================================

MEV_TO_JOULE = 1.602e-13    # J/MeV

# Prompt fission gamma energy groups (approximate spectrum)
# Group structure: [E_low, E_high] in MeV, fraction of total prompt gamma energy
PROMPT_GAMMA_GROUPS = [
    (0.1, 0.5, 0.10),    # soft gammas
    (0.5, 1.0, 0.25),    # medium
    (1.0, 2.0, 0.30),    # medium-high
    (2.0, 4.0, 0.20),    # high
    (4.0, 7.0, 0.10),    # very high
    (7.0, 10.0, 0.05),   # ultra-high
]

# Fission product decay gamma energy groups (simplified, at equilibrium)
FP_GAMMA_GROUPS = [
    (0.1, 0.5, 0.35),
    (0.5, 1.0, 0.30),
    (1.0, 2.0, 0.20),
    (2.0, 4.0, 0.10),
    (4.0, 7.0, 0.05),
]

# Energy per fission breakdown (MeV)
PROMPT_GAMMA_ENERGY = 7.0       # MeV per fission (prompt gammas)
FP_GAMMA_ENERGY = 7.0           # MeV per fission (fission product decay gammas, eq.)
CAPTURE_GAMMA_ENERGY = 5.0      # MeV per capture (structural activation)
BETA_ENERGY = 8.0               # MeV per fission (beta + neutrino)
TOTAL_RECOVERABLE = 200.0       # MeV per fission


# =============================================================================
# SourceTerm Dataclass
# =============================================================================

@dataclass
class SourceTerm:
    """Radiation source terms for shielding design."""

    # --- Neutron source ---
    fission_rate: float             # fissions/s
    S_neutron: float                # n/s, total neutron source rate
    neutron_per_fission: float      # nu

    # --- Prompt gamma source ---
    S_gamma_prompt: float           # photons/s, total prompt fission gammas
    P_gamma_prompt: float           # W, prompt gamma power
    prompt_spectrum: list           # [(E_low, E_high, source_rate), ...]

    # --- Fission product decay gamma source ---
    S_gamma_fp: float               # photons/s, FP decay gammas
    P_gamma_fp: float               # W, FP decay gamma power
    fp_spectrum: list               # [(E_low, E_high, source_rate), ...]

    # --- Structural activation gamma ---
    S_gamma_activation: float       # photons/s (approximate)
    P_gamma_activation: float       # W

    # --- Totals ---
    S_gamma_total: float            # photons/s, all gamma sources
    P_gamma_total: float            # W, total gamma power

    # --- Source geometry ---
    core_volume: float              # m3
    S_neutron_volumetric: float     # n/(s-m3)
    S_gamma_volumetric: float       # photons/(s-m3)


# =============================================================================
# Source Calculations
# =============================================================================

def compute_fission_rate(P_thermal):
    """Compute fission rate from thermal power.

    Fission rate = P / E_per_fission

    Args:
        P_thermal: Thermal power in W

    Returns:
        float: Fissions per second
    """
    return P_thermal / ENERGY_PER_FISSION


def compute_neutron_source(fission_rate, nu=None):
    """Compute total neutron source rate.

    S_n = fission_rate * nu

    Args:
        fission_rate: Fissions per second
        nu: Neutrons per fission (default from config)

    Returns:
        float: Neutrons per second
    """
    if nu is None:
        nu = NEUTRONS_PER_FISSION
    return fission_rate * nu


def compute_prompt_gamma_source(fission_rate):
    """Compute prompt fission gamma source.

    Total prompt gamma energy: ~7 MeV/fission
    Average prompt gamma energy: ~1.5 MeV/photon
    -> ~4.7 prompt gamma photons per fission

    Args:
        fission_rate: Fissions per second

    Returns:
        tuple: (total_source_rate, power_W, spectrum_list)
    """
    avg_gamma_energy_MeV = 1.5  # average photon energy
    photons_per_fission = PROMPT_GAMMA_ENERGY / avg_gamma_energy_MeV

    S_total = fission_rate * photons_per_fission
    P_total = fission_rate * PROMPT_GAMMA_ENERGY * MEV_TO_JOULE

    # Distribute into energy groups
    spectrum = []
    for E_low, E_high, fraction in PROMPT_GAMMA_GROUPS:
        E_avg = (E_low + E_high) / 2.0
        photons_in_group = fraction * PROMPT_GAMMA_ENERGY / E_avg * fission_rate
        spectrum.append((E_low, E_high, photons_in_group))

    return S_total, P_total, spectrum


def compute_fp_gamma_source(fission_rate):
    """Compute fission product decay gamma source at equilibrium.

    At steady-state, FP decay gammas contribute ~7 MeV/fission.
    Average decay gamma energy: ~0.7 MeV/photon
    -> ~10 decay gamma photons per fission

    Args:
        fission_rate: Fissions per second

    Returns:
        tuple: (total_source_rate, power_W, spectrum_list)
    """
    avg_gamma_energy_MeV = 0.7
    photons_per_fission = FP_GAMMA_ENERGY / avg_gamma_energy_MeV

    S_total = fission_rate * photons_per_fission
    P_total = fission_rate * FP_GAMMA_ENERGY * MEV_TO_JOULE

    spectrum = []
    for E_low, E_high, fraction in FP_GAMMA_GROUPS:
        E_avg = (E_low + E_high) / 2.0
        photons_in_group = fraction * FP_GAMMA_ENERGY / E_avg * fission_rate
        spectrum.append((E_low, E_high, photons_in_group))

    return S_total, P_total, spectrum


def compute_activation_gamma(fission_rate, capture_fraction=0.05):
    """Estimate structural activation gamma source.

    Neutron capture in Hastelloy-N, FLiBe carrier, etc. produces
    capture gammas. This is a rough parametric estimate.

    Args:
        fission_rate: Fissions per second
        capture_fraction: Fraction of neutrons captured in structure

    Returns:
        tuple: (source_rate, power_W)
    """
    neutron_rate = fission_rate * NEUTRONS_PER_FISSION
    capture_rate = neutron_rate * capture_fraction

    # Each capture releases ~5 MeV in gammas, average ~2 MeV/photon
    avg_photon_energy = 2.0  # MeV
    photons_per_capture = CAPTURE_GAMMA_ENERGY / avg_photon_energy

    S_total = capture_rate * photons_per_capture
    P_total = capture_rate * CAPTURE_GAMMA_ENERGY * MEV_TO_JOULE

    return S_total, P_total


def decay_gamma_source_vs_time(P_thermal, t_shutdown, T_operating=3.156e7):
    """Decay gamma source strength as a function of time after shutdown.

    Uses ANS-5.1 decay heat approximation:
    P_decay(t) = P0 * 0.066 * [t^(-0.2) - (t + T_op)^(-0.2)]

    Assumes ~50% of decay heat is emitted as gammas.

    Args:
        P_thermal: Nominal thermal power (W)
        t_shutdown: Array of times after shutdown (s)
        T_operating: Operating time before shutdown (s, default 1 year)

    Returns:
        tuple: (t_array, P_decay_gamma_W, S_gamma_array)
    """
    t = np.asarray(t_shutdown, dtype=float)

    # ANS standard decay heat
    P_decay = P_thermal * 0.066 * (t**(-0.2) - (t + T_operating)**(-0.2))

    # Gamma fraction of decay heat (~50%)
    gamma_fraction = 0.50
    P_gamma = P_decay * gamma_fraction

    # Convert to photon rate (average decay gamma ~0.7 MeV)
    avg_energy = 0.7 * MEV_TO_JOULE  # J
    S_gamma = P_gamma / avg_energy

    return t, P_gamma, S_gamma


# =============================================================================
# Main Computation
# =============================================================================

def compute_source_term(design_params=None):
    """Compute all radiation source terms for the MSR.

    Args:
        design_params: DerivedParameters from config (computed if None)

    Returns:
        SourceTerm dataclass
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    fission_rate = compute_fission_rate(THERMAL_POWER)
    S_neutron = compute_neutron_source(fission_rate)

    S_prompt, P_prompt, prompt_spec = compute_prompt_gamma_source(fission_rate)
    S_fp, P_fp, fp_spec = compute_fp_gamma_source(fission_rate)
    S_act, P_act = compute_activation_gamma(fission_rate)

    S_gamma_total = S_prompt + S_fp + S_act
    P_gamma_total = P_prompt + P_fp + P_act

    S_n_vol = S_neutron / d.core_volume
    S_g_vol = S_gamma_total / d.core_volume

    return SourceTerm(
        fission_rate=fission_rate,
        S_neutron=S_neutron,
        neutron_per_fission=NEUTRONS_PER_FISSION,
        S_gamma_prompt=S_prompt,
        P_gamma_prompt=P_prompt,
        prompt_spectrum=prompt_spec,
        S_gamma_fp=S_fp,
        P_gamma_fp=P_fp,
        fp_spectrum=fp_spec,
        S_gamma_activation=S_act,
        P_gamma_activation=P_act,
        S_gamma_total=S_gamma_total,
        P_gamma_total=P_gamma_total,
        core_volume=d.core_volume,
        S_neutron_volumetric=S_n_vol,
        S_gamma_volumetric=S_g_vol,
    )


# =============================================================================
# Printing
# =============================================================================

def print_source_term(st):
    """Print formatted radiation source term summary.

    Args:
        st: SourceTerm dataclass instance
    """
    print("=" * 72)
    print("   RADIATION SOURCE TERMS - 40 MWth Marine MSR")
    print("=" * 72)

    print("\n--- Neutron Source ---")
    print(f"  Fission rate:               {st.fission_rate:14.3e} fissions/s")
    print(f"  Nu (neutrons/fission):      {st.neutron_per_fission:14.2f}")
    print(f"  Total neutron source:       {st.S_neutron:14.3e} n/s")
    print(f"  Volumetric neutron source:  {st.S_neutron_volumetric:14.3e} n/(s-m3)")

    print("\n--- Prompt Fission Gammas ---")
    print(f"  Total source:               {st.S_gamma_prompt:14.3e} photons/s")
    print(f"  Power:                      {st.P_gamma_prompt / 1e6:14.3f} MW")
    print(f"  Energy groups:")
    print(f"    {'E range [MeV]':>16s}  {'Source [ph/s]':>14s}")
    for E_low, E_high, S in st.prompt_spectrum:
        print(f"    {E_low:5.1f} - {E_high:5.1f}       {S:14.3e}")

    print("\n--- Fission Product Decay Gammas (equilibrium) ---")
    print(f"  Total source:               {st.S_gamma_fp:14.3e} photons/s")
    print(f"  Power:                      {st.P_gamma_fp / 1e6:14.3f} MW")
    print(f"  Energy groups:")
    print(f"    {'E range [MeV]':>16s}  {'Source [ph/s]':>14s}")
    for E_low, E_high, S in st.fp_spectrum:
        print(f"    {E_low:5.1f} - {E_high:5.1f}       {S:14.3e}")

    print("\n--- Structural Activation Gammas ---")
    print(f"  Total source:               {st.S_gamma_activation:14.3e} photons/s")
    print(f"  Power:                      {st.P_gamma_activation / 1e3:14.3f} kW")

    print("\n--- Total Gamma Source ---")
    print(f"  Total gamma source:         {st.S_gamma_total:14.3e} photons/s")
    print(f"  Total gamma power:          {st.P_gamma_total / 1e6:14.3f} MW")
    print(f"  Volumetric gamma source:    {st.S_gamma_volumetric:14.3e} ph/(s-m3)")

    print("\n--- Source Geometry ---")
    print(f"  Core volume:                {st.core_volume:14.3f} m3")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()
    st = compute_source_term(d)
    print_source_term(st)

    # Post-shutdown decay source
    print("\n--- Post-Shutdown Decay Gamma Source ---")
    times = [1, 10, 100, 1000, 3600, 86400, 604800]  # s
    labels = ["1 s", "10 s", "100 s", "1000 s", "1 hr", "1 day", "1 week"]

    t_arr, P_gamma_arr, S_gamma_arr = decay_gamma_source_vs_time(
        THERMAL_POWER, np.array(times, dtype=float)
    )

    print(f"  {'Time':>10s}  {'P_decay_gamma [kW]':>18s}  {'S_gamma [ph/s]':>14s}  "
          f"{'% of nominal':>12s}")
    print(f"  {'-' * 10}  {'-' * 18}  {'-' * 14}  {'-' * 12}")
    for lbl, P_g, S_g in zip(labels, P_gamma_arr, S_gamma_arr):
        pct = P_g / st.P_gamma_total * 100
        print(f"  {lbl:>10s}  {P_g / 1e3:18.2f}  {S_g:14.3e}  {pct:12.2f}%")
