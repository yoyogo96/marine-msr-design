"""
MSR-Specific Point Kinetics with Circulating Fuel Correction
==============================================================

The key difference between an MSR and a solid-fuel reactor is that
delayed neutron precursors circulate with the fuel salt. Precursors
generated in-core are carried into the external loop, where their
decays do not contribute to the chain reaction.

This reduces the effective delayed neutron fraction:
  beta_eff_i = beta_i * [1 - exp(-lambda_i * tau_core)] / (lambda_i * tau_core)

The point kinetics equations become:
  dn/dt = (rho - beta_eff) / Lambda * n + sum(lambda_i * C_i)
  dC_i/dt = beta_eff_i / Lambda * n - lambda_i * C_i - C_i / tau_loop

where the last term represents precursors leaving the core and
entering the external loop.

Temperature feedback:
  rho = alpha_fuel * (T_fuel - T_fuel_nom) + alpha_graphite * (T_graphite - T_graphite_nom)

Uses adaptive sub-stepping when |eigenvalue * dt| > 2.0 to maintain
numerical stability of the stiff system.

References:
  - ORNL-4541, Appendix E (MSBR Kinetics Model)
  - Kerlin et al., "Theoretical Dynamics Analysis of the MSRE" (ORNL-TM-0567)
  - Haubenreich & Engel, "Experience with the MSRE" (Nuclear Applications, 1970)
  - Keepin, "Physics of Nuclear Kinetics" (delayed neutron data)
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from config import (
    THERMAL_POWER, DELAYED_NEUTRON_GROUPS, PROMPT_NEUTRON_LIFETIME,
    CORE_AVG_TEMP, compute_derived,
)


# =============================================================================
# Constants
# =============================================================================

N_GROUPS = 6  # Keepin 6-group delayed neutron data


# =============================================================================
# MSR Kinetics Class
# =============================================================================

class MSRKinetics:
    """MSR-specific point kinetics with circulating fuel correction.

    State vector: [n, C_1, C_2, ..., C_6]
      n: normalized neutron power (1.0 = nominal)
      C_i: normalized precursor concentration for group i

    Args:
        alpha_fuel: Fuel salt temperature coefficient of reactivity (dk/k per K)
                    Typical: -3.0e-5 to -5.0e-5 for FLiBe+UF4
        alpha_graphite: Graphite temperature coefficient of reactivity (dk/k per K)
                        Typical: -2.0e-5 to -3.0e-5
        tau_core: Fuel salt residence time in core (s)
        tau_loop: Fuel salt transit time in external loop (s)
        Lambda: Prompt neutron generation time (s)
        T_fuel_nom: Nominal fuel salt temperature (K)
        T_graphite_nom: Nominal graphite temperature (K)
    """

    def __init__(self, alpha_fuel=-4.0e-5, alpha_graphite=-2.5e-5,
                 tau_core=5.0, tau_loop=10.0,
                 Lambda=None, T_fuel_nom=None, T_graphite_nom=None):
        """Initialize MSR kinetics model."""

        self.alpha_fuel = alpha_fuel
        self.alpha_graphite = alpha_graphite
        self.tau_core = tau_core
        self.tau_loop = tau_loop

        # Nuclear data
        self.beta = np.array(DELAYED_NEUTRON_GROUPS['beta'])
        self.lam = np.array(DELAYED_NEUTRON_GROUPS['lambda'])
        self.beta_total_static = DELAYED_NEUTRON_GROUPS['beta_total']
        self.Lambda = Lambda if Lambda is not None else PROMPT_NEUTRON_LIFETIME

        # Nominal temperatures
        self.T_fuel_nom = T_fuel_nom if T_fuel_nom is not None else CORE_AVG_TEMP
        self.T_graphite_nom = T_graphite_nom if T_graphite_nom is not None else CORE_AVG_TEMP + 5.0

        # Compute effective delayed neutron fractions
        self.beta_eff = self.compute_effective_beta()
        self.beta_eff_total = float(np.sum(self.beta_eff))

        # State: [n, C_1, ..., C_6]
        self._state = np.zeros(1 + N_GROUPS)
        self._state[0] = 1.0  # nominal power

        # Compute steady-state reactivity offset.
        # In an MSR, precursors that decay outside the core are lost.
        # The reactor must maintain a small positive excess reactivity
        # to compensate. At steady state with dn/dt = 0:
        #   rho_ss = beta_eff - Lambda * sum(lambda_i * C_i) / n
        # where C_i = beta_eff_i / [Lambda * (lambda_i + 1/tau_loop)]
        # So: rho_ss = beta_eff - sum[ beta_eff_i * lambda_i / (lambda_i + 1/tau_loop) ]
        self.rho_ss_offset = float(
            self.beta_eff_total
            - np.sum(self.beta_eff * self.lam / (self.lam + 1.0 / self.tau_loop))
        )

        # Initialize precursors at equilibrium
        for i in range(N_GROUPS):
            # At steady state: dC_i/dt = 0
            # beta_eff_i/Lambda * n = (lambda_i + 1/tau_loop) * C_i
            # C_i = beta_eff_i * n / (Lambda * (lambda_i + 1/tau_loop))
            self._state[1 + i] = (self.beta_eff[i] * self._state[0] /
                                   (self.Lambda * (self.lam[i] + 1.0 / self.tau_loop)))

        self._time = 0.0
        # Start with the steady-state offset as external reactivity
        # (represents the built-in excess reactivity for circulating fuel)
        self._rho_external = self.rho_ss_offset

    @property
    def state(self):
        """Current state vector [n, C_1, ..., C_6]."""
        return self._state.copy()

    @property
    def power(self):
        """Current normalized power."""
        return self._state[0]

    @property
    def time(self):
        """Current simulation time (s)."""
        return self._time

    def compute_effective_beta(self):
        """Compute effective delayed neutron fractions with circulating fuel correction.

        For each group i:
          beta_eff_i = beta_i * (1 - exp(-lambda_i * tau_core)) / (lambda_i * tau_core)

        This accounts for precursors that are generated in-core but decay
        outside the core during their transit through the external loop.

        Returns:
            np.ndarray: Effective delayed neutron fractions for each group
        """
        beta_eff = np.zeros(N_GROUPS)
        for i in range(N_GROUPS):
            x = self.lam[i] * self.tau_core
            if x < 1e-10:
                beta_eff[i] = self.beta[i]  # No correction for very short-lived groups
            else:
                beta_eff[i] = self.beta[i] * (1.0 - math.exp(-x)) / x

        return beta_eff

    def reactivity(self, T_fuel, T_graphite):
        """Compute total reactivity from temperature feedback + external insertion.

        rho = alpha_fuel * (T_fuel - T_fuel_nom) + alpha_graphite * (T_graphite - T_graphite_nom)
              + rho_external

        Args:
            T_fuel: Current fuel salt average temperature (K)
            T_graphite: Current graphite average temperature (K)

        Returns:
            float: Total reactivity in dk/k
        """
        rho_fuel = self.alpha_fuel * (T_fuel - self.T_fuel_nom)
        rho_graphite = self.alpha_graphite * (T_graphite - self.T_graphite_nom)
        return rho_fuel + rho_graphite + self._rho_external

    def set_external_reactivity(self, rho_ext):
        """Set external reactivity insertion (e.g., control rods).

        The total external reactivity includes the steady-state offset
        needed to compensate for precursor loss in the external loop,
        plus any additional insertion.

        Args:
            rho_ext: Additional external reactivity in dk/k (0 = nominal)
        """
        self._rho_external = self.rho_ss_offset + rho_ext

    def derivatives(self, t, state, T_fuel, T_graphite):
        """Compute time derivatives of the state vector.

        dn/dt = (rho - beta_eff) / Lambda * n + sum(lambda_i * C_i)
        dC_i/dt = beta_eff_i / Lambda * n - lambda_i * C_i - C_i / tau_loop

        Args:
            t: Time (s)
            state: State vector [n, C_1, ..., C_6]
            T_fuel: Fuel salt temperature (K)
            T_graphite: Graphite temperature (K)

        Returns:
            np.ndarray: Time derivatives [dn/dt, dC_1/dt, ..., dC_6/dt]
        """
        n = state[0]
        C = state[1:]

        rho = self.reactivity(T_fuel, T_graphite)
        beta_eff_total = self.beta_eff_total

        # Power equation
        dn_dt = (rho - beta_eff_total) / self.Lambda * n + np.sum(self.lam * C)

        # Precursor equations
        dC_dt = np.zeros(N_GROUPS)
        for i in range(N_GROUPS):
            dC_dt[i] = (self.beta_eff[i] / self.Lambda * n
                        - self.lam[i] * C[i]
                        - C[i] / self.tau_loop)

        dydt = np.zeros(1 + N_GROUPS)
        dydt[0] = dn_dt
        dydt[1:] = dC_dt

        return dydt

    def _dominant_eigenvalue(self, T_fuel, T_graphite):
        """Estimate the dominant eigenvalue of the kinetics system.

        The dominant eigenvalue is approximately:
          eigenvalue ~ (rho - beta_eff) / Lambda

        This is used to determine the required sub-stepping for stability.

        Args:
            T_fuel: Fuel temperature (K)
            T_graphite: Graphite temperature (K)

        Returns:
            float: Dominant eigenvalue (1/s)
        """
        rho = self.reactivity(T_fuel, T_graphite)
        return (rho - self.beta_eff_total) / self.Lambda

    def _rk4_substep(self, dt, T_fuel, T_graphite):
        """Single RK4 time step.

        Args:
            dt: Time step (s)
            T_fuel: Fuel temperature (K)
            T_graphite: Graphite temperature (K)
        """
        t = self._time
        y = self._state

        k1 = self.derivatives(t, y, T_fuel, T_graphite)
        k2 = self.derivatives(t + dt / 2, y + dt / 2 * k1, T_fuel, T_graphite)
        k3 = self.derivatives(t + dt / 2, y + dt / 2 * k2, T_fuel, T_graphite)
        k4 = self.derivatives(t + dt, y + dt * k3, T_fuel, T_graphite)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._time += dt

        # Prevent negative power (unphysical)
        if self._state[0] < 0:
            self._state[0] = 0.0

    def step(self, dt, T_fuel, T_graphite):
        """Advance the kinetics by dt with adaptive sub-stepping.

        If |eigenvalue * dt| > 2.0, subdivides the timestep to maintain
        RK4 stability. The RK4 stability limit is |h*lambda| < 2.78,
        so we use a conservative threshold of 2.0.

        Args:
            dt: Desired time step (s)
            T_fuel: Fuel salt average temperature (K)
            T_graphite: Graphite average temperature (K)

        Returns:
            float: Normalized power after the step
        """
        eigenvalue = self._dominant_eigenvalue(T_fuel, T_graphite)
        stability_param = abs(eigenvalue * dt)

        # Adaptive sub-stepping
        STABILITY_LIMIT = 2.0
        if stability_param > STABILITY_LIMIT and abs(eigenvalue) > 1e-10:
            n_substeps = int(math.ceil(stability_param / STABILITY_LIMIT))
            n_substeps = min(n_substeps, 10000)  # safety limit
            dt_sub = dt / n_substeps
        else:
            n_substeps = 1
            dt_sub = dt

        for _ in range(n_substeps):
            self._rk4_substep(dt_sub, T_fuel, T_graphite)

        return self._state[0]

    def reset(self, n0=1.0):
        """Reset to steady state at given power level.

        Args:
            n0: Initial normalized power (default 1.0)
        """
        self._state[0] = n0
        for i in range(N_GROUPS):
            self._state[1 + i] = (self.beta_eff[i] * n0 /
                                   (self.Lambda * (self.lam[i] + 1.0 / self.tau_loop)))
        self._time = 0.0
        self._rho_external = self.rho_ss_offset


# =============================================================================
# Printing Utilities
# =============================================================================

def print_kinetics_parameters(kin):
    """Print MSR kinetics model parameters.

    Args:
        kin: MSRKinetics instance
    """
    print("=" * 72)
    print("   MSR POINT KINETICS PARAMETERS")
    print("=" * 72)

    print("\n--- Temperature Feedback Coefficients ---")
    print(f"  alpha_fuel:                 {kin.alpha_fuel * 1e5:10.2f} x10^-5 dk/k/K ({kin.alpha_fuel * 1e5:.2f} pcm/K)")
    print(f"  alpha_graphite:             {kin.alpha_graphite * 1e5:10.2f} x10^-5 dk/k/K ({kin.alpha_graphite * 1e5:.2f} pcm/K)")
    print(f"  Total (at equal dT):        {(kin.alpha_fuel + kin.alpha_graphite) * 1e5:10.2f} pcm/K")

    print("\n--- Circulating Fuel Parameters ---")
    print(f"  Core transit time:          {kin.tau_core:10.2f} s")
    print(f"  External loop time:         {kin.tau_loop:10.2f} s")
    print(f"  Total loop time:            {kin.tau_core + kin.tau_loop:10.2f} s")

    print("\n--- Prompt Neutron Generation ---")
    print(f"  Lambda:                     {kin.Lambda:10.2e} s")

    print("\n--- Delayed Neutron Fractions ---")
    print(f"  {'Group':>6s}  {'beta_static':>12s}  {'beta_eff':>12s}  {'lambda [1/s]':>12s}  "
          f"{'Reduction':>10s}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 10}")
    for i in range(N_GROUPS):
        reduction = (1.0 - kin.beta_eff[i] / kin.beta[i]) * 100 if kin.beta[i] > 0 else 0
        print(f"  {i + 1:6d}  {kin.beta[i]:12.6f}  {kin.beta_eff[i]:12.6f}  "
              f"{kin.lam[i]:12.4f}  {reduction:9.1f}%")
    print(f"  {'Total':>6s}  {kin.beta_total_static:12.6f}  {kin.beta_eff_total:12.6f}")

    reduction_total = (1.0 - kin.beta_eff_total / kin.beta_total_static) * 100
    print(f"\n  Effective beta reduction:   {reduction_total:10.1f} %")
    print(f"  Static beta:               {kin.beta_total_static * 1e5:10.1f} pcm")
    print(f"  Effective beta:            {kin.beta_eff_total * 1e5:10.1f} pcm")

    print("\n--- Nominal Temperatures ---")
    print(f"  T_fuel_nominal:             {kin.T_fuel_nom - 273.15:10.1f} C")
    print(f"  T_graphite_nominal:         {kin.T_graphite_nom - 273.15:10.1f} C")

    print("\n" + "=" * 72)


# =============================================================================
# Standalone Test
# =============================================================================

def _test_step_response():
    """Test: +100 pcm reactivity step at constant temperature (no feedback)."""
    kin = MSRKinetics()
    kin.set_external_reactivity(100e-5)  # +100 pcm above nominal

    print("\n--- Step Response Test: +100 pcm (no feedback) ---")
    print(f"  {'t [s]':>8s}  {'n/n0':>10s}  {'rho [pcm]':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}")

    dt = 0.01
    t_print = [0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    idx = 0

    T_f = kin.T_fuel_nom
    T_g = kin.T_graphite_nom

    for step_num in range(int(20.0 / dt)):
        t = step_num * dt
        if idx < len(t_print) and t >= t_print[idx] - dt / 2:
            rho = kin.reactivity(T_f, T_g) * 1e5
            print(f"  {t:8.3f}  {kin.power:10.4f}  {rho:10.1f}")
            idx += 1
        kin.step(dt, T_f, T_g)


def _test_temperature_feedback():
    """Test: Power step with temperature feedback."""
    kin = MSRKinetics()

    print("\n--- Temperature Feedback Test ---")
    print("  +200 pcm insertion, fuel heats proportionally to power excess")

    dt = 0.01
    T_f = kin.T_fuel_nom
    T_g = kin.T_graphite_nom

    # Insert +200 pcm above nominal at t=0
    kin.set_external_reactivity(200e-5)

    print(f"\n  {'t [s]':>8s}  {'n/n0':>10s}  {'T_fuel [C]':>10s}  {'rho_net [pcm]':>13s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 13}")

    # Simple thermal model: m*cp*dT/dt = (n - 1) * P0
    # Simplified: dT_fuel/dt = K * (n - 1), K ~ 2 K/s per unit power excess
    K_fuel = 2.0  # K/s per unit power excess

    for step_num in range(int(30.0 / dt)):
        t = step_num * dt

        if step_num % int(1.0 / dt) == 0 or step_num < 10:
            rho = kin.reactivity(T_f, T_g) * 1e5
            print(f"  {t:8.3f}  {kin.power:10.4f}  {T_f - 273.15:10.1f}  {rho:13.1f}")

        # Kinetics step
        kin.step(dt, T_f, T_g)

        # Temperature update (simple explicit coupling)
        T_f += K_fuel * (kin.power - 1.0) * dt
        T_g += 0.5 * K_fuel * (kin.power - 1.0) * dt  # graphite responds slower


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()

    # Create kinetics model with design parameters
    kin = MSRKinetics(
        tau_core=d.residence_time_core,
        tau_loop=d.residence_time_core * 1.5,  # external loop ~ 1.5x core
    )

    print_kinetics_parameters(kin)
    _test_step_response()
    _test_temperature_feedback()
