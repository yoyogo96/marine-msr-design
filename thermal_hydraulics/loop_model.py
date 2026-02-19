"""
Lumped-Parameter Transient Loop Model
======================================

Coupled ODE system for the primary (FLiBe) and intermediate (FLiNaK) loops.
Each thermal node is a lumped-parameter element whose temperature evolves
according to energy balance ODEs.

Nodes
-----
0. Core fuel salt (average temperature)
1. Core graphite (average temperature)
2. Hot leg pipe
3. HX primary side (FLiBe in shell/tube)
4. Cold leg pipe
5. HX secondary side (FLiNaK in shell/tube)

Transport delays between nodes are modelled as first-order lags:
    tau * dT_pipe/dt = T_in - T_pipe

The model supports time-dependent power and flow perturbations for
transient analysis (pump trip, power excursion, loss of heat sink, etc.).

References
----------
* Kerlin et al., "Dynamic Modeling of Nuclear Reactor Systems" (ORNL-TM)
* ORNL-4541, Appendix E (MSBR Kinetics Model)
* Todreas & Kazimi, "Nuclear Systems I"
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import (
    THERMAL_POWER, CORE_INLET_TEMP, CORE_OUTLET_TEMP, CORE_AVG_TEMP,
    SECONDARY_INLET_TEMP, SECONDARY_OUTLET_TEMP,
    CHANNEL_DIAMETER, CHANNEL_PITCH, FUEL_SALT_FRACTION,
    GRAPHITE, OPERATING_PRESSURE,
    compute_derived,
)
from thermal_hydraulics.salt_properties import (
    flibe_density, flibe_viscosity, flibe_specific_heat,
    flibe_thermal_conductivity,
    flinak_density, flinak_specific_heat, flinak_thermal_conductivity,
)


# ==========================================================================
# Data classes
# ==========================================================================

@dataclass
class LoopModelParams:
    """Physical parameters for the lumped-parameter loop model."""

    # --- Core ---
    m_salt_core: float          # kg, fuel salt mass in core
    cp_salt: float              # J/(kg-K), fuel salt specific heat
    m_graphite: float           # kg, graphite moderator mass in core
    cp_graphite: float          # J/(kg-K), graphite specific heat
    UA_core: float              # W/K, salt-to-graphite heat transfer
    m_dot_primary: float        # kg/s, primary loop mass flow rate

    # --- Hot leg pipe ---
    m_salt_hotleg: float        # kg, salt mass in hot leg
    tau_hotleg: float           # s, transport time through hot leg

    # --- Primary HX ---
    m_salt_hx: float            # kg, salt mass in primary side of HX
    UA_hx: float                # W/K, overall HX heat transfer coefficient

    # --- Cold leg pipe ---
    m_salt_coldleg: float       # kg, salt mass in cold leg
    tau_coldleg: float          # s, transport time through cold leg

    # --- Intermediate (secondary) ---
    m_flinak_hx: float          # kg, FLiNaK mass in HX secondary side
    cp_flinak: float            # J/(kg-K), FLiNaK specific heat
    m_dot_secondary: float      # kg/s, secondary loop mass flow rate
    T_secondary_cold: float     # K, secondary cold-leg return temperature

    # --- Nominal conditions ---
    P_nominal: float            # W, nominal thermal power
    T_core_salt_ss: float       # K, steady-state core salt average temp
    T_graphite_ss: float        # K, steady-state graphite average temp


@dataclass
class TransientResult:
    """Time histories from a transient simulation."""

    t: np.ndarray               # s, time array
    T_core_salt: np.ndarray     # K, core fuel salt average temperature
    T_graphite: np.ndarray      # K, core graphite average temperature
    T_hotleg: np.ndarray        # K, hot leg pipe temperature
    T_hx_primary: np.ndarray    # K, HX primary side temperature
    T_coldleg: np.ndarray       # K, cold leg pipe temperature
    T_hx_secondary: np.ndarray  # K, HX secondary side temperature
    power: np.ndarray           # W, fission power history
    flow_primary: np.ndarray    # kg/s, primary flow history


# ==========================================================================
# Loop Model
# ==========================================================================

class LoopModel:
    """Lumped-parameter coupled loop transient model.

    State vector [6 elements]:
        y[0] = T_core_salt      (core fuel salt average)
        y[1] = T_graphite       (core graphite average)
        y[2] = T_hotleg         (hot leg pipe)
        y[3] = T_hx_primary     (HX primary side)
        y[4] = T_coldleg        (cold leg pipe)
        y[5] = T_hx_secondary   (HX secondary side)
    """

    N_STATES = 6

    def __init__(self, params: LoopModelParams):
        """Initialize the loop model with physical parameters.

        Args:
            params: LoopModelParams with all physical constants.
        """
        self.p = params

    # ------------------------------------------------------------------
    # Steady-state solver
    # ------------------------------------------------------------------

    def steady_state(self):
        """Compute nominal steady-state temperatures for all nodes.

        At steady state, all dT/dt = 0.  The temperatures are found by
        solving the algebraic energy balance equations.

        Returns:
            np.ndarray of shape (6,) with steady-state node temperatures in K.
        """
        p = self.p
        P = p.P_nominal
        m_dot_p = p.m_dot_primary
        cp_s = p.cp_salt
        cp_fn = p.cp_flinak
        m_dot_s = p.m_dot_secondary

        # Core salt: average of inlet and outlet
        # T_core_out = T_core_in + P / (m_dot * cp)
        delta_T_core = P / (m_dot_p * cp_s)
        T_core_in_ss = CORE_INLET_TEMP
        T_core_out_ss = T_core_in_ss + delta_T_core
        T_core_salt_ss = (T_core_in_ss + T_core_out_ss) / 2.0

        # Graphite: UA_core * (T_salt - T_graphite) = gamma_heat
        # Gamma heat in graphite ~ 5% of total power
        gamma_frac = 0.05
        Q_gamma = gamma_frac * P
        T_graphite_ss = T_core_salt_ss + Q_gamma / p.UA_core

        # Hot leg: first-order lag, at SS: T_hotleg = T_core_out
        T_hotleg_ss = T_core_out_ss

        # HX primary: at steady state the derivative equations give:
        #   2*m_dot_p*cp*(T_hl - T_hx_p) = UA_hx*(T_hx_p - T_hx_s)    ... (A)
        #   UA_hx*(T_hx_p - T_hx_s) = 2*m_dot_s*cp_fn*(T_hx_s - T_sec_cold) ... (B)
        #
        # From (B): UA_hx*(T_hx_p - T_hx_s) = P  (total power removed)
        # But also from (A): 2*m_dot_p*cp*(T_hl - T_hx_p) = P
        # => T_hx_p = T_hl - P / (2*m_dot_p*cp) = T_core_out - delta_T_core/2
        #           = T_core_in + delta_T_core - delta_T_core/2
        #           = T_core_in + delta_T_core/2  =  T_core_salt_ss  (the average)

        T_sec_cold = p.T_secondary_cold

        # HX primary average: from 2*m_dot*cp*(T_hl - T_hx_p) = P
        T_hx_p_ss = T_hotleg_ss - P / (2.0 * m_dot_p * cp_s)

        # HX secondary average: from 2*m_dot_s*cp_fn*(T_hx_s - T_sec_cold) = P
        T_hx_sec_ss = T_sec_cold + P / (2.0 * m_dot_s * cp_fn)

        # Cold leg: first-order lag, at SS: T_coldleg = T_hx_p_out
        # T_hx_p_out = T_hotleg - P / (m_dot_p * cp_s) = T_core_in
        T_coldleg_ss = T_core_in_ss

        y_ss = np.array([
            T_core_salt_ss,
            T_graphite_ss,
            T_hotleg_ss,
            T_hx_p_ss,
            T_coldleg_ss,
            T_hx_sec_ss,
        ])

        return y_ss

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def derivatives(self, t, state, power_fraction=1.0, flow_fraction=1.0):
        """Compute time derivatives of all state variables.

        Args:
            t: Time in seconds.
            state: State vector [6] (node temperatures in K).
            power_fraction: Fission power as fraction of nominal (1.0 = 100%).
            flow_fraction: Primary loop flow as fraction of nominal
                           (1.0 = nominal, 0.0 = pump trip).

        Returns:
            np.ndarray of shape (6,) with dstate/dt.
        """
        p = self.p
        T_salt, T_gr, T_hl, T_hx_p, T_cl, T_hx_s = state

        P = power_fraction * p.P_nominal
        m_dot_p = flow_fraction * p.m_dot_primary
        m_dot_s = flow_fraction * p.m_dot_secondary  # assume secondary follows

        cp_s = p.cp_salt
        cp_g = p.cp_graphite
        cp_fn = p.cp_flinak

        # ---- Core fuel salt ----
        # Energy balance:
        #   m*cp*dT/dt = P*(1-gamma_frac) - m_dot*cp*(T_out - T_in) - UA*(T_salt - T_gr)
        # With lumped model: T_out = 2*T_salt - T_in, where T_in = T_coldleg
        gamma_frac = 0.05
        T_core_in = T_cl
        T_core_out = 2.0 * T_salt - T_core_in  # from average definition

        dT_salt = (P * (1.0 - gamma_frac)
                   - m_dot_p * cp_s * (T_core_out - T_core_in)
                   - p.UA_core * (T_salt - T_gr)
                   ) / (p.m_salt_core * cp_s)

        # ---- Core graphite ----
        # m_gr*cp_gr*dT/dt = gamma_heat + UA_core*(T_salt - T_gr)
        dT_gr = (P * gamma_frac
                 + p.UA_core * (T_salt - T_gr)
                 ) / (p.m_graphite * cp_g)
        # Note: sign convention means graphite gains heat from salt when T_salt > T_gr
        # and also gains direct gamma heating.  Net: graphite is slightly hotter.
        # Correction: graphite gives heat TO salt via conduction if T_gr > T_salt.
        # The UA term already handles the direction.  Gamma heat goes IN.
        # Actually the energy balance should be:
        #   m_gr*cp_gr*dT_gr/dt = P*gamma_frac - UA_core*(T_gr - T_salt)
        # which equals:
        #   = P*gamma_frac + UA_core*(T_salt - T_gr)
        # This is correct as written above.

        # ---- Hot leg pipe (first-order lag) ----
        # tau * dT/dt = T_in - T_pipe
        # T_in for hot leg = core outlet
        tau_hl = max(p.tau_hotleg, 0.1)  # avoid division by zero
        if flow_fraction > 0.01:
            tau_hl_eff = tau_hl / flow_fraction  # slower flow = longer transit
        else:
            tau_hl_eff = tau_hl / 0.01
        dT_hl = (T_core_out - T_hl) / tau_hl_eff

        # ---- HX primary side ----
        # Lumped-node energy balance:
        #   m*cp*dT/dt = m_dot*cp*(T_in - T_out) - UA*(T_hx_p - T_hx_s)
        # With T_in = T_hl and T_out = 2*T_hx_p - T_hl (from average def.):
        #   m_dot*cp*(T_in - T_out) = 2*m_dot*cp*(T_hl - T_hx_p)
        dT_hx_p = (2.0 * m_dot_p * cp_s * (T_hl - T_hx_p)
                    - p.UA_hx * (T_hx_p - T_hx_s)
                    ) / (p.m_salt_hx * cp_s)

        # ---- Cold leg pipe (first-order lag) ----
        # T_in for cold leg = HX primary outlet
        # HX primary outlet ~ 2*T_hx_p - T_hl (from lumped HX average)
        T_hx_p_out = 2.0 * T_hx_p - T_hl
        # Clamp to prevent unphysical values
        T_hx_p_out = max(T_hx_p_out, p.T_secondary_cold)

        tau_cl = max(p.tau_coldleg, 0.1)
        if flow_fraction > 0.01:
            tau_cl_eff = tau_cl / flow_fraction
        else:
            tau_cl_eff = tau_cl / 0.01
        dT_cl = (T_hx_p_out - T_cl) / tau_cl_eff

        # ---- HX secondary side (FLiNaK) ----
        # Lumped-node energy balance:
        #   m*cp*dT/dt = UA*(T_hx_p - T_hx_s) - m_dot_s*cp_fn*(T_out - T_in)
        # With T_in = T_sec_cold and T_out = 2*T_hx_s - T_sec_cold:
        #   m_dot_s*cp_fn*(T_out - T_in) = 2*m_dot_s*cp_fn*(T_hx_s - T_sec_cold)
        T_sec_cold = p.T_secondary_cold
        dT_hx_s = (p.UA_hx * (T_hx_p - T_hx_s)
                    - 2.0 * m_dot_s * cp_fn * (T_hx_s - T_sec_cold)
                    ) / (p.m_flinak_hx * cp_fn)

        return np.array([dT_salt, dT_gr, dT_hl, dT_hx_p, dT_cl, dT_hx_s])

    # ------------------------------------------------------------------
    # Transient simulation
    # ------------------------------------------------------------------

    def simulate(self, t_span, power_fn=None, flow_fn=None,
                 y0=None, max_step=0.5, method='RK45'):
        """Run a transient simulation with time-dependent perturbations.

        Args:
            t_span: (t_start, t_end) in seconds.
            power_fn: Callable(t) -> power_fraction (default: constant 1.0).
            flow_fn: Callable(t) -> flow_fraction (default: constant 1.0).
            y0: Initial state vector [6].  If None, uses steady_state().
            max_step: Maximum ODE solver step size in seconds.
            method: ODE integration method (default 'RK45').

        Returns:
            TransientResult with time histories.
        """
        if power_fn is None:
            power_fn = lambda t: 1.0
        if flow_fn is None:
            flow_fn = lambda t: 1.0
        if y0 is None:
            y0 = self.steady_state()

        def rhs(t, y):
            pf = power_fn(t)
            ff = flow_fn(t)
            return self.derivatives(t, y, power_fraction=pf, flow_fraction=ff)

        # Dense output for smooth interpolation
        t_eval = np.linspace(t_span[0], t_span[1],
                             max(200, int((t_span[1] - t_span[0]) / max_step)))

        sol = solve_ivp(rhs, t_span, y0, method=method,
                        t_eval=t_eval, max_step=max_step,
                        rtol=1e-6, atol=1e-8)

        if not sol.success:
            print(f"WARNING: ODE solver did not converge: {sol.message}")

        # Reconstruct power and flow histories
        power_hist = np.array([power_fn(ti) * self.p.P_nominal for ti in sol.t])
        flow_hist = np.array([flow_fn(ti) * self.p.m_dot_primary for ti in sol.t])

        return TransientResult(
            t=sol.t,
            T_core_salt=sol.y[0],
            T_graphite=sol.y[1],
            T_hotleg=sol.y[2],
            T_hx_primary=sol.y[3],
            T_coldleg=sol.y[4],
            T_hx_secondary=sol.y[5],
            power=power_hist,
            flow_primary=flow_hist,
        )


# ==========================================================================
# Factory function
# ==========================================================================

def create_loop_model(design_params=None):
    """Create a LoopModel with parameters derived from the central config.

    Args:
        design_params: DerivedParameters from config.  Computed if None.

    Returns:
        LoopModel instance ready for steady-state or transient analysis.
    """
    if design_params is None:
        design_params = compute_derived()
    d = design_params

    T_avg = CORE_AVG_TEMP

    # --- Salt properties at average temperature ---
    rho_salt = flibe_density(T_avg, uf4_mol_fraction=0.05)
    cp_salt = flibe_specific_heat(T_avg)

    # --- Salt masses ---
    m_salt_core = d.fuel_salt_volume_core * rho_salt
    # Hot/cold legs: assume 5 m pipe, 0.3 m diameter each
    pipe_L = 5.0  # m
    pipe_D = 0.3  # m
    A_pipe = math.pi / 4.0 * pipe_D**2
    V_pipe = A_pipe * pipe_L
    m_salt_pipe = V_pipe * rho_salt

    # HX primary side: assume ~0.5 * core salt volume
    V_hx_primary = 0.5 * d.fuel_salt_volume_core
    m_salt_hx = V_hx_primary * rho_salt

    # --- Graphite ---
    m_graphite = d.graphite_mass
    cp_graphite = GRAPHITE['specific_heat']

    # --- UA_core: salt-to-graphite ----
    # At steady state: UA*(T_salt - T_graphite) balances heat distribution.
    # For the lumped model, UA_core should be large enough that T_graphite
    # is close to T_salt (tight thermal coupling in graphite-moderated core).
    # Estimate: UA = n_channels * h * pi * D * H
    # h ~ 1000 W/(m2-K) (typical for molten salt convection)
    # Use config values
    h_est = 1000.0  # W/(m2-K) initial estimate
    UA_core = d.n_channels * h_est * math.pi * CHANNEL_DIAMETER * d.core_height

    # --- UA_hx: sized from LMTD ----
    # Q = UA * LMTD  =>  UA = Q / LMTD
    LMTD = d.intermediate_lmtd
    if LMTD > 0:
        UA_hx = THERMAL_POWER / LMTD
    else:
        UA_hx = THERMAL_POWER / 50.0  # fallback

    # --- Transport times ---
    v_pipe = d.volumetric_flow_rate / A_pipe
    tau_pipe = pipe_L / max(v_pipe, 0.01)

    # --- Secondary (FLiNaK) ---
    T_avg_sec = (SECONDARY_INLET_TEMP + SECONDARY_OUTLET_TEMP) / 2.0
    rho_flinak = flinak_density(T_avg_sec)
    cp_flinak = flinak_specific_heat(T_avg_sec)
    delta_T_sec = SECONDARY_OUTLET_TEMP - SECONDARY_INLET_TEMP
    m_dot_secondary = THERMAL_POWER / (cp_flinak * delta_T_sec)
    # FLiNaK mass in HX secondary side (similar volume to primary side)
    m_flinak_hx = V_hx_primary * rho_flinak

    params = LoopModelParams(
        m_salt_core=m_salt_core,
        cp_salt=cp_salt,
        m_graphite=m_graphite,
        cp_graphite=cp_graphite,
        UA_core=UA_core,
        m_dot_primary=d.mass_flow_rate,
        m_salt_hotleg=m_salt_pipe,
        tau_hotleg=tau_pipe,
        m_salt_hx=m_salt_hx,
        UA_hx=UA_hx,
        m_salt_coldleg=m_salt_pipe,
        tau_coldleg=tau_pipe,
        m_flinak_hx=m_flinak_hx,
        cp_flinak=cp_flinak,
        m_dot_secondary=m_dot_secondary,
        T_secondary_cold=SECONDARY_INLET_TEMP,
        P_nominal=THERMAL_POWER,
        T_core_salt_ss=CORE_AVG_TEMP,
        T_graphite_ss=CORE_AVG_TEMP + 5.0,  # rough estimate
    )

    return LoopModel(params)


# ==========================================================================
# Printing utilities
# ==========================================================================

def print_model_parameters(model):
    """Print lumped model parameters.

    Args:
        model: LoopModel instance.
    """
    p = model.p
    print("=" * 68)
    print("   LUMPED-PARAMETER LOOP MODEL PARAMETERS")
    print("=" * 68)

    print("\n--- Core ---")
    print(f"  Salt mass (core):         {p.m_salt_core:10.1f} kg")
    print(f"  Salt cp:                  {p.cp_salt:10.1f} J/(kg-K)")
    print(f"  Graphite mass:            {p.m_graphite:10.1f} kg")
    print(f"  Graphite cp:              {p.cp_graphite:10.1f} J/(kg-K)")
    print(f"  UA_core:                  {p.UA_core/1e3:10.1f} kW/K")

    print("\n--- Loop ---")
    print(f"  Primary mass flow:        {p.m_dot_primary:10.2f} kg/s")
    print(f"  Hot leg salt mass:        {p.m_salt_hotleg:10.1f} kg")
    print(f"  Hot leg transport time:   {p.tau_hotleg:10.2f} s")
    print(f"  HX primary salt mass:     {p.m_salt_hx:10.1f} kg")
    print(f"  UA_hx:                    {p.UA_hx/1e3:10.1f} kW/K")
    print(f"  Cold leg salt mass:       {p.m_salt_coldleg:10.1f} kg")
    print(f"  Cold leg transport time:  {p.tau_coldleg:10.2f} s")

    print("\n--- Secondary ---")
    print(f"  FLiNaK mass (HX):        {p.m_flinak_hx:10.1f} kg")
    print(f"  FLiNaK cp:               {p.cp_flinak:10.1f} J/(kg-K)")
    print(f"  Secondary mass flow:      {p.m_dot_secondary:10.2f} kg/s")
    print(f"  Secondary cold temp:      {p.T_secondary_cold - 273.15:10.1f} C")

    print(f"\n  Nominal power:            {p.P_nominal/1e6:10.1f} MW")
    print()


def print_steady_state(model, y_ss):
    """Print steady-state temperatures.

    Args:
        model: LoopModel instance.
        y_ss: Steady-state state vector.
    """
    labels = [
        "Core fuel salt (avg)",
        "Core graphite (avg)",
        "Hot leg pipe",
        "HX primary side",
        "Cold leg pipe",
        "HX secondary side",
    ]
    print("=" * 68)
    print("   STEADY-STATE NODE TEMPERATURES")
    print("=" * 68)
    for i, (label, T) in enumerate(zip(labels, y_ss)):
        print(f"  [{i}] {label:<25s}  {T - 273.15:8.1f} C  ({T:8.1f} K)")
    print()


def print_transient_summary(result):
    """Print a brief summary of transient results.

    Args:
        result: TransientResult instance.
    """
    print("=" * 68)
    print("   TRANSIENT SIMULATION SUMMARY")
    print("=" * 68)

    print(f"\n  Time span:  {result.t[0]:.1f} - {result.t[-1]:.1f} s  "
          f"({len(result.t)} points)")

    labels = [
        ("Core salt", result.T_core_salt),
        ("Graphite", result.T_graphite),
        ("Hot leg", result.T_hotleg),
        ("HX primary", result.T_hx_primary),
        ("Cold leg", result.T_coldleg),
        ("HX secondary", result.T_hx_secondary),
    ]

    print(f"\n  {'Node':<16s}  {'T_init [C]':>10s}  {'T_final [C]':>11s}  "
          f"{'T_max [C]':>10s}  {'T_min [C]':>10s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*10}")
    for label, T_arr in labels:
        print(f"  {label:<16s}  {T_arr[0]-273.15:10.1f}  {T_arr[-1]-273.15:11.1f}  "
              f"{np.max(T_arr)-273.15:10.1f}  {np.min(T_arr)-273.15:10.1f}")

    print(f"\n  Power: {result.power[0]/1e6:.1f} -> {result.power[-1]/1e6:.1f} MW")
    print(f"  Flow:  {result.flow_primary[0]:.1f} -> {result.flow_primary[-1]:.1f} kg/s")
    print()


# ==========================================================================
# Main - demonstration transients
# ==========================================================================

if __name__ == '__main__':
    d = compute_derived()
    model = create_loop_model(d)
    print_model_parameters(model)

    # --- Steady state ---
    y_ss = model.steady_state()
    print_steady_state(model, y_ss)

    # --- Verify steady state (derivatives should be ~0) ---
    dy = model.derivatives(0, y_ss, 1.0, 1.0)
    print("Steady-state derivative check (should be ~0):")
    print(f"  |dy/dt|_max = {np.max(np.abs(dy)):.3e} K/s\n")

    # =========================================================
    # Transient 1: 10% power step increase
    # =========================================================
    print("=" * 68)
    print("   TRANSIENT 1: 10% POWER STEP (t=10 s)")
    print("=" * 68)

    def power_step(t):
        return 1.1 if t >= 10.0 else 1.0

    result_step = model.simulate(
        t_span=(0, 300),
        power_fn=power_step,
        flow_fn=lambda t: 1.0,
        y0=y_ss,
    )
    print_transient_summary(result_step)

    # =========================================================
    # Transient 2: Pump coastdown (ULOF)
    # =========================================================
    print("=" * 68)
    print("   TRANSIENT 2: PUMP COASTDOWN (ULOF, t=10 s)")
    print("=" * 68)

    def pump_coastdown(t):
        """Exponential pump coastdown with ~10 s halving time."""
        if t < 10.0:
            return 1.0
        tau_coast = 10.0  # halving time
        frac = math.exp(-(t - 10.0) / tau_coast)
        return max(frac, 0.05)  # 5% natural circulation floor

    result_ulof = model.simulate(
        t_span=(0, 300),
        power_fn=lambda t: 1.0,
        flow_fn=pump_coastdown,
        y0=y_ss,
    )
    print_transient_summary(result_ulof)

    # =========================================================
    # Transient 3: Loss of heat sink (LOHS)
    # =========================================================
    print("=" * 68)
    print("   TRANSIENT 3: LOSS OF HEAT SINK (LOHS, t=10 s)")
    print("=" * 68)

    # Simulate LOHS by reducing secondary flow to 5%
    def lohs_secondary_flow(t):
        if t < 10.0:
            return 1.0
        return 0.05  # 5% residual

    # For LOHS, we modify the model slightly: reduce secondary flow
    # Using the flow_fn for primary, but also need to handle secondary.
    # As a simplification, reduce overall flow fraction which affects both loops.
    result_lohs = model.simulate(
        t_span=(0, 600),
        power_fn=lambda t: 1.0 if t < 10.0 else 0.07,  # scram to decay heat
        flow_fn=lambda t: 1.0 if t < 10.0 else 0.10,    # coastdown with NC
        y0=y_ss,
    )
    print_transient_summary(result_lohs)
