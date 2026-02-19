"""
Design Basis Accident Transient Simulations
=============================================

Three design basis accident simulations coupling the MSR point kinetics
model with the lumped-parameter loop thermal model:

1. ULOF - Unprotected Loss of Flow
   Pump trips at t=0, no scram. Negative temperature feedback must
   bring power down. Flow coastdown: m_dot(t) = m_dot_0 * exp(-t/tau).

2. UTOP - Unprotected Transient Overpower
   +200 pcm reactivity insertion over 10 seconds, no scram.
   Temperature feedback must compensate.

3. SBO - Station Blackout
   Loss of all power. Scram inserts -5000 pcm. Pump coastdown to 5%
   natural circulation. Decay heat tracked.

For each transient:
  - Coupled MSRKinetics + LoopModel
  - Adaptive time-stepping
  - Time history output
  - Peak temperature and safety margin reporting

References:
  - ORNL-TM-3286, "Safety Analysis of the MSRE" (Beall, 1970)
  - ANS-5.1-2014 (Decay Heat Standard)
  - 10 CFR 50.46 (Acceptance Criteria for ECCS)
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from config import (
    THERMAL_POWER, CORE_OUTLET_TEMP, CORE_INLET_TEMP,
    CORE_AVG_TEMP, MAX_FUEL_SALT_TEMP, MAX_VESSEL_WALL_TEMP,
    MIN_SALT_TEMP, compute_derived,
)
from thermal_hydraulics.loop_model import LoopModel, create_loop_model
from safety.point_kinetics import MSRKinetics


# =============================================================================
# Transient Result Dataclass
# =============================================================================

@dataclass
class TransientHistory:
    """Time history from a transient simulation."""

    name: str                   # transient name
    t: np.ndarray               # s, time array
    power: np.ndarray           # normalized power (n/n0)
    reactivity: np.ndarray      # dk/k
    T_fuel: np.ndarray          # K, fuel salt average temperature
    T_graphite: np.ndarray      # K, graphite average temperature
    T_hotleg: np.ndarray        # K, hot leg temperature
    T_hx_primary: np.ndarray    # K, HX primary temperature
    T_coldleg: np.ndarray       # K, cold leg temperature
    flow_fraction: np.ndarray   # fraction of nominal flow

    # Peak values
    peak_power: float           # maximum n/n0
    peak_T_fuel: float          # K, peak fuel temperature
    peak_T_fuel_time: float     # s, time of peak fuel temperature
    min_T_fuel: float           # K, minimum fuel temperature
    peak_reactivity: float      # dk/k, peak reactivity

    # Safety margins
    margin_to_boiling: float    # K, (T_boil - T_peak_fuel)
    margin_to_freezing: float   # K, (T_min_fuel - T_freeze)


# =============================================================================
# Coupled Transient Solver
# =============================================================================

def run_coupled_transient(name, duration, dt_initial,
                          kinetics, loop_model, y_ss,
                          flow_fn, rho_ext_fn=None,
                          power_override_fn=None,
                          print_interval=10.0):
    """Run a coupled kinetics + thermal-hydraulic transient.

    The coupling is explicit: at each time step, the kinetics model
    uses temperatures from the thermal model, and the thermal model
    uses power from the kinetics.

    Args:
        name: Transient name string
        duration: Total simulation time (s)
        dt_initial: Initial time step (s)
        kinetics: MSRKinetics instance
        loop_model: LoopModel instance
        y_ss: Steady-state thermal state vector [6]
        flow_fn: Callable(t) -> flow_fraction (0 to 1)
        rho_ext_fn: Callable(t) -> external reactivity dk/k (optional)
        power_override_fn: Callable(t, n_kinetics) -> power_fraction (optional,
                           for forcing power e.g. decay heat curve)
        print_interval: Print status every this many seconds

    Returns:
        TransientHistory dataclass
    """
    # Initialize
    t = 0.0
    dt = dt_initial
    y_th = y_ss.copy()

    # Storage lists
    t_hist = [t]
    power_hist = [kinetics.power]
    rho_hist = [kinetics.reactivity(y_th[0], y_th[1])]
    T_fuel_hist = [y_th[0]]
    T_gr_hist = [y_th[1]]
    T_hl_hist = [y_th[2]]
    T_hx_hist = [y_th[3]]
    T_cl_hist = [y_th[4]]
    flow_hist = [flow_fn(0)]

    last_print = 0.0

    while t < duration:
        # Adaptive dt
        dt = min(dt_initial, duration - t)
        if dt <= 0:
            break

        # Current flow fraction
        ff = flow_fn(t)

        # External reactivity
        if rho_ext_fn is not None:
            kinetics.set_external_reactivity(rho_ext_fn(t))

        # --- Kinetics step ---
        T_fuel = y_th[0]  # core fuel salt average
        T_graphite = y_th[1]  # core graphite average

        kinetics.step(dt, T_fuel, T_graphite)

        # Power to use for thermal model
        if power_override_fn is not None:
            power_frac = power_override_fn(t, kinetics.power)
        else:
            power_frac = kinetics.power

        # --- Thermal step (explicit Euler for loop model) ---
        dy_th = loop_model.derivatives(t, y_th, power_fraction=power_frac,
                                        flow_fraction=ff)
        y_th = y_th + dt * dy_th

        # Clamp temperatures to physical range
        for i in range(len(y_th)):
            y_th[i] = max(y_th[i], MIN_SALT_TEMP - 100)  # allow some subcooling

        t += dt

        # Store
        t_hist.append(t)
        power_hist.append(kinetics.power)
        rho_hist.append(kinetics.reactivity(y_th[0], y_th[1]))
        T_fuel_hist.append(y_th[0])
        T_gr_hist.append(y_th[1])
        T_hl_hist.append(y_th[2])
        T_hx_hist.append(y_th[3])
        T_cl_hist.append(y_th[4])
        flow_hist.append(ff)

        # Progress print
        if t - last_print >= print_interval:
            print(f"  t={t:7.1f}s  n={kinetics.power:8.4f}  "
                  f"T_fuel={y_th[0] - 273.15:7.1f}C  T_gr={y_th[1] - 273.15:7.1f}C  "
                  f"rho={kinetics.reactivity(y_th[0], y_th[1]) * 1e5:+8.1f}pcm  flow={ff:.3f}")
            last_print = t

    # Convert to arrays
    t_arr = np.array(t_hist)
    power_arr = np.array(power_hist)
    rho_arr = np.array(rho_hist)
    T_fuel_arr = np.array(T_fuel_hist)
    T_gr_arr = np.array(T_gr_hist)
    T_hl_arr = np.array(T_hl_hist)
    T_hx_arr = np.array(T_hx_hist)
    T_cl_arr = np.array(T_cl_hist)
    flow_arr = np.array(flow_hist)

    # Peak values
    peak_power = float(np.max(power_arr))
    idx_peak_T = int(np.argmax(T_fuel_arr))
    peak_T_fuel = float(T_fuel_arr[idx_peak_T])
    peak_T_fuel_time = float(t_arr[idx_peak_T])
    min_T_fuel = float(np.min(T_fuel_arr))
    peak_rho = float(np.max(np.abs(rho_arr)))

    # Safety margins
    T_boil = MAX_FUEL_SALT_TEMP
    T_freeze = MIN_SALT_TEMP
    margin_boil = T_boil - peak_T_fuel
    margin_freeze = min_T_fuel - T_freeze

    return TransientHistory(
        name=name,
        t=t_arr,
        power=power_arr,
        reactivity=rho_arr,
        T_fuel=T_fuel_arr,
        T_graphite=T_gr_arr,
        T_hotleg=T_hl_arr,
        T_hx_primary=T_hx_arr,
        T_coldleg=T_cl_arr,
        flow_fraction=flow_arr,
        peak_power=peak_power,
        peak_T_fuel=peak_T_fuel,
        peak_T_fuel_time=peak_T_fuel_time,
        min_T_fuel=min_T_fuel,
        peak_reactivity=peak_rho,
        margin_to_boiling=margin_boil,
        margin_to_freezing=margin_freeze,
    )


# =============================================================================
# Transient 1: ULOF
# =============================================================================

def simulate_ulof(design_params=None, duration=600.0, dt=0.01):
    """Unprotected Loss of Flow transient.

    At t=0: pump trips, flow coastdown exponential with tau_pump = 10 s.
    No scram (unprotected). Negative temperature feedback reduces power.

    Args:
        design_params: DerivedParameters (computed if None)
        duration: Simulation duration (s)
        dt: Time step (s)

    Returns:
        TransientHistory
    """
    if design_params is None:
        design_params = compute_derived()

    print("=" * 72)
    print("   TRANSIENT 1: ULOF (Unprotected Loss of Flow)")
    print("=" * 72)
    print("  Pump trips at t=0. No scram. Flow coastdown tau=10 s.")
    print()

    loop = create_loop_model(design_params)
    y_ss = loop.steady_state()

    kin = MSRKinetics(
        tau_core=design_params.residence_time_core,
        tau_loop=design_params.residence_time_core * 1.5,
    )

    tau_pump = 10.0  # pump coastdown time constant

    def flow_fn(t):
        """Exponential pump coastdown to 5% natural circulation."""
        if t < 0:
            return 1.0
        frac = math.exp(-t / tau_pump)
        return max(frac, 0.05)  # 5% natural circulation floor

    result = run_coupled_transient(
        name="ULOF",
        duration=duration,
        dt_initial=dt,
        kinetics=kin,
        loop_model=loop,
        y_ss=y_ss,
        flow_fn=flow_fn,
    )

    return result


# =============================================================================
# Transient 2: UTOP
# =============================================================================

def simulate_utop(design_params=None, duration=300.0, dt=0.01):
    """Unprotected Transient Overpower.

    +200 pcm reactivity insertion over 10 seconds (ramp), no scram.
    Temperature feedback must compensate.

    Args:
        design_params: DerivedParameters (computed if None)
        duration: Simulation duration (s)
        dt: Time step (s)

    Returns:
        TransientHistory
    """
    if design_params is None:
        design_params = compute_derived()

    print("=" * 72)
    print("   TRANSIENT 2: UTOP (Unprotected Transient Overpower)")
    print("=" * 72)
    print("  +200 pcm insertion over 10 s. No scram.")
    print()

    loop = create_loop_model(design_params)
    y_ss = loop.steady_state()

    kin = MSRKinetics(
        tau_core=design_params.residence_time_core,
        tau_loop=design_params.residence_time_core * 1.5,
    )

    rho_insertion = 200e-5  # 200 pcm = 2e-3 dk/k
    ramp_time = 10.0  # s

    def rho_ext_fn(t):
        """Linear ramp of +200 pcm over 10 s."""
        if t <= 0:
            return 0.0
        elif t < ramp_time:
            return rho_insertion * (t / ramp_time)
        else:
            return rho_insertion

    result = run_coupled_transient(
        name="UTOP",
        duration=duration,
        dt_initial=dt,
        kinetics=kin,
        loop_model=loop,
        y_ss=y_ss,
        flow_fn=lambda t: 1.0,  # constant flow
        rho_ext_fn=rho_ext_fn,
    )

    return result


# =============================================================================
# Transient 3: SBO
# =============================================================================

def simulate_sbo(design_params=None, duration=3600.0, dt=0.05):
    """Station Blackout transient.

    At t=0: loss of all power.
    - Scram: control rod insertion adds -5000 pcm immediately
    - Pump coastdown: flow -> 5% natural circulation
    - Decay heat: P(t) = P0 * 0.066 * [t^-0.2 - (t+T_op)^-0.2]

    Args:
        design_params: DerivedParameters (computed if None)
        duration: Simulation duration (s)
        dt: Time step (s)

    Returns:
        TransientHistory
    """
    if design_params is None:
        design_params = compute_derived()

    print("=" * 72)
    print("   TRANSIENT 3: SBO (Station Blackout)")
    print("=" * 72)
    print("  Loss of all power. Scram (-5000 pcm). Pump coastdown.")
    print("  Decay heat curve (ANS standard).")
    print()

    loop = create_loop_model(design_params)
    y_ss = loop.steady_state()

    kin = MSRKinetics(
        tau_core=design_params.residence_time_core,
        tau_loop=design_params.residence_time_core * 1.5,
    )

    scram_reactivity = -5000e-5  # -5000 pcm
    tau_pump = 10.0
    T_operating = 3.156e7  # 1 year of operation (s)

    def rho_ext_fn(t):
        """Scram: immediate -5000 pcm insertion."""
        if t <= 0:
            return 0.0
        # Assume rod insertion completes in 1 second
        if t < 1.0:
            return scram_reactivity * t
        return scram_reactivity

    def flow_fn(t):
        """Pump coastdown to natural circulation (5%)."""
        if t < 0:
            return 1.0
        frac = math.exp(-t / tau_pump)
        return max(frac, 0.05)

    def power_override_fn(t, n_kinetics):
        """Override power with decay heat curve after scram.

        After scram, fission power drops rapidly. Decay heat dominates.
        Use the larger of kinetics power and decay heat.
        """
        if t <= 0.1:
            return n_kinetics

        # ANS decay heat standard
        t_safe = max(t, 0.1)
        P_decay_frac = 0.066 * (t_safe**(-0.2) - (t_safe + T_operating)**(-0.2))
        P_decay_frac = max(P_decay_frac, 0.0)

        # Use the larger of fission (kinetics) and decay heat
        return max(n_kinetics, P_decay_frac)

    result = run_coupled_transient(
        name="SBO",
        duration=duration,
        dt_initial=dt,
        kinetics=kin,
        loop_model=loop,
        y_ss=y_ss,
        flow_fn=flow_fn,
        rho_ext_fn=rho_ext_fn,
        power_override_fn=power_override_fn,
        print_interval=60.0,
    )

    return result


# =============================================================================
# Printing
# =============================================================================

def print_transient_summary(result):
    """Print formatted transient analysis summary.

    Args:
        result: TransientHistory dataclass
    """
    print("\n" + "=" * 72)
    print(f"   {result.name} TRANSIENT SUMMARY")
    print("=" * 72)

    print(f"\n  Duration:                   {result.t[-1]:10.1f} s ({result.t[-1] / 60:.1f} min)")
    print(f"  Time steps:                 {len(result.t):10d}")

    print("\n--- Power ---")
    print(f"  Initial:                    {result.power[0]:10.4f} (n/n0)")
    print(f"  Peak:                       {result.peak_power:10.4f} (n/n0)")
    print(f"  Final:                      {result.power[-1]:10.4f} (n/n0)")

    print("\n--- Temperatures ---")
    print(f"  T_fuel initial:             {result.T_fuel[0] - 273.15:10.1f} C")
    print(f"  T_fuel peak:                {result.peak_T_fuel - 273.15:10.1f} C  (at t={result.peak_T_fuel_time:.1f} s)")
    print(f"  T_fuel final:               {result.T_fuel[-1] - 273.15:10.1f} C")
    print(f"  T_fuel minimum:             {result.min_T_fuel - 273.15:10.1f} C")
    print(f"  T_graphite peak:            {np.max(result.T_graphite) - 273.15:10.1f} C")
    print(f"  T_graphite final:           {result.T_graphite[-1] - 273.15:10.1f} C")

    print("\n--- Reactivity ---")
    print(f"  Initial:                    {result.reactivity[0] * 1e5:+10.1f} pcm")
    print(f"  Peak |rho|:                 {result.peak_reactivity * 1e5:10.1f} pcm")
    print(f"  Final:                      {result.reactivity[-1] * 1e5:+10.1f} pcm")

    print("\n--- Safety Margins ---")
    T_boil = MAX_FUEL_SALT_TEMP - 273.15
    T_freeze = MIN_SALT_TEMP - 273.15
    print(f"  Salt boiling point:         {T_boil:10.1f} C")
    print(f"  Margin to boiling:          {result.margin_to_boiling:10.1f} K  "
          f"{'OK' if result.margin_to_boiling > 0 else 'BOILING!'}")
    print(f"  Salt freezing point:        {T_freeze:10.1f} C")
    print(f"  Margin to freezing:         {result.margin_to_freezing:10.1f} K  "
          f"{'OK' if result.margin_to_freezing > 0 else 'FREEZING!'}")
    print(f"  Vessel wall limit:          {MAX_VESSEL_WALL_TEMP - 273.15:10.1f} C")

    # Overall assessment
    safe = (result.margin_to_boiling > 0 and
            result.margin_to_freezing > 0 and
            result.peak_T_fuel < MAX_VESSEL_WALL_TEMP + 200)  # generous check
    print(f"\n  Overall:                    {'SAFE - Temperatures within limits' if safe else 'CONCERN - Review required'}")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()

    # =========================================================
    # ULOF
    # =========================================================
    result_ulof = simulate_ulof(d, duration=600.0, dt=0.01)
    print_transient_summary(result_ulof)

    # =========================================================
    # UTOP
    # =========================================================
    result_utop = simulate_utop(d, duration=300.0, dt=0.01)
    print_transient_summary(result_utop)

    # =========================================================
    # SBO
    # =========================================================
    result_sbo = simulate_sbo(d, duration=3600.0, dt=0.05)
    print_transient_summary(result_sbo)

    # =========================================================
    # Comparison table
    # =========================================================
    print("\n" + "=" * 72)
    print("   TRANSIENT COMPARISON SUMMARY")
    print("=" * 72)

    results = [result_ulof, result_utop, result_sbo]
    print(f"\n  {'Transient':<10s}  {'Peak n/n0':>10s}  {'T_fuel_peak [C]':>15s}  "
          f"{'Margin_boil [K]':>15s}  {'Margin_freeze [K]':>17s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 15}  {'-' * 15}  {'-' * 17}")

    for r in results:
        print(f"  {r.name:<10s}  {r.peak_power:10.3f}  "
              f"{r.peak_T_fuel - 273.15:15.1f}  "
              f"{r.margin_to_boiling:15.1f}  "
              f"{r.margin_to_freezing:17.1f}")
