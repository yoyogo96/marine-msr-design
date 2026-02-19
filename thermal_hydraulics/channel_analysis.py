"""
Single-Channel Thermal-Hydraulic Analysis for Graphite-Moderated MSR
====================================================================

1D axial analysis of a single fuel salt channel surrounded by a graphite
moderator annulus.  The power profile is assumed to follow a chopped cosine
distribution along the active core height.

Physics models
--------------
* Axial power:  q'''(z) = q'''_max * cos(pi*z / H_e)
  where H_e = H * extrapolation_factor, z in [-H/2, +H/2]
* Bulk salt temperature rise from energy balance.
* Graphite temperatures from cylindrical conduction through the annulus.
* Convective heat transfer via Gnielinski / Dittus-Boelter correlations.
* Hot channel factors for conservative margin assessment.

References
----------
* ORNL-4541 (MSBR Conceptual Design)
* El-Wakil, "Nuclear Heat Transport"
* Todreas & Kazimi, "Nuclear Systems I"
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup so the module can be run standalone or imported as a package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import (
    THERMAL_POWER, CORE_INLET_TEMP, CORE_OUTLET_TEMP, CORE_AVG_TEMP,
    CHANNEL_DIAMETER, CHANNEL_PITCH, GRAPHITE, FUEL_SALT_FRACTION,
    MAX_FUEL_SALT_TEMP, MAX_GRAPHITE_TEMP, MAX_VESSEL_WALL_TEMP,
    compute_derived,
)
from thermal_hydraulics.salt_properties import (
    flibe_density, flibe_viscosity, flibe_thermal_conductivity,
    flibe_specific_heat, flibe_prandtl,
    heat_transfer_coefficient, friction_factor as calc_friction_factor,
    gnielinski_nu, dittus_boelter_nu,
)


# ==========================================================================
# Data classes
# ==========================================================================

@dataclass
class ChannelResult:
    """Results from a single-channel thermal-hydraulic analysis."""

    # Axial mesh
    z_points: np.ndarray            # m, axial positions from -H/2 to +H/2
    n_axial: int                    # number of axial nodes

    # Temperature distributions (arrays along z)
    T_salt: np.ndarray              # K, bulk salt temperature
    T_wall: np.ndarray              # K, channel wall (salt-side) temperature
    T_graphite_inner: np.ndarray    # K, graphite inner surface temperature
    T_graphite_outer: np.ndarray    # K, graphite outer surface temperature

    # Linear heat rate distribution
    q_linear: np.ndarray            # W/m, linear heat rate along z

    # Peak values
    peak_salt_temp: float           # K
    peak_wall_temp: float           # K
    peak_graphite_temp: float       # K

    # Dimensionless numbers (channel-average)
    Re: float
    Pr: float
    Nu: float
    htc: float                      # W/(m2-K), convective heat transfer coefficient

    # Pressure drop
    pressure_drop: float            # Pa, frictional pressure drop through core
    friction_factor: float          # Darcy friction factor

    # Hot channel factors
    F_q: float                      # power peaking factor
    F_eng: float                    # engineering uncertainty factor
    peak_salt_temp_hot: float       # K, with hot channel factors
    peak_graphite_temp_hot: float   # K, with hot channel factors


@dataclass
class HotChannelFactors:
    """Conservative multipliers for hot-channel safety margin assessment."""
    F_q: float = 1.15               # power peaking uncertainty
    F_eng: float = 1.10             # engineering / manufacturing tolerance
    F_flow: float = 1.05            # flow maldistribution factor


# ==========================================================================
# Core analysis functions
# ==========================================================================

def axial_power_profile(z, q_linear_max, H, extrapolation_factor=1.15):
    """Chopped-cosine axial power profile.

    Args:
        z: Axial position(s) in m, origin at core midplane.
           Range: -H/2 to +H/2.
        q_linear_max: Peak linear heat rate in W/m.
        H: Active core height in m.
        extrapolation_factor: He/H ratio accounting for reflector savings.

    Returns:
        Linear heat rate q'(z) in W/m (array or scalar matching z).
    """
    H_e = H * extrapolation_factor
    return q_linear_max * np.cos(np.pi * z / H_e)


def salt_temperature_profile(z, T_inlet, Q_channel, m_dot, cp, H,
                             extrapolation_factor=1.15):
    """Bulk salt temperature as a function of axial position.

    Derived by integrating the chopped-cosine heat flux from -H/2 to z.

    T_salt(z) = T_inlet + (Q / (m_dot * cp)) *
                [ sin(pi*z/He) / (2*sin(pi*H/(2*He))) + 0.5 ]

    Args:
        z: Axial position(s) in m (origin at midplane).
        T_inlet: Core inlet temperature in K.
        Q_channel: Total power per channel in W.
        m_dot: Mass flow rate per channel in kg/s.
        cp: Specific heat in J/(kg-K).
        H: Active core height in m.
        extrapolation_factor: He/H ratio.

    Returns:
        Bulk salt temperature(s) in K.
    """
    H_e = H * extrapolation_factor
    delta_T_total = Q_channel / (m_dot * cp)
    sin_term = np.sin(np.pi * z / H_e) / (2.0 * np.sin(np.pi * H / (2.0 * H_e)))
    return T_inlet + delta_T_total * (sin_term + 0.5)


def single_channel_analysis(power_per_channel, mass_flow_per_channel,
                            channel_diameter, core_height, T_inlet,
                            n_axial=50, extrapolation_factor=1.15,
                            channel_pitch=None, hot_channel_factors=None):
    """Perform a complete 1-D single-channel thermal-hydraulic analysis.

    MSR thermal model
    -----------------
    In a molten salt reactor the fuel salt IS the heat source.  Fission heat
    is generated volumetrically inside the salt channel and must be
    transported outward:

        salt centreline  ->  salt bulk  ->  channel wall  ->  graphite

    The channel wall (graphite inner surface) is COOLER than the bulk salt
    because heat flows from the salt toward the graphite moderator.

    Heat transfer at the salt-wall interface:
        q''_wall = h * (T_bulk - T_wall)
    where q''_wall is the heat flux leaving the salt through the channel
    wall surface (pi * D * dz per unit length).

    By symmetry in a regular hexagonal lattice where every channel
    produces the same power, the heat flux at the outer graphite boundary
    (r = pitch/2) is zero.  The graphite temperature distribution between
    channels is governed only by its own volumetric gamma heating (small,
    ~5% of fission power).  With zero external heat flux and small internal
    generation, the graphite temperature is close to the wall temperature,
    with the peak (at r2) only slightly above T_wall.

    Args:
        power_per_channel: Total thermal power deposited in one channel, W.
        mass_flow_per_channel: Salt mass flow rate through one channel, kg/s.
        channel_diameter: Fuel salt channel hydraulic diameter, m.
        core_height: Active core height, m.
        T_inlet: Salt inlet temperature, K.
        n_axial: Number of axial mesh points (default 50).
        extrapolation_factor: Cosine extrapolation ratio He/H (default 1.15).
        channel_pitch: Hexagonal lattice pitch, m.  Defaults to config value.
        hot_channel_factors: HotChannelFactors instance (defaults created if None).

    Returns:
        ChannelResult dataclass with all computed fields.
    """
    if channel_pitch is None:
        channel_pitch = CHANNEL_PITCH
    if hot_channel_factors is None:
        hot_channel_factors = HotChannelFactors()

    H = core_height
    H_e = H * extrapolation_factor
    D = channel_diameter
    r1 = D / 2.0                            # channel inner radius
    r2 = channel_pitch / 2.0                # outer radius of graphite cell (hex approx)

    # ---- Axial mesh (midplane = 0) ----
    z = np.linspace(-H / 2.0, H / 2.0, n_axial)

    # ---- Average salt properties at midpoint temperature ----
    T_avg = (T_inlet + T_inlet + power_per_channel /
             (mass_flow_per_channel * flibe_specific_heat(T_inlet))) / 2.0
    # Clamp to reasonable range
    T_avg = min(max(T_avg, T_inlet), T_inlet + 200.0)

    rho = flibe_density(T_avg, uf4_mol_fraction=0.05)
    mu = flibe_viscosity(T_avg)
    cp = flibe_specific_heat(T_avg)
    k_salt = flibe_thermal_conductivity(T_avg)
    Pr = flibe_prandtl(T_avg)

    A_flow = math.pi / 4.0 * D**2
    v = mass_flow_per_channel / (rho * A_flow)
    Re = rho * v * D / mu

    # ---- Friction factor and pressure drop ----
    f = calc_friction_factor(Re, roughness=0.0, D=D)
    dp = f * (H / D) * 0.5 * rho * v**2      # Pa

    # ---- Heat transfer coefficient ----
    htc = heat_transfer_coefficient(Re, Pr, D, k_salt)
    if Re > 10000:
        Nu = gnielinski_nu(Re, Pr)
    elif Re > 2300:
        Nu_lam = 3.66
        Nu_turb = gnielinski_nu(10000, Pr)
        frac = (Re - 2300.0) / (10000.0 - 2300.0)
        Nu = Nu_lam * (1.0 - frac) + Nu_turb * frac
    else:
        Nu = 3.66

    # ---- Linear heat rate ----
    # Total power = integral of q'(z) dz over [-H/2, H/2]
    # Integral of cos(pi z / He) from -H/2 to H/2 = 2 He/pi * sin(pi H / (2 He))
    integral_factor = (2.0 * H_e / math.pi) * math.sin(math.pi * H / (2.0 * H_e))
    q_linear_max = power_per_channel / integral_factor   # W/m

    q_lin = axial_power_profile(z, q_linear_max, H, extrapolation_factor)

    # ---- Salt bulk temperature ----
    T_salt = salt_temperature_profile(z, T_inlet, power_per_channel,
                                      mass_flow_per_channel, cp, H,
                                      extrapolation_factor)

    # ---- Wall temperature (graphite inner surface) ----
    # Heat generated volumetrically in the salt must exit through the channel
    # wall.  The wall heat flux per unit length equals the linear heat rate:
    #     q''_wall = q'(z) / (pi * D)     [W/m2]
    #
    # Heat flows FROM salt TO wall, so T_wall < T_salt:
    #     q''_wall = h * (T_salt - T_wall)
    #     T_wall   = T_salt - q''_wall / h
    q_flux = q_lin / (math.pi * D)          # W/m2 on channel wall
    T_wall = T_salt - q_flux / htc

    # ---- Graphite temperatures ----
    # By symmetry in a uniform lattice, there is zero heat flux at the
    # outer graphite boundary (r2 = pitch/2).  The only heat source in
    # the graphite is gamma heating, which is approximately 5% of the
    # fission power deposited uniformly in the graphite annulus.
    #
    # For an annular region with uniform volumetric heat generation q'''_g,
    # insulated at r2 and held at T_wall at r1, the temperature distribution
    # peaks at r2:
    #
    #   T(r2) - T(r1) = (q'''_g / 4k_g) * [r2^2 - r1^2
    #                     - 2*r2^2 * ln(r2/r1)]         (hollow cylinder)
    #
    # This is a small correction (a few degrees) because k_graphite ~ 120 W/m-K.

    k_graphite = GRAPHITE['thermal_conductivity']     # W/(m-K)
    gamma_heat_fraction = 0.05

    # Volumetric gamma heat generation rate in graphite [W/m3]
    # = gamma_fraction * q_linear / A_graphite_annulus
    A_graphite = math.pi * (r2**2 - r1**2)     # graphite annulus area per channel
    q_triple_prime_graphite = gamma_heat_fraction * q_lin / A_graphite  # W/m3 per axial node

    # Temperature rise from r1 to r2 in insulated-outer-boundary annulus
    # with uniform volumetric generation:
    #   dT = (q''' / 4k) * [ (r2^2 - r1^2) - 2*r2^2 * ln(r2/r1) ]
    ln_ratio = math.log(r2 / r1)
    geom_factor = (r2**2 - r1**2) - 2.0 * r2**2 * ln_ratio

    # Graphite inner surface = channel wall temperature
    T_graphite_inner = T_wall.copy()

    # Graphite outer surface (peak in graphite, at symmetry plane)
    delta_T_graphite = (q_triple_prime_graphite / (4.0 * k_graphite)) * geom_factor
    T_graphite_outer = T_wall + delta_T_graphite

    # ---- Peak temperatures ----
    peak_salt = float(np.max(T_salt))
    peak_wall = float(np.max(T_wall))
    peak_graphite = float(np.max(T_graphite_outer))

    # ---- Hot channel factors ----
    F_q = hot_channel_factors.F_q
    F_eng = hot_channel_factors.F_eng

    # Hot channel peak salt:  delta_T above inlet is scaled by F_q * F_eng
    delta_T_salt = peak_salt - T_inlet
    peak_salt_hot = T_inlet + delta_T_salt * F_q * F_eng

    # Hot channel peak graphite (use delta from inlet as well)
    delta_T_graphite_hot = peak_graphite - T_inlet
    peak_graphite_hot = T_inlet + delta_T_graphite_hot * F_q * F_eng

    return ChannelResult(
        z_points=z,
        n_axial=n_axial,
        T_salt=T_salt,
        T_wall=T_wall,
        T_graphite_inner=T_graphite_inner,
        T_graphite_outer=T_graphite_outer,
        q_linear=q_lin,
        peak_salt_temp=peak_salt,
        peak_wall_temp=peak_wall,
        peak_graphite_temp=peak_graphite,
        Re=Re,
        Pr=Pr,
        Nu=Nu,
        htc=htc,
        pressure_drop=dp,
        friction_factor=f,
        F_q=F_q,
        F_eng=F_eng,
        peak_salt_temp_hot=peak_salt_hot,
        peak_graphite_temp_hot=peak_graphite_hot,
    )


# ==========================================================================
# Convenience wrapper using config parameters
# ==========================================================================

def run_nominal_channel_analysis():
    """Run single-channel analysis at nominal design conditions.

    Retrieves all parameters from the central config module.

    Returns:
        ChannelResult for the average channel at nominal power.
    """
    d = compute_derived()

    power_per_channel = THERMAL_POWER / d.n_channels
    mass_flow_per_channel = d.mass_flow_rate / d.n_channels

    result = single_channel_analysis(
        power_per_channel=power_per_channel,
        mass_flow_per_channel=mass_flow_per_channel,
        channel_diameter=CHANNEL_DIAMETER,
        core_height=d.core_height,
        T_inlet=CORE_INLET_TEMP,
        channel_pitch=CHANNEL_PITCH,
    )
    return result


def print_channel_results(res):
    """Print a formatted summary of single-channel analysis results.

    Args:
        res: ChannelResult instance.
    """
    print("=" * 68)
    print("   SINGLE-CHANNEL THERMAL-HYDRAULIC ANALYSIS RESULTS")
    print("=" * 68)

    print("\n--- Flow Conditions ---")
    print(f"  Reynolds number:          {res.Re:12.0f}")
    print(f"  Prandtl number:           {res.Pr:12.1f}")
    print(f"  Nusselt number:           {res.Nu:12.1f}")
    print(f"  Heat transfer coeff:      {res.htc:12.1f} W/(m2-K)")
    print(f"  Darcy friction factor:    {res.friction_factor:12.5f}")
    print(f"  Core pressure drop:       {res.pressure_drop/1e3:12.2f} kPa")

    print("\n--- Peak Temperatures (Nominal Channel) ---")
    print(f"  Salt inlet:               {CORE_INLET_TEMP - 273.15:12.1f} C")
    print(f"  Peak salt:                {res.peak_salt_temp - 273.15:12.1f} C")
    print(f"  Peak wall:                {res.peak_wall_temp - 273.15:12.1f} C")
    print(f"  Peak graphite (inner):    {np.max(res.T_graphite_inner) - 273.15:12.1f} C")
    print(f"  Peak graphite (outer):    {np.max(res.T_graphite_outer) - 273.15:12.1f} C")

    print("\n--- Hot Channel Temperatures ---")
    print(f"  F_q (power peaking):      {res.F_q:12.2f}")
    print(f"  F_eng (engineering):       {res.F_eng:12.2f}")
    print(f"  Peak salt (hot ch.):      {res.peak_salt_temp_hot - 273.15:12.1f} C")
    print(f"  Peak graphite (hot ch.):  {res.peak_graphite_temp_hot - 273.15:12.1f} C")

    print("\n--- Safety Margin Check ---")
    salt_margin = MAX_FUEL_SALT_TEMP - res.peak_salt_temp_hot
    graphite_margin = MAX_GRAPHITE_TEMP - res.peak_graphite_temp_hot
    print(f"  Salt temp margin:         {salt_margin:12.1f} K   "
          f"({'PASS' if salt_margin > 0 else '** FAIL **'})")
    print(f"  Graphite temp margin:     {graphite_margin:12.1f} K   "
          f"({'PASS' if graphite_margin > 0 else '** FAIL **'})")

    print("\n--- Axial Profile (selected nodes) ---")
    indices = np.linspace(0, len(res.z_points) - 1, 11, dtype=int)
    print(f"  {'z [m]':>8s}  {'T_salt [C]':>11s}  {'T_wall [C]':>11s}  "
          f"{'T_gr_in [C]':>11s}  {'T_gr_out [C]':>12s}  {'q_lin [kW/m]':>12s}")
    print(f"  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*12}  {'-'*12}")
    for i in indices:
        print(f"  {res.z_points[i]:8.3f}  "
              f"{res.T_salt[i] - 273.15:11.1f}  "
              f"{res.T_wall[i] - 273.15:11.1f}  "
              f"{res.T_graphite_inner[i] - 273.15:11.1f}  "
              f"{res.T_graphite_outer[i] - 273.15:12.1f}  "
              f"{res.q_linear[i] / 1e3:12.2f}")

    print()


# ==========================================================================
# Main
# ==========================================================================

if __name__ == '__main__':
    result = run_nominal_channel_analysis()
    print_channel_results(result)
