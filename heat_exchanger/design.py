"""
Shell-and-Tube Heat Exchanger Design for Marine MSR
====================================================

Primary-to-intermediate heat exchanger sizing using LMTD method.

Configuration:
  - Tube-side: FLiBe fuel salt (higher pressure, corrosive)
  - Shell-side: FLiNaK intermediate salt
  - Tube material: Hastelloy-N
  - Flow arrangement: Counterflow

Design basis: Q = 40 MWth transferred from FLiBe (700->600 C)
              to FLiNaK (550->650 C) via counterflow shell-and-tube HX.

References:
  - Kern, "Process Heat Transfer" (shell-side correlations)
  - Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer"
  - ORNL-TM-3832, "MSBR Heat Exchangers" (Robertson, 1972)
  - INL-EXT-10-18297 (salt property data)
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
    THERMAL_POWER, CORE_INLET_TEMP, CORE_OUTLET_TEMP,
    SECONDARY_INLET_TEMP, SECONDARY_OUTLET_TEMP,
    HASTELLOY_N, compute_derived,
)
from thermal_hydraulics.salt_properties import (
    flibe_density, flibe_viscosity, flibe_specific_heat,
    flibe_thermal_conductivity, flibe_prandtl,
    flinak_density, flinak_viscosity, flinak_specific_heat,
    flinak_thermal_conductivity, flinak_prandtl,
    gnielinski_nu, heat_transfer_coefficient, friction_factor,
)


# =============================================================================
# Tube Geometry Constants
# =============================================================================

TUBE_OD = 19.05e-3          # m (3/4 inch standard)
TUBE_WALL = 1.65e-3         # m
TUBE_ID = TUBE_OD - 2 * TUBE_WALL  # 15.75 mm
TUBE_PITCH = 25.4e-3        # m (triangular pitch, 1.33 * OD)
MAX_TUBE_LENGTH = 4.0       # m (marine constraint)
PITCH_RATIO = TUBE_PITCH / TUBE_OD  # ~1.33

# Fouling resistance (conservative for molten salts)
FOULING_TUBE = 0.0001       # m2-K/W (tube-side, FLiBe)
FOULING_SHELL = 0.0001      # m2-K/W (shell-side, FLiNaK)


# =============================================================================
# HXDesign Dataclass
# =============================================================================

@dataclass
class HXDesign:
    """Complete heat exchanger design output."""

    # --- Thermal Duty ---
    Q: float                    # W, thermal duty
    LMTD: float                 # K, log-mean temperature difference
    F_correction: float         # dimensionless, LMTD correction factor (1.0 for counterflow)

    # --- Overall Heat Transfer ---
    U_overall: float            # W/(m2-K), overall heat transfer coefficient
    UA: float                   # W/K, overall conductance
    A_required: float           # m2, required heat transfer area

    # --- Tube-side ---
    h_tube: float               # W/(m2-K), tube-side HTC
    Re_tube: float              # Reynolds number, tube-side
    Pr_tube: float              # Prandtl number, tube-side
    v_tube: float               # m/s, tube-side velocity

    # --- Shell-side ---
    h_shell: float              # W/(m2-K), shell-side HTC
    Re_shell: float             # Reynolds number, shell-side
    Pr_shell: float             # Prandtl number, shell-side
    v_shell: float              # m/s, shell-side velocity

    # --- Geometry ---
    N_tubes: int                # number of tubes
    tube_length: float          # m, effective tube length
    tube_OD: float              # m
    tube_ID: float              # m
    tube_wall: float            # m
    tube_pitch: float           # m
    shell_diameter: float       # m, inner shell diameter
    N_baffles: int              # number of baffles
    baffle_spacing: float       # m
    baffle_cut: float           # fraction (typically 0.25)

    # --- Performance ---
    effectiveness: float        # dimensionless, HX effectiveness
    NTU: float                  # number of transfer units
    Cr: float                   # capacity ratio C_min/C_max
    dp_tube: float              # Pa, tube-side pressure drop
    dp_shell: float             # Pa, shell-side pressure drop

    # --- Masses & Dimensions ---
    tube_mass: float            # kg, total tube mass
    shell_mass: float           # kg, shell mass (approximate)
    total_mass: float           # kg, total HX dry mass estimate
    hx_length_overall: float    # m, overall HX length including headers


# =============================================================================
# LMTD Calculation
# =============================================================================

def compute_lmtd(T_h_in, T_h_out, T_c_in, T_c_out):
    """Compute log-mean temperature difference for counterflow.

    Args:
        T_h_in: Hot fluid inlet temperature (K)
        T_h_out: Hot fluid outlet temperature (K)
        T_c_in: Cold fluid inlet temperature (K)
        T_c_out: Cold fluid outlet temperature (K)

    Returns:
        float: LMTD in K
    """
    dT1 = T_h_in - T_c_out    # hot end
    dT2 = T_h_out - T_c_in    # cold end

    if dT1 <= 0 or dT2 <= 0:
        raise ValueError(f"Invalid temperature cross: dT1={dT1:.1f} K, dT2={dT2:.1f} K")

    if abs(dT1 - dT2) < 0.01:
        # Equal temperature differences: LMTD = dT (avoid 0/0)
        return (dT1 + dT2) / 2.0

    return (dT1 - dT2) / math.log(dT1 / dT2)


# =============================================================================
# Shell Diameter from Tube Count
# =============================================================================

def shell_diameter_from_tubes(N_tubes, tube_OD, tube_pitch, layout='triangular'):
    """Estimate shell inner diameter from tube count using bundle diameter.

    Uses the CTP (tube count constant) method:
      D_bundle = tube_OD * (N_tubes / K1)^(1/n1)
      D_shell = D_bundle + clearance

    For triangular pitch:
      K1 = 0.319, n1 = 2.142 (1-pass)

    Args:
        N_tubes: Number of tubes
        tube_OD: Tube outer diameter (m)
        tube_pitch: Tube pitch (m)
        layout: 'triangular' or 'square'

    Returns:
        float: Shell inner diameter in m
    """
    if layout == 'triangular':
        # Approximation: bundle area = N_tubes * (sqrt(3)/2) * pitch^2
        # D_bundle = sqrt(4 * A_bundle / pi)
        A_bundle = N_tubes * (math.sqrt(3) / 2.0) * tube_pitch**2
        D_bundle = math.sqrt(4.0 * A_bundle / math.pi)
    else:
        A_bundle = N_tubes * tube_pitch**2
        D_bundle = math.sqrt(4.0 * A_bundle / math.pi)

    # Shell-to-bundle diametral clearance (~20-50 mm for fixed tubesheet)
    clearance = 0.030  # m
    D_shell = D_bundle + clearance

    return D_shell


# =============================================================================
# Tube-side Heat Transfer Coefficient
# =============================================================================

def tube_side_htc(m_dot_total, N_tubes, T_avg):
    """Calculate tube-side (FLiBe) heat transfer coefficient using Gnielinski.

    Args:
        m_dot_total: Total tube-side mass flow rate (kg/s)
        N_tubes: Number of tubes
        T_avg: Average fluid temperature (K)

    Returns:
        tuple: (h, Re, Pr, velocity) in (W/(m2-K), -, -, m/s)
    """
    rho = flibe_density(T_avg, uf4_mol_fraction=0.05)
    mu = flibe_viscosity(T_avg)
    cp = flibe_specific_heat(T_avg)
    k = flibe_thermal_conductivity(T_avg)

    # Per-tube mass flow
    m_dot_per_tube = m_dot_total / N_tubes
    A_tube = math.pi / 4.0 * TUBE_ID**2
    velocity = m_dot_per_tube / (rho * A_tube)

    Re = rho * velocity * TUBE_ID / mu
    Pr = mu * cp / k

    # Gnielinski correlation (valid Re > 2300)
    h = heat_transfer_coefficient(Re, Pr, TUBE_ID, k)

    return h, Re, Pr, velocity


# =============================================================================
# Shell-side Heat Transfer Coefficient
# =============================================================================

def shell_side_htc(m_dot_shell, N_tubes, D_shell, baffle_spacing, T_avg):
    """Calculate shell-side (FLiNaK) heat transfer coefficient using Kern method.

    Kern method uses an equivalent diameter for the shell side and applies
    the modified Gnielinski/Dittus-Boelter correlation.

    Args:
        m_dot_shell: Shell-side mass flow rate (kg/s)
        N_tubes: Number of tubes
        D_shell: Shell inner diameter (m)
        baffle_spacing: Baffle spacing (m)
        T_avg: Average fluid temperature (K)

    Returns:
        tuple: (h, Re, Pr, velocity) in (W/(m2-K), -, -, m/s)
    """
    rho = flinak_density(T_avg)
    mu = flinak_viscosity(T_avg)
    cp = flinak_specific_heat(T_avg)
    k = flinak_thermal_conductivity(T_avg)

    # Shell-side equivalent diameter for triangular pitch (Kern)
    # D_e = 4 * (free flow area per unit cell) / (wetted perimeter per unit cell)
    # For triangular pitch:
    #   D_e = 4 * [(sqrt(3)/4)*P^2 - (pi/8)*D_o^2] / [(pi/2)*D_o]
    A_cell = (math.sqrt(3) / 4.0) * TUBE_PITCH**2 - (math.pi / 8.0) * TUBE_OD**2
    P_wet = (math.pi / 2.0) * TUBE_OD
    D_e = 4.0 * A_cell / P_wet

    # Shell-side crossflow area at baffle
    # A_s = D_shell * baffle_spacing * (P - D_o) / P
    A_crossflow = D_shell * baffle_spacing * (TUBE_PITCH - TUBE_OD) / TUBE_PITCH

    # Velocity at baffle crossflow
    velocity = m_dot_shell / (rho * A_crossflow)

    Re = rho * velocity * D_e / mu
    Pr = mu * cp / k

    # Kern shell-side correlation: Nu = 0.36 * Re^0.55 * Pr^(1/3)
    # This is the standard Kern method for shell-side with segmental baffles
    if Re > 2000:
        Nu = 0.36 * Re**0.55 * Pr**(1.0 / 3.0)
    else:
        # Low-Re correction
        Nu = max(3.66, 0.36 * Re**0.55 * Pr**(1.0 / 3.0))

    h = Nu * k / D_e

    return h, Re, Pr, velocity


# =============================================================================
# Overall Heat Transfer Coefficient
# =============================================================================

def overall_U(h_tube, h_shell, r_i, r_o, k_wall, fouling_i=0.0, fouling_o=0.0):
    """Calculate overall heat transfer coefficient based on outer tube area.

    U_o = 1 / [ r_o/(r_i*h_i) + r_o*ln(r_o/r_i)/k_wall
                + 1/h_o + r_o*fouling_i/r_i + fouling_o ]

    Args:
        h_tube: Tube-side HTC (W/(m2-K))
        h_shell: Shell-side HTC (W/(m2-K))
        r_i: Tube inner radius (m)
        r_o: Tube outer radius (m)
        k_wall: Tube wall thermal conductivity (W/(m-K))
        fouling_i: Tube-side fouling resistance (m2-K/W)
        fouling_o: Shell-side fouling resistance (m2-K/W)

    Returns:
        float: Overall U in W/(m2-K) based on outer area
    """
    R_tube = r_o / (r_i * h_tube)
    R_wall = r_o * math.log(r_o / r_i) / k_wall
    R_shell = 1.0 / h_shell
    R_fouling = r_o * fouling_i / r_i + fouling_o

    return 1.0 / (R_tube + R_wall + R_shell + R_fouling)


# =============================================================================
# Pressure Drop Calculations
# =============================================================================

def tube_side_pressure_drop(Re, v, rho, L, N_passes=1):
    """Tube-side pressure drop including entry/exit losses.

    dp = N_passes * [f * L/D * rho*v^2/2 + 4 * rho*v^2/2]

    Args:
        Re: Tube-side Reynolds number
        v: Tube velocity (m/s)
        rho: Fluid density (kg/m3)
        L: Tube length (m)
        N_passes: Number of tube passes

    Returns:
        float: Pressure drop in Pa
    """
    f = friction_factor(Re)
    dp_friction = f * (L / TUBE_ID) * 0.5 * rho * v**2
    dp_entry_exit = 4.0 * 0.5 * rho * v**2  # ~4 velocity heads per pass
    return N_passes * (dp_friction + dp_entry_exit)


def shell_side_pressure_drop(Re_s, v_s, rho, D_shell, D_e, N_baffles):
    """Shell-side pressure drop using Kern method.

    dp = f * D_shell/D_e * (N_baffles + 1) * rho * v^2 / 2

    Args:
        Re_s: Shell-side Reynolds number
        v_s: Shell-side crossflow velocity (m/s)
        rho: Fluid density (kg/m3)
        D_shell: Shell inner diameter (m)
        D_e: Shell equivalent diameter (m)
        N_baffles: Number of baffles

    Returns:
        float: Pressure drop in Pa
    """
    # Kern friction factor for shell side
    if Re_s > 500:
        f_s = np.exp(0.576 - 0.19 * np.log(Re_s))
    else:
        f_s = 48.0 / max(Re_s, 1.0)

    dp = f_s * (D_shell / D_e) * (N_baffles + 1) * 0.5 * rho * v_s**2
    return dp


# =============================================================================
# Main Design Function
# =============================================================================

def design_heat_exchanger(Q=None, design_params=None):
    """Size the primary-to-intermediate shell-and-tube heat exchanger.

    Uses the LMTD method for counterflow arrangement. Iterates on tube count
    to satisfy the required heat transfer area within the maximum tube length
    constraint.

    Args:
        Q: Thermal duty in W (default: THERMAL_POWER from config)
        design_params: DerivedParameters (computed if None)

    Returns:
        HXDesign dataclass with complete design
    """
    if Q is None:
        Q = THERMAL_POWER
    if design_params is None:
        design_params = compute_derived()

    d = design_params

    # --- Temperature boundary conditions ---
    T_h_in = CORE_OUTLET_TEMP       # 700 C (973.15 K)
    T_h_out = CORE_INLET_TEMP       # 600 C (873.15 K)
    T_c_in = SECONDARY_INLET_TEMP   # 550 C (823.15 K)
    T_c_out = SECONDARY_OUTLET_TEMP # 650 C (923.15 K)

    T_h_avg = (T_h_in + T_h_out) / 2.0  # 650 C
    T_c_avg = (T_c_in + T_c_out) / 2.0  # 600 C

    # --- LMTD ---
    LMTD = compute_lmtd(T_h_in, T_h_out, T_c_in, T_c_out)
    F_correction = 1.0  # Pure counterflow, F = 1.0

    # --- Mass flow rates ---
    cp_h = flibe_specific_heat(T_h_avg)
    m_dot_h = Q / (cp_h * (T_h_in - T_h_out))  # primary (tube-side)

    cp_c = flinak_specific_heat(T_c_avg)
    m_dot_c = Q / (cp_c * (T_c_out - T_c_in))  # secondary (shell-side)

    # --- Hastelloy-N wall conductivity at average temperature ---
    T_wall_avg = (T_h_avg + T_c_avg) / 2.0
    k_wall = config.hastelloy_thermal_conductivity(T_wall_avg)

    # --- Iterative sizing: find N_tubes that gives tube_length <= MAX ---
    # Start with an initial estimate from typical U ~ 2000 W/(m2-K)
    U_est = 2000.0  # W/(m2-K)
    A_est = Q / (U_est * LMTD * F_correction)
    N_tubes_est = int(A_est / (math.pi * TUBE_OD * MAX_TUBE_LENGTH)) + 1

    # Search range
    N_min = max(100, N_tubes_est // 2)
    N_max = N_tubes_est * 3

    best_design = None

    for N_tubes in range(N_min, N_max + 1, 10):
        # Tube-side HTC
        h_t, Re_t, Pr_t, v_t = tube_side_htc(m_dot_h, N_tubes, T_h_avg)

        # Shell geometry
        D_shell = shell_diameter_from_tubes(N_tubes, TUBE_OD, TUBE_PITCH)

        # Baffle spacing: typically 0.3-0.5 * D_shell
        baffle_spacing = 0.4 * D_shell
        baffle_spacing = max(baffle_spacing, 0.05)  # minimum 50 mm

        # Shell-side HTC
        h_s, Re_s, Pr_s, v_s = shell_side_htc(m_dot_c, N_tubes, D_shell,
                                                baffle_spacing, T_c_avg)

        # Overall U
        r_i = TUBE_ID / 2.0
        r_o = TUBE_OD / 2.0
        U = overall_U(h_t, h_s, r_i, r_o, k_wall, FOULING_TUBE, FOULING_SHELL)

        # Required area
        A_req = Q / (U * LMTD * F_correction)

        # Tube length for this N_tubes
        A_per_tube = math.pi * TUBE_OD  # per unit length
        L_tube = A_req / (N_tubes * A_per_tube)

        if L_tube <= MAX_TUBE_LENGTH and L_tube >= 1.0:
            # Acceptable design; pick the one closest to 3.5 m (preferred)
            if best_design is None or abs(L_tube - 3.5) < abs(best_design[0] - 3.5):
                best_design = (L_tube, N_tubes, h_t, Re_t, Pr_t, v_t,
                               h_s, Re_s, Pr_s, v_s, U, A_req, D_shell,
                               baffle_spacing)

    if best_design is None:
        # Fallback: use maximum tube length and solve for N_tubes
        print("WARNING: Could not find design within tube length constraint.")
        print("         Using maximum tube length and computing required tube count.")
        L_tube = MAX_TUBE_LENGTH
        # Re-estimate with higher tube count
        N_tubes = N_max
        h_t, Re_t, Pr_t, v_t = tube_side_htc(m_dot_h, N_tubes, T_h_avg)
        D_shell = shell_diameter_from_tubes(N_tubes, TUBE_OD, TUBE_PITCH)
        baffle_spacing = 0.4 * D_shell
        h_s, Re_s, Pr_s, v_s = shell_side_htc(m_dot_c, N_tubes, D_shell,
                                                baffle_spacing, T_c_avg)
        r_i = TUBE_ID / 2.0
        r_o = TUBE_OD / 2.0
        U = overall_U(h_t, h_s, r_i, r_o, k_wall, FOULING_TUBE, FOULING_SHELL)
        A_req = Q / (U * LMTD * F_correction)
        best_design = (L_tube, N_tubes, h_t, Re_t, Pr_t, v_t,
                       h_s, Re_s, Pr_s, v_s, U, A_req, D_shell,
                       baffle_spacing)

    (L_tube, N_tubes, h_t, Re_t, Pr_t, v_t,
     h_s, Re_s, Pr_s, v_s, U, A_req, D_shell, baffle_spacing) = best_design

    UA = U * A_req

    # --- Number of baffles ---
    N_baffles = max(1, int(L_tube / baffle_spacing) - 1)
    baffle_cut = 0.25  # 25% baffle cut (standard)

    # --- Effectiveness-NTU ---
    C_h = m_dot_h * cp_h
    C_c = m_dot_c * cp_c
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    Cr = C_min / C_max

    NTU = UA / C_min

    # Counterflow effectiveness
    if abs(Cr - 1.0) < 1e-6:
        effectiveness = NTU / (1.0 + NTU)
    else:
        exp_term = math.exp(-NTU * (1.0 - Cr))
        effectiveness = (1.0 - exp_term) / (1.0 - Cr * exp_term)

    # --- Pressure drops ---
    rho_h = flibe_density(T_h_avg, uf4_mol_fraction=0.05)
    rho_c = flinak_density(T_c_avg)

    dp_tube = tube_side_pressure_drop(Re_t, v_t, rho_h, L_tube)

    # Shell-side equivalent diameter
    A_cell = (math.sqrt(3) / 4.0) * TUBE_PITCH**2 - (math.pi / 8.0) * TUBE_OD**2
    P_wet = (math.pi / 2.0) * TUBE_OD
    D_e = 4.0 * A_cell / P_wet

    dp_shell = shell_side_pressure_drop(Re_s, v_s, rho_c, D_shell, D_e, N_baffles)

    # --- Mass estimates ---
    rho_hastelloy = HASTELLOY_N['density']

    # Tube mass
    A_tube_metal = math.pi / 4.0 * (TUBE_OD**2 - TUBE_ID**2)
    tube_mass = N_tubes * A_tube_metal * L_tube * rho_hastelloy

    # Shell mass (cylindrical shell, ~10 mm wall)
    shell_wall = 0.010  # m
    shell_mass = (math.pi * D_shell * shell_wall * L_tube * rho_hastelloy
                  + 2 * math.pi / 4.0 * D_shell**2 * shell_wall * rho_hastelloy)  # + tubesheets

    # Total (tubes + shell + baffles + headers, ~1.5x tube+shell)
    total_mass = 1.5 * (tube_mass + shell_mass)

    # Overall HX length (tube length + 2 * header depth)
    header_depth = 0.2  # m per end
    hx_length_overall = L_tube + 2 * header_depth

    return HXDesign(
        Q=Q,
        LMTD=LMTD,
        F_correction=F_correction,
        U_overall=U,
        UA=UA,
        A_required=A_req,
        h_tube=h_t,
        Re_tube=Re_t,
        Pr_tube=Pr_t,
        v_tube=v_t,
        h_shell=h_s,
        Re_shell=Re_s,
        Pr_shell=Pr_s,
        v_shell=v_s,
        N_tubes=N_tubes,
        tube_length=L_tube,
        tube_OD=TUBE_OD,
        tube_ID=TUBE_ID,
        tube_wall=TUBE_WALL,
        tube_pitch=TUBE_PITCH,
        shell_diameter=D_shell,
        N_baffles=N_baffles,
        baffle_spacing=baffle_spacing,
        baffle_cut=baffle_cut,
        effectiveness=effectiveness,
        NTU=NTU,
        Cr=Cr,
        dp_tube=dp_tube,
        dp_shell=dp_shell,
        tube_mass=tube_mass,
        shell_mass=shell_mass,
        total_mass=total_mass,
        hx_length_overall=hx_length_overall,
    )


# =============================================================================
# Printing
# =============================================================================

def print_hx_design(hx):
    """Print formatted heat exchanger design summary.

    Args:
        hx: HXDesign dataclass instance
    """
    print("=" * 72)
    print("   PRIMARY HEAT EXCHANGER DESIGN - Shell & Tube (FLiBe / FLiNaK)")
    print("=" * 72)

    print("\n--- Thermal Duty ---")
    print(f"  Thermal power:              {hx.Q / 1e6:10.1f} MW")
    print(f"  LMTD:                       {hx.LMTD:10.2f} K")
    print(f"  F correction factor:        {hx.F_correction:10.3f}")
    print(f"  Overall U:                  {hx.U_overall:10.1f} W/(m2-K)")
    print(f"  UA:                         {hx.UA / 1e3:10.1f} kW/K")
    print(f"  Required area:              {hx.A_required:10.1f} m2")

    print("\n--- Tube-side (FLiBe) ---")
    print(f"  Heat transfer coeff:        {hx.h_tube:10.1f} W/(m2-K)")
    print(f"  Reynolds number:            {hx.Re_tube:10.0f}")
    print(f"  Prandtl number:             {hx.Pr_tube:10.1f}")
    print(f"  Velocity:                   {hx.v_tube:10.3f} m/s")
    print(f"  Pressure drop:              {hx.dp_tube / 1e3:10.2f} kPa")

    print("\n--- Shell-side (FLiNaK) ---")
    print(f"  Heat transfer coeff:        {hx.h_shell:10.1f} W/(m2-K)")
    print(f"  Reynolds number:            {hx.Re_shell:10.0f}")
    print(f"  Prandtl number:             {hx.Pr_shell:10.1f}")
    print(f"  Velocity:                   {hx.v_shell:10.3f} m/s")
    print(f"  Pressure drop:              {hx.dp_shell / 1e3:10.2f} kPa")

    print("\n--- Geometry ---")
    print(f"  Number of tubes:            {hx.N_tubes:10d}")
    print(f"  Tube length:                {hx.tube_length:10.3f} m")
    print(f"  Tube OD:                    {hx.tube_OD * 1e3:10.2f} mm")
    print(f"  Tube ID:                    {hx.tube_ID * 1e3:10.2f} mm")
    print(f"  Tube wall:                  {hx.tube_wall * 1e3:10.2f} mm")
    print(f"  Tube pitch:                 {hx.tube_pitch * 1e3:10.2f} mm")
    print(f"  Shell diameter:             {hx.shell_diameter:10.3f} m")
    print(f"  Number of baffles:          {hx.N_baffles:10d}")
    print(f"  Baffle spacing:             {hx.baffle_spacing * 1e3:10.1f} mm")
    print(f"  Baffle cut:                 {hx.baffle_cut * 100:10.1f} %")

    print("\n--- Performance ---")
    print(f"  Effectiveness:              {hx.effectiveness:10.4f}")
    print(f"  NTU:                        {hx.NTU:10.3f}")
    print(f"  Capacity ratio (Cr):        {hx.Cr:10.4f}")

    print("\n--- Mass & Dimensions ---")
    print(f"  Tube mass:                  {hx.tube_mass:10.1f} kg")
    print(f"  Shell mass:                 {hx.shell_mass:10.1f} kg")
    print(f"  Total HX mass (est.):       {hx.total_mass:10.1f} kg")
    print(f"  Overall HX length:          {hx.hx_length_overall:10.3f} m")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    d = compute_derived()
    hx = design_heat_exchanger(design_params=d)
    print_hx_design(hx)

    # Verify energy balance
    print("\n--- Verification ---")
    print(f"  Q = U*A*LMTD = {hx.U_overall * hx.A_required * hx.LMTD / 1e6:.2f} MW")
    print(f"  Design Q     = {hx.Q / 1e6:.2f} MW")
    ratio = (hx.U_overall * hx.A_required * hx.LMTD) / hx.Q
    print(f"  Ratio:         {ratio:.4f} (should be ~1.0)")

    # Check marine constraint
    print(f"\n  Tube length:   {hx.tube_length:.3f} m  (max {MAX_TUBE_LENGTH:.1f} m)  "
          f"{'OK' if hx.tube_length <= MAX_TUBE_LENGTH else 'EXCEEDS LIMIT'}")
