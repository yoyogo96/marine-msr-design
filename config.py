"""
Central Configuration for 40 MWth Marine Molten Salt Reactor (MSR)

This file defines ALL design parameters for the MSR in a tiered structure:
  Tier 1: Fixed Design Basis (constants)
  Tier 2: Material Properties (temperature-dependent)
  Tier 3: Derived Parameters (computed from Tier 1 + Tier 2)
  Tier 4: Nuclear Data (cross-sections, delayed neutron groups)
  Tier 5: Ship Motion Parameters (DNV Rules)
  Tier 6: Safety Limits

All values in SI units. Units noted in comments.

Usage:
    from config import compute_derived, print_summary
    design = compute_derived()
    print_summary(design)
"""

import math
from dataclasses import dataclass

import numpy as np


# =============================================================================
# TIER 1: FIXED DESIGN BASIS
# =============================================================================

# --- REACTOR DESIGN BASIS ---
THERMAL_POWER = 40e6              # W (40 MWth)
ELECTRICAL_POWER = 16e6           # W (~16 MWe at 40% efficiency)
THERMAL_EFFICIENCY = 0.40         # sCO2 Brayton cycle

# --- SHIP PARAMETERS ---
SHIP_CLASS = "6,000 TEU Panamax Container Ship"
ENGINE_ROOM_LENGTH = 25.0         # m
ENGINE_ROOM_WIDTH = 20.0          # m
ENGINE_ROOM_HEIGHT = 12.0         # m
DESIGN_SPEED_KNOTS = 18.0         # knots
SHIP_DISPLACEMENT = 80000         # tonnes (approximate)

# --- SALT COMPOSITION ---
SALT_TYPE = "FLiBe + UF4"
LIF_MOLE_FRACTION = 0.645         # mol fraction LiF (adjusted for UF4)
BEF2_MOLE_FRACTION = 0.305        # mol fraction BeF2
UF4_MOLE_FRACTION = 0.05          # mol fraction UF4 (~5 mol%)
LI7_ENRICHMENT = 0.99995          # 7Li enrichment fraction

# --- ENRICHMENT ---
U235_ENRICHMENT = 0.12            # 12% U-235 (HALEU, initial estimate - iterated by criticality)
ENRICHMENT_MIN = 0.05             # Search bounds (lower)
ENRICHMENT_MAX = 0.20             # Search bounds (upper)

# --- OPERATING CONDITIONS ---
CORE_INLET_TEMP = 600 + 273.15    # K (600 C)
CORE_OUTLET_TEMP = 700 + 273.15   # K (700 C)
CORE_AVG_TEMP = 650 + 273.15      # K (650 C)
OPERATING_PRESSURE = 0.2e6        # Pa (~2 atm, near atmospheric)
SECONDARY_INLET_TEMP = 550 + 273.15   # K
SECONDARY_OUTLET_TEMP = 650 + 273.15  # K

# --- GEOMETRY TARGETS ---
CORE_HEIGHT_TO_DIAMETER = 1.2     # H/D ratio (slightly elongated)
GRAPHITE_VOLUME_FRACTION = 0.77   # Based on MSRE experience
FUEL_SALT_FRACTION = 0.23         # 1 - graphite fraction
CHANNEL_PITCH = 0.05              # m (hexagonal pitch, initial estimate)
CHANNEL_DIAMETER = 0.025          # m (salt channel diameter)

# --- DESIGN LIFE ---
DESIGN_LIFE_YEARS = 20            # years
CAPACITY_FACTOR = 0.85            # fraction


# =============================================================================
# TIER 2: MATERIAL PROPERTIES
# =============================================================================

# --- Hastelloy-N (Ni-Mo-Cr alloy, primary structural material) ---
HASTELLOY_N = {
    'density': 8860,                   # kg/m3
    'melting_point': 1372 + 273.15,    # K
    'thermal_expansion': 12.3e-6,      # 1/K (21-316 C avg)
    'elastic_modulus': 219e9,          # Pa (at room temp)
    'poisson_ratio': 0.32,            # dimensionless
    'max_service_temp': 704 + 273.15,  # K
    # Thermal conductivity: k(T) W/m-K (curve fit from data)
    # At 280 C: 16.1, 400 C: 18.4, 500 C: 19.6, 600 C: 21.7, 700 C: 26.2
    'creep_rupture_700C_10khr': 83e6,  # Pa
    'corrosion_rate': 25e-6,           # m/yr in FLiBe
    'allowable_stress_design': 55e6,   # Pa (conservative, ASME)
}

# --- Nuclear Graphite (IG-110 type) ---
GRAPHITE = {
    'density': 1780,                   # kg/m3
    'thermal_conductivity': 120,       # W/m-K (BOL, unirradiated)
    'specific_heat': 1700,             # J/kg-K (at ~650 C)
    'thermal_expansion': 4.5e-6,       # 1/K
    'moderating_ratio': 200,           # dimensionless (approximate)
}

# --- Biological Shield Materials ---
CONCRETE = {
    'density': 2350,                   # kg/m3 (regular concrete)
    'density_heavy': 3500,             # kg/m3 (baryte concrete)
}

B4C = {
    'density': 2520,                   # kg/m3
}

STEEL = {
    'density': 7850,                   # kg/m3
    'thermal_conductivity': 50,        # W/m-K
}


# =============================================================================
# TIER 2 (continued): SALT THERMOPHYSICAL PROPERTY CORRELATIONS
# =============================================================================

def salt_density(T):
    """FLiBe salt density as function of temperature.

    Args:
        T: Temperature in K

    Returns:
        Density in kg/m3

    Reference: Williams et al., ORNL/TM-2006/12
    """
    return 2413.0 - 0.488 * (T - 273.15)


def salt_viscosity(T):
    """FLiBe salt dynamic viscosity as function of temperature.

    Args:
        T: Temperature in K

    Returns:
        Dynamic viscosity in Pa-s

    Reference: Williams et al., ORNL/TM-2006/12
    """
    return 1.16e-4 * math.exp(3755.0 / T)


def salt_thermal_conductivity(T):
    """FLiBe salt thermal conductivity as function of temperature.

    Args:
        T: Temperature in K

    Returns:
        Thermal conductivity in W/m-K

    Reference: Williams et al., ORNL/TM-2006/12
    Note: FLiBe thermal conductivity is relatively constant ~1.0 W/m-K
    """
    return 1.1


def salt_specific_heat(T):
    """FLiBe salt specific heat as function of temperature.

    Args:
        T: Temperature in K

    Returns:
        Specific heat in J/kg-K

    Reference: Williams et al., ORNL/TM-2006/12
    Note: Approximately constant for FLiBe in the operating range.
    """
    return 2386.0


def hastelloy_thermal_conductivity(T):
    """Hastelloy-N thermal conductivity as function of temperature.

    Args:
        T: Temperature in K

    Returns:
        Thermal conductivity in W/m-K

    Data points (from ORNL reports):
        280 C -> 16.1, 400 C -> 18.4, 500 C -> 19.6, 600 C -> 21.7, 700 C -> 26.2
    Linear interpolation from polynomial fit.
    """
    T_C = T - 273.15  # Convert to Celsius
    # Quadratic fit to data points
    return 12.29 + 0.01058 * T_C + 1.429e-5 * T_C**2


# =============================================================================
# TIER 3: DERIVED PARAMETERS
# =============================================================================

@dataclass
class DerivedParameters:
    """All derived design parameters computed from Tier 1 and Tier 2."""

    # --- Core Geometry ---
    core_volume: float                 # m3 (total core volume)
    core_diameter: float               # m
    core_radius: float                 # m
    core_height: float                 # m
    core_power_density: float          # W/m3 (volumetric, whole core)
    fuel_power_density: float          # W/m3 (volumetric, fuel salt only)

    # --- Channel Geometry ---
    n_channels: int                    # number of fuel channels
    channel_flow_area: float           # m2 (single channel)
    total_flow_area: float             # m2 (all channels)

    # --- Salt Inventory ---
    fuel_salt_volume_core: float       # m3 (salt in core only)
    fuel_salt_volume_total: float      # m3 (core + plena + piping, est. 2x core)
    fuel_salt_mass_core: float         # kg
    fuel_salt_mass_total: float        # kg
    fuel_salt_density: float           # kg/m3 (at average temp)

    # --- Graphite Inventory ---
    graphite_volume: float             # m3
    graphite_mass: float               # kg

    # --- Flow Parameters ---
    mass_flow_rate: float              # kg/s
    volumetric_flow_rate: float        # m3/s
    salt_velocity: float               # m/s (in channels)
    residence_time_core: float         # s (fuel salt in core)
    salt_viscosity: float              # Pa-s (at average temp)
    salt_cp: float                     # J/kg-K (at average temp)
    salt_k: float                      # W/m-K (at average temp)
    reynolds_number: float             # dimensionless
    prandtl_number: float              # dimensionless

    # --- Pressure Drop ---
    friction_factor: float             # Darcy friction factor
    pressure_drop_core: float          # Pa (core only)

    # --- Pump ---
    pump_power_hydraulic: float        # W (hydraulic power)
    pump_power_shaft: float            # W (assuming 80% efficiency)

    # --- Vessel ---
    vessel_inner_radius: float         # m
    vessel_wall_thickness: float       # m (estimated from pressure + safety)
    vessel_outer_radius: float         # m
    vessel_height: float               # m (core + plena)

    # --- Intermediate Loop ---
    intermediate_dt: float             # K (temperature difference)
    intermediate_lmtd: float           # K (log-mean temperature difference)

    # --- Uranium Loading ---
    uranium_mass: float                # kg (total U in salt)
    u235_mass: float                   # kg (U-235 in salt)


def compute_derived():
    """Compute all derived parameters from Tier 1 and Tier 2 data.

    Returns:
        DerivedParameters dataclass with all computed values.
    """
    # --- Salt properties at average temperature ---
    T_avg = CORE_AVG_TEMP
    rho_salt = salt_density(T_avg)
    mu_salt = salt_viscosity(T_avg)
    cp_salt = salt_specific_heat(T_avg)
    k_salt = salt_thermal_conductivity(T_avg)

    # --- Core Geometry ---
    # Power density target: ~20-40 MW/m3 for graphite-moderated MSR (MSRE was ~5)
    # We size from H/D ratio and power density
    # Q = q''' * V_core  =>  V_core = Q / q'''
    # Use fuel power density approach: all power deposited in fuel salt fraction
    # Reasonable average power density for whole core: ~22 MW/m3
    # (MSBR was ~22 MW/m3, MSRE was ~5 MW/m3; 40 MWth is between)
    # V_core = pi/4 * D^2 * H,  H = HD_ratio * D
    # V_core = pi/4 * HD_ratio * D^3
    # Solve for D:

    # Target volumetric power density for the whole core
    # We'll set a target and derive geometry, then check
    target_power_density = 22.0e6  # W/m3 (whole core average)

    core_volume = THERMAL_POWER / target_power_density  # m3
    # V = pi/4 * H/D * D^3  =>  D = (4*V / (pi * H/D))^(1/3)
    core_diameter = (4.0 * core_volume / (math.pi * CORE_HEIGHT_TO_DIAMETER))**(1.0 / 3.0)
    core_radius = core_diameter / 2.0
    core_height = CORE_HEIGHT_TO_DIAMETER * core_diameter

    # Actual core volume (recalculate for consistency)
    core_volume = math.pi / 4.0 * core_diameter**2 * core_height
    core_power_density = THERMAL_POWER / core_volume  # W/m3

    # Fuel salt volume in core
    fuel_salt_volume_core = core_volume * FUEL_SALT_FRACTION
    fuel_power_density = THERMAL_POWER / fuel_salt_volume_core  # W/m3

    # --- Channel Geometry ---
    # Hexagonal lattice: area per cell = (sqrt(3)/2) * pitch^2
    cell_area = (math.sqrt(3) / 2.0) * CHANNEL_PITCH**2  # m2
    channel_flow_area = math.pi / 4.0 * CHANNEL_DIAMETER**2  # m2

    # Core cross-sectional area
    core_cross_section = math.pi / 4.0 * core_diameter**2  # m2

    # Number of channels
    n_channels = int(core_cross_section / cell_area)
    total_flow_area = n_channels * channel_flow_area  # m2

    # --- Salt Inventory ---
    fuel_salt_mass_core = fuel_salt_volume_core * rho_salt  # kg
    # Total salt inventory: core + upper/lower plena + piping + HX
    # Typically ~2x core volume
    fuel_salt_volume_total = fuel_salt_volume_core * 2.0  # m3
    fuel_salt_mass_total = fuel_salt_volume_total * rho_salt  # kg

    # --- Graphite Inventory ---
    graphite_volume = core_volume * GRAPHITE_VOLUME_FRACTION  # m3
    graphite_mass = graphite_volume * GRAPHITE['density']  # kg

    # --- Flow Parameters ---
    delta_T = CORE_OUTLET_TEMP - CORE_INLET_TEMP  # K
    mass_flow_rate = THERMAL_POWER / (cp_salt * delta_T)  # kg/s
    volumetric_flow_rate = mass_flow_rate / rho_salt  # m3/s

    # Velocity in channels
    salt_velocity = volumetric_flow_rate / total_flow_area  # m/s

    # Residence time in core
    residence_time_core = fuel_salt_volume_core / volumetric_flow_rate  # s

    # --- Dimensionless Numbers ---
    reynolds_number = rho_salt * salt_velocity * CHANNEL_DIAMETER / mu_salt
    prandtl_number = mu_salt * cp_salt / k_salt

    # --- Pressure Drop (Darcy-Weisbach) ---
    if reynolds_number < 2300:
        friction_factor = 64.0 / reynolds_number  # Laminar
    else:
        # Colebrook approximation (smooth pipe)
        friction_factor = 0.316 * reynolds_number**(-0.25)  # Blasius

    pressure_drop_core = (friction_factor * core_height / CHANNEL_DIAMETER *
                          0.5 * rho_salt * salt_velocity**2)  # Pa

    # --- Pump Power ---
    pump_efficiency = 0.80
    pump_power_hydraulic = volumetric_flow_rate * pressure_drop_core  # W
    # Add factor of ~3 for total loop losses (HX, piping, plena)
    total_pressure_drop = pressure_drop_core * 3.0  # Pa
    pump_power_shaft = (volumetric_flow_rate * total_pressure_drop) / pump_efficiency  # W

    # --- Vessel Sizing ---
    # Vessel must contain core + reflector + downcomer
    reflector_thickness = 0.15  # m (graphite reflector)
    downcomer_gap = 0.05  # m
    vessel_inner_radius = core_radius + reflector_thickness + downcomer_gap  # m

    # Wall thickness from thin-wall pressure vessel: t = P*r / (sigma_allow - 0.6*P)
    P = OPERATING_PRESSURE  # Pa
    sigma = HASTELLOY_N['allowable_stress_design']  # Pa
    vessel_wall_thickness_pressure = P * vessel_inner_radius / (sigma - 0.6 * P)  # m
    # Minimum practical thickness
    vessel_wall_thickness = max(vessel_wall_thickness_pressure, 0.02)  # m (min 20 mm)
    vessel_outer_radius = vessel_inner_radius + vessel_wall_thickness  # m

    # Vessel height: core + upper plenum (0.5m) + lower plenum (0.5m)
    vessel_height = core_height + 1.0  # m

    # --- Intermediate Loop LMTD ---
    # Primary: T_hot_in = 700 C, T_cold_out = 600 C
    # Secondary: T_cold_in = 550 C, T_hot_out = 650 C
    # Counter-flow
    dt_hot = CORE_OUTLET_TEMP - SECONDARY_OUTLET_TEMP  # hot end delta
    dt_cold = CORE_INLET_TEMP - SECONDARY_INLET_TEMP   # cold end delta
    intermediate_dt = (dt_hot + dt_cold) / 2.0  # K (arithmetic mean)
    if abs(dt_hot - dt_cold) < 1e-6:
        intermediate_lmtd = dt_hot
    else:
        intermediate_lmtd = (dt_hot - dt_cold) / math.log(dt_hot / dt_cold)  # K

    # --- Uranium Loading ---
    # Molecular weights
    MW_LiF = 25.94   # g/mol (7Li)
    MW_BeF2 = 47.01  # g/mol
    MW_UF4 = 314.02  # g/mol (average U)

    # Average molecular weight of salt mixture
    MW_avg = (LIF_MOLE_FRACTION * MW_LiF +
              BEF2_MOLE_FRACTION * MW_BeF2 +
              UF4_MOLE_FRACTION * MW_UF4)  # g/mol

    # Mass fraction of uranium in salt
    # UF4 mass fraction * (U mass / UF4 mass)
    MW_U = 238.03  # g/mol (mostly U-238)
    uf4_mass_fraction = (UF4_MOLE_FRACTION * MW_UF4) / MW_avg
    u_mass_fraction = uf4_mass_fraction * (MW_U / MW_UF4)

    uranium_mass = fuel_salt_mass_total * u_mass_fraction  # kg
    u235_mass = uranium_mass * U235_ENRICHMENT  # kg

    return DerivedParameters(
        core_volume=core_volume,
        core_diameter=core_diameter,
        core_radius=core_radius,
        core_height=core_height,
        core_power_density=core_power_density,
        fuel_power_density=fuel_power_density,
        n_channels=n_channels,
        channel_flow_area=channel_flow_area,
        total_flow_area=total_flow_area,
        fuel_salt_volume_core=fuel_salt_volume_core,
        fuel_salt_volume_total=fuel_salt_volume_total,
        fuel_salt_mass_core=fuel_salt_mass_core,
        fuel_salt_mass_total=fuel_salt_mass_total,
        fuel_salt_density=rho_salt,
        graphite_volume=graphite_volume,
        graphite_mass=graphite_mass,
        mass_flow_rate=mass_flow_rate,
        volumetric_flow_rate=volumetric_flow_rate,
        salt_velocity=salt_velocity,
        residence_time_core=residence_time_core,
        salt_viscosity=mu_salt,
        salt_cp=cp_salt,
        salt_k=k_salt,
        reynolds_number=reynolds_number,
        prandtl_number=prandtl_number,
        friction_factor=friction_factor,
        pressure_drop_core=pressure_drop_core,
        pump_power_hydraulic=pump_power_hydraulic,
        pump_power_shaft=pump_power_shaft,
        vessel_inner_radius=vessel_inner_radius,
        vessel_wall_thickness=vessel_wall_thickness,
        vessel_outer_radius=vessel_outer_radius,
        vessel_height=vessel_height,
        intermediate_dt=intermediate_dt,
        intermediate_lmtd=intermediate_lmtd,
        uranium_mass=uranium_mass,
        u235_mass=u235_mass,
    )


# =============================================================================
# TIER 4: NUCLEAR DATA
# =============================================================================

# Delayed neutron data (Keepin 6-group, U-235 thermal fission)
DELAYED_NEUTRON_GROUPS = {
    'beta': [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],  # group fractions
    'lambda': [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01],                  # 1/s (decay constants)
    'beta_total': 0.006502,                                                  # total delayed fraction
}
PROMPT_NEUTRON_LIFETIME = 4.0e-4   # s (thermal spectrum MSR, longer than PWR)
NEUTRONS_PER_FISSION = 2.43        # nu for U-235 thermal
ENERGY_PER_FISSION = 200e6 * 1.602e-19  # J (200 MeV)

# 1-group cross-sections (spectrum-averaged for graphite-moderated FLiBe+UF4)
# These are approximate values for a thermal MSR spectrum.
# Source: Estimated from ORNL-4541 and ENDF/B-VIII data
SIGMA_FISSION_U235 = 430e-28       # m2 (430 barns, spectrum-averaged)
SIGMA_ABSORPTION_U235 = 520e-28    # m2 (includes fission + capture)
SIGMA_ABSORPTION_U238 = 8.0e-28    # m2
SIGMA_ABSORPTION_LI7 = 0.045e-28   # m2 (very low, enriched 7Li)
SIGMA_ABSORPTION_BE = 0.0076e-28   # m2
SIGMA_ABSORPTION_F = 0.0096e-28    # m2
SIGMA_SCATTER_GRAPHITE = 4.7e-28   # m2
SIGMA_ABSORPTION_GRAPHITE = 0.0035e-28  # m2
TRANSPORT_MEAN_FREE_PATH = 0.027   # m (in graphite-moderated MSR lattice)
DIFFUSION_COEFFICIENT = TRANSPORT_MEAN_FREE_PATH / 3  # m
MIGRATION_LENGTH_SQUARED = 0.032   # m2 (thermal MSR)
FERMI_AGE = 0.030                  # m2 (graphite)

# Resonance integral for U-238 (for resonance escape probability)
RESONANCE_INTEGRAL_U238 = 275e-28  # m2 (275 barns)


# =============================================================================
# TIER 5: SHIP MOTION PARAMETERS (DNV GL Rules)
# =============================================================================

# Design sea state parameters
ROLL_ANGLE_MAX = 30.0              # degrees
PITCH_ANGLE_MAX = 10.0             # degrees
HEAVE_ACCELERATION = 0.5           # g (fraction of gravitational acceleration)
ROLL_PERIOD = 15.0                 # seconds (typical for container ship)
PITCH_PERIOD = 8.0                 # seconds
SWAY_ACCELERATION = 0.3           # g (lateral)


# =============================================================================
# TIER 6: SAFETY LIMITS
# =============================================================================

# Regulatory dose limits
OCCUPATIONAL_DOSE_LIMIT = 20e-3    # Sv/yr (ICRP recommendation)
PUBLIC_DOSE_LIMIT = 1e-3           # Sv/yr
EMERGENCY_DOSE_LIMIT = 100e-3      # Sv (single event)

# Design temperature limits
MAX_FUEL_SALT_TEMP = 1400 + 273.15     # K (boiling point of FLiBe)
MAX_VESSEL_WALL_TEMP = 704 + 273.15    # K (Hastelloy-N service limit)
MAX_GRAPHITE_TEMP = 1000 + 273.15      # K (conservative)
MIN_SALT_TEMP = 459 + 273.15           # K (freezing point of FLiBe)

# Nuclear safety limits
DRAIN_TANK_KEFF_LIMIT = 0.95      # Subcriticality requirement for drain tank
SHUTDOWN_MARGIN = 0.01             # dk/k minimum shutdown margin


# =============================================================================
# SUMMARY OUTPUT
# =============================================================================

def print_summary(d=None):
    """Print a formatted summary of key design parameters.

    Args:
        d: DerivedParameters instance (computed if not provided)
    """
    if d is None:
        d = compute_derived()

    print("=" * 72)
    print("     40 MWth MARINE MOLTEN SALT REACTOR - DESIGN SUMMARY")
    print("=" * 72)

    print("\n--- Reactor Design Basis ---")
    print(f"  Thermal Power:          {THERMAL_POWER/1e6:10.1f} MW")
    print(f"  Electrical Power:       {ELECTRICAL_POWER/1e6:10.1f} MW")
    print(f"  Thermal Efficiency:     {THERMAL_EFFICIENCY*100:10.1f} %")
    print(f"  Salt Type:              {SALT_TYPE}")
    print(f"  U-235 Enrichment:       {U235_ENRICHMENT*100:10.1f} %")

    print("\n--- Core Geometry ---")
    print(f"  Core Diameter:          {d.core_diameter:10.3f} m")
    print(f"  Core Height:            {d.core_height:10.3f} m")
    print(f"  Core Volume:            {d.core_volume:10.3f} m3")
    print(f"  H/D Ratio:              {CORE_HEIGHT_TO_DIAMETER:10.2f}")
    print(f"  Fuel Channels:          {d.n_channels:10d}")
    print(f"  Core Power Density:     {d.core_power_density/1e6:10.2f} MW/m3")
    print(f"  Fuel Power Density:     {d.fuel_power_density/1e6:10.2f} MW/m3")

    print("\n--- Fuel Salt ---")
    print(f"  Density (at Tavg):      {d.fuel_salt_density:10.1f} kg/m3")
    print(f"  Viscosity (at Tavg):    {d.salt_viscosity*1e3:10.3f} mPa-s")
    print(f"  Specific Heat:          {d.salt_cp:10.1f} J/kg-K")
    print(f"  Thermal Cond.:          {d.salt_k:10.2f} W/m-K")
    print(f"  Volume (core):          {d.fuel_salt_volume_core:10.3f} m3")
    print(f"  Volume (total):         {d.fuel_salt_volume_total:10.3f} m3")
    print(f"  Mass (core):            {d.fuel_salt_mass_core:10.1f} kg")
    print(f"  Mass (total):           {d.fuel_salt_mass_total:10.1f} kg")

    print("\n--- Flow Parameters ---")
    print(f"  Mass Flow Rate:         {d.mass_flow_rate:10.2f} kg/s")
    print(f"  Volumetric Flow Rate:   {d.volumetric_flow_rate*1e3:10.3f} L/s")
    print(f"  Salt Velocity:          {d.salt_velocity:10.3f} m/s")
    print(f"  Residence Time (core):  {d.residence_time_core:10.2f} s")
    print(f"  Reynolds Number:        {d.reynolds_number:10.0f}")
    print(f"  Prandtl Number:         {d.prandtl_number:10.1f}")

    print("\n--- Pressure Drop & Pumping ---")
    print(f"  Friction Factor:        {d.friction_factor:10.4f}")
    print(f"  Core dP:                {d.pressure_drop_core/1e3:10.2f} kPa")
    print(f"  Pump Power (hydraulic): {d.pump_power_hydraulic/1e3:10.2f} kW")
    print(f"  Pump Power (shaft):     {d.pump_power_shaft/1e3:10.2f} kW")

    print("\n--- Graphite ---")
    print(f"  Volume:                 {d.graphite_volume:10.3f} m3")
    print(f"  Mass:                   {d.graphite_mass:10.1f} kg")

    print("\n--- Vessel ---")
    print(f"  Inner Radius:           {d.vessel_inner_radius:10.3f} m")
    print(f"  Wall Thickness:         {d.vessel_wall_thickness*1e3:10.1f} mm")
    print(f"  Outer Radius:           {d.vessel_outer_radius:10.3f} m")
    print(f"  Vessel Height:          {d.vessel_height:10.3f} m")

    print("\n--- Heat Exchanger ---")
    print(f"  LMTD:                   {d.intermediate_lmtd:10.2f} K")

    print("\n--- Uranium Loading ---")
    print(f"  Total Uranium:          {d.uranium_mass:10.1f} kg")
    print(f"  U-235 Mass:             {d.u235_mass:10.1f} kg")

    print("\n--- Operating Temperatures ---")
    print(f"  Core Inlet:             {CORE_INLET_TEMP - 273.15:10.1f} C  ({CORE_INLET_TEMP:.1f} K)")
    print(f"  Core Outlet:            {CORE_OUTLET_TEMP - 273.15:10.1f} C  ({CORE_OUTLET_TEMP:.1f} K)")
    print(f"  Core Average:           {CORE_AVG_TEMP - 273.15:10.1f} C  ({CORE_AVG_TEMP:.1f} K)")

    print("\n--- Safety Limits ---")
    print(f"  Max Salt Temp:          {MAX_FUEL_SALT_TEMP - 273.15:10.1f} C")
    print(f"  Min Salt Temp:          {MIN_SALT_TEMP - 273.15:10.1f} C")
    print(f"  Max Vessel Wall Temp:   {MAX_VESSEL_WALL_TEMP - 273.15:10.1f} C")
    print(f"  Shutdown Margin:        {SHUTDOWN_MARGIN*100:10.2f} %dk/k")

    print("\n" + "=" * 72)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    design = compute_derived()
    print_summary(design)
