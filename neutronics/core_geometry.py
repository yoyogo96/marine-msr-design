"""
Core Geometry Design for Graphite-Moderated Marine MSR
======================================================

Defines the core geometry with a hexagonal channel lattice embedded in a
graphite moderator matrix. Fuel salt flows upward through cylindrical
channels drilled in hexagonal graphite blocks.

Geometry Design Algorithm:
  1. Core volume from thermal power / volumetric power density target
  2. Core diameter and height from H/D ratio constraint
  3. Hexagonal lattice: channel pitch and diameter set the graphite/salt
     volume fractions in each unit cell
  4. Number of channels from core cross-section / unit cell area
  5. Actual volume fractions recalculated from discrete channel count
  6. External salt inventory estimated as ~2x core salt volume
     (plena + downcomer + piping + heat exchanger)

The design balances neutron moderation (graphite fraction) against heat
removal and salt inventory. Typical graphite fraction is 0.70-0.80 for
well-thermalized MSR spectra.

Sources:
  - ORNL-4541: MSBR Conceptual Design Study
  - ORNL-TM-0728: MSRE Design and Operations Report
  - Robertson (1971): MSR lattice design principles
"""

import math
from dataclasses import dataclass

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    THERMAL_POWER, CORE_HEIGHT_TO_DIAMETER,
    GRAPHITE_VOLUME_FRACTION, FUEL_SALT_FRACTION,
    CHANNEL_PITCH, CHANNEL_DIAMETER,
    GRAPHITE, CORE_AVG_TEMP,
)
from thermal_hydraulics.salt_properties import flibe_density


@dataclass
class CoreGeometry:
    """Complete core geometry specification.

    All dimensions in SI units (meters, m^2, m^3).
    """
    # --- Overall core dimensions ---
    core_diameter: float          # m
    core_height: float            # m
    core_volume: float            # m^3
    core_radius: float            # m

    # --- Channel lattice ---
    n_channels: int               # number of fuel salt channels
    channel_pitch: float          # m (hexagonal pitch)
    channel_diameter: float       # m (salt channel diameter)

    # --- Volume fractions (actual, from discrete channel count) ---
    fuel_salt_fraction_actual: float    # dimensionless
    graphite_fraction_actual: float     # dimensionless

    # --- Volumes and inventory ---
    fuel_salt_volume_core: float        # m^3 (salt in core channels)
    fuel_salt_volume_total: float       # m^3 (core + external loop)
    graphite_volume: float              # m^3
    reflector_thickness: float          # m (graphite reflector)

    # --- Derived quantities ---
    core_cross_section: float           # m^2 (pi/4 * D^2)
    channel_flow_area: float            # m^2 (single channel)
    total_flow_area: float              # m^2 (all channels)
    unit_cell_area: float               # m^2 (hex cell)
    power_density: float                # W/m^3 (whole core)
    fuel_power_density: float           # W/m^3 (fuel salt only)
    graphite_mass: float                # kg
    fuel_salt_mass_core: float          # kg
    fuel_salt_mass_total: float         # kg


def _hexagonal_cell_area(pitch):
    """Area of a regular hexagonal unit cell with given flat-to-flat pitch.

    For a hexagonal lattice, the area associated with each channel is:
        A_cell = (sqrt(3) / 2) * pitch^2

    This is the area of a regular hexagon with the given pitch as the
    distance between parallel faces (flat-to-flat distance).

    Args:
        pitch: Hexagonal pitch (flat-to-flat distance) in m

    Returns:
        float: Cell area in m^2
    """
    return (math.sqrt(3) / 2.0) * pitch**2


def _channel_area(diameter):
    """Cross-sectional area of a cylindrical fuel channel.

    Args:
        diameter: Channel diameter in m

    Returns:
        float: Channel cross-sectional area in m^2
    """
    return math.pi / 4.0 * diameter**2


def _compute_lattice_volume_fractions(pitch, channel_dia):
    """Compute fuel salt and graphite volume fractions from lattice dimensions.

    In the hexagonal unit cell, the fuel fraction is:
        f_fuel = pi * d^2 / (2 * sqrt(3) * p^2)

    where d is the channel diameter and p is the pitch.

    Args:
        pitch: Hexagonal lattice pitch in m
        channel_dia: Fuel channel diameter in m

    Returns:
        tuple: (fuel_salt_fraction, graphite_fraction)
    """
    cell_area = _hexagonal_cell_area(pitch)
    chan_area = _channel_area(channel_dia)
    f_fuel = chan_area / cell_area
    f_graphite = 1.0 - f_fuel
    return f_fuel, f_graphite


def design_core(thermal_power=None, power_density_target=22.0e6,
                h_d_ratio=None, graphite_fraction=None,
                channel_pitch=None, channel_diameter=None):
    """Design the MSR core geometry from top-level requirements.

    Algorithm:
      1. V_core = P_thermal / q'''_target
      2. D_core = (4 V / (pi * H/D))^(1/3)
      3. H_core = (H/D) * D_core
      4. Lay out hexagonal channels: N = A_core / A_cell
      5. Recalculate actual volume fractions from discrete N

    Args:
        thermal_power: Thermal power in W (default from config)
        power_density_target: Volumetric power density target in W/m^3
            (default 22 MW/m^3, consistent with MSBR-class)
        h_d_ratio: Core height-to-diameter ratio (default from config)
        graphite_fraction: Target graphite volume fraction (default from config)
        channel_pitch: Hex lattice pitch in m (default from config)
        channel_diameter: Salt channel diameter in m (default from config)

    Returns:
        CoreGeometry: Fully specified core geometry
    """
    # Apply defaults from config
    if thermal_power is None:
        thermal_power = THERMAL_POWER
    if h_d_ratio is None:
        h_d_ratio = CORE_HEIGHT_TO_DIAMETER
    if graphite_fraction is None:
        graphite_fraction = GRAPHITE_VOLUME_FRACTION
    if channel_pitch is None:
        channel_pitch = CHANNEL_PITCH
    if channel_diameter is None:
        channel_diameter = CHANNEL_DIAMETER

    # --- Step 1: Core volume from power and power density ---
    core_volume = thermal_power / power_density_target  # m^3

    # --- Step 2: Core diameter from volume and H/D ratio ---
    # V = pi/4 * D^2 * H = pi/4 * D^2 * (H/D * D) = pi/4 * H/D * D^3
    core_diameter = (4.0 * core_volume / (math.pi * h_d_ratio))**(1.0 / 3.0)
    core_radius = core_diameter / 2.0

    # --- Step 3: Core height ---
    core_height = h_d_ratio * core_diameter

    # Recalculate exact volume
    core_volume = math.pi / 4.0 * core_diameter**2 * core_height

    # --- Step 4: Channel lattice ---
    core_cross_section = math.pi / 4.0 * core_diameter**2
    cell_area = _hexagonal_cell_area(channel_pitch)
    chan_area = _channel_area(channel_diameter)

    # Number of channels that fit in the core cross-section
    n_channels = int(core_cross_section / cell_area)

    # --- Step 5: Actual volume fractions from discrete channel count ---
    total_channel_area = n_channels * chan_area
    total_cell_area = n_channels * cell_area

    # The channels occupy this fraction of the core cross-section
    fuel_salt_fraction_actual = total_channel_area / core_cross_section
    graphite_fraction_actual = 1.0 - fuel_salt_fraction_actual

    # --- Volumes ---
    fuel_salt_volume_core = core_volume * fuel_salt_fraction_actual
    graphite_volume = core_volume * graphite_fraction_actual

    # Reflector: graphite annulus around the core
    reflector_thickness = 0.15  # m

    # External loop salt volume: plena (top + bottom) + downcomer + piping + HX
    # Typically ~1.0x core salt volume for external inventory
    fuel_salt_volume_external = fuel_salt_volume_core * 1.0
    fuel_salt_volume_total = fuel_salt_volume_core + fuel_salt_volume_external

    # --- Masses ---
    rho_salt = flibe_density(CORE_AVG_TEMP, uf4_mol_fraction=0.05)
    rho_graphite = GRAPHITE['density']

    graphite_mass = graphite_volume * rho_graphite
    fuel_salt_mass_core = fuel_salt_volume_core * rho_salt
    fuel_salt_mass_total = fuel_salt_volume_total * rho_salt

    # --- Power densities ---
    power_density = thermal_power / core_volume
    fuel_power_density = thermal_power / fuel_salt_volume_core

    total_flow_area = n_channels * chan_area

    return CoreGeometry(
        core_diameter=core_diameter,
        core_height=core_height,
        core_volume=core_volume,
        core_radius=core_radius,
        n_channels=n_channels,
        channel_pitch=channel_pitch,
        channel_diameter=channel_diameter,
        fuel_salt_fraction_actual=fuel_salt_fraction_actual,
        graphite_fraction_actual=graphite_fraction_actual,
        fuel_salt_volume_core=fuel_salt_volume_core,
        fuel_salt_volume_total=fuel_salt_volume_total,
        graphite_volume=graphite_volume,
        reflector_thickness=reflector_thickness,
        core_cross_section=core_cross_section,
        channel_flow_area=chan_area,
        total_flow_area=total_flow_area,
        unit_cell_area=cell_area,
        power_density=power_density,
        fuel_power_density=fuel_power_density,
        graphite_mass=graphite_mass,
        fuel_salt_mass_core=fuel_salt_mass_core,
        fuel_salt_mass_total=fuel_salt_mass_total,
    )


def print_core_geometry(geom=None):
    """Print formatted core geometry summary.

    Args:
        geom: CoreGeometry instance (computed with defaults if None)
    """
    if geom is None:
        geom = design_core()

    print("=" * 72)
    print("  CORE GEOMETRY DESIGN")
    print("=" * 72)

    print(f"\n  --- Overall Dimensions ---")
    print(f"    Core Diameter:        {geom.core_diameter:10.4f} m")
    print(f"    Core Radius:          {geom.core_radius:10.4f} m")
    print(f"    Core Height:          {geom.core_height:10.4f} m")
    print(f"    Core Volume:          {geom.core_volume:10.4f} m^3")
    print(f"    H/D Ratio:            {geom.core_height/geom.core_diameter:10.3f}")

    print(f"\n  --- Hexagonal Channel Lattice ---")
    print(f"    Channel Pitch:        {geom.channel_pitch*100:10.2f} cm")
    print(f"    Channel Diameter:     {geom.channel_diameter*100:10.2f} cm")
    print(f"    Number of Channels:   {geom.n_channels:10d}")
    print(f"    Unit Cell Area:       {geom.unit_cell_area*1e4:10.4f} cm^2")
    print(f"    Channel Area:         {geom.channel_flow_area*1e4:10.4f} cm^2")
    print(f"    Total Flow Area:      {geom.total_flow_area*1e4:10.2f} cm^2")

    print(f"\n  --- Volume Fractions (actual from discrete channels) ---")
    print(f"    Fuel Salt Fraction:   {geom.fuel_salt_fraction_actual:10.4f}")
    print(f"    Graphite Fraction:    {geom.graphite_fraction_actual:10.4f}")

    print(f"\n  --- Volumes ---")
    print(f"    Fuel Salt (core):     {geom.fuel_salt_volume_core:10.4f} m^3")
    print(f"    Fuel Salt (total):    {geom.fuel_salt_volume_total:10.4f} m^3")
    print(f"    Graphite:             {geom.graphite_volume:10.4f} m^3")
    print(f"    Reflector Thickness:  {geom.reflector_thickness*100:10.1f} cm")

    print(f"\n  --- Masses ---")
    print(f"    Graphite:             {geom.graphite_mass:10.1f} kg")
    print(f"    Fuel Salt (core):     {geom.fuel_salt_mass_core:10.1f} kg")
    print(f"    Fuel Salt (total):    {geom.fuel_salt_mass_total:10.1f} kg")

    print(f"\n  --- Power Density ---")
    print(f"    Core Average:         {geom.power_density/1e6:10.2f} MW/m^3")
    print(f"    Fuel Salt:            {geom.fuel_power_density/1e6:10.2f} MW/m^3")

    print()


if __name__ == '__main__':
    print("Designing core with default config parameters...\n")
    geom = design_core()
    print_core_geometry(geom)

    # Sensitivity to graphite fraction
    print("\n  --- Graphite Fraction Sensitivity ---")
    print(f"  {'f_graphite':>12s}  {'f_fuel':>10s}  {'N_chan':>8s}  {'V_core':>10s}  {'q_fuel':>12s}")
    print(f"  {'':>12s}  {'':>10s}  {'':>8s}  {'(m^3)':>10s}  {'(MW/m^3)':>12s}")
    for fg in [0.65, 0.70, 0.75, 0.77, 0.80, 0.85]:
        g = design_core(graphite_fraction=fg)
        print(f"  {fg:10.2f}    {g.fuel_salt_fraction_actual:10.4f}  "
              f"{g.n_channels:8d}  {g.core_volume:10.4f}  "
              f"{g.fuel_power_density/1e6:12.2f}")
