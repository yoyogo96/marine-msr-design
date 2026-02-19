"""
Molten Salt Thermophysical Properties
=====================================
Temperature-dependent correlations for FLiBe (2LiF-BeF2) and FLiNaK (LiF-NaF-KF).

Sources:
  - INL-EXT-10-18297: "Engineering Database of Liquid Salt Thermophysical and
    Thermochemical Properties" (Allen, 2010)
  - Romatoski & Forsberg (2017): "Fluoride salt coolant properties for nuclear
    reactor applications: A review", Annals of Nuclear Energy 109, 635-647
  - ORNL-4541: "Conceptual Design Study of a Single-Fluid MSBR" (Robertson, 1971)

All correlations use SI units. Temperature T in Kelvin unless noted otherwise.
"""

import numpy as np


# ============================================================
# FLiBe (2LiF-BeF2) - Primary Fuel Salt Carrier
# Composition: 67 mol% LiF / 33 mol% BeF2 (standard 2:1 molar ratio)
# ============================================================

def flibe_density(T, uf4_mol_fraction=0.0):
    """FLiBe density as a function of temperature.

    For pure FLiBe the linear fit from INL-EXT-10-18297 is used.
    When UF4 is dissolved, a linear mixing correction is applied:
    UF4 has a much higher molecular mass (~314 g/mol vs ~33 g/mol
    average for FLiBe) so small mole fractions noticeably increase
    bulk density.

    Args:
        T: Temperature in K
        uf4_mol_fraction: UF4 mole fraction in the mixture (0 to ~0.05)

    Returns:
        float: Density in kg/m³

    Source: INL-EXT-10-18297, Eq. 2.1
    Validity: 732 K (459°C) to 1573 K (1300°C)
    Uncertainty: ±2% (pure FLiBe), ±5% (with UF4)
    """
    # Pure FLiBe: rho = 2413 - 0.488 * T_C  [kg/m³], T_C in °C
    # Source: INL-EXT-10-18297 correlation is defined in Celsius.
    rho_flibe = 2413.0 - 0.488 * (T - 273.15)

    # UF4 addition increases density; linear approximation adequate for
    # uf4_mol_fraction < 5 mol%.  The ~1200 kg/m³ factor is an empirical
    # coefficient derived from molecular-weight-weighted volume mixing.
    if uf4_mol_fraction > 0:
        rho_flibe += uf4_mol_fraction * 1200.0

    return rho_flibe


def flibe_viscosity(T):
    """FLiBe dynamic viscosity as a function of temperature.

    Arrhenius form: mu = A * exp(B / T)
    where A = 1.16e-4 Pa·s, B = 3755 K.

    Args:
        T: Temperature in K

    Returns:
        float: Dynamic viscosity in Pa·s

    Source: Romatoski & Forsberg (2017), Table 3
    Validity: 773 K to 1073 K
    Uncertainty: ±20%
    """
    return 1.16e-4 * np.exp(3755.0 / T)


def flibe_thermal_conductivity(T):
    """FLiBe thermal conductivity.

    Thermal conductivity of FLiBe shows weak temperature dependence
    over the operating range; the INL recommended constant value is
    used here.  Measurement scatter is large so a ±15-20% uncertainty
    band should be carried in sensitivity studies.

    Args:
        T: Temperature in K (accepted but value is currently constant)

    Returns:
        float: Thermal conductivity in W/(m·K)

    Source: INL-EXT-10-18297
    Validity: 773-1073 K
    Uncertainty: ±15-20%
    """
    # Suppress "unused argument" tools warning while keeping API consistent
    _ = T
    return 1.1  # W/(m·K) - INL recommended value


def flibe_specific_heat(T):
    """FLiBe isobaric specific heat capacity.

    Specific heat is nearly constant over the MSR operating temperature
    range; the Romatoski & Forsberg recommended value is used.

    Args:
        T: Temperature in K (accepted for API consistency)

    Returns:
        float: Specific heat in J/(kg·K)

    Source: Romatoski & Forsberg (2017)
    Validity: 732-1200 K
    Uncertainty: ±3%
    """
    _ = T
    return 2386.0  # J/(kg·K)


def flibe_prandtl(T):
    """FLiBe Prandtl number derived from constituent property correlations.

    Pr = mu * cp / k

    Args:
        T: Temperature in K

    Returns:
        float: Dimensionless Prandtl number
    """
    mu = flibe_viscosity(T)
    cp = flibe_specific_heat(T)
    k = flibe_thermal_conductivity(T)
    return mu * cp / k


def flibe_melting_point():
    """FLiBe equilibrium melting point.

    Returns:
        float: Melting point in K (459°C = 732.15 K)

    Source: ORNL-4541
    """
    return 459.0 + 273.15  # 732.15 K


def flibe_boiling_point():
    """FLiBe atmospheric boiling point.

    Returns:
        float: Boiling point in K (1430°C = 1703.15 K)

    Source: INL-EXT-10-18297
    """
    return 1430.0 + 273.15  # 1703.15 K


# ============================================================
# FLiNaK (LiF-NaF-KF) - Intermediate Loop Salt
# Composition: 46.5-11.5-42.0 mol% (eutectic)
# ============================================================

def flinak_density(T):
    """FLiNaK density as a function of temperature.

    Args:
        T: Temperature in K

    Returns:
        float: Density in kg/m³

    Source: INL-EXT-10-18297, Table 3.1
    Validity: 773-1173 K
    Uncertainty: ±2%
    """
    # INL-EXT-10-18297 correlation is defined in Celsius.
    return 2729.0 - 0.73 * (T - 273.15)


def flinak_viscosity(T):
    """FLiNaK dynamic viscosity.

    Arrhenius form: mu = A * exp(B / T)
    where A = 4.0e-5 Pa·s, B = 4170 K.

    Args:
        T: Temperature in K

    Returns:
        float: Dynamic viscosity in Pa·s

    Source: INL-EXT-10-18297
    Validity: 773-1073 K
    Uncertainty: ±15%
    """
    return 4.0e-5 * np.exp(4170.0 / T)


def flinak_thermal_conductivity(T):
    """FLiNaK thermal conductivity.

    Linear in temperature: k = 0.36 + 5.6e-4 * T  [W/(m·K)]

    Args:
        T: Temperature in K

    Returns:
        float: Thermal conductivity in W/(m·K)

    Source: INL-EXT-10-18297
    Validity: 790-1080 K
    Uncertainty: ±20%
    """
    return 0.36 + 5.6e-4 * T


def flinak_specific_heat(T):
    """FLiNaK isobaric specific heat capacity.

    Args:
        T: Temperature in K (accepted for API consistency)

    Returns:
        float: Specific heat in J/(kg·K)

    Source: INL-EXT-10-18297
    Uncertainty: ±10%
    """
    _ = T
    return 1880.0  # J/(kg·K), nearly constant over operating range


def flinak_prandtl(T):
    """FLiNaK Prandtl number derived from constituent property correlations.

    Pr = mu * cp / k

    Args:
        T: Temperature in K

    Returns:
        float: Dimensionless Prandtl number
    """
    mu = flinak_viscosity(T)
    cp = flinak_specific_heat(T)
    k = flinak_thermal_conductivity(T)
    return mu * cp / k


def flinak_melting_point():
    """FLiNaK eutectic melting point.

    Returns:
        float: Melting point in K (454°C = 727.15 K)

    Source: INL-EXT-10-18297
    """
    return 454.0 + 273.15  # 727.15 K


# ============================================================
# Heat Transfer Correlations for Molten Salts
# ============================================================

def dittus_boelter_nu(Re, Pr, heating=True):
    """Dittus-Boelter correlation for turbulent forced convection in a pipe.

    Nu = 0.023 * Re^0.8 * Pr^n
    where n = 0.4 for heating, n = 0.3 for cooling.

    Note: For molten salts (Pr ~ 10-15) this correlation can overestimate
    the Nusselt number by 10-20%.  The Gnielinski correlation is preferred
    for higher accuracy.

    Args:
        Re: Reynolds number (should be > 10000)
        Pr: Prandtl number (valid for 0.6 < Pr < 160)
        heating: True if fluid is being heated, False if cooled

    Returns:
        float: Nusselt number

    Validity: Re > 10000, 0.6 < Pr < 160, L/D > 10
    """
    n = 0.4 if heating else 0.3
    return 0.023 * Re**0.8 * Pr**n


def gnielinski_nu(Re, Pr, f=None):
    """Gnielinski correlation for turbulent pipe convection.

    Provides improved accuracy over Dittus-Boelter, especially in the
    transition region and for fluids with moderate-to-high Prandtl numbers.

    Nu = (f/8)(Re - 1000)Pr / [1 + 12.7 * sqrt(f/8) * (Pr^(2/3) - 1)]

    When friction factor f is not supplied the smooth-pipe Petukhov
    approximation is used: f = (0.790 * ln(Re) - 1.64)^(-2).

    Args:
        Re: Reynolds number (valid for 2300 < Re < 5e6)
        Pr: Prandtl number (valid for 0.5 < Pr < 2000)
        f: Darcy friction factor; computed internally if None

    Returns:
        float: Nusselt number
    """
    if f is None:
        # Petukhov smooth-pipe friction factor
        f = (0.790 * np.log(Re) - 1.64)**(-2)

    numerator = (f / 8.0) * (Re - 1000.0) * Pr
    denominator = 1.0 + 12.7 * np.sqrt(f / 8.0) * (Pr**(2.0 / 3.0) - 1.0)
    return numerator / denominator


def heat_transfer_coefficient(Re, Pr, D, k_fluid):
    """Calculate the convective heat transfer coefficient h = Nu * k / D.

    Selects the appropriate Nusselt correlation based on flow regime:
      - Turbulent (Re > 10000): Gnielinski
      - Transition (2300 < Re <= 10000): linear interpolation between
        fully-developed laminar (Nu = 3.66) and Gnielinski at Re = 10000
      - Laminar (Re <= 2300): constant wall-temperature value Nu = 3.66

    Args:
        Re: Reynolds number
        Pr: Prandtl number
        D: Hydraulic diameter in m
        k_fluid: Fluid thermal conductivity in W/(m·K)

    Returns:
        float: Convective heat transfer coefficient in W/(m²·K)
    """
    if Re > 10000:
        Nu = gnielinski_nu(Re, Pr)
    elif Re > 2300:
        # Transition region: linear blend
        Nu_lam = 3.66
        Nu_turb = gnielinski_nu(10000, Pr)
        frac = (Re - 2300.0) / (10000.0 - 2300.0)
        Nu = Nu_lam * (1.0 - frac) + Nu_turb * frac
    else:
        Nu = 3.66  # Laminar, constant wall temperature

    return Nu * k_fluid / D


def friction_factor(Re, roughness=0.0, D=1.0):
    """Darcy-Weisbach friction factor for internal pipe flow.

    Selects correlation based on flow regime:
      - Laminar (Re < 2300): f = 64 / Re
      - Turbulent, smooth (roughness ~ 0): Petukhov approximation
      - Turbulent, rough: Swamee-Jain explicit approximation to
        the Colebrook-White equation

    Args:
        Re: Reynolds number
        roughness: Absolute surface roughness in m (default 0 = hydraulically smooth)
        D: Pipe inner diameter in m (used to compute relative roughness e/D)

    Returns:
        float: Darcy friction factor f (dimensionless)
    """
    if Re < 2300:
        return 64.0 / Re

    e_D = roughness / D if D > 0 else 0.0

    if e_D < 1.0e-10:
        # Smooth pipe: Petukhov correlation
        return (0.790 * np.log(Re) - 1.64)**(-2)
    else:
        # Swamee-Jain explicit approximation to Colebrook-White
        term1 = e_D / 3.7
        term2 = 5.74 / Re**0.9
        return 0.25 / (np.log10(term1 + term2))**2


# ============================================================
# Summary / Diagnostic Function
# ============================================================

def print_salt_properties(T_celsius=650):
    """Print all salt properties at a given temperature to stdout.

    Useful for quick sanity checks and design point verification.

    Args:
        T_celsius: Temperature in °C (default 650°C = typical MSR mid-core)
    """
    T = T_celsius + 273.15

    print(f"\n{'=' * 60}")
    print(f"  Molten Salt Properties at {T_celsius}°C ({T:.1f} K)")
    print(f"{'=' * 60}")

    print(f"\n  FLiBe (2LiF-BeF2):")
    print(f"    Density:              {flibe_density(T):.1f} kg/m³")
    print(f"    Density (5% UF4):     {flibe_density(T, 0.05):.1f} kg/m³")
    print(f"    Viscosity:            {flibe_viscosity(T) * 1000:.2f} mPa·s")
    print(f"    Thermal conductivity: {flibe_thermal_conductivity(T):.2f} W/(m·K)")
    print(f"    Specific heat:        {flibe_specific_heat(T):.0f} J/(kg·K)")
    print(f"    Prandtl number:       {flibe_prandtl(T):.1f}")
    print(f"    Melting point:        {flibe_melting_point() - 273.15:.1f}°C")
    print(f"    Boiling point:        {flibe_boiling_point() - 273.15:.1f}°C")

    print(f"\n  FLiNaK (LiF-NaF-KF):")
    print(f"    Density:              {flinak_density(T):.1f} kg/m³")
    print(f"    Viscosity:            {flinak_viscosity(T) * 1000:.2f} mPa·s")
    print(f"    Thermal conductivity: {flinak_thermal_conductivity(T):.2f} W/(m·K)")
    print(f"    Specific heat:        {flinak_specific_heat(T):.0f} J/(kg·K)")
    print(f"    Prandtl number:       {flinak_prandtl(T):.1f}")
    print(f"    Melting point:        {flinak_melting_point() - 273.15:.1f}°C")


if __name__ == '__main__':
    for T_C in [600, 650, 700]:
        print_salt_properties(T_C)
