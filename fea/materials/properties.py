"""
Material property manager for the 40 MWth Marine MSR FEA analysis.

Provides temperature-dependent material properties keyed by material
zone ID, wrapping the correlations and constants defined in config.py.

Two property regimes based on geometric level:

    Level 1 (R-Z axisymmetric full core):
        Zone 0 = Homogenized core (23% fuel salt + 77% graphite)
        Zone 1 = Radial reflector (graphite)
        Zone 2 = Axial reflector (graphite)
        Zone 3 = Hastelloy-N vessel wall

    Level 2/3 (X-Y unit cell or rosette):
        Zone 0 = Fuel salt (FLiBe + 5 mol% UF4)
        Zone 1 = Graphite moderator (IG-110)

Temperature conventions: All temperatures in Kelvin. Config.py
correlations (salt_density, salt_thermal_conductivity, etc.) accept
and return SI units with temperature in Kelvin.

Usage:
    from fea.materials.properties import MaterialLibrary
    mat = MaterialLibrary(level=1)
    k = mat.thermal_conductivity(zone=0, T=923.15)
"""

import sys
import os
import numpy as np

# Ensure project root is on the path so config can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config


class MaterialLibrary:
    """Central material property manager for FEA.

    Wraps config.py functions and constants to provide temperature-dependent
    properties keyed by material zone ID.

    Level 1 zones (R-Z axisymmetric):
        0 = homogenized core (23% fuel salt + 77% graphite)
        1 = graphite reflector (radial)
        2 = graphite reflector (axial)
        3 = Hastelloy-N vessel wall

    Level 2/3 zones (X-Y unit cell):
        0 = fuel salt (FLiBe + 5 mol% UF4)
        1 = graphite moderator (IG-110)

    Parameters
    ----------
    level : int
        Geometric level (1, 2, or 3). Determines zone-to-material mapping.
    """

    # Volume fractions for homogenized core (Level 1, zone 0)
    FUEL_FRACTION = config.FUEL_SALT_FRACTION        # 0.23
    GRAPHITE_FRACTION = config.GRAPHITE_VOLUME_FRACTION  # 0.77

    def __init__(self, level=1):
        if level not in (1, 2, 3):
            raise ValueError(f"level must be 1, 2, or 3, got {level}")
        self.level = level

        # Pre-compute derived quantities from config
        self._derived = config.compute_derived()

        # Nuclear data from config (one-group, spectrum-averaged)
        self._sigma_a_fuel = self._compute_fuel_sigma_a()
        self._nu_sigma_f_fuel = self._compute_fuel_nu_sigma_f()
        self._sigma_a_graphite = (config.SIGMA_ABSORPTION_GRAPHITE
                                  * config.GRAPHITE['density']
                                  / (12.011e-3) * 6.022e23)
        self._D_core = config.DIFFUSION_COEFFICIENT  # m (transport MFP / 3)

    # -----------------------------------------------------------------
    #  Thermal conductivity  [W/(m*K)]
    # -----------------------------------------------------------------

    def thermal_conductivity(self, zone, T=923.15):
        """Return thermal conductivity k [W/(m*K)] for zone at temperature T.

        Salt:        ~1.1 W/m-K (nearly constant, from config correlation)
        Graphite:    120 W/m-K at BOL, ~80 W/m-K at 650 C after irradiation
                     Simple model: k_g(T) = 120 * (923.15 / T)^0.5
        Hastelloy-N: Quadratic fit from ORNL data (config function)
        Homogenized: Volume-weighted harmonic mean

        Parameters
        ----------
        zone : int
            Material zone ID.
        T : float
            Temperature [K]. Default 923.15 (650 C).

        Returns
        -------
        k : float
            Thermal conductivity [W/(m*K)].
        """
        if self.level == 1:
            return self._k_level1(zone, T)
        else:
            return self._k_level23(zone, T)

    def _k_salt(self, T):
        """Fuel salt thermal conductivity from config."""
        return config.salt_thermal_conductivity(T)

    def _k_graphite(self, T):
        """Nuclear graphite thermal conductivity with temperature dependence.

        BOL value from config (120 W/m-K at ~20 C). Decreases with
        temperature approximately as T^(-0.5) for nuclear graphite.

        Reference: ORNL/TM-2005/519, IG-110 data.
        """
        k_ref = config.GRAPHITE['thermal_conductivity']  # 120 W/m-K (BOL, ~293 K)
        T_ref = 293.15  # K (reference temperature for BOL value)
        # Temperature dependence: k ~ 1/T for phonon-dominated conduction
        # Use k(T) = k_ref * (T_ref / T)^0.5 to be less aggressive
        # At 650 C (923 K): 120 * (293/923)^0.5 = 120 * 0.563 = 67.6 W/m-K
        # This is reasonable for irradiated IG-110 at high temperature.
        return k_ref * (T_ref / T) ** 0.5

    def _k_hastelloy(self, T):
        """Hastelloy-N thermal conductivity from config quadratic fit."""
        return config.hastelloy_thermal_conductivity(T)

    def _k_level1(self, zone, T):
        """Level 1 thermal conductivity by zone."""
        if zone == 0:
            # Homogenized core: volume-weighted arithmetic mean
            # (appropriate when phases are in parallel slabs along heat flow)
            k_s = self._k_salt(T)
            k_g = self._k_graphite(T)
            return self.FUEL_FRACTION * k_s + self.GRAPHITE_FRACTION * k_g
        elif zone in (1, 2):
            return self._k_graphite(T)
        elif zone == 3:
            return self._k_hastelloy(T)
        else:
            raise ValueError(f"Level 1 zone must be 0-3, got {zone}")

    def _k_level23(self, zone, T):
        """Level 2/3 thermal conductivity by zone."""
        if zone == 0:
            return self._k_salt(T)
        elif zone == 1:
            return self._k_graphite(T)
        else:
            raise ValueError(f"Level 2/3 zone must be 0 or 1, got {zone}")

    # -----------------------------------------------------------------
    #  Density  [kg/m3]
    # -----------------------------------------------------------------

    def density(self, zone, T=923.15):
        """Return density rho [kg/m3] for zone at temperature T.

        Salt: temperature-dependent from config correlation.
        Graphite: 1780 kg/m3 (constant, from config).
        Hastelloy-N: 8860 kg/m3 (constant, from config).
        Homogenized core: volume-weighted average.

        Parameters
        ----------
        zone : int
            Material zone ID.
        T : float
            Temperature [K]. Default 923.15 (650 C).

        Returns
        -------
        rho : float
            Density [kg/m3].
        """
        if self.level == 1:
            return self._rho_level1(zone, T)
        else:
            return self._rho_level23(zone, T)

    def _rho_salt(self, T):
        """Fuel salt density from config."""
        return config.salt_density(T)

    def _rho_level1(self, zone, T):
        """Level 1 density by zone."""
        if zone == 0:
            rho_s = self._rho_salt(T)
            rho_g = config.GRAPHITE['density']
            return self.FUEL_FRACTION * rho_s + self.GRAPHITE_FRACTION * rho_g
        elif zone in (1, 2):
            return config.GRAPHITE['density']
        elif zone == 3:
            return config.HASTELLOY_N['density']
        else:
            raise ValueError(f"Level 1 zone must be 0-3, got {zone}")

    def _rho_level23(self, zone, T):
        """Level 2/3 density by zone."""
        if zone == 0:
            return self._rho_salt(T)
        elif zone == 1:
            return config.GRAPHITE['density']
        else:
            raise ValueError(f"Level 2/3 zone must be 0 or 1, got {zone}")

    # -----------------------------------------------------------------
    #  Specific heat  [J/(kg*K)]
    # -----------------------------------------------------------------

    def specific_heat(self, zone, T=923.15):
        """Return specific heat cp [J/(kg*K)] for zone at temperature T.

        Salt: ~2386 J/kg-K (constant, from config).
        Graphite: ~1700 J/kg-K at 650 C (from config).
        Hastelloy-N: ~440 J/kg-K (typical Ni-alloy).

        Parameters
        ----------
        zone : int
            Material zone ID.
        T : float
            Temperature [K].

        Returns
        -------
        cp : float
            Specific heat [J/(kg*K)].
        """
        if self.level == 1:
            return self._cp_level1(zone, T)
        else:
            return self._cp_level23(zone, T)

    def _cp_level1(self, zone, T):
        if zone == 0:
            cp_s = config.salt_specific_heat(T)
            cp_g = config.GRAPHITE['specific_heat']
            rho_s = self._rho_salt(T)
            rho_g = config.GRAPHITE['density']
            # Mass-weighted specific heat for homogenized core
            rho_mix = self.FUEL_FRACTION * rho_s + self.GRAPHITE_FRACTION * rho_g
            return (self.FUEL_FRACTION * rho_s * cp_s
                    + self.GRAPHITE_FRACTION * rho_g * cp_g) / rho_mix
        elif zone in (1, 2):
            return config.GRAPHITE['specific_heat']
        elif zone == 3:
            return 440.0  # J/kg-K, typical for Hastelloy-N at 650 C
        else:
            raise ValueError(f"Level 1 zone must be 0-3, got {zone}")

    def _cp_level23(self, zone, T):
        if zone == 0:
            return config.salt_specific_heat(T)
        elif zone == 1:
            return config.GRAPHITE['specific_heat']
        else:
            raise ValueError(f"Level 2/3 zone must be 0 or 1, got {zone}")

    # -----------------------------------------------------------------
    #  Elastic modulus  [Pa]
    # -----------------------------------------------------------------

    def elastic_modulus(self, zone, T=923.15):
        """Return Young's modulus E [Pa] for zone at temperature T.

        Hastelloy-N: 219 GPa at RT, decreases ~15% at 650 C.
            E(T) = 219e9 * (1 - 2.0e-4 * (T - 293.15))
        Graphite (IG-110): ~10 GPa (weak T dependence).
        Salt: effectively zero (fluid).

        Parameters
        ----------
        zone : int
            Material zone ID.
        T : float
            Temperature [K].

        Returns
        -------
        E : float
            Young's modulus [Pa].
        """
        if self.level == 1:
            return self._E_level1(zone, T)
        else:
            return self._E_level23(zone, T)

    def _E_level1(self, zone, T):
        if zone == 0:
            # Homogenized core: dominated by graphite (salt is fluid, E~0)
            return self._E_graphite(T) * self.GRAPHITE_FRACTION
        elif zone in (1, 2):
            return self._E_graphite(T)
        elif zone == 3:
            return self._E_hastelloy(T)
        else:
            raise ValueError(f"Level 1 zone must be 0-3, got {zone}")

    def _E_level23(self, zone, T):
        if zone == 0:
            return 0.0  # Fluid
        elif zone == 1:
            return self._E_graphite(T)
        else:
            raise ValueError(f"Level 2/3 zone must be 0 or 1, got {zone}")

    def _E_hastelloy(self, T):
        """Hastelloy-N elastic modulus with temperature dependence."""
        E_rt = config.HASTELLOY_N['elastic_modulus']  # 219e9 Pa at RT
        # Linear decrease: ~15% reduction from RT to 650 C
        return E_rt * max(0.5, 1.0 - 2.0e-4 * (T - 293.15))

    def _E_graphite(self, T):
        """Nuclear graphite elastic modulus."""
        # IG-110: ~10 GPa, weak temperature dependence
        return 10.0e9

    # -----------------------------------------------------------------
    #  Poisson's ratio  [-]
    # -----------------------------------------------------------------

    def poisson_ratio(self, zone):
        """Return Poisson's ratio nu [-] for zone.

        Hastelloy-N: 0.32 (from config).
        Graphite: 0.20 (typical for nuclear graphite).
        Salt: 0.5 (incompressible fluid, but not used in structural).

        Parameters
        ----------
        zone : int
            Material zone ID.

        Returns
        -------
        nu : float
            Poisson's ratio [-].
        """
        if self.level == 1:
            if zone == 0:
                return 0.20  # Dominated by graphite
            elif zone in (1, 2):
                return 0.20
            elif zone == 3:
                return config.HASTELLOY_N['poisson_ratio']  # 0.32
            else:
                raise ValueError(f"Level 1 zone must be 0-3, got {zone}")
        else:
            if zone == 0:
                return 0.50  # Fluid (incompressible limit)
            elif zone == 1:
                return 0.20
            else:
                raise ValueError(f"Level 2/3 zone must be 0 or 1, got {zone}")

    # -----------------------------------------------------------------
    #  Thermal expansion coefficient  [1/K]
    # -----------------------------------------------------------------

    def thermal_expansion(self, zone, T=923.15):
        """Return coefficient of thermal expansion alpha [1/K] for zone.

        Hastelloy-N: 12.3e-6 1/K (from config).
        Graphite: 4.5e-6 1/K (from config).
        Salt: volumetric expansion from density correlation.
        Homogenized core: volume-weighted average.

        Parameters
        ----------
        zone : int
            Material zone ID.
        T : float
            Temperature [K].

        Returns
        -------
        alpha : float
            Coefficient of thermal expansion [1/K].
        """
        if self.level == 1:
            if zone == 0:
                # Volume-weighted average
                alpha_s = self._alpha_salt()
                alpha_g = config.GRAPHITE['thermal_expansion']
                return (self.FUEL_FRACTION * alpha_s
                        + self.GRAPHITE_FRACTION * alpha_g)
            elif zone in (1, 2):
                return config.GRAPHITE['thermal_expansion']  # 4.5e-6
            elif zone == 3:
                return config.HASTELLOY_N['thermal_expansion']  # 12.3e-6
            else:
                raise ValueError(f"Level 1 zone must be 0-3, got {zone}")
        else:
            if zone == 0:
                return self._alpha_salt()
            elif zone == 1:
                return config.GRAPHITE['thermal_expansion']
            else:
                raise ValueError(f"Level 2/3 zone must be 0 or 1, got {zone}")

    def _alpha_salt(self):
        """Volumetric thermal expansion for salt, derived from density correlation.

        rho(T) = 2413.0 - 0.488 * (T - 273.15)
        alpha_vol = -(1/rho) * drho/dT = 0.488 / rho
        alpha_linear = alpha_vol / 3
        """
        rho_avg = config.salt_density(config.CORE_AVG_TEMP)
        alpha_vol = 0.488 / rho_avg
        return alpha_vol / 3.0

    # -----------------------------------------------------------------
    #  Neutron diffusion properties
    # -----------------------------------------------------------------

    def diffusion_coefficient(self, zone):
        """Return neutron diffusion coefficient D [m] for zone.

        D = transport_mean_free_path / 3. Varies by material.

        Homogenized core: config value (spectrum-averaged for lattice).
        Graphite reflector: D_graphite = 1 / (3 * Sigma_tr_graphite)
                           ~ 0.84 cm for pure graphite.
        Hastelloy-N: not meaningful (absorber), set to small value.

        Parameters
        ----------
        zone : int
            Material zone ID (Level 1 only).

        Returns
        -------
        D : float
            Diffusion coefficient [m].
        """
        if self.level != 1:
            raise ValueError("Diffusion properties only defined for Level 1")

        if zone == 0:
            return config.DIFFUSION_COEFFICIENT  # ~0.009 m
        elif zone in (1, 2):
            # Pure graphite: D = 1/(3*N*sigma_tr) ~ 0.84 cm
            # sigma_tr ~ sigma_scatter (graphite is mostly scattering)
            N_graphite = config.GRAPHITE['density'] / (12.011e-3) * 6.022e23
            sigma_tr = config.SIGMA_SCATTER_GRAPHITE  # ~4.7 barn
            Sigma_tr = N_graphite * sigma_tr
            return 1.0 / (3.0 * Sigma_tr)
        elif zone == 3:
            # Vessel: strong absorber, D is small
            return 0.001  # 1 mm (placeholder, neutrons absorbed quickly)
        else:
            raise ValueError(f"Level 1 zone must be 0-3, got {zone}")

    def sigma_a(self, zone):
        """Return macroscopic absorption cross-section Sigma_a [1/m] for zone.

        Parameters
        ----------
        zone : int
            Material zone ID (Level 1 only).

        Returns
        -------
        Sigma_a : float
            Macroscopic absorption cross-section [1/m].
        """
        if self.level != 1:
            raise ValueError("Nuclear properties only defined for Level 1")

        if zone == 0:
            return self._sigma_a_fuel
        elif zone in (1, 2):
            return self._sigma_a_graphite
        elif zone == 3:
            # Hastelloy-N (Ni-Mo-Cr): strong absorber
            # Approximate: Sigma_a ~ 10 /m (high due to Ni, Mo)
            return 10.0
        else:
            raise ValueError(f"Level 1 zone must be 0-3, got {zone}")

    def nu_sigma_f(self, zone):
        """Return nu*Sigma_f [1/m] for zone. Zero outside core.

        Parameters
        ----------
        zone : int
            Material zone ID (Level 1 only).

        Returns
        -------
        nu_Sigma_f : float
            Production cross-section [1/m].
        """
        if self.level != 1:
            raise ValueError("Nuclear properties only defined for Level 1")

        if zone == 0:
            return self._nu_sigma_f_fuel
        else:
            return 0.0

    def _compute_fuel_sigma_a(self):
        """Compute macroscopic absorption cross-section for homogenized core.

        Contributions from fuel salt (U-235, U-238, Li, Be, F) and graphite,
        volume-weighted.
        """
        derived = self._derived

        # Salt component number densities
        rho_salt = derived.fuel_salt_density  # kg/m3 at Tavg
        MW_avg = (config.LIF_MOLE_FRACTION * 25.94
                  + config.BEF2_MOLE_FRACTION * 47.01
                  + config.UF4_MOLE_FRACTION * 314.02) * 1e-3  # kg/mol

        N_A = 6.022e23
        N_mol = rho_salt / MW_avg * N_A  # molecules/m3

        # Number densities of each species
        N_U = N_mol * config.UF4_MOLE_FRACTION  # U atoms / m3
        N_U235 = N_U * config.U235_ENRICHMENT
        N_U238 = N_U * (1.0 - config.U235_ENRICHMENT)
        N_Li = N_mol * config.LIF_MOLE_FRACTION
        N_Be = N_mol * config.BEF2_MOLE_FRACTION
        N_F = N_mol * (config.LIF_MOLE_FRACTION
                       + 2.0 * config.BEF2_MOLE_FRACTION
                       + 4.0 * config.UF4_MOLE_FRACTION)

        # Salt macroscopic absorption
        Sigma_a_salt = (N_U235 * config.SIGMA_ABSORPTION_U235
                        + N_U238 * config.SIGMA_ABSORPTION_U238
                        + N_Li * config.SIGMA_ABSORPTION_LI7
                        + N_Be * config.SIGMA_ABSORPTION_BE
                        + N_F * config.SIGMA_ABSORPTION_F)

        # Graphite macroscopic absorption
        N_C = config.GRAPHITE['density'] / (12.011e-3) * N_A
        Sigma_a_graphite = N_C * config.SIGMA_ABSORPTION_GRAPHITE

        # Volume-weighted homogenization
        return (self.FUEL_FRACTION * Sigma_a_salt
                + self.GRAPHITE_FRACTION * Sigma_a_graphite)

    def _compute_fuel_nu_sigma_f(self):
        """Compute nu*Sigma_f for homogenized core.

        Only U-235 contributes to fission (U-238 fast fission negligible
        in thermal spectrum).
        """
        derived = self._derived
        rho_salt = derived.fuel_salt_density
        MW_avg = (config.LIF_MOLE_FRACTION * 25.94
                  + config.BEF2_MOLE_FRACTION * 47.01
                  + config.UF4_MOLE_FRACTION * 314.02) * 1e-3

        N_A = 6.022e23
        N_mol = rho_salt / MW_avg * N_A
        N_U235 = N_mol * config.UF4_MOLE_FRACTION * config.U235_ENRICHMENT

        Sigma_f = N_U235 * config.SIGMA_FISSION_U235
        nu = config.NEUTRONS_PER_FISSION

        # Volume-weighted (only salt contributes)
        return self.FUEL_FRACTION * nu * Sigma_f

    # -----------------------------------------------------------------
    #  Mesh-level property access
    # -----------------------------------------------------------------

    def get_element_properties(self, mesh, property_name, T_nodal=None):
        """Get property array for all elements in a mesh.

        Evaluates the named property for each element based on its
        material zone ID. If T_nodal is provided, the element centroid
        temperature is used for temperature-dependent properties.

        Parameters
        ----------
        mesh : Mesh
            FEA mesh with material_ids.
        property_name : str
            Property name: 'k', 'rho', 'cp', 'E', 'nu', 'alpha', 'D',
            'sigma_a', 'nu_sigma_f'.
        T_nodal : ndarray or None, optional
            Nodal temperature field [K]. Shape (N_nodes,).
            If None, uses default temperature (923.15 K).

        Returns
        -------
        prop : ndarray, shape (N_elem,)
            Property values for each element.

        Raises
        ------
        ValueError
            If property_name is unrecognized.
        """
        n_elem = mesh.n_elements
        prop = np.empty(n_elem, dtype=np.float64)

        # Property dispatch table
        prop_func_map = {
            'k': self.thermal_conductivity,
            'rho': self.density,
            'cp': self.specific_heat,
            'E': self.elastic_modulus,
            'alpha': self.thermal_expansion,
        }
        prop_func_no_T = {
            'nu': self.poisson_ratio,
            'D': self.diffusion_coefficient,
            'sigma_a': self.sigma_a,
            'nu_sigma_f': self.nu_sigma_f,
        }

        if property_name in prop_func_no_T:
            func = prop_func_no_T[property_name]
            for e in range(n_elem):
                zone = int(mesh.material_ids[e])
                prop[e] = func(zone)
            return prop

        if property_name not in prop_func_map:
            raise ValueError(
                f"Unknown property '{property_name}'. "
                f"Valid: {sorted(list(prop_func_map.keys()) + list(prop_func_no_T.keys()))}"
            )

        func = prop_func_map[property_name]

        if T_nodal is not None:
            T_nodal = np.asarray(T_nodal, dtype=np.float64)
            for e in range(n_elem):
                zone = int(mesh.material_ids[e])
                # Centroid temperature: average of element node temperatures
                conn = mesh.elements[e]
                T_centroid = np.mean(T_nodal[conn])
                prop[e] = func(zone, T_centroid)
        else:
            for e in range(n_elem):
                zone = int(mesh.material_ids[e])
                prop[e] = func(zone)

        return prop
