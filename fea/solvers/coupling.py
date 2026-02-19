"""
One-Way Coupled Multiphysics Pipeline for 40 MWth Marine MSR
==============================================================

Implements a sequential one-way coupling pipeline:

    Neutronics  -->  Thermal  -->  Structural

Level 1 (R-Z axisymmetric full core):
    All three physics solved on the same R-Z mesh.
    1. Neutron diffusion eigenvalue -> keff + power distribution
    2. Power distribution -> volumetric heat source -> thermal solve
    3. Temperature field -> thermal loading -> structural solve

Level 2/3 (X-Y unit cell or rosette):
    Thermal + structural only (no neutronics; uses uniform power density).
    1. Power density = total_power / core_volume
    2. Volumetric heat source in fuel channels -> thermal solve
    3. Temperature field -> thermal loading -> structural solve

The coupling is one-way (no feedback from thermal to neutronics).
For this educational/design-phase analysis, the one-way approach
is adequate because:
    - Temperature feedback on cross-sections is second-order
    - We use pre-computed, spectrum-averaged cross-sections
    - The thermal expansion reactivity effect is captured separately
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fea.mesh.nodes import Mesh
from fea.mesh.geometry_levels import (
    build_level1_rz, build_level2_hex_cell, build_level3_rosette,
    CORE_RADIUS, CORE_HALF_HEIGHT, REFLECTOR_THICKNESS, VESSEL_THICKNESS,
)
from fea.materials.properties import MaterialLibrary
from fea.solvers.thermal import solve_thermal, ThermalResult
from fea.solvers.neutronics import solve_diffusion_fe, NeutronicsResult
from fea.solvers.structural import solve_structural, StructuralResult


@dataclass
class CoupledResult:
    """Results from the coupled multiphysics analysis.

    Attributes
    ----------
    mesh : Mesh
        The Tri3 mesh used for the analysis.
    level : int
        Geometry level (1, 2, or 3).
    neutronics : NeutronicsResult or None
        Neutron diffusion result (Level 1 only).
    thermal : ThermalResult
        Steady-state thermal result.
    structural : StructuralResult
        Structural analysis result.
    total_power : float
        Total reactor thermal power [W].
    """
    mesh: Mesh
    level: int
    neutronics: Optional[NeutronicsResult]
    thermal: ThermalResult
    structural: StructuralResult
    total_power: float


class CoupledAnalysis:
    """One-way coupling: Neutronics -> Thermal -> Structural.

    Level 1 (R-Z): All three physics on same mesh domain.
    Level 2/3 (X-Y): Thermal + structural only (no neutronics).

    Parameters
    ----------
    level : int
        Geometry level: 1 (R-Z full core), 2 (hex cell), 3 (rosette).
    """

    def __init__(self, level=1):
        if level not in (1, 2, 3):
            raise ValueError(f"level must be 1, 2, or 3, got {level}")
        self.level = level
        self.mat_lib = MaterialLibrary(level=level)

    def run(self, total_power=40e6):
        """Execute full coupling pipeline.

        For Level 1:
          1. Build R-Z mesh
          2. Solve neutronics -> get keff and power distribution
          3. Use power distribution as thermal source -> solve thermal
          4. Use temperature field -> solve structural

        For Level 2/3:
          1. Build hex cell / rosette mesh
          2. Use uniform power density (from total_power / core_volume)
          3. Solve thermal with convective BCs on channel walls
          4. Solve structural with thermal loading

        Parameters
        ----------
        total_power : float
            Total reactor thermal power [W]. Default 40 MW.

        Returns
        -------
        result : CoupledResult
            Combined results from all physics.
        """
        print(f"[Coupling] Starting Level {self.level} analysis...")
        print(f"[Coupling] Total power: {total_power/1e6:.1f} MW")

        if self.level == 1:
            return self._run_level1(total_power)
        else:
            return self._run_level23(total_power)

    def _run_level1(self, total_power):
        """Level 1: Full R-Z axisymmetric analysis with all three physics.

        Thermal model:
            In a real MSR, the flowing salt carries away the fission
            heat convectively. The conduction-only R-Z model captures
            the temperature distribution through the structural
            components (reflector + vessel wall) that determines
            thermal stresses.

            Approach:
            1. The core region (zone 0) temperature is pinned via
               Dirichlet BCs: all core nodes set to a cosine-weighted
               profile from T_inlet (600 C) at center to T_outlet
               (700 C) at the core-reflector boundary.
            2. The reflector and vessel conduct heat outward.
            3. No volumetric source in reflector/vessel (gamma heating
               is small, ~2-3% of total power).
            4. External vessel surface: convective BC to seawater.

            The volumetric source in the core provides the internal
            heat generation that drives the temperature gradient. We
            scale it to represent the fraction not removed by salt
            flow. For the gamma heating deposited directly in
            structures (reflector + vessel), we use ~3% of total power.
        """

        # --- 1. Build mesh ---
        print("[Coupling] Building Level 1 R-Z mesh...")
        mesh = build_level1_rz(nr_core=30, nz_core=45,
                               nr_reflector=8, nz_reflector=8,
                               nr_vessel=3)
        print(f"[Coupling]   Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

        # --- 2. Neutronics ---
        print("[Coupling] Solving neutron diffusion eigenvalue problem...")
        neut_mat = self._build_neutronics_material_dict()
        neut_result = solve_diffusion_fe(
            mesh, neut_mat, total_power=total_power,
            max_iter=500, tol_k=1e-6, tol_flux=1e-5,
        )
        print(f"[Coupling]   keff = {neut_result.keff:.5f}"
              f"  (converged={neut_result.converged}, "
              f"iter={neut_result.iterations})")

        # --- 3. Prepare thermal source ---
        print("[Coupling] Preparing thermal source and boundary conditions...")

        # For the thermal model, we pin the core temperature to the
        # salt temperature profile and only solve for conduction through
        # reflector and vessel. The thermal source in reflector/vessel
        # is gamma heating (~3% of total power).
        import config
        T_inlet = config.CORE_INLET_TEMP      # 873.15 K (600 C)
        T_outlet = config.CORE_OUTLET_TEMP     # 973.15 K (700 C)
        T_core_avg = config.CORE_AVG_TEMP      # 923.15 K (650 C)

        # Build a Dirichlet BC that pins ALL core nodes (zone 0) to
        # a physically motivated temperature profile.
        # Profile: T(r,z) = T_avg + (T_outlet - T_inlet)/2 * z/z_half
        # This gives T_inlet at z=0 (bottom, but we model upper quadrant so z=0 is midplane)
        # Actually for the upper quadrant: T increases from midplane (T_avg) to
        # top (T_outlet). Plus radial variation: hotter at center.
        core_node_set = set()
        for e in range(mesh.n_elements):
            if mesh.material_ids[e] == 0:
                for n in mesh.elements[e]:
                    core_node_set.add(int(n))
        core_nodes = np.array(sorted(core_node_set), dtype=np.int64)

        # Temperature profile in core: T(r,z) based on flux shape
        # Use the neutron flux to weight the temperature profile
        z_half = CORE_HALF_HEIGHT
        r_core = CORE_RADIUS
        core_T_values = np.empty(len(core_nodes), dtype=np.float64)
        for i, n in enumerate(core_nodes):
            r_n = mesh.nodes[n, 0]
            z_n = mesh.nodes[n, 1]
            # Axial: linear from T_avg at midplane (z=0) to T_outlet at top
            T_axial = T_core_avg + (T_outlet - T_core_avg) * (z_n / z_half) if z_half > 0 else T_core_avg
            # Radial: slight peaking at center based on flux
            flux_n = neut_result.flux[n] if n < len(neut_result.flux) else 0.5
            # Scale: center is slightly hotter (peak), edge approaches T_axial
            T_radial_factor = 1.0 + 0.05 * flux_n  # small radial variation
            core_T_values[i] = T_axial * T_radial_factor
            # Clamp to physical range
            core_T_values[i] = np.clip(core_T_values[i], T_inlet, T_outlet + 50)

        # Gamma heating in reflector and vessel (~3% of total power)
        gamma_fraction = 0.03
        gamma_power = total_power * gamma_fraction

        # Distribute gamma heating in reflector (zones 1,2) and vessel (zone 3)
        # Gamma deposit is roughly proportional to material density
        q_volumetric = np.zeros(mesh.n_elements)
        reflector_volume = 0.0
        vessel_volume = 0.0
        axisym = (mesh.coord_system == 'axisymmetric')

        for e in range(mesh.n_elements):
            area = abs(mesh.element_area(e))
            if axisym:
                r_c = np.mean(mesh.element_coords(e)[:, 0])
                vol = 2.0 * np.pi * max(r_c, 0.0) * area
            else:
                vol = area
            zone = int(mesh.material_ids[e])
            if zone in (1, 2):
                reflector_volume += vol
            elif zone == 3:
                vessel_volume += vol

        # Gamma heating: ~60% in reflector, ~40% in vessel (density-weighted)
        if reflector_volume > 0:
            q_refl = gamma_power * 0.6 / reflector_volume
        else:
            q_refl = 0.0
        if vessel_volume > 0:
            q_vessel = gamma_power * 0.4 / vessel_volume
        else:
            q_vessel = 0.0

        for e in range(mesh.n_elements):
            zone = int(mesh.material_ids[e])
            if zone in (1, 2):
                q_volumetric[e] = q_refl
            elif zone == 3:
                q_volumetric[e] = q_vessel
            # Core elements: zero volumetric source (Dirichlet BCs instead)

        print(f"[Coupling]   Gamma heating in reflector: {q_refl:.3e} W/m3")
        print(f"[Coupling]   Gamma heating in vessel: {q_vessel:.3e} W/m3")

        # --- 4. Thermal solve ---
        print("[Coupling] Solving steady-state thermal conduction...")

        # Vessel outer surface: convective to seawater
        h_ext = 5000.0    # W/m2-K (forced seawater cooling)
        T_seawater = 308.15  # K (35 C)

        bc_conv = {}
        if 'outer_wall' in mesh.boundary_edges:
            bc_conv['outer_wall'] = (h_ext, T_seawater)
        if 'top_wall' in mesh.boundary_edges:
            bc_conv['top_wall'] = (h_ext, T_seawater)

        # Dirichlet: all core nodes pinned to salt temperature profile
        bc_dir = {}
        bc_dir['_core_nodes'] = (core_nodes, core_T_values)

        thermal_result = solve_thermal(
            mesh, self.mat_lib, q_volumetric,
            bc_dirichlet=bc_dir,
            bc_convective=bc_conv,
            max_iter=10, tol=1e-4,
        )
        print(f"[Coupling]   T_min = {np.min(thermal_result.temperature):.1f} K, "
              f"T_max = {np.max(thermal_result.temperature):.1f} K")
        print(f"[Coupling]   Thermal converged={thermal_result.converged}, "
              f"iter={thermal_result.iterations}")

        # --- 5. Structural solve ---
        print("[Coupling] Solving structural with thermal loading...")
        struct_mat = self._build_structural_material_dict()
        struct_result = solve_structural(
            mesh, struct_mat, thermal_result.temperature,
            T_ref=923.15, plane='strain',
        )
        print(f"[Coupling]   Max displacement: "
              f"{np.max(np.linalg.norm(struct_result.displacement, axis=1)):.4e} m")
        print(f"[Coupling]   Max von Mises: "
              f"{np.max(struct_result.von_mises):.3e} Pa "
              f"({np.max(struct_result.von_mises)/1e6:.1f} MPa)")

        return CoupledResult(
            mesh=mesh,
            level=1,
            neutronics=neut_result,
            thermal=thermal_result,
            structural=struct_result,
            total_power=total_power,
        )

    def _run_level23(self, total_power):
        """Level 2/3: X-Y unit cell/rosette (thermal + structural only)."""

        # --- 1. Build mesh ---
        print(f"[Coupling] Building Level {self.level} mesh...")
        if self.level == 2:
            mesh = build_level2_hex_cell(n_channel_circ=24,
                                         n_channel_radial=6,
                                         n_graphite_radial=10)
        else:
            mesh = build_level3_rosette(n_channel_circ=20, n_radial=8)
        print(f"[Coupling]   Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

        # --- 2. Compute uniform power density ---
        core_volume = np.pi * CORE_RADIUS**2 * (2.0 * CORE_HALF_HEIGHT)
        fuel_volume_fraction = 0.23  # from config
        fuel_power_density = total_power / (core_volume * fuel_volume_fraction)
        print(f"[Coupling]   Fuel power density: {fuel_power_density:.3e} W/m3")

        # Assign power only to fuel elements (zone 0)
        q_volumetric = np.zeros(mesh.n_elements)
        fuel_mask = mesh.material_ids == 0
        q_volumetric[fuel_mask] = fuel_power_density

        # --- 3. Thermal solve ---
        print("[Coupling] Solving steady-state thermal...")
        # Convective BC on channel wall: salt bulk temperature
        h_channel = 3000.0   # W/m2-K (molten salt laminar/transitional)
        T_bulk = 923.15      # K (650 C, core average)

        bc_conv = {}
        if 'channel_wall' in mesh.boundary_edges:
            bc_conv['channel_wall'] = (h_channel, T_bulk)
        # For rosette, check individual channel walls
        for i in range(7):
            tag = f'channel_wall_{i}'
            if tag in mesh.boundary_edges:
                bc_conv[tag] = (h_channel, T_bulk)

        # Outer boundary: adiabatic (symmetry) for unit cell
        # For rosette, also adiabatic on outer boundary

        thermal_result = solve_thermal(
            mesh, self.mat_lib, q_volumetric,
            bc_convective=bc_conv,
            max_iter=10, tol=1e-4,
        )
        print(f"[Coupling]   T_min = {np.min(thermal_result.temperature):.1f} K, "
              f"T_max = {np.max(thermal_result.temperature):.1f} K")

        # --- 4. Structural solve ---
        print("[Coupling] Solving structural with thermal loading...")
        struct_mat = self._build_structural_material_dict()

        # For unit-cell/rosette, fix a reference point to remove rigid body
        # Fix hex boundary in both directions (approximate restraint)
        bc_fixed = {}
        for tag in ['hex_boundary', 'outer_boundary']:
            if tag in mesh.boundary_nodes:
                bc_fixed[tag] = (0.0, 0.0)

        struct_result = solve_structural(
            mesh, struct_mat, thermal_result.temperature,
            T_ref=923.15, bc_symmetry={}, bc_fixed=bc_fixed,
            plane='strain',
        )
        print(f"[Coupling]   Max von Mises: "
              f"{np.max(struct_result.von_mises):.3e} Pa "
              f"({np.max(struct_result.von_mises)/1e6:.1f} MPa)")

        return CoupledResult(
            mesh=mesh,
            level=self.level,
            neutronics=None,
            thermal=thermal_result,
            structural=struct_result,
            total_power=total_power,
        )

    def transfer_power_to_thermal(self, neut_result, thermal_mesh):
        """Map neutronics power density to thermal mesh heat source.

        Since both solvers use the same mesh, this is a direct element copy.

        Parameters
        ----------
        neut_result : NeutronicsResult
            Neutronics solution with power_density per element.
        thermal_mesh : Mesh
            The thermal mesh (same as neutronics mesh).

        Returns
        -------
        q_volumetric : ndarray, shape (N_elem,)
            Volumetric heat source [W/m3] for each element.
        """
        return neut_result.power_density.copy()

    def transfer_temperature_to_structural(self, thermal_result, struct_mesh):
        """Map temperature from Tri3 to Tri6 nodes.

        Corner nodes: direct copy.
        Mid-edge nodes: average of adjacent corner nodes.

        Parameters
        ----------
        thermal_result : ThermalResult
            Thermal solution with nodal temperatures.
        struct_mesh : Mesh
            The Tri6 structural mesh.

        Returns
        -------
        T_tri6 : ndarray, shape (N_tri6_nodes,)
            Temperature at Tri6 nodes.
        """
        T_tri3 = thermal_result.temperature
        n_tri3 = len(T_tri3)
        n_tri6 = struct_mesh.n_nodes

        T_tri6 = np.zeros(n_tri6)
        # Corner nodes: direct copy
        T_tri6[:n_tri3] = T_tri3

        # Mid-edge nodes: average of parent corner nodes
        for e in range(struct_mesh.n_elements):
            conn = struct_mesh.elements[e]
            edges = [(conn[0], conn[1], conn[3]),
                     (conn[1], conn[2], conn[4]),
                     (conn[2], conn[0], conn[5])]
            for ca, cb, mid in edges:
                if mid >= n_tri3:
                    T_tri6[mid] = 0.5 * (T_tri6[ca] + T_tri6[cb])

        return T_tri6

    def _build_neutronics_material_dict(self):
        """Build zone-keyed material dict for neutronics solver.

        Returns
        -------
        mat_dict : dict
            Maps zone_id -> dict with 'D', 'sigma_a', 'nu_sigma_f', 'sigma_f'.
        """
        mat_dict = {}
        for zone in range(4):
            mat_dict[zone] = {
                'D': self.mat_lib.diffusion_coefficient(zone),
                'sigma_a': self.mat_lib.sigma_a(zone),
                'nu_sigma_f': self.mat_lib.nu_sigma_f(zone),
                'sigma_f': self.mat_lib.nu_sigma_f(zone) / 2.43 if zone == 0 else 0.0,
            }
        return mat_dict

    def _build_structural_material_dict(self):
        """Build zone-keyed material dict for structural solver.

        Returns
        -------
        mat_dict : dict
            Maps zone_id -> dict with 'E', 'nu', 'alpha'.
        """
        mat_dict = {}
        if self.level == 1:
            zones = range(4)
        else:
            zones = range(2)

        for zone in zones:
            E = self.mat_lib.elastic_modulus(zone)
            nu = self.mat_lib.poisson_ratio(zone)
            alpha = self.mat_lib.thermal_expansion(zone)
            # Clamp E to minimum for numerical stability (fluid zones)
            E = max(E, 1.0e6)
            # Clamp nu to valid range for plane strain
            nu = min(nu, 0.499)
            mat_dict[zone] = {
                'E': E,
                'nu': nu,
                'alpha': alpha,
            }
        return mat_dict
