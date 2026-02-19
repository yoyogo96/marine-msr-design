"""
2D Plane-Strain Linear Elastic Solver with Thermal Loading
============================================================

Solves the linear elasticity BVP for the 40 MWth Marine MSR vessel
and internals under thermal loading conditions.

The structural solver operates on Tri6 (quadratic triangle) meshes
to achieve accurate stress recovery. If the input mesh is Tri3, it
must be converted via tri3_to_tri6() before calling solve_structural().

Governing equation:
    div(sigma) = 0      (static equilibrium, no body forces)

Constitutive law (plane strain):
    sigma = D * (epsilon - epsilon_th)

where:
    epsilon_th = alpha * dT * {1, 1, 0}^T
    D = plane strain constitutive matrix

Finite element formulation:
    K * u = f_th

where:
    K[I,J]  = integral B_I^T * D * B_J dA       (elastic stiffness)
    f_th[I] = integral B_I^T * D * eps_th dA     (thermal load)

Post-processing:
    epsilon = B * u_e                  (strain at element centroid)
    epsilon_mech = epsilon - epsilon_th
    sigma = D * epsilon_mech
    sigma_vm = sqrt(s11^2 + s22^2 - s11*s22 + 3*s12^2)

For ASME stress intensity (plane strain):
    sigma_zz = nu*(sigma_xx + sigma_yy) - E*alpha*dT
    Principal stresses from 2D Mohr's circle + sigma_zz
    SI = max(|s1-s2|, |s2-s3|, |s1-s3|)

Boundary conditions:
    Symmetry at r=0: u_r = 0 (radial displacement fixed)
    Symmetry at z=0: u_z = 0 (axial displacement fixed)
    Free surfaces: natural BC (no constraint)

DOF ordering: [u0_r, u0_z, u1_r, u1_z, ...] (interleaved)
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fea.mesh.nodes import Mesh, tri3_to_tri6
from fea.elements.tri6 import (
    B_matrix_tri6,
    D_matrix_plane_strain,
    D_matrix_plane_stress,
    shape_functions_tri6,
)
from fea.assembly.sparse_assembler import (
    assemble_elastic_stiffness,
    assemble_thermal_load,
)
from fea.assembly.boundary_conditions import (
    apply_dirichlet_elimination,
    get_boundary_dofs,
)


@dataclass
class StructuralResult:
    """Results from the structural analysis solve.

    Attributes
    ----------
    displacement : ndarray, shape (N_nodes, 2)
        Nodal displacements [m]. Column 0 = u_x (or u_r), Column 1 = u_y (or u_z).
    stress : ndarray, shape (N_elem, 3)
        Element centroid stress [Pa]: [sigma_xx, sigma_yy, tau_xy].
    strain : ndarray, shape (N_elem, 3)
        Element centroid total strain: [eps_xx, eps_yy, gamma_xy].
    von_mises : ndarray, shape (N_elem,)
        Von Mises equivalent stress [Pa].
    stress_intensity : ndarray, shape (N_elem,)
        ASME Tresca stress intensity [Pa].
    mesh : Mesh
        The Tri6 mesh used for the solve.
    """
    displacement: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    von_mises: np.ndarray
    stress_intensity: np.ndarray
    mesh: 'Mesh'


def _map_temperature_tri3_to_tri6(mesh_tri6, T_tri3, n_tri3_nodes):
    """Map temperature field from Tri3 nodes to Tri6 nodes.

    Corner nodes of Tri6 share the same indices as the original Tri3
    nodes, so temperatures are copied directly. Mid-edge nodes get
    the average of their two parent corner nodes.

    Parameters
    ----------
    mesh_tri6 : Mesh
        The Tri6 mesh (produced by tri3_to_tri6).
    T_tri3 : ndarray, shape (N_tri3_nodes,)
        Temperature at each Tri3 node.
    n_tri3_nodes : int
        Number of nodes in the original Tri3 mesh.

    Returns
    -------
    T_tri6 : ndarray, shape (N_tri6_nodes,)
        Temperature at each Tri6 node.
    """
    n_tri6_nodes = mesh_tri6.n_nodes
    T_tri6 = np.zeros(n_tri6_nodes)

    # Copy corner node temperatures directly (first n_tri3_nodes nodes)
    T_tri6[:n_tri3_nodes] = T_tri3

    # For mid-edge nodes, compute from element connectivity
    # Build a mapping: mid_node -> (corner_a, corner_b)
    mid_node_parents = {}
    for e in range(mesh_tri6.n_elements):
        conn = mesh_tri6.elements[e]
        # corners: conn[0], conn[1], conn[2]
        # mid-edge: conn[3] (between 0-1), conn[4] (between 1-2), conn[5] (between 2-0)
        edges = [(conn[0], conn[1], conn[3]),
                 (conn[1], conn[2], conn[4]),
                 (conn[2], conn[0], conn[5])]
        for c_a, c_b, mid in edges:
            if mid >= n_tri3_nodes and mid not in mid_node_parents:
                mid_node_parents[mid] = (c_a, c_b)

    for mid, (c_a, c_b) in mid_node_parents.items():
        T_tri6[mid] = 0.5 * (T_tri6[c_a] + T_tri6[c_b])

    return T_tri6


def _build_material_arrays_structural(mesh, material_lib):
    """Extract per-element structural material properties.

    The material_lib maps zone_id -> dict with keys:
        'E'     : Young's modulus [Pa]
        'nu'    : Poisson's ratio [-]
        'alpha' : Coefficient of thermal expansion [1/K]

    Parameters
    ----------
    mesh : Mesh
    material_lib : dict

    Returns
    -------
    E_arr : ndarray, shape (N_elem,)
    nu_arr : ndarray, shape (N_elem,)
    alpha_arr : ndarray, shape (N_elem,)
    """
    n_elem = mesh.n_elements
    E_arr = np.empty(n_elem)
    nu_arr = np.empty(n_elem)
    alpha_arr = np.empty(n_elem)

    for e in range(n_elem):
        zone = int(mesh.material_ids[e])
        mat = material_lib[zone]
        E_arr[e] = mat['E']
        nu_arr[e] = mat['nu']
        alpha_arr[e] = mat['alpha']

    return E_arr, nu_arr, alpha_arr


def _compute_stresses(mesh, u_global, E_arr, nu_arr, alpha_arr,
                      dT_nodal, plane='strain'):
    """Post-process element centroid stresses and strains.

    Evaluates strain and stress at each element centroid (xi=1/3, eta=1/3).

    Parameters
    ----------
    mesh : Mesh (Tri6)
    u_global : ndarray, shape (2*N_nodes,)
        Global displacement vector.
    E_arr : ndarray, shape (N_elem,)
    nu_arr : ndarray, shape (N_elem,)
    alpha_arr : ndarray, shape (N_elem,)
    dT_nodal : ndarray, shape (N_nodes,)
        Temperature change at each Tri6 node.
    plane : str
        'strain' or 'stress'.

    Returns
    -------
    stress : ndarray, shape (N_elem, 3)
        [sigma_xx, sigma_yy, tau_xy] at centroid.
    strain : ndarray, shape (N_elem, 3)
        [eps_xx, eps_yy, gamma_xy] at centroid (total, including thermal).
    von_mises : ndarray, shape (N_elem,)
    stress_intensity : ndarray, shape (N_elem,)
    """
    n_elem = mesh.n_elements
    stress = np.zeros((n_elem, 3))
    strain = np.zeros((n_elem, 3))
    von_mises = np.zeros(n_elem)
    stress_intensity = np.zeros(n_elem)

    # Centroid in parametric coords
    xi_c, eta_c = 1.0 / 3.0, 1.0 / 3.0

    for e in range(n_elem):
        coords = mesh.element_coords(e)
        conn = mesh.elements[e]

        # Extract element displacement vector (12,)
        u_e = np.empty(12)
        for i in range(6):
            u_e[2 * i] = u_global[2 * conn[i]]
            u_e[2 * i + 1] = u_global[2 * conn[i] + 1]

        # B matrix and strain at centroid
        B, detJ = B_matrix_tri6(xi_c, eta_c, coords)
        eps_total = B @ u_e  # (3,) total strain

        # Thermal strain at centroid
        N = shape_functions_tri6(xi_c, eta_c)
        dT_centroid = N @ dT_nodal[conn]
        eps_th = alpha_arr[e] * dT_centroid * np.array([1.0, 1.0, 0.0])

        # Mechanical strain
        eps_mech = eps_total - eps_th

        # Constitutive matrix
        E_e = E_arr[e]
        nu_e = nu_arr[e]
        if plane == 'strain':
            D = D_matrix_plane_strain(E_e, nu_e)
        else:
            D = D_matrix_plane_stress(E_e, nu_e)

        # Stress
        sig = D @ eps_mech  # [sigma_xx, sigma_yy, tau_xy]

        strain[e] = eps_total
        stress[e] = sig

        s11 = sig[0]
        s22 = sig[1]
        s12 = sig[2]

        # Von Mises (2D, generalized for plane strain/stress)
        von_mises[e] = np.sqrt(s11**2 + s22**2 - s11 * s22 + 3.0 * s12**2)

        # ASME Stress Intensity (Tresca-based)
        # For plane strain, the out-of-plane stress is:
        #   sigma_zz = nu*(sigma_xx + sigma_yy) - E*alpha*dT
        if plane == 'strain':
            s33 = nu_e * (s11 + s22) - E_e * alpha_arr[e] * dT_centroid
        else:
            s33 = 0.0

        # 2D principal stresses from Mohr's circle
        s_avg = 0.5 * (s11 + s22)
        R = np.sqrt(((s11 - s22) / 2.0)**2 + s12**2)
        sp1 = s_avg + R
        sp2 = s_avg - R

        # Three principal stresses
        principals = np.array([sp1, sp2, s33])

        # Stress intensity = max principal stress difference
        stress_intensity[e] = max(
            abs(principals[0] - principals[1]),
            abs(principals[1] - principals[2]),
            abs(principals[0] - principals[2]),
        )

    return stress, strain, von_mises, stress_intensity


def solve_structural(mesh, material_lib, temperature_field,
                     T_ref=923.15, bc_symmetry=None, bc_fixed=None,
                     plane='strain'):
    """Solve 2D linear elasticity with thermal loading.

    Parameters
    ----------
    mesh : Mesh
        Tri6 mesh (if Tri3, automatically converted via tri3_to_tri6).
    material_lib : dict
        Maps zone_id -> dict with keys:
            'E'     : Young's modulus [Pa]
            'nu'    : Poisson's ratio [-]
            'alpha' : Coefficient of thermal expansion [1/K]
    temperature_field : ndarray, shape (N_nodes_input,)
        Nodal temperatures [K]. If mesh is Tri3, these are Tri3 node
        temperatures and will be mapped to Tri6.
    T_ref : float
        Stress-free reference temperature [K]. Default 923.15 K (650 C).
    bc_symmetry : dict or None
        Symmetry boundary conditions. Keys are boundary tag names,
        values are 'r' (fix radial/x component) or 'z' (fix axial/y component).
        Example: {'symmetry_r0': 'r', 'symmetry_z0': 'z'}
        If None, automatically detected from mesh boundary tags.
    bc_fixed : dict or None
        Fixed displacement BCs. Keys are boundary tags, values are
        (u_x, u_y) tuples. E.g., {'fixed_base': (0.0, 0.0)}.
    plane : str
        'strain' or 'stress'. Default 'strain'.

    Returns
    -------
    result : StructuralResult
        Solution dataclass with displacements, stresses, von Mises, etc.

    Raises
    ------
    ValueError
        If material_lib is missing required keys.
    """
    # --- Convert Tri3 to Tri6 if needed ---
    n_input_nodes = mesh.n_nodes
    if mesh.element_type == 'tri3':
        mesh_tri6 = tri3_to_tri6(mesh)
        T_tri6 = _map_temperature_tri3_to_tri6(
            mesh_tri6, temperature_field, n_input_nodes
        )
    elif mesh.element_type == 'tri6':
        mesh_tri6 = mesh
        T_tri6 = temperature_field.copy()
    else:
        raise ValueError(
            f"solve_structural requires 'tri3' or 'tri6' mesh, "
            f"got '{mesh.element_type}'"
        )

    n_nodes = mesh_tri6.n_nodes
    n_dof = 2 * n_nodes

    # --- Temperature change from reference ---
    dT = T_tri6 - T_ref

    # --- Material arrays ---
    E_arr, nu_arr, alpha_arr = _build_material_arrays_structural(
        mesh_tri6, material_lib
    )

    # --- Assemble stiffness matrix ---
    K = assemble_elastic_stiffness(mesh_tri6, E_arr, nu_arr, plane=plane)

    # --- Assemble thermal load vector ---
    f_th = assemble_thermal_load(
        mesh_tri6, E_arr, nu_arr, alpha_arr, dT, plane=plane
    )

    # --- Collect boundary conditions ---
    bc_dofs = []
    bc_vals = []

    # Auto-detect symmetry BCs if not provided
    if bc_symmetry is None:
        bc_symmetry = {}
        if 'symmetry_r0' in mesh_tri6.boundary_nodes:
            bc_symmetry['symmetry_r0'] = 'r'
        if 'symmetry_z0' in mesh_tri6.boundary_nodes:
            bc_symmetry['symmetry_z0'] = 'z'

    # Apply symmetry BCs
    for tag, direction in bc_symmetry.items():
        if tag not in mesh_tri6.boundary_nodes:
            continue
        nodes = mesh_tri6.boundary_nodes[tag]
        if direction in ('r', 'x'):
            # Fix radial (x) component: DOF = 2*node + 0
            component = 0
        elif direction in ('z', 'y'):
            # Fix axial (y) component: DOF = 2*node + 1
            component = 1
        else:
            raise ValueError(
                f"Unknown symmetry direction '{direction}' for tag '{tag}'. "
                f"Use 'r'/'x' or 'z'/'y'."
            )
        dofs = get_boundary_dofs(mesh_tri6, tag, dofs_per_node=2,
                                 component=component)
        bc_dofs.extend(dofs.tolist())
        bc_vals.extend([0.0] * len(dofs))

    # Apply fixed displacement BCs
    if bc_fixed is not None:
        for tag, (ux, uy) in bc_fixed.items():
            if tag not in mesh_tri6.boundary_nodes:
                continue
            nodes = mesh_tri6.boundary_nodes[tag]
            for node in nodes:
                bc_dofs.append(2 * node)
                bc_vals.append(ux)
                bc_dofs.append(2 * node + 1)
                bc_vals.append(uy)

    # Check that we have some BCs (otherwise system is singular)
    if len(bc_dofs) == 0:
        raise ValueError(
            "No boundary conditions applied. The system is singular. "
            "Provide bc_symmetry or bc_fixed, or ensure mesh has "
            "'symmetry_r0'/'symmetry_z0' boundary tags."
        )

    bc_dofs = np.array(bc_dofs, dtype=np.int64)
    bc_vals = np.array(bc_vals, dtype=np.float64)

    # Remove duplicates (keep last value for each DOF)
    unique_dofs, unique_idx = np.unique(bc_dofs, return_index=True)
    bc_dofs = unique_dofs
    bc_vals = bc_vals[unique_idx]

    # --- Apply BCs via elimination ---
    K_mod, f_mod = apply_dirichlet_elimination(K, f_th, bc_dofs, bc_vals)

    # --- Solve ---
    u = spsolve(K_mod, f_mod)

    # --- Reshape displacement ---
    displacement = u.reshape(n_nodes, 2)

    # --- Post-process stresses ---
    stress, strain_arr, von_mises, stress_intensity = _compute_stresses(
        mesh_tri6, u, E_arr, nu_arr, alpha_arr, dT, plane=plane
    )

    return StructuralResult(
        displacement=displacement,
        stress=stress,
        strain=strain_arr,
        von_mises=von_mises,
        stress_intensity=stress_intensity,
        mesh=mesh_tri6,
    )
