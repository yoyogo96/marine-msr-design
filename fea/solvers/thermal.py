"""
Steady-state heat conduction solver for the 40 MWth Marine MSR.

Solves the steady-state heat equation:

    -div(k(T) * grad(T)) = q'''

using Picard iteration for nonlinear (temperature-dependent) thermal
conductivity. For materials with essentially constant k (salt, graphite
at narrow temperature ranges), the solver converges in one iteration.

Boundary conditions:
    - Dirichlet: prescribed temperature on boundary nodes
    - Convective Robin: -k * dT/dn = h * (T - T_inf)
    - Neumann (zero-flux): natural BC, no explicit treatment needed

For axisymmetric meshes (Level 1, R-Z), all element integrals include
the 2*pi*r factor automatically through the Tri3 element routines.

Usage:
    from fea.solvers.thermal import solve_thermal
    from fea.materials.properties import MaterialLibrary

    mesh = build_level1_rz()
    mat = MaterialLibrary(level=1)
    q = np.zeros(mesh.n_elements)
    q[mesh.material_ids == 0] = 22e6  # core power density

    result = solve_thermal(
        mesh, mat, q,
        bc_convective={'outer_wall': (500.0, 573.15),
                       'top_wall': (500.0, 573.15)},
    )
"""

from dataclasses import dataclass
import numpy as np
import scipy.sparse.linalg as spla

import sys
import os

# Ensure project root is on path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fea.assembly.sparse_assembler import (
    assemble_scalar_stiffness,
    assemble_scalar_load,
)
from fea.assembly.boundary_conditions import (
    apply_dirichlet_penalty,
    apply_convective_robin,
)
from fea.elements.tri3 import shape_gradients_tri3


@dataclass
class ThermalResult:
    """Result from steady-state thermal solve.

    Attributes
    ----------
    temperature : ndarray, shape (N_nodes,)
        Nodal temperature field [K].
    heat_flux : ndarray, shape (N_elem, 2)
        Element-average heat flux vector [W/m2].
        heat_flux[e] = -k_e * grad(T)_e
    mesh : Mesh
        Reference to the mesh used in the solve.
    iterations : int
        Number of Picard iterations performed.
    converged : bool
        Whether the nonlinear iteration converged.
    residual : float
        Final relative residual ||T^{n+1} - T^n|| / ||T^n||.
    """
    temperature: np.ndarray
    heat_flux: np.ndarray
    mesh: object
    iterations: int
    converged: bool
    residual: float


def solve_thermal(mesh, material_lib, q_volumetric,
                  bc_dirichlet=None, bc_convective=None,
                  T_initial=None, max_iter=20, tol=1e-4):
    """Solve steady-state heat conduction with nonlinear Picard iteration.

    Equation:  -div(k(T) * grad(T)) = q'''

    Algorithm (Picard iteration):
        1. Initialize T^0 (uniform or from T_initial)
        2. For n = 0, 1, ...:
            a. Evaluate k(T^n) for each element
            b. Assemble global stiffness K and load f
            c. Apply boundary conditions
            d. Solve K * T^{n+1} = f
            e. Check convergence: ||T^{n+1} - T^n|| / ||T^n|| < tol
            f. If converged, exit. Otherwise continue.

    Parameters
    ----------
    mesh : Mesh
        Triangular (Tri3) finite element mesh.
    material_lib : MaterialLibrary
        Material property manager.
    q_volumetric : float or ndarray, shape (N_elem,)
        Volumetric heat generation rate [W/m3]. If scalar, applied
        uniformly to all elements.
    bc_dirichlet : dict or None, optional
        Dirichlet boundary conditions. Format:
            {boundary_tag: T_value}  - uniform T on boundary
            {boundary_tag: (nodes_array, values_array)} - per-node values
        Example: {'outer_wall': 573.15}
    bc_convective : dict or None, optional
        Convective Robin boundary conditions. Format:
            {boundary_tag: (h_conv, T_inf)}
        where h_conv [W/m2/K] is the convection coefficient and
        T_inf [K] is the ambient/coolant temperature.
        Example: {'outer_wall': (500.0, 573.15)}
    T_initial : ndarray or None, optional
        Initial temperature field [K], shape (N_nodes,).
        Default: 923.15 K (650 C) everywhere.
    max_iter : int, optional
        Maximum Picard iterations. Default 20.
    tol : float, optional
        Convergence tolerance on relative temperature change. Default 1e-4.

    Returns
    -------
    result : ThermalResult
        Solution container with temperature, heat flux, and convergence info.
    """
    if bc_dirichlet is None:
        bc_dirichlet = {}
    if bc_convective is None:
        bc_convective = {}

    n_nodes = mesh.n_nodes
    n_elem = mesh.n_elements

    # Initialize temperature field
    if T_initial is not None:
        T = np.asarray(T_initial, dtype=np.float64).copy()
    else:
        T = np.full(n_nodes, 923.15, dtype=np.float64)

    # Prepare volumetric heat source
    if np.isscalar(q_volumetric):
        q_elem = np.full(n_elem, float(q_volumetric), dtype=np.float64)
    else:
        q_elem = np.asarray(q_volumetric, dtype=np.float64)
        if q_elem.shape != (n_elem,):
            raise ValueError(
                f"q_volumetric shape {q_elem.shape} != ({n_elem},)"
            )

    # Check if conductivity is temperature-dependent for any zone
    # If not, we can skip Picard iteration
    has_T_dependent_k = _check_T_dependent_k(material_lib, mesh)

    if not has_T_dependent_k:
        max_iter = 1  # No need for iteration

    # --- Picard iteration ---
    converged = False
    residual = float('inf')
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # 1. Evaluate element conductivities at current temperature
        k_elem = material_lib.get_element_properties(mesh, 'k', T_nodal=T)

        # 2. Assemble global stiffness and load
        K = assemble_scalar_stiffness(mesh, k_elem)
        f = assemble_scalar_load(mesh, q_elem)

        # 3. Apply convective (Robin) BCs
        for tag, (h_conv, T_inf) in bc_convective.items():
            K, f = apply_convective_robin(K, f, mesh, tag, h_conv, T_inf)

        # 4. Apply Dirichlet BCs
        bc_dofs_all = []
        bc_vals_all = []
        for tag, bc_spec in bc_dirichlet.items():
            if isinstance(bc_spec, (int, float)):
                # Uniform temperature on boundary
                if tag not in mesh.boundary_nodes:
                    raise KeyError(
                        f"Boundary tag '{tag}' not found in mesh. "
                        f"Available: {list(mesh.boundary_nodes.keys())}"
                    )
                nodes = mesh.boundary_nodes[tag]
                bc_dofs_all.append(nodes)
                bc_vals_all.append(np.full(len(nodes), float(bc_spec)))
            elif isinstance(bc_spec, tuple) and len(bc_spec) == 2:
                nodes, values = bc_spec
                bc_dofs_all.append(np.asarray(nodes, dtype=np.int64))
                bc_vals_all.append(np.asarray(values, dtype=np.float64))
            else:
                raise ValueError(
                    f"Invalid Dirichlet BC format for tag '{tag}': {bc_spec}"
                )

        if bc_dofs_all:
            all_dofs = np.concatenate(bc_dofs_all)
            all_vals = np.concatenate(bc_vals_all)
            K, f = apply_dirichlet_penalty(K, f, all_dofs, all_vals)

        # 5. Solve the linear system
        T_new = spla.spsolve(K, f)

        # 6. Check convergence
        T_norm = np.linalg.norm(T)
        if T_norm > 0:
            residual = np.linalg.norm(T_new - T) / T_norm
        else:
            residual = np.linalg.norm(T_new - T)

        T = T_new

        if residual < tol:
            converged = True
            break

    # --- Post-process: compute element heat flux ---
    heat_flux = _compute_heat_flux(mesh, T, material_lib)

    return ThermalResult(
        temperature=T,
        heat_flux=heat_flux,
        mesh=mesh,
        iterations=iteration,
        converged=converged,
        residual=residual,
    )


def _check_T_dependent_k(material_lib, mesh):
    """Check if any material zone has temperature-dependent conductivity.

    Tests by evaluating k at two temperatures and checking for difference.

    Parameters
    ----------
    material_lib : MaterialLibrary
    mesh : Mesh

    Returns
    -------
    has_T_dep : bool
        True if any zone's conductivity changes with temperature.
    """
    unique_zones = np.unique(mesh.material_ids)
    T1 = 873.15  # 600 C
    T2 = 973.15  # 700 C

    for zone in unique_zones:
        zone = int(zone)
        k1 = material_lib.thermal_conductivity(zone, T1)
        k2 = material_lib.thermal_conductivity(zone, T2)
        if abs(k1 - k2) / max(abs(k1), 1e-30) > 1e-6:
            return True

    return False


def _compute_heat_flux(mesh, T_nodal, material_lib):
    """Compute element-average heat flux from the temperature solution.

    For each Tri3 element:
        q_flux = -k * grad(T)
        grad(T) = dN * T_e  (constant within element)

    Parameters
    ----------
    mesh : Mesh
    T_nodal : ndarray, shape (N_nodes,)
        Nodal temperatures [K].
    material_lib : MaterialLibrary

    Returns
    -------
    heat_flux : ndarray, shape (N_elem, 2)
        Element heat flux vectors [W/m2].
    """
    n_elem = mesh.n_elements
    heat_flux = np.empty((n_elem, 2), dtype=np.float64)

    for e in range(n_elem):
        coords = mesh.element_coords(e)
        conn = mesh.elements[e]
        T_elem = T_nodal[conn]

        dN, _ = shape_gradients_tri3(coords)

        # grad(T) = dN @ T_elem  (dN is 2x3, T_elem is 3)
        # but dN[0,:] = dNi/dx, dN[1,:] = dNi/dy
        # grad(T) = [sum(dNi/dx * Ti), sum(dNi/dy * Ti)]
        grad_T = dN @ T_elem  # shape (2,)

        # Get conductivity at centroid temperature
        zone = int(mesh.material_ids[e])
        T_centroid = np.mean(T_elem)
        k = material_lib.thermal_conductivity(zone, T_centroid)

        heat_flux[e] = -k * grad_T

    return heat_flux
