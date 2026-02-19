"""
1-Group Finite Element Diffusion Eigenvalue Solver
===================================================

Solves the 1-group neutron diffusion equation in eigenvalue form
for the 40 MWth Marine MSR using Tri3 finite elements.

Weak form (multiply by test function N_i, integrate by parts):

    integral(D * grad(N_i) . grad(phi) + Sigma_a * N_i * phi) dA
        = (1/keff) * integral(nu*Sigma_f * N_i * phi) dA

Matrix form:
    A * phi = (1/keff) * F * phi

where:
    A = K_diff + M_abs    (diffusion stiffness + absorption mass matrix)
    F = M_fiss             (fission source mass matrix)
    K_diff[i,j] = integral(D * grad(N_i) . grad(N_j)) dA
    M_abs[i,j]  = integral(Sigma_a * N_i * N_j) dA
    M_fiss[i,j] = integral(nu*Sigma_f * N_i * N_j) dA

Solution via inverse power iteration:
    1. Initialize phi_0 (cosine * Bessel shape for R-Z)
    2. Compute source: s = F * phi_n
    3. Solve: A * phi_(n+1) = (1/k_n) * s
    4. Update keff via Rayleigh quotient
    5. Normalize phi_(n+1)
    6. Check convergence

Boundary conditions:
    - Vacuum: phi = 0 on outer boundary (via elimination)
    - Symmetry: dphi/dn = 0 (natural BC, no action needed)

For axisymmetric geometry (Level 1 R-Z):
    Element matrices include 2*pi*r factor automatically via
    the axisymmetric flag in tri3 element routines.

After convergence, power density is computed per element:
    q'''_e = Sigma_f_e * phi_avg_e * E_fission * normalization
    normalized so integral(q''' * dV) = total_power

Sources:
    - Duderstadt & Hamilton, "Nuclear Reactor Analysis"
    - Lewis & Miller, "Computational Methods of Neutron Transport"
    - ORNL-4541 for MSR cross-section estimates
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import splu
from scipy.special import j0 as bessel_j0
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fea.mesh.nodes import Mesh
from fea.elements.tri3 import shape_gradients_tri3, mass_tri3, stiffness_scalar_tri3
from config import ENERGY_PER_FISSION


@dataclass
class NeutronicsResult:
    """Results from the neutron diffusion eigenvalue solve.

    Attributes
    ----------
    keff : float
        Effective multiplication factor.
    flux : ndarray, shape (N_nodes,)
        Normalized neutron flux (peak = 1.0).
    power_density : ndarray, shape (N_elem,)
        Fission power density per element [W/m^3].
    mesh : Mesh
        The mesh used for the solve.
    iterations : int
        Number of power iterations performed.
    converged : bool
        Whether the solver achieved convergence.
    """
    keff: float
    flux: np.ndarray
    power_density: np.ndarray
    mesh: 'Mesh'
    iterations: int
    converged: bool


def _build_material_arrays(mesh, material_lib):
    """Extract per-element material properties from the material library.

    The material_lib must be a dict mapping zone_id -> dict with keys:
        'D'          : diffusion coefficient [m]
        'sigma_a'    : macroscopic absorption cross-section [1/m]
        'nu_sigma_f' : nu * macroscopic fission cross-section [1/m]
        'sigma_f'    : macroscopic fission cross-section [1/m]

    Parameters
    ----------
    mesh : Mesh
        FE mesh with material_ids per element.
    material_lib : dict
        Material zone properties.

    Returns
    -------
    D_arr : ndarray, shape (N_elem,)
    sigma_a_arr : ndarray, shape (N_elem,)
    nu_sigma_f_arr : ndarray, shape (N_elem,)
    sigma_f_arr : ndarray, shape (N_elem,)
    """
    n_elem = mesh.n_elements
    D_arr = np.empty(n_elem)
    sigma_a_arr = np.empty(n_elem)
    nu_sigma_f_arr = np.empty(n_elem)
    sigma_f_arr = np.empty(n_elem)

    for e in range(n_elem):
        zone = int(mesh.material_ids[e])
        mat = material_lib[zone]
        D_arr[e] = mat['D']
        sigma_a_arr[e] = mat['sigma_a']
        nu_sigma_f_arr[e] = mat.get('nu_sigma_f', 0.0)
        sigma_f_arr[e] = mat.get('sigma_f', 0.0)

    return D_arr, sigma_a_arr, nu_sigma_f_arr, sigma_f_arr


def _assemble_diffusion_matrices(mesh, D_arr, sigma_a_arr, nu_sigma_f_arr):
    """Assemble the three global sparse matrices for the diffusion problem.

    A = K_diff + M_abs
    F = M_fiss

    Parameters
    ----------
    mesh : Mesh
    D_arr : ndarray, shape (N_elem,)
    sigma_a_arr : ndarray, shape (N_elem,)
    nu_sigma_f_arr : ndarray, shape (N_elem,)

    Returns
    -------
    A : csc_matrix, shape (N_nodes, N_nodes)
    F : csc_matrix, shape (N_nodes, N_nodes)
    """
    n_elem = mesh.n_elements
    n_nodes = mesh.n_nodes
    axisym = (mesh.coord_system == 'axisymmetric')

    # Pre-allocate COO arrays: each Tri3 element -> 9 entries per matrix
    nnz_est = n_elem * 9
    # A matrix COO
    a_rows = np.empty(nnz_est, dtype=np.int64)
    a_cols = np.empty(nnz_est, dtype=np.int64)
    a_vals = np.empty(nnz_est, dtype=np.float64)
    # F matrix COO
    f_rows = np.empty(nnz_est, dtype=np.int64)
    f_cols = np.empty(nnz_est, dtype=np.int64)
    f_vals = np.empty(nnz_est, dtype=np.float64)

    idx = 0
    for e in range(n_elem):
        coords = mesh.element_coords(e)
        conn = mesh.elements[e]

        # Diffusion stiffness: K_diff_e = D * A * dN^T @ dN (with axisym)
        K_diff_e = stiffness_scalar_tri3(coords, D_arr[e], axisymmetric=axisym)

        # Absorption mass: M_abs_e = sigma_a * consistent mass
        M_abs_e = mass_tri3(coords, rho=sigma_a_arr[e], axisymmetric=axisym)

        # Fission source mass: M_fiss_e = nu_sigma_f * consistent mass
        M_fiss_e = mass_tri3(coords, rho=nu_sigma_f_arr[e], axisymmetric=axisym)

        # A_e = K_diff_e + M_abs_e
        A_e = K_diff_e + M_abs_e

        for i in range(3):
            for j in range(3):
                a_rows[idx] = conn[i]
                a_cols[idx] = conn[j]
                a_vals[idx] = A_e[i, j]
                f_rows[idx] = conn[i]
                f_cols[idx] = conn[j]
                f_vals[idx] = M_fiss_e[i, j]
                idx += 1

    # Trim and build CSC
    a_rows = a_rows[:idx]
    a_cols = a_cols[:idx]
    a_vals = a_vals[:idx]
    f_rows = f_rows[:idx]
    f_cols = f_cols[:idx]
    f_vals = f_vals[:idx]

    A = coo_matrix((a_vals, (a_rows, a_cols)),
                   shape=(n_nodes, n_nodes)).tocsc()
    F = coo_matrix((f_vals, (f_rows, f_cols)),
                   shape=(n_nodes, n_nodes)).tocsc()

    return A, F


def _apply_vacuum_bc(A, F, bc_nodes):
    """Apply vacuum BC (phi=0) via row/column elimination.

    Zeroes rows and columns for BC nodes in both A and F,
    sets diagonal of A to 1.0 and F to 0.0 at those DOFs.
    This preserves eigenvalue structure for the free DOFs.

    Parameters
    ----------
    A : csc_matrix
    F : csc_matrix
    bc_nodes : ndarray of int
        Node indices where phi = 0.

    Returns
    -------
    A_mod : csc_matrix
    F_mod : csc_matrix
    """
    A_mod = lil_matrix(A)
    F_mod = lil_matrix(F)

    for dof in bc_nodes:
        A_mod[dof, :] = 0
        A_mod[:, dof] = 0
        A_mod[dof, dof] = 1.0
        F_mod[dof, :] = 0
        F_mod[:, dof] = 0
        # F diagonal stays 0 -> no fission source at boundary

    return A_mod.tocsc(), F_mod.tocsc()


def _initial_flux_rz(mesh, D_core, sigma_a_core):
    """Generate initial flux guess for R-Z axisymmetric geometry.

    Uses the fundamental mode shape for a finite cylinder:
        phi(r, z) = J0(2.405 * r / R_ext) * cos(pi * z / H_ext)

    where R_ext and H_ext include extrapolation distances:
        d_ext = 2 * D / Sigma_tr  (approximately 2*D for graphite-moderated)

    Parameters
    ----------
    mesh : Mesh
        Axisymmetric mesh with nodes[:,0]=r and nodes[:,1]=z.
    D_core : float
        Diffusion coefficient in the core region.
    sigma_a_core : float
        Absorption cross-section in the core (for transport estimate).

    Returns
    -------
    phi0 : ndarray, shape (N_nodes,)
        Initial flux estimate (non-negative, normalized).
    """
    r = mesh.nodes[:, 0]
    z = mesh.nodes[:, 1]

    # Determine core extent from node coordinates
    R_core = np.max(r) * 0.7  # approximate core radius
    H_half = np.max(z) * 0.7  # approximate core half-height

    # Extrapolation distance: d ~ 2.13 * D (for planar vacuum interface)
    d_ext = 2.13 * D_core

    R_ext = R_core + d_ext
    H_ext = 2.0 * (H_half + d_ext)  # full height with extrapolation

    # Fundamental mode
    phi0 = bessel_j0(2.405 * r / R_ext) * np.cos(np.pi * z / (H_ext))

    # Ensure non-negative and normalize
    phi0 = np.maximum(phi0, 0.0)
    norm = np.max(phi0)
    if norm > 0:
        phi0 /= norm

    # Set minimum value for non-zero initial guess
    phi0 = np.maximum(phi0, 1e-10)

    return phi0


def _initial_flux_generic(mesh):
    """Flat initial flux guess for non-axisymmetric meshes.

    Parameters
    ----------
    mesh : Mesh

    Returns
    -------
    phi0 : ndarray, shape (N_nodes,)
    """
    return np.ones(mesh.n_nodes)


def _compute_element_volumes(mesh):
    """Compute element volumes (areas for 2D, with 2*pi*r for axisymmetric).

    Parameters
    ----------
    mesh : Mesh

    Returns
    -------
    volumes : ndarray, shape (N_elem,)
        Element volumes in m^3 (axisym) or m^2 (cartesian).
    """
    axisym = (mesh.coord_system == 'axisymmetric')
    volumes = np.empty(mesh.n_elements)

    for e in range(mesh.n_elements):
        area = abs(mesh.element_area(e))
        if axisym:
            coords = mesh.element_coords(e)
            r_c = np.mean(coords[:, 0])
            r_c = max(r_c, 0.0)
            volumes[e] = 2.0 * np.pi * r_c * area
        else:
            volumes[e] = area

    return volumes


def solve_diffusion_fe(mesh, material_lib, total_power=40e6,
                       max_iter=500, tol_k=1e-6, tol_flux=1e-5):
    """Solve 1-group FE diffusion eigenvalue problem.

    Parameters
    ----------
    mesh : Mesh
        Tri3 mesh (cartesian or axisymmetric).
    material_lib : dict
        Maps zone_id (int) -> dict with keys:
            'D'          : diffusion coefficient [m]
            'sigma_a'    : macroscopic absorption [1/m]
            'nu_sigma_f' : nu * Sigma_f [1/m]
            'sigma_f'    : Sigma_f [1/m]
    total_power : float
        Total thermal power for normalization [W]. Default 40 MW.
    max_iter : int
        Maximum power iterations. Default 500.
    tol_k : float
        Convergence tolerance on |dk/k|. Default 1e-6.
    tol_flux : float
        Convergence tolerance on ||dphi||/||phi||. Default 1e-5.

    Returns
    -------
    result : NeutronicsResult
        Solution dataclass with keff, flux, power_density, etc.

    Raises
    ------
    ValueError
        If mesh is not Tri3 or material_lib is missing required zones.
    """
    if mesh.element_type != 'tri3':
        raise ValueError(
            f"solve_diffusion_fe requires 'tri3' mesh, got '{mesh.element_type}'"
        )

    # --- Extract material arrays ---
    D_arr, sigma_a_arr, nu_sigma_f_arr, sigma_f_arr = \
        _build_material_arrays(mesh, material_lib)

    # --- Assemble global matrices ---
    A, F = _assemble_diffusion_matrices(mesh, D_arr, sigma_a_arr, nu_sigma_f_arr)

    # --- Identify vacuum BC nodes ---
    # Vacuum BC on outer_wall and top_wall boundaries
    bc_node_set = set()
    for tag in ['outer_wall', 'top_wall']:
        if tag in mesh.boundary_nodes:
            bc_node_set.update(mesh.boundary_nodes[tag].tolist())
    # Also check generic boundary tags
    for tag in ['outer_boundary', 'outer']:
        if tag in mesh.boundary_nodes:
            bc_node_set.update(mesh.boundary_nodes[tag].tolist())

    bc_nodes = np.array(sorted(bc_node_set), dtype=np.int64)

    # --- Apply vacuum BCs ---
    A_bc, F_bc = _apply_vacuum_bc(A, F, bc_nodes)

    # --- Pre-factor A for repeated solves ---
    A_lu = splu(A_bc)

    # --- Initial flux guess ---
    # Get representative core D for initial guess
    core_zones = [z for z in material_lib if material_lib[z].get('nu_sigma_f', 0) > 0]
    if core_zones:
        D_core = material_lib[core_zones[0]]['D']
        sigma_a_core = material_lib[core_zones[0]]['sigma_a']
    else:
        D_core = 0.01
        sigma_a_core = 1.0

    if mesh.coord_system == 'axisymmetric':
        phi = _initial_flux_rz(mesh, D_core, sigma_a_core)
    else:
        phi = _initial_flux_generic(mesh)

    # Set BC nodes to zero
    phi[bc_nodes] = 0.0

    # Normalize
    phi_norm = np.linalg.norm(phi)
    if phi_norm > 0:
        phi /= phi_norm

    # --- Power iteration ---
    # Standard formulation:
    #   Given A*phi = (1/k)*F*phi, we iterate:
    #     s = F*phi_n                       (fission source)
    #     A*phi_(n+1) = (1/k_n)*s           (solve for new flux)
    #     k_(n+1) = k_n * sum(F*phi_(n+1)) / sum(s)  (update eigenvalue)
    #     normalize phi_(n+1)
    keff = 1.0
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # 1. Compute fission source: s = F * phi
        source = F_bc @ phi
        source_total = np.sum(source)

        # 2. Solve: A * phi_new = (1/k) * source
        rhs = source / keff
        phi_new = A_lu.solve(rhs)

        # 3. Enforce BCs
        phi_new[bc_nodes] = 0.0

        # 4. Update keff: ratio of new to old fission source totals
        new_source = F_bc @ phi_new
        new_source_total = np.sum(new_source)

        if abs(source_total) > 0:
            keff_new = keff * new_source_total / source_total
        else:
            keff_new = keff

        # 5. Normalize phi_new
        norm_new = np.linalg.norm(phi_new)
        if norm_new > 0:
            phi_new /= norm_new

        # 6. Check convergence
        dk_rel = abs(keff_new - keff) / max(abs(keff_new), 1e-30)
        dphi_rel = np.linalg.norm(phi_new - phi) / max(np.linalg.norm(phi_new), 1e-30)

        keff = keff_new
        phi = phi_new

        if dk_rel < tol_k and dphi_rel < tol_flux:
            converged = True
            break

    # --- Normalize flux so peak = 1.0 ---
    phi_max = np.max(np.abs(phi))
    if phi_max > 0:
        phi /= phi_max

    # --- Compute power density per element ---
    volumes = _compute_element_volumes(mesh)
    power_density = np.empty(mesh.n_elements)

    for e in range(mesh.n_elements):
        conn = mesh.elements[e]
        phi_avg = np.mean(phi[conn])
        # q''' = Sigma_f * phi * E_fission (unnormalized)
        power_density[e] = sigma_f_arr[e] * phi_avg * ENERGY_PER_FISSION

    # Normalize so total power matches
    total_power_raw = np.sum(power_density * volumes)
    if total_power_raw > 0:
        power_density *= total_power / total_power_raw

    return NeutronicsResult(
        keff=keff,
        flux=phi,
        power_density=power_density,
        mesh=mesh,
        iterations=n_iter,
        converged=converged,
    )
