"""
Sparse matrix assembly for finite element global systems.

Assembles element-level matrices and vectors into global sparse systems
using COO (coordinate) format accumulation followed by CSC conversion
for efficient solving.

Strategy:
    1. Loop over all elements
    2. Compute element matrix/vector via element-level routines
    3. Map local DOFs to global DOFs
    4. Accumulate (row, col, val) triplets into COO arrays
    5. Convert to CSC for solver compatibility

For the 40 MWth Marine MSR, typical problem sizes:
    - Thermal (Tri3, scalar): ~10k-50k DOFs
    - Structural (Tri6, vector): ~50k-200k DOFs

The COO-to-CSC conversion automatically sums duplicate entries,
which is exactly the finite element assembly operation.
"""

import numpy as np
from scipy.sparse import coo_matrix

from ..elements.tri3 import (
    stiffness_scalar_tri3,
    mass_tri3,
    load_vector_tri3,
)
from ..elements.tri6 import (
    stiffness_elastic_tri6,
    thermal_load_tri6,
    D_matrix_plane_strain,
    D_matrix_plane_stress,
    B_matrix_tri6,
    shape_functions_tri6,
)
from ..elements.quadrature import gauss_triangle_3pt


# ---------------------------------------------------------------------------
#  Generic assemblers (function-callback API)
# ---------------------------------------------------------------------------

def assemble_global_matrix(mesh, element_matrix_func, dofs_per_node=1, **kwargs):
    """
    Assemble a global sparse matrix from element contributions.

    For each element e:
        1. coords = mesh.element_coords(e)
        2. K_e = element_matrix_func(coords, **kwargs_for_element)
        3. Map local DOFs to global DOFs
        4. Accumulate (row, col, val) triplets

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    element_matrix_func : callable
        Function signature: element_matrix_func(coords, **kwargs) -> ndarray.
        Returns element matrix of shape (n_local_dof, n_local_dof).
    dofs_per_node : int, optional
        Number of DOFs per node (1 for scalar, 2 for 2D vector). Default 1.
    **kwargs
        Additional keyword arguments passed to element_matrix_func.
        If a kwarg value is an ndarray of length N_elem, the e-th element
        receives the e-th entry. Otherwise the value is passed as-is.

    Returns
    -------
    K : scipy.sparse.csc_matrix
        Global assembled matrix in CSC format.
    """
    n_elem = mesh.n_elements
    n_per_elem = mesh.elements.shape[1]
    n_local_dof = n_per_elem * dofs_per_node
    n_global_dof = mesh.n_nodes * dofs_per_node

    # Pre-allocate COO arrays (upper bound: n_elem * n_local_dof^2 entries)
    nnz_est = n_elem * n_local_dof * n_local_dof
    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    vals = np.empty(nnz_est, dtype=np.float64)

    idx = 0
    for e in range(n_elem):
        coords = mesh.element_coords(e)

        # Build element-specific kwargs: extract per-element values
        elem_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray) and val.shape == (n_elem,):
                elem_kwargs[key] = val[e]
            else:
                elem_kwargs[key] = val

        K_e = element_matrix_func(coords, **elem_kwargs)

        # Map local DOFs to global DOFs
        conn = mesh.elements[e]
        if dofs_per_node == 1:
            global_dofs = conn
        else:
            global_dofs = np.empty(n_local_dof, dtype=np.int64)
            for i, node in enumerate(conn):
                for d in range(dofs_per_node):
                    global_dofs[i * dofs_per_node + d] = node * dofs_per_node + d

        # Fill COO triplets
        for i_local in range(n_local_dof):
            for j_local in range(n_local_dof):
                rows[idx] = global_dofs[i_local]
                cols[idx] = global_dofs[j_local]
                vals[idx] = K_e[i_local, j_local]
                idx += 1

    # Trim to actual size
    rows = rows[:idx]
    cols = cols[:idx]
    vals = vals[:idx]

    K_coo = coo_matrix((vals, (rows, cols)), shape=(n_global_dof, n_global_dof))
    return K_coo.tocsc()


def assemble_global_vector(mesh, element_vector_func, dofs_per_node=1, **kwargs):
    """
    Assemble a global load vector from element contributions.

    For each element e:
        1. coords = mesh.element_coords(e)
        2. f_e = element_vector_func(coords, **kwargs_for_element)
        3. Map local DOFs to global DOFs
        4. Scatter into global vector

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    element_vector_func : callable
        Function signature: element_vector_func(coords, **kwargs) -> ndarray.
        Returns element vector of shape (n_local_dof,).
    dofs_per_node : int, optional
        Number of DOFs per node. Default 1.
    **kwargs
        Additional keyword arguments. Per-element arrays (length N_elem)
        are indexed by element number.

    Returns
    -------
    f : ndarray, shape (N_dof,)
        Global assembled load vector.
    """
    n_elem = mesh.n_elements
    n_per_elem = mesh.elements.shape[1]
    n_local_dof = n_per_elem * dofs_per_node
    n_global_dof = mesh.n_nodes * dofs_per_node

    f = np.zeros(n_global_dof)

    for e in range(n_elem):
        coords = mesh.element_coords(e)

        elem_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray) and val.shape == (n_elem,):
                elem_kwargs[key] = val[e]
            else:
                elem_kwargs[key] = val

        f_e = element_vector_func(coords, **elem_kwargs)

        conn = mesh.elements[e]
        if dofs_per_node == 1:
            global_dofs = conn
        else:
            global_dofs = np.empty(n_local_dof, dtype=np.int64)
            for i, node in enumerate(conn):
                for d in range(dofs_per_node):
                    global_dofs[i * dofs_per_node + d] = node * dofs_per_node + d

        for i_local in range(n_local_dof):
            f[global_dofs[i_local]] += f_e[i_local]

    return f


# ---------------------------------------------------------------------------
#  Convenience assemblers for specific physics
# ---------------------------------------------------------------------------

def assemble_scalar_stiffness(mesh, k_per_element):
    """
    Assemble global stiffness matrix for scalar conduction/diffusion.

    K_e = k_e * A_e * B^T * B   (Tri3 elements)

    Parameters
    ----------
    mesh : Mesh
        Must have element_type='tri3'.
    k_per_element : ndarray, shape (N_elem,)
        Thermal conductivity (or diffusion coefficient) for each element.

    Returns
    -------
    K : scipy.sparse.csc_matrix, shape (N_nodes, N_nodes)
        Global stiffness matrix.

    Raises
    ------
    ValueError
        If mesh is not Tri3 or array sizes mismatch.
    """
    if mesh.element_type != 'tri3':
        raise ValueError(
            f"assemble_scalar_stiffness requires 'tri3' mesh, "
            f"got '{mesh.element_type}'"
        )
    k_per_element = np.asarray(k_per_element, dtype=np.float64)
    if k_per_element.shape != (mesh.n_elements,):
        raise ValueError(
            f"k_per_element shape {k_per_element.shape} != ({mesh.n_elements},)"
        )

    n_elem = mesh.n_elements
    n_nodes = mesh.n_nodes
    axisym = (mesh.coord_system == 'axisymmetric')

    # Each Tri3 element contributes 3x3 = 9 entries
    nnz_est = n_elem * 9
    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    vals = np.empty(nnz_est, dtype=np.float64)

    idx = 0
    for e in range(n_elem):
        coords = mesh.element_coords(e)
        K_e = stiffness_scalar_tri3(coords, k_per_element[e], axisymmetric=axisym)
        conn = mesh.elements[e]

        for i in range(3):
            for j in range(3):
                rows[idx] = conn[i]
                cols[idx] = conn[j]
                vals[idx] = K_e[i, j]
                idx += 1

    K_coo = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return K_coo.tocsc()


def assemble_scalar_mass(mesh, rho_per_element):
    """
    Assemble global consistent mass matrix for scalar field.

    M_e = rho_e * A_e/12 * [[2,1,1],[1,2,1],[1,1,2]]   (Tri3)

    Parameters
    ----------
    mesh : Mesh
        Must have element_type='tri3'.
    rho_per_element : ndarray, shape (N_elem,)
        Density or mass coefficient for each element.

    Returns
    -------
    M : scipy.sparse.csc_matrix, shape (N_nodes, N_nodes)
        Global mass matrix.
    """
    if mesh.element_type != 'tri3':
        raise ValueError(
            f"assemble_scalar_mass requires 'tri3' mesh, "
            f"got '{mesh.element_type}'"
        )
    rho_per_element = np.asarray(rho_per_element, dtype=np.float64)
    if rho_per_element.shape != (mesh.n_elements,):
        raise ValueError(
            f"rho_per_element shape {rho_per_element.shape} != ({mesh.n_elements},)"
        )

    n_elem = mesh.n_elements
    n_nodes = mesh.n_nodes
    axisym = (mesh.coord_system == 'axisymmetric')

    nnz_est = n_elem * 9
    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    vals = np.empty(nnz_est, dtype=np.float64)

    idx = 0
    for e in range(n_elem):
        coords = mesh.element_coords(e)
        M_e = mass_tri3(coords, rho=rho_per_element[e], axisymmetric=axisym)
        conn = mesh.elements[e]

        for i in range(3):
            for j in range(3):
                rows[idx] = conn[i]
                cols[idx] = conn[j]
                vals[idx] = M_e[i, j]
                idx += 1

    M_coo = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return M_coo.tocsc()


def assemble_scalar_load(mesh, q_per_element):
    """
    Assemble global load vector for volumetric source term.

    f_e = q_e * A_e/3 * [1, 1, 1]   (Tri3)

    Parameters
    ----------
    mesh : Mesh
        Must have element_type='tri3'.
    q_per_element : ndarray, shape (N_elem,)
        Volumetric source (e.g., heat generation W/m^3) for each element.

    Returns
    -------
    f : ndarray, shape (N_nodes,)
        Global load vector.
    """
    if mesh.element_type != 'tri3':
        raise ValueError(
            f"assemble_scalar_load requires 'tri3' mesh, "
            f"got '{mesh.element_type}'"
        )
    q_per_element = np.asarray(q_per_element, dtype=np.float64)
    if q_per_element.shape != (mesh.n_elements,):
        raise ValueError(
            f"q_per_element shape {q_per_element.shape} != ({mesh.n_elements},)"
        )

    n_nodes = mesh.n_nodes
    axisym = (mesh.coord_system == 'axisymmetric')
    f = np.zeros(n_nodes)

    for e in range(mesh.n_elements):
        coords = mesh.element_coords(e)
        f_e = load_vector_tri3(coords, q_per_element[e], axisymmetric=axisym)
        conn = mesh.elements[e]
        for i in range(3):
            f[conn[i]] += f_e[i]

    return f


def assemble_elastic_stiffness(mesh, E_per_element, nu_per_element, plane='strain'):
    """
    Assemble global stiffness matrix for 2D elasticity (Tri6).

    K_e = integral B^T * D * B dA   (via 3-point Gauss quadrature)

    DOF ordering: [u0, v0, u1, v1, ..., u_{N-1}, v_{N-1}]

    Parameters
    ----------
    mesh : Mesh
        Must have element_type='tri6'.
    E_per_element : ndarray, shape (N_elem,)
        Young's modulus for each element.
    nu_per_element : ndarray, shape (N_elem,)
        Poisson's ratio for each element.
    plane : str, optional
        'strain' or 'stress'. Default 'strain'.

    Returns
    -------
    K : scipy.sparse.csc_matrix, shape (2*N_nodes, 2*N_nodes)
        Global elastic stiffness matrix.
    """
    if mesh.element_type != 'tri6':
        raise ValueError(
            f"assemble_elastic_stiffness requires 'tri6' mesh, "
            f"got '{mesh.element_type}'"
        )
    E_per_element = np.asarray(E_per_element, dtype=np.float64)
    nu_per_element = np.asarray(nu_per_element, dtype=np.float64)

    n_elem = mesh.n_elements
    n_nodes = mesh.n_nodes
    n_dof = 2 * n_nodes

    # Each Tri6 element: 12x12 = 144 entries
    nnz_est = n_elem * 144
    rows = np.empty(nnz_est, dtype=np.int64)
    cols = np.empty(nnz_est, dtype=np.int64)
    vals = np.empty(nnz_est, dtype=np.float64)

    idx = 0
    for e in range(n_elem):
        coords = mesh.element_coords(e)
        K_e = stiffness_elastic_tri6(
            coords, E_per_element[e], nu_per_element[e], plane=plane
        )
        conn = mesh.elements[e]

        # Global DOF map: node i -> [2*node, 2*node+1]
        global_dofs = np.empty(12, dtype=np.int64)
        for i in range(6):
            global_dofs[2 * i] = 2 * conn[i]
            global_dofs[2 * i + 1] = 2 * conn[i] + 1

        for i_local in range(12):
            for j_local in range(12):
                rows[idx] = global_dofs[i_local]
                cols[idx] = global_dofs[j_local]
                vals[idx] = K_e[i_local, j_local]
                idx += 1

    K_coo = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K_coo.tocsc()


def assemble_thermal_load(mesh, E_per_element, nu_per_element,
                          alpha_per_element, dT_nodal, plane='strain'):
    """
    Assemble global thermal load vector for 2D elasticity (Tri6).

    f_th = integral B^T * D * eps_th dA

    where eps_th = alpha * dT * {1, 1, 0}^T

    Temperature is interpolated from nodal values using Tri6 shape functions.

    Parameters
    ----------
    mesh : Mesh
        Must have element_type='tri6'.
    E_per_element : ndarray, shape (N_elem,)
        Young's modulus for each element.
    nu_per_element : ndarray, shape (N_elem,)
        Poisson's ratio for each element.
    alpha_per_element : ndarray, shape (N_elem,)
        Coefficient of thermal expansion for each element.
    dT_nodal : ndarray, shape (N_nodes,)
        Temperature change at each node (from reference temperature).
    plane : str, optional
        'strain' or 'stress'. Default 'strain'.

    Returns
    -------
    f_th : ndarray, shape (2*N_nodes,)
        Global thermal load vector.
    """
    if mesh.element_type != 'tri6':
        raise ValueError(
            f"assemble_thermal_load requires 'tri6' mesh, "
            f"got '{mesh.element_type}'"
        )

    n_elem = mesh.n_elements
    n_nodes = mesh.n_nodes
    n_dof = 2 * n_nodes
    f_th = np.zeros(n_dof)

    for e in range(n_elem):
        coords = mesh.element_coords(e)
        conn = mesh.elements[e]
        dT_elem = dT_nodal[conn]

        f_e = thermal_load_tri6(
            coords, E_per_element[e], nu_per_element[e],
            alpha_per_element[e], dT_elem, plane=plane
        )

        global_dofs = np.empty(12, dtype=np.int64)
        for i in range(6):
            global_dofs[2 * i] = 2 * conn[i]
            global_dofs[2 * i + 1] = 2 * conn[i] + 1

        for i_local in range(12):
            f_th[global_dofs[i_local]] += f_e[i_local]

    return f_th
