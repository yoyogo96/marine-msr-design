"""
Boundary condition application for finite element systems.

Supports three types of boundary conditions:

1. Dirichlet (essential) via penalty method:
   - Fast, preserves matrix sparsity pattern
   - Adds large penalty to diagonal, adjusts RHS
   - Not suitable for eigenvalue problems (corrupts eigenvalues)

2. Dirichlet via row/column elimination:
   - Exact enforcement, no penalty parameter
   - Suitable for eigenvalue problems
   - Modifies sparsity pattern (converts to lil_matrix internally)

3. Robin (convective) boundary condition:
   - Natural BC: -k*dT/dn = h*(T - T_inf)
   - Adds boundary mass matrix h*integral(N_i*N_j) to stiffness
   - Adds boundary load h*T_inf*integral(N_i) to RHS
   - Works for both cartesian and axisymmetric meshes

For the 40 MWth Marine MSR:
    - Dirichlet: fixed temperature at coolant inlet
    - Robin: convection at vessel outer wall, fuel-coolant interface
    - Symmetry: zero normal displacement on symmetry planes
"""

import numpy as np
from scipy.sparse import lil_matrix

from ..elements.tri3 import boundary_mass_tri3, boundary_load_tri3


def apply_dirichlet_penalty(K, f, bc_dofs, bc_values, penalty=1e20):
    """
    Apply Dirichlet boundary conditions via the penalty method.

    For each constrained DOF i with prescribed value u_i:
        K[i, i] += penalty
        f[i]    += penalty * u_i

    This drives the solution at DOF i toward u_i with error O(1/penalty).
    The penalty should be ~1e10 to 1e20 times the largest diagonal of K.

    Parameters
    ----------
    K : scipy.sparse matrix
        Global stiffness matrix. A copy is made to avoid modifying the
        original (converting to lil_matrix for efficient diagonal access).
    f : ndarray, shape (N,)
        Global load vector. Modified in-place.
    bc_dofs : array_like of int
        Indices of constrained DOFs.
    bc_values : array_like of float
        Prescribed values at constrained DOFs.
    penalty : float, optional
        Penalty parameter. Default 1e20.

    Returns
    -------
    K_mod : scipy.sparse.csc_matrix
        Modified stiffness matrix with penalty applied.
    f_mod : ndarray, shape (N,)
        Modified load vector (same object as input f).
    """
    bc_dofs = np.asarray(bc_dofs, dtype=np.int64)
    bc_values = np.asarray(bc_values, dtype=np.float64)

    if len(bc_dofs) != len(bc_values):
        raise ValueError(
            f"bc_dofs length ({len(bc_dofs)}) != bc_values length ({len(bc_values)})"
        )
    if len(bc_dofs) == 0:
        return K.tocsc(), f

    K_mod = lil_matrix(K)
    for dof, val in zip(bc_dofs, bc_values):
        K_mod[dof, dof] += penalty
        f[dof] += penalty * val

    return K_mod.tocsc(), f


def apply_dirichlet_elimination(K, f, bc_dofs, bc_values):
    """
    Apply Dirichlet boundary conditions via row/column elimination.

    For each constrained DOF i with prescribed value u_i:
        1. Subtract column i * u_i from the load vector
        2. Zero out row i and column i
        3. Set K[i, i] = 1, f[i] = u_i

    This preserves the eigenvalue structure (no artificial large values)
    and is preferred for eigenvalue problems.

    Parameters
    ----------
    K : scipy.sparse matrix
        Global stiffness matrix. A copy is made internally.
    f : ndarray, shape (N,)
        Global load vector. A copy is made internally.
    bc_dofs : array_like of int
        Indices of constrained DOFs.
    bc_values : array_like of float
        Prescribed values at constrained DOFs.

    Returns
    -------
    K_mod : scipy.sparse.csc_matrix
        Modified stiffness matrix with BC rows/columns eliminated.
    f_mod : ndarray, shape (N,)
        Modified load vector.
    """
    bc_dofs = np.asarray(bc_dofs, dtype=np.int64)
    bc_values = np.asarray(bc_values, dtype=np.float64)

    if len(bc_dofs) != len(bc_values):
        raise ValueError(
            f"bc_dofs length ({len(bc_dofs)}) != bc_values length ({len(bc_values)})"
        )
    if len(bc_dofs) == 0:
        return K.tocsc(), f.copy()

    K_mod = lil_matrix(K.copy())
    f_mod = f.copy()
    n = K.shape[0]

    # Create a set for fast lookup
    bc_set = set(bc_dofs.tolist())

    for dof, val in zip(bc_dofs, bc_values):
        # Modify RHS: f_j -= K[j, dof] * val for all free DOFs j
        # Use CSC column access for efficiency
        K_csc = K_mod.tocsc()
        col_start = K_csc.indptr[dof]
        col_end = K_csc.indptr[dof + 1]
        row_indices = K_csc.indices[col_start:col_end]
        col_values = K_csc.data[col_start:col_end]

        for r, v in zip(row_indices, col_values):
            if r not in bc_set:
                f_mod[r] -= v * val

        # Zero out row and column
        K_mod[dof, :] = 0
        K_mod[:, dof] = 0
        # Set diagonal to 1
        K_mod[dof, dof] = 1.0
        f_mod[dof] = val

    return K_mod.tocsc(), f_mod


def apply_convective_robin(K, f, mesh, boundary_tag, h_conv, T_inf):
    """
    Apply Robin (convective) boundary condition on a named boundary.

    Robin BC:  -k * dT/dn = h * (T - T_inf)

    This adds to the global system:
        K += h * integral_boundary(N_i * N_j) ds      (boundary mass)
        f += h * T_inf * integral_boundary(N_i) ds     (boundary load)

    For axisymmetric problems, the boundary integrals include the
    2*pi*r factor automatically (handled by element routines).

    Parameters
    ----------
    K : scipy.sparse matrix
        Global stiffness matrix. Modified via returned copy.
    f : ndarray, shape (N,)
        Global load vector. Modified in-place.
    mesh : Mesh
        The finite element mesh.
    boundary_tag : str
        Key in mesh.boundary_edges identifying the boundary.
    h_conv : float
        Convection heat transfer coefficient (W/m^2/K).
    T_inf : float
        Ambient/fluid temperature (K or C).

    Returns
    -------
    K_mod : scipy.sparse.csc_matrix
        Stiffness matrix with convection boundary mass added.
    f_mod : ndarray, shape (N,)
        Load vector with convection boundary load added (same object as input f).

    Raises
    ------
    KeyError
        If boundary_tag is not found in mesh.boundary_edges.
    """
    if boundary_tag not in mesh.boundary_edges:
        raise KeyError(
            f"Boundary tag '{boundary_tag}' not found in mesh. "
            f"Available tags: {list(mesh.boundary_edges.keys())}"
        )

    edges = mesh.boundary_edges[boundary_tag]
    axisym = (mesh.coord_system == 'axisymmetric')

    K_mod = lil_matrix(K)

    for edge in edges:
        # Extract the two endpoint nodes of the edge
        # For Tri3: edge = (node_a, node_b)
        # For Tri6: edge = (node_a, node_mid, node_b) -- use endpoints only
        if len(edge) == 2:
            node_a, node_b = edge
        elif len(edge) == 3:
            node_a, _, node_b = edge
        else:
            raise ValueError(f"Unexpected edge format: {edge}")

        coords_edge = mesh.nodes[np.array([node_a, node_b])]

        # Element boundary mass matrix (2x2)
        H_e = boundary_mass_tri3(coords_edge, h_conv, axisymmetric=axisym)
        # Element boundary load vector (2,)
        f_e = boundary_load_tri3(coords_edge, h_conv, T_inf, axisymmetric=axisym)

        # Assemble into global system
        edge_nodes = [node_a, node_b]
        for i in range(2):
            f[edge_nodes[i]] += f_e[i]
            for j in range(2):
                K_mod[edge_nodes[i], edge_nodes[j]] += H_e[i, j]

    return K_mod.tocsc(), f


def get_boundary_dofs(mesh, boundary_tag, dofs_per_node=1, component=None):
    """
    Get global DOF indices for nodes on a named boundary.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    boundary_tag : str
        Key in mesh.boundary_nodes identifying the boundary.
    dofs_per_node : int, optional
        Number of DOFs per node. Default 1.
    component : int or None, optional
        For vector problems (dofs_per_node > 1):
        - component=0: return x-DOFs only (u)
        - component=1: return y-DOFs only (v)
        - component=None: return all DOFs for each node
        Ignored when dofs_per_node=1.

    Returns
    -------
    dofs : ndarray of int
        Sorted array of global DOF indices.

    Raises
    ------
    KeyError
        If boundary_tag is not found in mesh.boundary_nodes.
    ValueError
        If component is out of range.
    """
    if boundary_tag not in mesh.boundary_nodes:
        raise KeyError(
            f"Boundary tag '{boundary_tag}' not found in mesh. "
            f"Available tags: {list(mesh.boundary_nodes.keys())}"
        )

    nodes = mesh.boundary_nodes[boundary_tag]

    if dofs_per_node == 1:
        return np.sort(nodes)

    if component is not None:
        if component < 0 or component >= dofs_per_node:
            raise ValueError(
                f"component={component} out of range for "
                f"dofs_per_node={dofs_per_node}"
            )
        dofs = nodes * dofs_per_node + component
    else:
        # All DOFs for each boundary node
        dofs = np.empty(len(nodes) * dofs_per_node, dtype=np.int64)
        for i, node in enumerate(nodes):
            for d in range(dofs_per_node):
                dofs[i * dofs_per_node + d] = node * dofs_per_node + d

    return np.sort(dofs)
