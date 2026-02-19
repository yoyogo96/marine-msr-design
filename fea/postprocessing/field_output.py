"""
Post-processing utilities for FEA field output.

Provides nodal averaging, line extraction, and stress computations
for the 40 MWth Marine MSR finite element analysis results.

Functions:
    nodal_average:     Element-centered values -> node-averaged values
    extract_line:      Extract field values along a line (barycentric interpolation)
    compute_von_mises: Von Mises equivalent stress from (N,3) stress array
    compute_principal: Principal stresses from (N,3) stress array
"""

import numpy as np
from typing import Tuple


def nodal_average(mesh, element_values):
    """Average element-centered values to nodes.

    For each node, average the values of all elements sharing that node.
    This produces a smoothed nodal field from element-centered data.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    element_values : ndarray, shape (N_elem,) or (N_elem, M)
        Element-centered field values. If 2D, each column is
        averaged independently.

    Returns
    -------
    nodal_values : ndarray, shape (N_nodes,) or (N_nodes, M)
        Node-averaged field values.
    """
    element_values = np.asarray(element_values, dtype=np.float64)
    n_nodes = mesh.n_nodes
    n_per_elem = mesh.elements.shape[1]

    is_1d = (element_values.ndim == 1)
    if is_1d:
        element_values = element_values[:, np.newaxis]

    n_components = element_values.shape[1]
    nodal_sum = np.zeros((n_nodes, n_components), dtype=np.float64)
    nodal_count = np.zeros(n_nodes, dtype=np.float64)

    for e in range(mesh.n_elements):
        conn = mesh.elements[e]
        # Use only corner nodes for Tri6 (first 3)
        n_corners = min(n_per_elem, 3)
        for i in range(n_corners):
            node = conn[i]
            nodal_sum[node] += element_values[e]
            nodal_count[node] += 1.0

        # For Tri6, also add to mid-edge nodes
        if n_per_elem == 6:
            for i in range(3, 6):
                node = conn[i]
                nodal_sum[node] += element_values[e]
                nodal_count[node] += 1.0

    # Avoid division by zero for isolated nodes
    nodal_count = np.maximum(nodal_count, 1.0)

    nodal_values = nodal_sum / nodal_count[:, np.newaxis]

    if is_1d:
        return nodal_values[:, 0]
    return nodal_values


def extract_line(mesh, field, start, end, n_points=100):
    """Extract field values along a line from start to end.

    Samples the field at n_points evenly spaced along the line segment
    using barycentric interpolation within elements.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh (Tri3).
    field : ndarray, shape (N_nodes,)
        Nodal field values.
    start : array_like, shape (2,)
        Start point of the line [x, y] or [r, z].
    end : array_like, shape (2,)
        End point of the line [x, y] or [r, z].
    n_points : int
        Number of sample points along the line. Default 100.

    Returns
    -------
    distances : ndarray, shape (n_found,)
        Arc-length distances from start for successfully interpolated points.
    values : ndarray, shape (n_found,)
        Interpolated field values at those points.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)

    line_vec = end - start
    total_length = np.linalg.norm(line_vec)
    if total_length < 1e-15:
        return np.array([0.0]), np.array([field[0]])

    t_values = np.linspace(0.0, 1.0, n_points)
    distances = []
    values = []

    for t in t_values:
        pt = start + t * line_vec
        # Find containing element
        val = _interpolate_at_point(mesh, field, pt)
        if val is not None:
            distances.append(t * total_length)
            values.append(val)

    return np.array(distances), np.array(values)


def compute_von_mises(stress):
    """Compute von Mises equivalent stress from 2D stress array.

    For 2D stress state {sigma_xx, sigma_yy, tau_xy}:
        sigma_vm = sqrt(s_xx^2 + s_yy^2 - s_xx*s_yy + 3*tau_xy^2)

    Parameters
    ----------
    stress : ndarray, shape (N, 3)
        Stress components [sigma_xx, sigma_yy, tau_xy] per element.

    Returns
    -------
    von_mises : ndarray, shape (N,)
        Von Mises equivalent stress.
    """
    stress = np.asarray(stress, dtype=np.float64)
    sxx = stress[:, 0]
    syy = stress[:, 1]
    txy = stress[:, 2]
    return np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * txy**2)


def compute_principal(stress):
    """Compute principal stresses from 2D stress array.

    For each stress state {sigma_xx, sigma_yy, tau_xy}, computes
    the two principal stresses from Mohr's circle:
        sigma_1 = (sxx + syy)/2 + R
        sigma_2 = (sxx + syy)/2 - R
        where R = sqrt(((sxx-syy)/2)^2 + txy^2)
        theta = 0.5 * atan2(2*txy, sxx - syy)

    Parameters
    ----------
    stress : ndarray, shape (N, 3)
        Stress components [sigma_xx, sigma_yy, tau_xy] per element.

    Returns
    -------
    sigma_1 : ndarray, shape (N,)
        Maximum principal stress.
    sigma_2 : ndarray, shape (N,)
        Minimum principal stress.
    theta : ndarray, shape (N,)
        Principal angle [rad] from x-axis to sigma_1 direction.
    """
    stress = np.asarray(stress, dtype=np.float64)
    sxx = stress[:, 0]
    syy = stress[:, 1]
    txy = stress[:, 2]

    s_avg = 0.5 * (sxx + syy)
    R = np.sqrt(((sxx - syy) / 2.0)**2 + txy**2)

    sigma_1 = s_avg + R
    sigma_2 = s_avg - R
    theta = 0.5 * np.arctan2(2.0 * txy, sxx - syy)

    return sigma_1, sigma_2, theta


def _interpolate_at_point(mesh, field, point):
    """Interpolate a nodal field at a given physical point.

    Searches all elements and uses barycentric coordinates for
    the containing triangle. Returns None if no element contains
    the point.

    Parameters
    ----------
    mesh : Mesh
    field : ndarray, shape (N_nodes,)
    point : ndarray, shape (2,)

    Returns
    -------
    value : float or None
    """
    px, py = point

    for e in range(mesh.n_elements):
        conn = mesh.elements[e]
        # Use corner nodes only (works for both Tri3 and Tri6)
        x0, y0 = mesh.nodes[conn[0]]
        x1, y1 = mesh.nodes[conn[1]]
        x2, y2 = mesh.nodes[conn[2]]

        # Compute barycentric coordinates
        detT = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(detT) < 1e-30:
            continue

        inv_detT = 1.0 / detT
        lam1 = ((y2 - y0) * (px - x0) - (x2 - x0) * (py - y0)) * inv_detT
        lam2 = (-(y1 - y0) * (px - x0) + (x1 - x0) * (py - y0)) * inv_detT
        lam0 = 1.0 - lam1 - lam2

        # Check if point is inside triangle (with tolerance)
        tol = -1e-6
        if lam0 >= tol and lam1 >= tol and lam2 >= tol:
            # Interpolate using corner node values
            value = lam0 * field[conn[0]] + lam1 * field[conn[1]] + lam2 * field[conn[2]]
            return value

    return None
