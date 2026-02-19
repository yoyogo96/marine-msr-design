"""
3-node linear triangle element for scalar field problems.

Used for thermal conduction, neutron diffusion, and other scalar PDEs
in the 40 MWth Marine MSR analysis.

Node numbering (counter-clockwise):
    2
    |\\
    | \\
    |  \\
    0---1

Parametric coordinates (xi, eta):
    Node 0: (0, 0)  ->  L1 = 1 - xi - eta
    Node 1: (1, 0)  ->  L2 = xi
    Node 2: (0, 1)  ->  L3 = eta

Shape functions:
    N1 = 1 - xi - eta
    N2 = xi
    N3 = eta

For linear triangles, shape function gradients are constant over the
element, enabling closed-form integration without quadrature.

Axisymmetric mode:
    (x, y) -> (r, z), multiply integrands by 2*pi*r_centroid
    r_centroid = (r_0 + r_1 + r_2) / 3
"""

import numpy as np
from .quadrature import gauss_line_2pt


def shape_functions_tri3(xi, eta):
    """
    Evaluate Tri3 shape functions at parametric point (xi, eta).

    N1 = 1 - xi - eta  (vertex 0)
    N2 = xi             (vertex 1)
    N3 = eta            (vertex 2)

    Parameters
    ----------
    xi : float
        First parametric coordinate (0 <= xi <= 1).
    eta : float
        Second parametric coordinate (0 <= eta <= 1-xi).

    Returns
    -------
    N : ndarray, shape (3,)
        Shape function values [N1, N2, N3].
    """
    return np.array([1.0 - xi - eta, xi, eta])


def shape_gradients_tri3(coords):
    """
    Compute shape function gradients in physical coordinates for Tri3.

    For linear triangles, the gradients are constant:
        dN/dx = [dN1/dx, dN2/dx, dN3/dx]
        dN/dy = [dN1/dy, dN2/dy, dN3/dy]

    Derived from the inverse Jacobian mapping:
        J = [[x2-x1, x3-x1],    dN/d(x,y) = J^{-T} * dN/d(xi,eta)
             [y2-y1, y3-y1]]

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Physical coordinates of the 3 nodes: [[x0,y0], [x1,y1], [x2,y2]].

    Returns
    -------
    dN : ndarray, shape (2, 3)
        Shape function gradients: dN[0,:] = dN_i/dx, dN[1,:] = dN_i/dy.
    area : float
        Area of the triangle (positive for CCW node ordering).

    Raises
    ------
    ValueError
        If the element has zero or negative area (degenerate triangle).
    """
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]

    # Jacobian determinant = 2 * area
    detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * detJ

    if abs(detJ) < 1e-30:
        raise ValueError(
            f"Degenerate triangle with near-zero area: {area:.2e}"
        )

    # Gradients of parametric shape functions: dN/d(xi) = [-1, 1, 0],
    # dN/d(eta) = [-1, 0, 1]
    # Physical gradients via inverse Jacobian:
    #   dN/dx = (1/detJ) * [ (y2-y0)*dN/dxi - (y1-y0)*dN/deta ]
    #   dN/dy = (1/detJ) * [-(x2-x0)*dN/dxi + (x1-x0)*dN/deta ]
    inv_detJ = 1.0 / detJ

    # dN_i/dxi  = [-1, 1, 0]
    # dN_i/deta = [-1, 0, 1]
    dy12 = y1 - y0  # dy for xi direction
    dy13 = y2 - y0  # dy for eta direction
    dx12 = x1 - x0
    dx13 = x2 - x0

    dN = np.empty((2, 3))

    # dN/dx = inv_detJ * [ dy13 * dN/dxi - dy12 * dN/deta ]
    dN[0, 0] = inv_detJ * (dy13 * (-1.0) - dy12 * (-1.0))  # = (dy13 - dy12) but with correct sign
    dN[0, 1] = inv_detJ * (dy13 * 1.0 - dy12 * 0.0)
    dN[0, 2] = inv_detJ * (dy13 * 0.0 - dy12 * 1.0)

    # dN/dy = inv_detJ * [-dx13 * dN/dxi + dx12 * dN/deta ]
    dN[1, 0] = inv_detJ * (-dx13 * (-1.0) + dx12 * (-1.0))
    dN[1, 1] = inv_detJ * (-dx13 * 1.0 + dx12 * 0.0)
    dN[1, 2] = inv_detJ * (-dx13 * 0.0 + dx12 * 1.0)

    return dN, area


def stiffness_scalar_tri3(coords, k, axisymmetric=False):
    """
    Element stiffness matrix for scalar diffusion/conduction on Tri3.

    Solves: -div(k * grad(u)) = f

    Cartesian:
        K_e = k * A * B^T * B
        where B = dN (2x3 gradient matrix), A = element area

    Axisymmetric (r-z):
        K_e = k * 2*pi*r_c * A * B^T * B
        where r_c = centroid radial coordinate

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates [[x0,y0], [x1,y1], [x2,y2]].
        In axisymmetric mode: [[r0,z0], [r1,z1], [r2,z2]].
    k : float
        Thermal conductivity or diffusion coefficient.
    axisymmetric : bool, optional
        If True, include 2*pi*r_centroid factor. Default False.

    Returns
    -------
    K_e : ndarray, shape (3, 3)
        Element stiffness matrix (symmetric positive semi-definite).
    """
    dN, area = shape_gradients_tri3(coords)

    # K_e = k * A * dN^T @ dN
    K_e = k * abs(area) * (dN.T @ dN)

    if axisymmetric:
        r_c = np.mean(coords[:, 0])
        # Clamp r_c to small positive value to handle axis elements
        r_c = max(r_c, 0.0)
        K_e *= 2.0 * np.pi * r_c

    return K_e


def mass_tri3(coords, rho=1.0, axisymmetric=False):
    """
    Consistent mass matrix for Tri3 element.

    Cartesian:
        M_e = rho * A/12 * [[2, 1, 1],
                             [1, 2, 1],
                             [1, 1, 2]]

    This is the exact integral of rho * N_i * N_j over the triangle
    for linear shape functions.

    Axisymmetric:
        M_e = rho * 2*pi*r_c * A/12 * [[2, 1, 1],
                                         [1, 2, 1],
                                         [1, 1, 2]]

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates.
    rho : float, optional
        Density or material coefficient. Default 1.0.
    axisymmetric : bool, optional
        If True, include 2*pi*r_centroid factor. Default False.

    Returns
    -------
    M_e : ndarray, shape (3, 3)
        Consistent mass matrix (symmetric positive definite).
    """
    _, area = shape_gradients_tri3(coords)
    A = abs(area)

    M_e = rho * A / 12.0 * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ])

    if axisymmetric:
        r_c = np.mean(coords[:, 0])
        r_c = max(r_c, 0.0)
        M_e *= 2.0 * np.pi * r_c

    return M_e


def load_vector_tri3(coords, q, axisymmetric=False):
    """
    Distributed load vector for Tri3 element.

    Cartesian:
        f_e = q * A/3 * [1, 1, 1]

    This is the exact integral of q * N_i over the triangle for
    constant source q and linear shape functions.

    Axisymmetric:
        f_e = q * 2*pi*r_c * A/3 * [1, 1, 1]

    Parameters
    ----------
    coords : ndarray, shape (3, 2)
        Node coordinates.
    q : float
        Uniform source term (e.g., volumetric heat generation W/m^3).
    axisymmetric : bool, optional
        If True, include 2*pi*r_centroid factor. Default False.

    Returns
    -------
    f_e : ndarray, shape (3,)
        Element load vector.
    """
    _, area = shape_gradients_tri3(coords)
    A = abs(area)

    f_e = q * A / 3.0 * np.ones(3)

    if axisymmetric:
        r_c = np.mean(coords[:, 0])
        r_c = max(r_c, 0.0)
        f_e *= 2.0 * np.pi * r_c

    return f_e


def boundary_mass_tri3(coords_edge, h, axisymmetric=False):
    """
    Boundary mass matrix for Robin (convection) BC on a Tri3 edge.

    Robin BC: k * dT/dn + h * T = h * T_inf
    This contributes h * integral(N_i * N_j) over the edge to the
    global stiffness matrix.

    Cartesian:
        H_e = h * L/6 * [[2, 1],
                          [1, 2]]
        where L = edge length.

    Axisymmetric:
        Uses 2-point Gauss quadrature with 2*pi*r at each point.

    Parameters
    ----------
    coords_edge : ndarray, shape (2, 2)
        Coordinates of the two edge nodes [[x_a, y_a], [x_b, y_b]].
    h : float
        Convection coefficient (W/m^2/K).
    axisymmetric : bool, optional
        If True, include 2*pi*r factor via numerical integration.

    Returns
    -------
    H_e : ndarray, shape (2, 2)
        Boundary mass matrix for the edge.
    """
    dx = coords_edge[1, 0] - coords_edge[0, 0]
    dy = coords_edge[1, 1] - coords_edge[0, 1]
    L = np.sqrt(dx * dx + dy * dy)

    if not axisymmetric:
        H_e = h * L / 6.0 * np.array([
            [2.0, 1.0],
            [1.0, 2.0],
        ])
    else:
        # Numerical integration with 2-point Gauss for 2*pi*r variation
        gp, gw = gauss_line_2pt()
        H_e = np.zeros((2, 2))
        for t, w in zip(gp, gw):
            N = np.array([1.0 - t, t])
            r = (1.0 - t) * coords_edge[0, 0] + t * coords_edge[1, 0]
            r = max(r, 0.0)
            # Jacobian of line parameterization = L
            H_e += w * h * 2.0 * np.pi * r * L * np.outer(N, N)

    return H_e


def boundary_load_tri3(coords_edge, h, T_inf, axisymmetric=False):
    """
    Boundary load vector for Robin (convection) BC on a Tri3 edge.

    Robin BC: k * dT/dn + h * T = h * T_inf
    This contributes h * T_inf * integral(N_i) over the edge to the
    global load vector.

    Cartesian:
        f_e = h * T_inf * L/2 * [1, 1]
        where L = edge length.

    Axisymmetric:
        Uses 2-point Gauss quadrature with 2*pi*r at each point.

    Parameters
    ----------
    coords_edge : ndarray, shape (2, 2)
        Coordinates of the two edge nodes.
    h : float
        Convection coefficient (W/m^2/K).
    T_inf : float
        Ambient/fluid temperature (K or C).
    axisymmetric : bool, optional
        If True, include 2*pi*r factor via numerical integration.

    Returns
    -------
    f_e : ndarray, shape (2,)
        Boundary load vector for the edge.
    """
    dx = coords_edge[1, 0] - coords_edge[0, 0]
    dy = coords_edge[1, 1] - coords_edge[0, 1]
    L = np.sqrt(dx * dx + dy * dy)

    if not axisymmetric:
        f_e = h * T_inf * L / 2.0 * np.ones(2)
    else:
        gp, gw = gauss_line_2pt()
        f_e = np.zeros(2)
        for t, w in zip(gp, gw):
            N = np.array([1.0 - t, t])
            r = (1.0 - t) * coords_edge[0, 0] + t * coords_edge[1, 0]
            r = max(r, 0.0)
            f_e += w * h * T_inf * 2.0 * np.pi * r * L * N

    return f_e
