"""
6-node quadratic triangle element for 2D elasticity.

Used for structural and thermo-mechanical analysis of the 40 MWth
Marine MSR vessel, internals, and support structures.

Node numbering (counter-clockwise, mid-edge nodes follow corners):

    2
    |\\
    5  4
    |    \\
    0--3--1

    Corner nodes: 0, 1, 2
    Mid-edge nodes: 3 (between 0-1), 4 (between 1-2), 5 (between 2-0)

Parametric coordinates (xi, eta):
    Node 0: (0, 0)      Node 3: (0.5, 0)
    Node 1: (1, 0)      Node 4: (0.5, 0.5)
    Node 2: (0, 1)      Node 5: (0, 0.5)

Area coordinates:
    L1 = 1 - xi - eta,  L2 = xi,  L3 = eta

Shape functions (quadratic):
    N0 = L1 * (2*L1 - 1)
    N1 = L2 * (2*L2 - 1)
    N2 = L3 * (2*L3 - 1)
    N3 = 4 * L1 * L2
    N4 = 4 * L2 * L3
    N5 = 4 * L3 * L1

DOF ordering for elasticity: [u0, v0, u1, v1, u2, v2, u3, v3, u4, v4, u5, v5]
"""

import numpy as np
from .quadrature import gauss_triangle_3pt, gauss_triangle_7pt


def shape_functions_tri6(xi, eta):
    """
    Evaluate Tri6 shape functions at parametric point (xi, eta).

    Parameters
    ----------
    xi : float
        First parametric coordinate.
    eta : float
        Second parametric coordinate.

    Returns
    -------
    N : ndarray, shape (6,)
        Shape function values [N0, N1, N2, N3, N4, N5].
    """
    L1 = 1.0 - xi - eta
    L2 = xi
    L3 = eta

    N = np.array([
        L1 * (2.0 * L1 - 1.0),  # N0 - corner (0,0)
        L2 * (2.0 * L2 - 1.0),  # N1 - corner (1,0)
        L3 * (2.0 * L3 - 1.0),  # N2 - corner (0,1)
        4.0 * L1 * L2,          # N3 - mid-edge 0-1
        4.0 * L2 * L3,          # N4 - mid-edge 1-2
        4.0 * L3 * L1,          # N5 - mid-edge 2-0
    ])
    return N


def shape_gradients_tri6(xi, eta):
    """
    Evaluate Tri6 shape function gradients in parametric space.

    dN/d(xi) and dN/d(eta) for all 6 nodes.

    Parameters
    ----------
    xi : float
        First parametric coordinate.
    eta : float
        Second parametric coordinate.

    Returns
    -------
    dN_dxi : ndarray, shape (2, 6)
        dN_dxi[0, :] = dN_i/d(xi), dN_dxi[1, :] = dN_i/d(eta).
    """
    L1 = 1.0 - xi - eta
    L2 = xi
    L3 = eta

    # Derivatives of area coordinates w.r.t. (xi, eta):
    # dL1/dxi = -1,  dL1/deta = -1
    # dL2/dxi =  1,  dL2/deta =  0
    # dL3/dxi =  0,  dL3/deta =  1

    dN_dxi = np.zeros((2, 6))

    # dN0/dxi = d[L1*(2L1-1)]/dxi = (2L1-1)*dL1/dxi + L1*2*dL1/dxi
    #         = dL1/dxi * (4L1 - 1) = -1 * (4L1 - 1)
    dN_dxi[0, 0] = -(4.0 * L1 - 1.0)
    dN_dxi[1, 0] = -(4.0 * L1 - 1.0)

    # dN1/dxi = dL2/dxi * (4L2 - 1) = 1 * (4L2 - 1)
    dN_dxi[0, 1] = 4.0 * L2 - 1.0
    dN_dxi[1, 1] = 0.0

    # dN2/dxi = dL3/dxi * (4L3 - 1) = 0
    dN_dxi[0, 2] = 0.0
    dN_dxi[1, 2] = 4.0 * L3 - 1.0

    # dN3/dxi = d[4*L1*L2]/dxi = 4*(dL1/dxi*L2 + L1*dL2/dxi) = 4*(-L2 + L1)
    dN_dxi[0, 3] = 4.0 * (L1 - L2)    # = 4*(-xi + (1-xi-eta)) = 4*(1 - 2xi - eta)
    dN_dxi[1, 3] = -4.0 * L2           # = 4*(-1*xi + L1*0) = -4*xi

    # dN4/dxi = d[4*L2*L3]/dxi = 4*(dL2/dxi*L3 + L2*dL3/dxi) = 4*(L3)
    dN_dxi[0, 4] = 4.0 * L3            # = 4*eta
    dN_dxi[1, 4] = 4.0 * L2            # = 4*xi

    # dN5/dxi = d[4*L3*L1]/dxi = 4*(dL3/dxi*L1 + L3*dL1/dxi) = 4*(0*L1 + L3*(-1))
    dN_dxi[0, 5] = -4.0 * L3           # = -4*eta
    dN_dxi[1, 5] = 4.0 * (L1 - L3)    # = 4*((1-xi-eta) - eta) = 4*(1-xi-2eta)

    return dN_dxi


def jacobian_tri6(xi, eta, coords):
    """
    Compute the Jacobian matrix and its determinant for a Tri6 element.

    J = dN_dxi @ coords = [[dx/dxi, dy/dxi],
                            [dx/deta, dy/deta]]

    Parameters
    ----------
    xi : float
        First parametric coordinate.
    eta : float
        Second parametric coordinate.
    coords : ndarray, shape (6, 2)
        Physical coordinates of all 6 nodes.

    Returns
    -------
    J : ndarray, shape (2, 2)
        Jacobian matrix.
    detJ : float
        Determinant of the Jacobian (must be positive for valid elements).

    Raises
    ------
    ValueError
        If the Jacobian determinant is non-positive.
    """
    dN_dxi = shape_gradients_tri6(xi, eta)  # (2, 6)
    J = dN_dxi @ coords  # (2, 2)
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    if detJ <= 0.0:
        raise ValueError(
            f"Non-positive Jacobian determinant: {detJ:.6e}. "
            "Check element node ordering (must be CCW) and mid-node positions."
        )

    return J, detJ


def B_matrix_tri6(xi, eta, coords):
    """
    Compute the strain-displacement matrix B for a Tri6 element.

    For plane strain/stress, the strain vector is:
        {eps_xx, eps_yy, gamma_xy}^T = B @ {u0, v0, u1, v1, ..., u5, v5}^T

    B has shape (3, 12) and is assembled from:
        B_i = [[dN_i/dx,    0   ],
               [   0,    dN_i/dy ],
               [dN_i/dy, dN_i/dx ]]

    Parameters
    ----------
    xi : float
        First parametric coordinate.
    eta : float
        Second parametric coordinate.
    coords : ndarray, shape (6, 2)
        Physical coordinates of all 6 nodes.

    Returns
    -------
    B : ndarray, shape (3, 12)
        Strain-displacement matrix.
    detJ : float
        Jacobian determinant at (xi, eta).
    """
    dN_dxi = shape_gradients_tri6(xi, eta)  # (2, 6)
    J, detJ = jacobian_tri6(xi, eta, coords)

    # Inverse Jacobian
    inv_detJ = 1.0 / detJ
    Jinv = inv_detJ * np.array([
        [ J[1, 1], -J[0, 1]],
        [-J[1, 0],  J[0, 0]],
    ])

    # Physical gradients: dN_dx = Jinv @ dN_dxi  -> (2, 6)
    dN_dx = Jinv @ dN_dxi  # dN_dx[0,:] = dN_i/dx, dN_dx[1,:] = dN_i/dy

    # Assemble B matrix (3 x 12)
    B = np.zeros((3, 12))
    for i in range(6):
        col = 2 * i
        B[0, col]     = dN_dx[0, i]   # dN_i/dx
        B[1, col + 1] = dN_dx[1, i]   # dN_i/dy
        B[2, col]     = dN_dx[1, i]   # dN_i/dy
        B[2, col + 1] = dN_dx[0, i]   # dN_i/dx

    return B, detJ


def D_matrix_plane_strain(E, nu):
    """
    Constitutive (material stiffness) matrix for plane strain.

    sigma = D * epsilon, where:
        D = E / ((1+nu)*(1-2*nu)) * [[1-nu,  nu,    0          ],
                                      [nu,    1-nu,  0          ],
                                      [0,     0,     (1-2*nu)/2 ]]

    Parameters
    ----------
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio (dimensionless, 0 < nu < 0.5).

    Returns
    -------
    D : ndarray, shape (3, 3)
        Plane strain constitutive matrix.
    """
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = factor * np.array([
        [1.0 - nu, nu,       0.0              ],
        [nu,       1.0 - nu, 0.0              ],
        [0.0,      0.0,      (1.0 - 2.0 * nu) / 2.0],
    ])
    return D


def D_matrix_plane_stress(E, nu):
    """
    Constitutive (material stiffness) matrix for plane stress.

    sigma = D * epsilon, where:
        D = E / (1 - nu^2) * [[1,  nu,  0        ],
                               [nu, 1,   0        ],
                               [0,  0,   (1-nu)/2 ]]

    Parameters
    ----------
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio (dimensionless, 0 < nu < 0.5).

    Returns
    -------
    D : ndarray, shape (3, 3)
        Plane stress constitutive matrix.
    """
    factor = E / (1.0 - nu * nu)
    D = factor * np.array([
        [1.0, nu,  0.0            ],
        [nu,  1.0, 0.0            ],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
    return D


def stiffness_elastic_tri6(coords, E, nu, plane='strain'):
    """
    Element stiffness matrix for 2D elasticity on Tri6.

    K_e = integral over element of B^T * D * B * t dA

    Uses 3-point Gauss quadrature (exact for quadratic elements with
    linear Jacobian; sufficient for isoparametric Tri6).

    For plane strain, unit thickness is assumed (t=1).

    Parameters
    ----------
    coords : ndarray, shape (6, 2)
        Physical coordinates of all 6 nodes.
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    plane : str, optional
        'strain' for plane strain, 'stress' for plane stress.
        Default 'strain'.

    Returns
    -------
    K_e : ndarray, shape (12, 12)
        Element stiffness matrix (symmetric positive semi-definite).
    """
    if plane == 'strain':
        D = D_matrix_plane_strain(E, nu)
    elif plane == 'stress':
        D = D_matrix_plane_stress(E, nu)
    else:
        raise ValueError(f"Unknown plane type: '{plane}'. Use 'strain' or 'stress'.")

    gpts, gwts = gauss_triangle_3pt()
    K_e = np.zeros((12, 12))

    for pt, wt in zip(gpts, gwts):
        xi, eta = pt
        B, detJ = B_matrix_tri6(xi, eta, coords)
        # K_e += wt * detJ * B^T @ D @ B
        # Note: wt already includes the 1/2 factor for the reference triangle,
        # and detJ maps from reference to physical coordinates.
        # The integral in physical space: integral f dA = integral f * detJ * d(xi,eta)
        # = sum_i w_i * f(xi_i, eta_i) * detJ(xi_i, eta_i)
        # But our weights already include the 1/2, so:
        # sum_i w_i * f_i * detJ_i  (with w_i having the 1/2 built in for ref tri)
        # Actually: integral over ref tri = sum w_i * f_i, where w_i includes 1/2.
        # Physical integral = integral f * detJ d(xi,eta) over ref domain
        # = sum w_i * f_i * detJ_i (w_i includes 1/2 for ref domain)
        # BUT our quadrature is: sum w_i * f_i = integral f dA_ref
        # So physical integral = sum w_i * f(xi_i) * detJ_i is WRONG because
        # w_i already accounts for the 1/2.
        # Correct: sum w_i * f_i * detJ_i (our weights include the 1/2 Jacobian
        # from parametric to reference, and detJ handles reference to physical).
        K_e += wt * B.T @ D @ B * detJ

    return K_e


def thermal_load_tri6(coords, E, nu, alpha, dT_nodes, plane='strain'):
    """
    Element thermal load vector for Tri6.

    Thermal strain: eps_th = alpha * dT * {1, 1, 0}^T

    Thermal stress (plane strain):
        sigma_th = -D * eps_th
        Note: For plane strain, eps_zz is constrained, creating thermal
        stress in z-direction as well.

    Load vector:
        f_th = integral B^T * D * eps_th dA

    Uses 3-point Gauss quadrature. Temperature dT is interpolated
    from nodal values using Tri6 shape functions.

    Parameters
    ----------
    coords : ndarray, shape (6, 2)
        Physical coordinates of all 6 nodes.
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    alpha : float
        Coefficient of thermal expansion (1/K).
    dT_nodes : ndarray, shape (6,)
        Temperature change at each node (K or C, from reference temperature).
    plane : str, optional
        'strain' or 'stress'. Default 'strain'.

    Returns
    -------
    f_th : ndarray, shape (12,)
        Element thermal load vector.
    """
    if plane == 'strain':
        D = D_matrix_plane_strain(E, nu)
    elif plane == 'stress':
        D = D_matrix_plane_stress(E, nu)
    else:
        raise ValueError(f"Unknown plane type: '{plane}'. Use 'strain' or 'stress'.")

    eps_th_unit = np.array([1.0, 1.0, 0.0])  # thermal strain direction

    gpts, gwts = gauss_triangle_3pt()
    f_th = np.zeros(12)

    for pt, wt in zip(gpts, gwts):
        xi, eta = pt
        B, detJ = B_matrix_tri6(xi, eta, coords)
        N = shape_functions_tri6(xi, eta)

        # Interpolate temperature change at Gauss point
        dT = N @ dT_nodes

        # Thermal strain at this point
        eps_th = alpha * dT * eps_th_unit

        # Contribution to load vector
        f_th += wt * B.T @ D @ eps_th * detJ

    return f_th


def mass_tri6(coords, rho=1.0):
    """
    Consistent mass matrix for Tri6 element (2D elasticity DOFs).

    M_e = integral rho * N^T * N dA

    where N is the (2, 12) shape function matrix:
        N = [[N0, 0, N1, 0, ..., N5, 0 ],
             [0, N0, 0, N1, ..., 0,  N5]]

    Uses 7-point Gauss quadrature for accurate integration of the
    quartic integrand (product of two quadratic shape functions).

    Parameters
    ----------
    coords : ndarray, shape (6, 2)
        Physical coordinates of all 6 nodes.
    rho : float, optional
        Material density (kg/m^3). Default 1.0.

    Returns
    -------
    M_e : ndarray, shape (12, 12)
        Consistent mass matrix (symmetric positive definite).
    """
    gpts, gwts = gauss_triangle_7pt()
    M_e = np.zeros((12, 12))

    for pt, wt in zip(gpts, gwts):
        xi, eta = pt
        N_vals = shape_functions_tri6(xi, eta)  # (6,)
        _, detJ = jacobian_tri6(xi, eta, coords)

        # Build the (2, 12) interpolation matrix
        N_mat = np.zeros((2, 12))
        for i in range(6):
            N_mat[0, 2 * i]     = N_vals[i]
            N_mat[1, 2 * i + 1] = N_vals[i]

        M_e += wt * rho * N_mat.T @ N_mat * detJ

    return M_e
