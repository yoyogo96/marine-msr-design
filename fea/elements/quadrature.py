"""
Gauss quadrature rules for triangular and line elements.

All triangle rules use the parametric coordinates (xi, eta) where the
area coordinates are:
    L1 = 1 - xi - eta
    L2 = xi
    L3 = eta

The reference triangle has vertices at (0,0), (1,0), (0,1) with area 1/2.
Weights include the 1/2 factor (area of reference triangle), so that:

    integral over ref triangle of f dA = sum_i w_i * f(xi_i, eta_i)

References:
    - Dunavant, D.A. "High degree efficient symmetrical Gaussian
      quadrature rules for the triangle." IJNME, 21(6), 1985.
    - Hammer, P.C. et al. "Numerical integration over simplexes
      and cones." Math Tables Aids Comput., 10(55), 1956.
"""

import numpy as np


def gauss_triangle_1pt():
    """
    1-point Gauss quadrature for triangle (exact for degree 1 polynomials).

    The single point is at the centroid (1/3, 1/3).
    Weight = 1/2 (area of reference triangle).

    Returns
    -------
    points : ndarray, shape (1, 2)
        Quadrature points in (xi, eta) coordinates.
    weights : ndarray, shape (1,)
        Quadrature weights (include 1/2 factor).
    """
    points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    weights = np.array([0.5])
    return points, weights


def gauss_triangle_3pt():
    """
    3-point Gauss quadrature for triangle (exact for degree 2 polynomials).

    Points at the midpoints of each edge in area coordinates:
        (1/2, 1/2, 0), (0, 1/2, 1/2), (1/2, 0, 1/2)
    Converted to (xi, eta):
        (1/6, 1/6), (2/3, 1/6), (1/6, 2/3)
    Each weight = 1/6 (= (1/3) * (1/2), where 1/3 is the barycentric
    weight and 1/2 is the reference triangle area).

    Returns
    -------
    points : ndarray, shape (3, 2)
        Quadrature points in (xi, eta) coordinates.
    weights : ndarray, shape (3,)
        Quadrature weights (include 1/2 factor).
    """
    points = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ])
    weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    return points, weights


def gauss_triangle_7pt():
    """
    7-point Gauss quadrature for triangle (exact for degree 5 polynomials).

    Uses the Hammer-Stroud rule with 3 symmetry orbits:
        - 1 point at centroid
        - 3 points at one orbit (a1, b1, b1) and permutations
        - 3 points at another orbit (a2, b2, b2) and permutations

    Area coordinates and weights from Dunavant (1985):
        Centroid weight: 0.225
        Orbit 1: a1 = 0.059715871789770, w1 = 0.132394152788506
        Orbit 2: a2 = 0.797426985353087, w2 = 0.125939180544827

    Weights are multiplied by 1/2 for the reference triangle area.

    Returns
    -------
    points : ndarray, shape (7, 2)
        Quadrature points in (xi, eta) coordinates.
    weights : ndarray, shape (7,)
        Quadrature weights (include 1/2 factor).
    """
    # Centroid
    p0 = [1.0 / 3.0, 1.0 / 3.0]
    w0 = 0.225

    # Orbit 1: area coords (a1, b1, b1) with b1 = (1 - a1) / 2
    a1 = 0.059715871789770
    b1 = 0.470142064105115  # (1 - a1) / 2
    w1 = 0.132394152788506
    # Permutations in (xi=L2, eta=L3) coordinates
    # (L1, L2, L3) = (a1, b1, b1) -> (xi, eta) = (b1, b1)
    # (L1, L2, L3) = (b1, a1, b1) -> (xi, eta) = (a1, b1)
    # (L1, L2, L3) = (b1, b1, a1) -> (xi, eta) = (b1, a1)
    p1 = [b1, b1]
    p2 = [a1, b1]
    p3 = [b1, a1]

    # Orbit 2: area coords (a2, b2, b2) with b2 = (1 - a2) / 2
    a2 = 0.797426985353087
    b2 = 0.101286507323456  # (1 - a2) / 2
    w2 = 0.125939180544827
    p4 = [b2, b2]
    p5 = [a2, b2]
    p6 = [b2, a2]

    points = np.array([p0, p1, p2, p3, p4, p5, p6])
    # Multiply all weights by 1/2 (reference triangle area)
    weights = np.array([w0, w1, w1, w1, w2, w2, w2]) * 0.5

    return points, weights


def gauss_line_2pt():
    """
    2-point Gauss quadrature on the reference line segment [0, 1].

    Mapped from the standard interval [-1, 1]:
        t = (1 + s) / 2,   dt = ds / 2
        s_i = +/- 1/sqrt(3),  w_i = 1 (on [-1,1])

    On [0, 1]:
        t_i = (1 +/- 1/sqrt(3)) / 2
        w_i = 1/2

    Used for boundary integrals on triangle edges parameterized by t in [0,1]:
        x(t) = (1-t) * x_a + t * x_b

    Returns
    -------
    points : ndarray, shape (2,)
        Quadrature points on [0, 1].
    weights : ndarray, shape (2,)
        Quadrature weights (include 1/2 Jacobian factor).
    """
    s = 1.0 / np.sqrt(3.0)
    points = np.array([0.5 * (1.0 - s), 0.5 * (1.0 + s)])
    weights = np.array([0.5, 0.5])
    return points, weights
