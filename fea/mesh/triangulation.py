"""
Triangular mesh generation using scipy.spatial.Delaunay with post-processing.

Provides constrained Delaunay-like triangulation for 2D finite element
meshes of the 40 MWth Marine MSR. The workflow is:

    1. Generate boundary and interior points for the domain
    2. Perform unconstrained Delaunay triangulation via scipy
    3. Post-process: remove triangles outside the domain or crossing
       internal boundary segments (e.g., channel walls)

This is NOT a true constrained Delaunay triangulation (CDT), but the
post-processing steps produce equivalent results for our reactor
geometry cases where:
    - Boundaries are well-sampled (many points on each boundary)
    - No extreme aspect ratios in the boundary point distribution

For production quality CDT, consider the `triangle` library (Shewchuk).
This implementation avoids that external dependency.

Conventions:
    - All angles in radians unless otherwise noted
    - Node ordering: counter-clockwise for positive area
    - Coordinate system: Cartesian (x, y) for unit-cell and rosette meshes
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Optional


def delaunay_with_boundary(points: np.ndarray,
                           boundary_segments: Optional[List[Tuple[int, int]]] = None
                           ) -> np.ndarray:
    """Constrained Delaunay-like triangulation.

    Uses scipy.spatial.Delaunay for base triangulation, then:
    1. Remove triangles whose edges cross boundary segments
    2. Optionally remove triangles outside the domain

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Node coordinates (x, y).
    boundary_segments : list of (i, j), optional
        Index pairs defining constrained edges that triangles must not
        cross. Typically the channel wall circle or hexagon perimeter.

    Returns
    -------
    connectivity : ndarray, shape (N_elem, 3)
        Triangle connectivity array (0-indexed node indices).
        Triangles are oriented counter-clockwise.
    """
    tri = Delaunay(points)
    simplices = tri.simplices.copy()

    # Ensure CCW orientation
    simplices = _orient_ccw(simplices, points)

    # Remove triangles crossing boundary segments
    if boundary_segments is not None and len(boundary_segments) > 0:
        simplices = remove_crossing_triangles(simplices, points,
                                              boundary_segments)

    return simplices


def remove_exterior_triangles(simplices: np.ndarray,
                              points: np.ndarray,
                              boundary_polygon: np.ndarray) -> np.ndarray:
    """Remove triangles with centroids outside the domain boundary.

    Uses a ray-casting (point-in-polygon) test on each triangle centroid.

    Parameters
    ----------
    simplices : ndarray, shape (N_tri, 3)
        Triangle connectivity.
    points : ndarray, shape (N, 2)
        Node coordinates.
    boundary_polygon : ndarray, shape (M, 2)
        Ordered vertices of the outer boundary polygon (closed or open;
        the first vertex is NOT repeated at the end).

    Returns
    -------
    filtered : ndarray, shape (N_kept, 3)
        Triangles whose centroids lie inside the polygon.
    """
    centroids = np.mean(points[simplices], axis=1)  # (N_tri, 2)
    mask = np.array([_point_in_polygon(c, boundary_polygon)
                     for c in centroids])
    return simplices[mask]


def remove_crossing_triangles(simplices: np.ndarray,
                              points: np.ndarray,
                              boundary_segments: List[Tuple[int, int]]
                              ) -> np.ndarray:
    """Remove triangles whose edges cross internal boundary segments.

    A triangle is removed if any of its three edges properly intersects
    (crosses, not just touches at an endpoint) a boundary segment that
    is NOT the same edge.

    Parameters
    ----------
    simplices : ndarray, shape (N_tri, 3)
        Triangle connectivity.
    points : ndarray, shape (N, 2)
        Node coordinates.
    boundary_segments : list of (i, j)
        Constrained edges that must not be crossed.

    Returns
    -------
    filtered : ndarray, shape (N_kept, 3)
        Triangles that do not cross any boundary segment.
    """
    seg_set = set()
    for i, j in boundary_segments:
        seg_set.add((min(i, j), max(i, j)))

    keep = np.ones(len(simplices), dtype=bool)
    for t_idx, tri in enumerate(simplices):
        # Extract the 3 edges of this triangle
        edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0])),
        ]
        for seg_i, seg_j in boundary_segments:
            seg_key = (min(seg_i, seg_j), max(seg_i, seg_j))
            # If the boundary segment IS one of the triangle edges, skip
            if seg_key in edges:
                continue
            # Check if any triangle edge crosses this boundary segment
            for e0, e1 in edges:
                if _segments_cross(points[e0], points[e1],
                                   points[seg_i], points[seg_j]):
                    keep[t_idx] = False
                    break
            if not keep[t_idx]:
                break

    return simplices[keep]


def remove_triangles_outside_circle(simplices: np.ndarray,
                                    points: np.ndarray,
                                    center: np.ndarray,
                                    radius: float,
                                    keep_inside: bool = True) -> np.ndarray:
    """Remove triangles based on centroid distance from a circle.

    Parameters
    ----------
    simplices : ndarray, shape (N_tri, 3)
        Triangle connectivity.
    points : ndarray, shape (N, 2)
        Node coordinates.
    center : ndarray, shape (2,)
        Circle center.
    radius : float
        Circle radius.
    keep_inside : bool
        If True, keep triangles with centroid inside the circle.
        If False, keep triangles with centroid outside the circle.

    Returns
    -------
    filtered : ndarray, shape (N_kept, 3)
    """
    centroids = np.mean(points[simplices], axis=1)
    dist = np.sqrt(np.sum((centroids - center) ** 2, axis=1))
    if keep_inside:
        mask = dist <= radius * (1.0 + 1e-10)
    else:
        mask = dist > radius * (1.0 - 1e-10)
    return simplices[mask]


def generate_circle_points(center: np.ndarray, radius: float,
                           n_points: int) -> np.ndarray:
    """Generate evenly spaced points on a circle.

    Parameters
    ----------
    center : ndarray, shape (2,)
        Circle center (x, y).
    radius : float
        Circle radius.
    n_points : int
        Number of points (equally spaced in angle).

    Returns
    -------
    pts : ndarray, shape (n_points, 2)
        Points on the circle, ordered counter-clockwise starting
        from angle 0 (positive x-direction).
    """
    center = np.asarray(center, dtype=np.float64)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
    ])
    return pts


def generate_hex_points(center: np.ndarray, pitch: float,
                        n_per_side: int) -> np.ndarray:
    """Generate points on hexagon boundary (flat-to-flat pitch).

    The hexagon is oriented with flats at top and bottom (flat-topped),
    which is the standard orientation for fuel element lattices.

    Flat-to-flat distance = pitch.
    Vertex-to-vertex distance = pitch / cos(30) = pitch * 2/sqrt(3).

    Parameters
    ----------
    center : ndarray, shape (2,)
        Hexagon center (x, y).
    pitch : float
        Flat-to-flat distance (m).
    n_per_side : int
        Number of points per hexagon side (including both endpoints
        of the side, so minimum is 2).

    Returns
    -------
    pts : ndarray, shape (6 * (n_per_side - 1), 2)
        Points on the hexagon boundary, ordered counter-clockwise.
        The 6 corner points are included; each side has n_per_side
        points but the last point of each side equals the first point
        of the next side, so only n_per_side - 1 unique points per side.
    """
    center = np.asarray(center, dtype=np.float64)
    # Circumradius (center to vertex)
    R = pitch / np.sqrt(3.0)

    # Hexagon vertices (flat-topped: first vertex at 30 degrees)
    # For flat-topped hex: vertices at 30, 90, 150, 210, 270, 330 degrees
    vertex_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    vertices = np.column_stack([
        center[0] + R * np.cos(vertex_angles),
        center[1] + R * np.sin(vertex_angles),
    ])

    # Generate points along each side
    pts_list = []
    n_sides = 6
    for i in range(n_sides):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % n_sides]
        # Parameterize: n_per_side points from v0 to v1, excluding v1
        t = np.linspace(0, 1, n_per_side, endpoint=False)
        side_pts = v0[np.newaxis, :] + t[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
        pts_list.append(side_pts)

    pts = np.vstack(pts_list)
    return pts


def generate_radial_points(center: np.ndarray, r_inner: float,
                           r_outer: float, n_circ: int,
                           n_radial: int,
                           include_inner: bool = False,
                           include_outer: bool = False) -> np.ndarray:
    """Generate points on concentric circles between two radii.

    Useful for filling the annular region between a channel wall
    and the hexagonal boundary with interior points.

    Parameters
    ----------
    center : ndarray, shape (2,)
        Center point.
    r_inner : float
        Inner radius.
    r_outer : float
        Outer radius.
    n_circ : int
        Number of points per ring (circumferential).
    n_radial : int
        Number of radial layers (excluding inner and outer boundaries).
    include_inner : bool
        If True, include points on the inner radius.
    include_outer : bool
        If True, include points on the outer radius.

    Returns
    -------
    pts : ndarray, shape (N, 2)
        Interior points on concentric circles.
    """
    center = np.asarray(center, dtype=np.float64)
    pts_list = []

    # Determine radial positions
    if include_inner and include_outer:
        radii = np.linspace(r_inner, r_outer, n_radial + 2)
    elif include_inner:
        radii = np.linspace(r_inner, r_outer, n_radial + 2)[:-1]
    elif include_outer:
        radii = np.linspace(r_inner, r_outer, n_radial + 2)[1:]
    else:
        radii = np.linspace(r_inner, r_outer, n_radial + 2)[1:-1]

    for i, r in enumerate(radii):
        # Offset alternate rings by half an angular step for better triangulation
        offset = 0.0 if i % 2 == 0 else np.pi / n_circ
        theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False) + offset
        ring = np.column_stack([
            center[0] + r * np.cos(theta),
            center[1] + r * np.sin(theta),
        ])
        pts_list.append(ring)

    if len(pts_list) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(pts_list)


def assign_materials_by_circles(simplices: np.ndarray,
                                points: np.ndarray,
                                circles: List[Tuple[np.ndarray, float]],
                                default_material: int = 1) -> np.ndarray:
    """Assign material IDs based on whether centroids fall inside circles.

    Each circle defines a material-0 region (fuel channel). Everything
    else gets the default material ID (graphite).

    Parameters
    ----------
    simplices : ndarray, shape (N_tri, 3)
        Triangle connectivity.
    points : ndarray, shape (N, 2)
        Node coordinates.
    circles : list of (center, radius)
        Each entry is (ndarray(2,), float) defining a fuel channel.
    default_material : int
        Material ID for elements outside all circles.

    Returns
    -------
    material_ids : ndarray, shape (N_tri,)
        Integer material zone ID for each triangle.
    """
    centroids = np.mean(points[simplices], axis=1)  # (N_tri, 2)
    material_ids = np.full(len(simplices), default_material, dtype=np.int64)

    for center, radius in circles:
        center = np.asarray(center, dtype=np.float64)
        dist = np.sqrt(np.sum((centroids - center) ** 2, axis=1))
        inside = dist <= radius * (1.0 + 1e-10)
        material_ids[inside] = 0  # fuel salt

    return material_ids


def extract_boundary_edges(simplices: np.ndarray,
                           boundary_node_indices: np.ndarray
                           ) -> List[Tuple[int, int]]:
    """Extract boundary edges from triangulation that lie on a given boundary.

    An edge is a boundary edge if BOTH its endpoints are in the
    boundary node set AND it appears as an edge of some triangle.

    Parameters
    ----------
    simplices : ndarray, shape (N_tri, 3)
        Triangle connectivity.
    boundary_node_indices : ndarray
        Node indices belonging to this boundary.

    Returns
    -------
    edges : list of (int, int)
        Boundary edges as (node_i, node_j) pairs.
    """
    bnd_set = set(boundary_node_indices.tolist())
    edge_list = []
    seen = set()

    for tri in simplices:
        for i in range(3):
            n0 = int(tri[i])
            n1 = int(tri[(i + 1) % 3])
            if n0 in bnd_set and n1 in bnd_set:
                key = (min(n0, n1), max(n0, n1))
                if key not in seen:
                    seen.add(key)
                    edge_list.append((n0, n1))

    return edge_list


def extract_exterior_edges(simplices: np.ndarray) -> List[Tuple[int, int]]:
    """Extract edges that appear in exactly one triangle (mesh boundary).

    Parameters
    ----------
    simplices : ndarray, shape (N_tri, 3)
        Triangle connectivity.

    Returns
    -------
    edges : list of (int, int)
        Exterior boundary edges.
    """
    edge_count = {}
    for tri in simplices:
        for i in range(3):
            n0 = int(tri[i])
            n1 = int(tri[(i + 1) % 3])
            key = (min(n0, n1), max(n0, n1))
            edge_count[key] = edge_count.get(key, 0) + 1

    return [(e[0], e[1]) for e, cnt in edge_count.items() if cnt == 1]


# =========================================================================
# Private helper functions
# =========================================================================

def _orient_ccw(simplices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Ensure all triangles have counter-clockwise orientation.

    Flips the node ordering (swaps nodes 1 and 2) for any triangle
    with negative signed area (clockwise).

    Parameters
    ----------
    simplices : ndarray, shape (N_tri, 3)
    points : ndarray, shape (N, 2)

    Returns
    -------
    simplices : ndarray, shape (N_tri, 3)
        Reoriented connectivity (modified in place and returned).
    """
    v0 = points[simplices[:, 0]]
    v1 = points[simplices[:, 1]]
    v2 = points[simplices[:, 2]]
    # Signed area = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
    cross = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
             (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    # Flip CW triangles
    cw_mask = cross < 0
    simplices[cw_mask, 1], simplices[cw_mask, 2] = (
        simplices[cw_mask, 2].copy(), simplices[cw_mask, 1].copy()
    )
    return simplices


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test.

    Casts a ray from the point in the +x direction and counts crossings
    with polygon edges. Odd number of crossings = inside.

    Parameters
    ----------
    point : ndarray, shape (2,)
    polygon : ndarray, shape (M, 2)
        Ordered vertices (last edge connects polygon[-1] to polygon[0]).

    Returns
    -------
    inside : bool
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = polygon[i, 1], polygon[j, 1]
        xi, xj = polygon[i, 0], polygon[j, 0]
        if ((yi > y) != (yj > y)):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _segments_cross(p1: np.ndarray, p2: np.ndarray,
                    p3: np.ndarray, p4: np.ndarray) -> bool:
    """Test if line segments (p1,p2) and (p3,p4) properly cross.

    Properly cross means they intersect at an interior point of both
    segments (not at endpoints). This avoids false positives when
    segments share an endpoint.

    Uses the signed-area orientation test.

    Parameters
    ----------
    p1, p2 : ndarray, shape (2,)
        Endpoints of segment 1.
    p3, p4 : ndarray, shape (2,)
        Endpoints of segment 2.

    Returns
    -------
    cross : bool
        True if the segments properly cross each other.
    """
    d1 = _cross2d(p3, p4, p1)
    d2 = _cross2d(p3, p4, p2)
    d3 = _cross2d(p1, p2, p3)
    d4 = _cross2d(p1, p2, p4)

    # Segments properly cross if the endpoints of each segment are on
    # opposite sides of the line containing the other segment.
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)):
        if ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

    return False


def _cross2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed area of triangle (a, b, c) times 2.

    Positive if c is to the left of line a->b (CCW).
    Negative if c is to the right (CW).
    Zero if collinear.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_hexagon(point: np.ndarray, center: np.ndarray,
                      pitch: float) -> bool:
    """Test if a point is inside a flat-topped hexagon.

    For a flat-topped hexagon with flat-to-flat distance = pitch,
    the apothem (center to flat) = pitch / 2.

    A point is inside if for all 3 pairs of parallel flats, the
    point's projected distance from center is <= apothem.

    Parameters
    ----------
    point : ndarray, shape (2,)
    center : ndarray, shape (2,)
    pitch : float
        Flat-to-flat distance.

    Returns
    -------
    inside : bool
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    apothem = pitch / 2.0

    # For a flat-topped hexagon, the 3 normal directions to the flats are:
    # (0, 1), (sqrt(3)/2, 1/2), (sqrt(3)/2, -1/2)
    s3_2 = np.sqrt(3.0) / 2.0

    # Check all 3 directions
    if abs(dy) > apothem * (1.0 + 1e-10):
        return False
    d2 = abs(s3_2 * dx + 0.5 * dy)
    if d2 > apothem * (1.0 + 1e-10):
        return False
    d3 = abs(s3_2 * dx - 0.5 * dy)
    if d3 > apothem * (1.0 + 1e-10):
        return False

    return True
