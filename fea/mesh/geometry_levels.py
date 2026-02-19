"""
Geometry builders for the 40 MWth Marine MSR FEA analysis.

Three levels of geometric fidelity for multi-scale thermal and
structural analysis:

    Level 1: 2D R-Z Axisymmetric Full Core
        - Homogenized core with reflector and vessel
        - For full-core thermal distribution and vessel stress
        - ~5000 elements (adjustable)

    Level 2: Single Hexagonal Unit Cell (X-Y)
        - One graphite hex block with central fuel channel
        - For local thermal analysis (fuel-graphite interface)
        - ~2000 elements (adjustable)

    Level 3: 7-Channel Rosette Cluster (X-Y)
        - 1 central + 6 surrounding hex cells
        - For multi-channel thermal interaction effects
        - ~10000 elements (adjustable)

Design dimensions (from config.py):
    Core radius:            0.6225 m  (124.5 cm diameter)
    Core half-height:       0.747 m   (149.4 cm total)
    Reflector thickness:    0.15 m    (15 cm graphite)
    Vessel wall thickness:  0.015 m   (1.5 cm Hastelloy-N)
    Hex pitch (flat-to-flat): 0.05 m  (5 cm)
    Channel diameter:       0.025 m   (2.5 cm, radius 1.25 cm)

Material zones:
    Level 1:
        0 = homogenized core (fuel 23% + graphite 77%)
        1 = radial reflector (graphite)
        2 = axial reflector (graphite)
        3 = vessel wall (Hastelloy-N)

    Levels 2 & 3:
        0 = fuel salt (inside channels)
        1 = graphite moderator (between channel and hex boundary)
"""

import numpy as np
from typing import Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fea.mesh.nodes import Mesh
from fea.mesh.triangulation import (
    delaunay_with_boundary,
    generate_circle_points,
    generate_hex_points,
    generate_radial_points,
    assign_materials_by_circles,
    extract_boundary_edges,
    extract_exterior_edges,
    remove_exterior_triangles,
    remove_triangles_outside_circle,
    _orient_ccw,
    _point_in_hexagon,
)


# =========================================================================
# MSR Design Constants
# =========================================================================
CORE_RADIUS = 0.6225           # m (124.5 cm / 2)
CORE_HALF_HEIGHT = 0.747       # m (149.4 cm / 2)
REFLECTOR_THICKNESS = 0.15     # m (15 cm)
VESSEL_THICKNESS = 0.015       # m (1.5 cm)
HEX_PITCH = 0.05              # m (5 cm flat-to-flat)
CHANNEL_RADIUS = 0.0125       # m (2.5 cm diameter / 2)


# =========================================================================
# Level 1: 2D R-Z Axisymmetric Full Core
# =========================================================================

def build_level1_rz(nr_core: int = 40, nz_core: int = 60,
                    nr_reflector: int = 10, nz_reflector: int = 10,
                    nr_vessel: int = 4) -> Mesh:
    """Level 1: R-Z axisymmetric full core (upper-right quadrant).

    Exploits symmetry: only the upper-right quadrant is meshed.
    The r=0 axis is a natural symmetry boundary, and z=0 is the
    core midplane symmetry.

    Domain:
        r: [0, R_outer]  where R_outer = core_radius + reflector + vessel
        z: [0, H/2 + reflector_z + vessel_z]

    Strategy: Structured quad grid with non-uniform spacing, then split
    each quad diagonally into 2 triangles.

    Material zones:
        0 = homogenized core (fuel 23% + graphite 77%)
        1 = radial reflector (graphite)
        2 = axial reflector (graphite)
        3 = vessel wall (Hastelloy-N)

    Boundary tags:
        'symmetry_r0': r=0 axis (Neumann: dT/dr = 0)
        'symmetry_z0': z=0 midplane (Neumann: dT/dz = 0)
        'outer_wall': vessel outer surface (r = R_outer)
        'top_wall': vessel top surface (z = Z_top)

    Parameters
    ----------
    nr_core : int
        Number of radial divisions in the core region.
    nz_core : int
        Number of axial divisions in the core region.
    nr_reflector : int
        Number of radial divisions in the reflector.
    nz_reflector : int
        Number of axial divisions in the axial reflector.
    nr_vessel : int
        Number of radial divisions in the vessel wall.

    Returns
    -------
    mesh : Mesh
        Axisymmetric Tri3 mesh of the upper-right quadrant.
    """
    r_core = CORE_RADIUS
    z_half = CORE_HALF_HEIGHT
    t_refl = REFLECTOR_THICKNESS
    t_vessel = VESSEL_THICKNESS

    r_refl = r_core + t_refl       # outer radius of reflector
    r_outer = r_refl + t_vessel    # outer radius of vessel
    z_refl = z_half + t_refl       # top of axial reflector
    z_top = z_refl + t_vessel      # top of vessel

    # --- Build non-uniform radial spacing ---
    # Core: slightly finer near center (for power peaking) and near edge
    r_core_pts = _graded_spacing(0.0, r_core, nr_core,
                                 ratio_start=0.8, ratio_end=0.8)
    # Reflector: uniform
    r_refl_pts = np.linspace(r_core, r_refl, nr_reflector + 1)[1:]
    # Vessel: uniform (4 elements through 15 mm)
    r_vessel_pts = np.linspace(r_refl, r_outer, nr_vessel + 1)[1:]

    r_all = np.concatenate([r_core_pts, r_refl_pts, r_vessel_pts])
    nr_total = len(r_all) - 1  # number of radial intervals

    # --- Build non-uniform axial spacing ---
    z_core_pts = _graded_spacing(0.0, z_half, nz_core,
                                 ratio_start=1.0, ratio_end=0.8)
    z_refl_pts = np.linspace(z_half, z_refl, nz_reflector + 1)[1:]
    z_vessel_pts = np.linspace(z_refl, z_top, max(nr_vessel, 3) + 1)[1:]

    z_all = np.concatenate([z_core_pts, z_refl_pts, z_vessel_pts])
    nz_total = len(z_all) - 1  # number of axial intervals

    # --- Generate nodes on structured grid ---
    nr_nodes = len(r_all)
    nz_nodes = len(z_all)
    n_nodes = nr_nodes * nz_nodes

    nodes = np.empty((n_nodes, 2), dtype=np.float64)
    for j in range(nz_nodes):
        for i in range(nr_nodes):
            idx = j * nr_nodes + i
            nodes[idx, 0] = r_all[i]  # r coordinate
            nodes[idx, 1] = z_all[j]  # z coordinate

    # --- Generate elements (quads -> 2 triangles each) ---
    n_quads = nr_total * nz_total
    n_elements = 2 * n_quads
    elements = np.empty((n_elements, 3), dtype=np.int64)
    material_ids = np.empty(n_elements, dtype=np.int64)

    # Count radial intervals in each zone
    nr_core_intervals = nr_core
    nr_refl_intervals = nr_reflector
    # nr_vessel_intervals = nr_vessel
    nz_core_intervals = nz_core
    nz_refl_intervals = nz_reflector
    # nz_vessel_intervals = len(z_vessel_pts)

    # Radial zone boundaries (interval indices)
    ir_core_end = nr_core_intervals    # [0, ir_core_end) = core
    ir_refl_end = ir_core_end + nr_refl_intervals   # reflector
    # ir_vessel_end = ir_refl_end + nr_vessel      # vessel

    # Axial zone boundaries (interval indices)
    iz_core_end = nz_core_intervals
    iz_refl_end = iz_core_end + nz_refl_intervals

    elem_idx = 0
    for j in range(nz_total):
        for i in range(nr_total):
            # Quad node indices
            n00 = j * nr_nodes + i           # bottom-left
            n10 = j * nr_nodes + (i + 1)     # bottom-right
            n01 = (j + 1) * nr_nodes + i     # top-left
            n11 = (j + 1) * nr_nodes + (i + 1)  # top-right

            # Split quad into 2 triangles (diagonal: n00 -> n11)
            elements[elem_idx] = [n00, n10, n11]
            elements[elem_idx + 1] = [n00, n11, n01]

            # Determine material zone
            mat = _get_material_zone_rz(i, j, ir_core_end, ir_refl_end,
                                        iz_core_end, iz_refl_end)
            material_ids[elem_idx] = mat
            material_ids[elem_idx + 1] = mat

            elem_idx += 2

    # --- Tag boundaries ---
    tol_r = 1e-10
    tol_z = 1e-10

    # symmetry_r0: all nodes at r = 0
    sym_r0_mask = nodes[:, 0] < tol_r
    sym_r0_nodes = np.where(sym_r0_mask)[0]

    # symmetry_z0: all nodes at z = 0
    sym_z0_mask = nodes[:, 1] < tol_z
    sym_z0_nodes = np.where(sym_z0_mask)[0]

    # outer_wall: all nodes at r = r_outer
    outer_mask = np.abs(nodes[:, 0] - r_outer) < tol_r + 1e-6 * r_outer
    outer_nodes = np.where(outer_mask)[0]

    # top_wall: all nodes at z = z_top
    top_mask = np.abs(nodes[:, 1] - z_top) < tol_z + 1e-6 * z_top
    top_nodes = np.where(top_mask)[0]

    # Build boundary edges
    boundary_edges = {}
    boundary_nodes_dict = {}

    for tag, bnodes in [('symmetry_r0', sym_r0_nodes),
                        ('symmetry_z0', sym_z0_nodes),
                        ('outer_wall', outer_nodes),
                        ('top_wall', top_nodes)]:
        edges = _extract_boundary_edges_structured(elements, bnodes)
        boundary_edges[tag] = edges
        boundary_nodes_dict[tag] = bnodes

    return Mesh(
        nodes=nodes,
        elements=elements,
        element_type='tri3',
        material_ids=material_ids,
        boundary_edges=boundary_edges,
        boundary_nodes=boundary_nodes_dict,
        coord_system='axisymmetric',
    )


# =========================================================================
# Level 2: Single Hexagonal Unit Cell
# =========================================================================

def build_level2_hex_cell(n_channel_circ: int = 24,
                          n_channel_radial: int = 6,
                          n_graphite_radial: int = 10) -> Mesh:
    """Level 2: Single hexagonal unit cell with circular fuel channel.

    Domain: Regular hexagon (flat-to-flat 5.0 cm = 0.05 m) centered
    at origin, with circular channel (radius 1.25 cm = 0.0125 m)
    at center.

    Strategy:
        1. Generate points on channel circle (n_channel_circ points)
        2. Generate radial layers from channel to ~hex inscribed circle
        3. Generate points on hex boundary edges
        4. Generate interior fill points in graphite region
        5. Delaunay triangulate
        6. Remove triangles outside hexagon
        7. Assign materials by centroid position (inside/outside channel)

    Material zones:
        0 = fuel salt (inside channel)
        1 = graphite moderator (between channel and hex boundary)

    Boundary tags:
        'channel_wall': edges on the circle (fuel-graphite interface)
        'hex_boundary': edges on hex perimeter

    Parameters
    ----------
    n_channel_circ : int
        Points around the channel circle. Default 24.
    n_channel_radial : int
        Radial layers of points inside the channel. Default 6.
    n_graphite_radial : int
        Radial layers between channel and hex boundary. Default 10.

    Returns
    -------
    mesh : Mesh
        Cartesian Tri3 mesh of the hexagonal unit cell.
    """
    pitch = HEX_PITCH
    r_channel = CHANNEL_RADIUS
    center = np.array([0.0, 0.0])

    # Apothem (center to flat) = pitch/2
    apothem = pitch / 2.0
    # Circumradius = pitch / sqrt(3)
    R_hex = pitch / np.sqrt(3.0)

    # --- 1. Channel circle boundary points ---
    circle_pts = generate_circle_points(center, r_channel, n_channel_circ)
    n_circle = len(circle_pts)

    # --- 2. Points inside the channel (radial layers) ---
    # Center point plus concentric rings
    inner_pts_list = [np.array([[0.0, 0.0]])]  # center point
    for i in range(1, n_channel_radial + 1):
        r = r_channel * i / (n_channel_radial + 1)
        n_ring = max(6, int(n_channel_circ * r / r_channel))
        offset = 0.0 if i % 2 == 0 else np.pi / n_ring
        theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + offset
        ring = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inner_pts_list.append(ring)
    channel_interior = np.vstack(inner_pts_list)

    # --- 3. Points in graphite region (concentric layers from channel to hex) ---
    graphite_pts_list = []
    for i in range(1, n_graphite_radial + 1):
        # Interpolate between channel radius and apothem
        frac = i / (n_graphite_radial + 1)
        r = r_channel + frac * (apothem * 0.95 - r_channel)
        n_ring = max(12, int(n_channel_circ * 1.2 * r / r_channel))
        n_ring = min(n_ring, n_channel_circ * 3)
        offset = 0.0 if i % 2 == 0 else np.pi / n_ring
        theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + offset
        ring = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        # Filter out points outside the hexagon
        keep = np.array([_point_in_hexagon(p, center, pitch) for p in ring])
        if np.any(keep):
            graphite_pts_list.append(ring[keep])

    # --- 4. Hex boundary points ---
    n_per_side = max(6, n_channel_circ // 3)
    hex_pts = generate_hex_points(center, pitch, n_per_side)
    n_hex = len(hex_pts)

    # --- 5. Assemble all points ---
    all_pts_list = [circle_pts, channel_interior]
    if graphite_pts_list:
        all_pts_list.append(np.vstack(graphite_pts_list))
    all_pts_list.append(hex_pts)

    points = np.vstack(all_pts_list)

    # Track index ranges for boundary identification
    idx_circle_start = 0
    idx_circle_end = n_circle
    idx_hex_start = len(points) - n_hex
    idx_hex_end = len(points)

    # Channel boundary segments (circle)
    channel_segments = []
    for i in range(n_circle):
        channel_segments.append((i, (i + 1) % n_circle))

    # Hex boundary segments
    hex_node_ids = np.arange(idx_hex_start, idx_hex_end)
    hex_segments = []
    for i in range(n_hex):
        hex_segments.append((idx_hex_start + i,
                             idx_hex_start + (i + 1) % n_hex))

    # Combined boundary segments
    all_segments = channel_segments + hex_segments

    # --- 6. Delaunay triangulation with boundary constraints ---
    simplices = delaunay_with_boundary(points, all_segments)

    # Remove triangles outside hexagon
    hex_polygon = hex_pts.copy()
    simplices = remove_exterior_triangles(simplices, points, hex_polygon)

    # --- 7. Assign materials ---
    material_ids = assign_materials_by_circles(
        simplices, points, [(center, r_channel)], default_material=1
    )

    # --- 8. Build boundary info ---
    circle_node_indices = np.arange(idx_circle_start, idx_circle_end)
    hex_node_indices = np.arange(idx_hex_start, idx_hex_end)

    channel_edges = extract_boundary_edges(simplices, circle_node_indices)
    hex_edges = extract_boundary_edges(simplices, hex_node_indices)

    boundary_edges = {
        'channel_wall': channel_edges,
        'hex_boundary': hex_edges,
    }
    boundary_nodes_dict = {
        'channel_wall': circle_node_indices,
        'hex_boundary': hex_node_indices,
    }

    return Mesh(
        nodes=points,
        elements=simplices,
        element_type='tri3',
        material_ids=material_ids,
        boundary_edges=boundary_edges,
        boundary_nodes=boundary_nodes_dict,
        coord_system='cartesian',
    )


# =========================================================================
# Level 3: 7-Channel Rosette Cluster
# =========================================================================

def build_level3_rosette(n_channel_circ: int = 20,
                         n_radial: int = 8) -> Mesh:
    """Level 3: 7-channel rosette cluster.

    1 central + 6 surrounding hexagonal unit cells, forming a
    miniature cluster that captures inter-channel thermal effects.

    Channel centers use the hexagonal lattice formula:
        center: (0, 0)
        ring 1 (6 positions):
            Axial coordinates (q, r) for |q| + |r| + |q+r| <= 2:
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            Cartesian: x = pitch * (q + 0.5*r)
                        y = pitch * sqrt(3)/2 * r

    Outer boundary: larger hexagon enclosing all 7 cells.

    Material zones:
        0 = fuel salt (inside any channel)
        1 = graphite moderator (everywhere else inside outer boundary)

    Boundary tags:
        'channel_wall_0' through 'channel_wall_6': each channel interface
        'outer_boundary': outer hexagonal perimeter

    Parameters
    ----------
    n_channel_circ : int
        Points around each channel circle. Default 20.
    n_radial : int
        Radial fill layers between channels. Default 8.

    Returns
    -------
    mesh : Mesh
        Cartesian Tri3 mesh of the 7-channel rosette.
    """
    pitch = HEX_PITCH
    r_channel = CHANNEL_RADIUS

    # --- Channel centers ---
    # Hex lattice coordinates (q, r) -> Cartesian
    hex_qr = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    channel_centers = []
    for q, r in hex_qr:
        x = pitch * (q + 0.5 * r)
        y = pitch * np.sqrt(3.0) / 2.0 * r
        channel_centers.append(np.array([x, y]))

    # Outer boundary: hexagon enclosing all 7 cells
    # The rosette outer boundary is a hexagon with flat-to-flat distance
    # equal to 3 * pitch (3 rows of hexes: center + 1 on each side)
    outer_pitch = 3.0 * pitch
    outer_center = np.array([0.0, 0.0])

    # --- Generate all points ---
    all_pts_list = []
    circle_index_ranges = []  # (start, end) for each channel circle
    current_idx = 0

    # 1. Channel boundary circles
    for ch_center in channel_centers:
        circle = generate_circle_points(ch_center, r_channel, n_channel_circ)
        start = current_idx
        all_pts_list.append(circle)
        current_idx += len(circle)
        circle_index_ranges.append((start, current_idx))

    # 2. Interior points inside each channel
    for ch_center in channel_centers:
        # Center point + radial layers
        pts_list = [ch_center.reshape(1, 2)]
        n_inner = max(3, n_radial // 2)
        for i in range(1, n_inner + 1):
            r = r_channel * i / (n_inner + 1)
            n_ring = max(6, int(n_channel_circ * r / r_channel))
            offset = 0.0 if i % 2 == 0 else np.pi / n_ring
            theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + offset
            ring = np.column_stack([
                ch_center[0] + r * np.cos(theta),
                ch_center[1] + r * np.sin(theta),
            ])
            pts_list.append(ring)
        interior = np.vstack(pts_list)
        all_pts_list.append(interior)
        current_idx += len(interior)

    # 3. Fill points in graphite region (between channels and outer boundary)
    # Use concentric rings around the rosette center
    outer_apothem = outer_pitch / 2.0
    max_fill_r = outer_apothem * 0.95
    n_fill_layers = n_radial * 2
    for i in range(1, n_fill_layers + 1):
        frac = i / (n_fill_layers + 1)
        r = r_channel * 1.5 + frac * (max_fill_r - r_channel * 1.5)
        n_ring = max(18, int(n_channel_circ * 2.0 * r / pitch))
        n_ring = min(n_ring, n_channel_circ * 6)
        offset = 0.0 if i % 2 == 0 else np.pi / n_ring
        theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + offset
        ring = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        # Filter: inside outer hex and NOT inside any channel circle
        keep = np.ones(len(ring), dtype=bool)
        for p_idx in range(len(ring)):
            if not _point_in_hexagon(ring[p_idx], outer_center, outer_pitch):
                keep[p_idx] = False
                continue
            for ch_center in channel_centers:
                dist = np.linalg.norm(ring[p_idx] - ch_center)
                if dist < r_channel * 0.9:
                    keep[p_idx] = False
                    break
        if np.any(keep):
            all_pts_list.append(ring[keep])
            current_idx += np.sum(keep)

    # 4. Outer boundary hex points
    n_per_side_outer = max(8, n_channel_circ // 2)
    outer_hex_pts = generate_hex_points(outer_center, outer_pitch,
                                        n_per_side_outer)
    n_outer_hex = len(outer_hex_pts)
    idx_outer_start = current_idx
    all_pts_list.append(outer_hex_pts)
    current_idx += n_outer_hex

    # Assemble
    points = np.vstack(all_pts_list)

    # --- Boundary segments ---
    # Channel circles
    all_segments = []
    for ch_idx, (start, end) in enumerate(circle_index_ranges):
        n_pts = end - start
        for i in range(n_pts):
            all_segments.append((start + i, start + (i + 1) % n_pts))

    # Outer hex boundary
    outer_node_ids = np.arange(idx_outer_start, idx_outer_start + n_outer_hex)
    for i in range(n_outer_hex):
        all_segments.append((idx_outer_start + i,
                             idx_outer_start + (i + 1) % n_outer_hex))

    # --- Triangulate ---
    simplices = delaunay_with_boundary(points, all_segments)

    # Remove triangles outside outer hex
    outer_polygon = outer_hex_pts.copy()
    simplices = remove_exterior_triangles(simplices, points, outer_polygon)

    # --- Assign materials ---
    circles_for_material = [(c, r_channel) for c in channel_centers]
    material_ids = assign_materials_by_circles(
        simplices, points, circles_for_material, default_material=1
    )

    # --- Boundary info ---
    boundary_edges = {}
    boundary_nodes_dict = {}

    for ch_idx, (start, end) in enumerate(circle_index_ranges):
        tag = f'channel_wall_{ch_idx}'
        ch_nodes = np.arange(start, end)
        edges = extract_boundary_edges(simplices, ch_nodes)
        boundary_edges[tag] = edges
        boundary_nodes_dict[tag] = ch_nodes

    outer_edges = extract_boundary_edges(simplices, outer_node_ids)
    boundary_edges['outer_boundary'] = outer_edges
    boundary_nodes_dict['outer_boundary'] = outer_node_ids

    return Mesh(
        nodes=points,
        elements=simplices,
        element_type='tri3',
        material_ids=material_ids,
        boundary_edges=boundary_edges,
        boundary_nodes=boundary_nodes_dict,
        coord_system='cartesian',
    )


# =========================================================================
# Private helper functions
# =========================================================================

def _graded_spacing(x_start: float, x_end: float, n: int,
                    ratio_start: float = 1.0,
                    ratio_end: float = 1.0) -> np.ndarray:
    """Generate non-uniform spacing with optional grading at ends.

    Creates n+1 points from x_start to x_end. The spacing is finer
    where ratio < 1.0 (at start or end).

    Uses a blended approach: uniform base plus sinusoidal bunching.

    Parameters
    ----------
    x_start : float
    x_end : float
    n : int
        Number of intervals.
    ratio_start : float
        Grading at start (< 1.0 = finer). Default 1.0 (uniform).
    ratio_end : float
        Grading at end (< 1.0 = finer). Default 1.0 (uniform).

    Returns
    -------
    pts : ndarray, shape (n+1,)
        Non-uniformly spaced points.
    """
    t = np.linspace(0, 1, n + 1)

    # Apply bunching function
    # Bunch toward start (t=0) and/or end (t=1) using tanh stretching
    if abs(ratio_start - 1.0) > 0.01 or abs(ratio_end - 1.0) > 0.01:
        # Simple sinusoidal bunching
        # Bunch toward both ends: t_new = 0.5 * (1 - cos(pi * t))
        # Bunch toward start only: t_new = 1 - cos(pi/2 * t)
        # Bunch toward end only: t_new = sin(pi/2 * t)

        if ratio_start < 1.0 and ratio_end < 1.0:
            # Bunch toward both ends
            beta = 1.5  # bunching parameter
            t = 0.5 * (1.0 - np.cos(np.pi * t))
        elif ratio_start < 1.0:
            # Bunch toward start
            beta_s = 1.0 - ratio_start
            t = t ** (1.0 / (1.0 + beta_s))
        elif ratio_end < 1.0:
            # Bunch toward end
            beta_e = 1.0 - ratio_end
            t = 1.0 - (1.0 - t) ** (1.0 / (1.0 + beta_e))

    pts = x_start + t * (x_end - x_start)
    return pts


def _get_material_zone_rz(i_r: int, j_z: int,
                           ir_core_end: int, ir_refl_end: int,
                           iz_core_end: int, iz_refl_end: int) -> int:
    """Determine material zone from (radial, axial) interval indices.

    Parameters
    ----------
    i_r : int
        Radial interval index (0-based).
    j_z : int
        Axial interval index (0-based).
    ir_core_end : int
        First radial interval in the reflector zone.
    ir_refl_end : int
        First radial interval in the vessel zone.
    iz_core_end : int
        First axial interval in the axial reflector zone.
    iz_refl_end : int
        First axial interval in the vessel zone.

    Returns
    -------
    material : int
        0=core, 1=radial reflector, 2=axial reflector, 3=vessel
    """
    in_core_r = i_r < ir_core_end
    in_refl_r = ir_core_end <= i_r < ir_refl_end
    in_vessel_r = i_r >= ir_refl_end

    in_core_z = j_z < iz_core_end
    in_refl_z = iz_core_end <= j_z < iz_refl_end
    in_vessel_z = j_z >= iz_refl_end

    # Vessel takes priority (any dimension in vessel zone)
    if in_vessel_r or in_vessel_z:
        return 3

    # Axial reflector (core radial range, reflector axial range)
    if in_core_r and in_refl_z:
        return 2

    # Radial reflector (reflector radial range, core+refl axial range)
    if in_refl_r and (in_core_z or in_refl_z):
        return 1

    # Core
    if in_core_r and in_core_z:
        return 0

    # Corner region (radial reflector + axial reflector) -> reflector
    return 1


def _extract_boundary_edges_structured(elements: np.ndarray,
                                       boundary_nodes: np.ndarray
                                       ) -> List[Tuple[int, int]]:
    """Extract boundary edges from a structured mesh.

    An edge is a boundary edge if both its nodes are in the boundary
    set and the edge belongs to at least one element.

    Parameters
    ----------
    elements : ndarray, shape (N_elem, 3)
    boundary_nodes : ndarray

    Returns
    -------
    edges : list of (int, int)
    """
    bnd_set = set(boundary_nodes.tolist())
    edges = []
    seen = set()

    for tri in elements:
        for i in range(3):
            n0 = int(tri[i])
            n1 = int(tri[(i + 1) % 3])
            if n0 in bnd_set and n1 in bnd_set:
                key = (min(n0, n1), max(n0, n1))
                if key not in seen:
                    seen.add(key)
                    edges.append((n0, n1))

    return edges
