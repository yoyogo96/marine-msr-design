"""
Mesh refinement strategies for FEA meshes.

Provides boundary-layer refinement (for resolving steep thermal gradients
at channel walls) and longest-edge bisection for local adaptive refinement.

Boundary-layer refinement:
    In MSR thermal analysis, the fuel-salt-to-graphite interface has a
    steep temperature gradient driven by the convective heat transfer
    coefficient. The first element layer at the channel wall must be
    thin enough to resolve this gradient (typically < 0.3 mm for
    Re ~ 1000 in the channel).

    Geometric growth ratio of 1.5 is used: each successive layer is
    1.5x thicker than the previous one, grading from fine (at wall)
    to coarse (in bulk).

Longest-edge bisection (Rivara):
    Local refinement by splitting the longest edge of marked elements.
    Neighboring elements sharing the split edge are also refined to
    maintain mesh conformity (no hanging nodes). This is useful for
    adaptive refinement driven by error indicators.

Conventions:
    - All input/output meshes use 0-indexed node numbering
    - Elements are Tri3 (3-node linear triangles)
    - Counter-clockwise node ordering is maintained
"""

import numpy as np
from typing import List, Set, Tuple, Dict


def refine_near_boundary(nodes: np.ndarray,
                         elements: np.ndarray,
                         boundary_nodes: np.ndarray,
                         n_layers: int = 3,
                         first_layer: float = 0.0003,
                         growth_ratio: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Add boundary-layer refinement near specified boundaries.

    Creates graded layers of thin elements near the boundary for
    resolving steep thermal gradients (e.g., at channel wall).

    The algorithm:
    1. Identify boundary elements (elements with >= 1 node on boundary)
    2. For each boundary element, compute the normal direction from
       the boundary edge toward the interior
    3. Insert new node layers at geometric progression distances
    4. Replace original boundary elements with thin layer elements

    Geometric growth: layer_i thickness = first_layer * growth_ratio^i

    Parameters
    ----------
    nodes : ndarray, shape (N, 2)
        Node coordinates.
    elements : ndarray, shape (N_elem, 3)
        Tri3 connectivity (0-indexed).
    boundary_nodes : ndarray
        Indices of nodes on the boundary to refine near.
    n_layers : int, optional
        Number of refinement layers. Default 3.
    first_layer : float, optional
        Thickness of the first (thinnest) layer in meters. Default 0.3 mm.
    growth_ratio : float, optional
        Each layer is this factor thicker than the previous. Default 1.5.

    Returns
    -------
    new_nodes : ndarray, shape (N_new, 2)
        Updated node array (original + new layer nodes).
    new_elements : ndarray, shape (N_elem_new, 3)
        Updated element connectivity.

    Notes
    -----
    This is a simplified boundary-layer insertion. For complex boundaries
    with high curvature, a more sophisticated normal-smoothing algorithm
    would be needed. Our MSR geometries (circles and straight hex edges)
    are sufficiently simple.
    """
    nodes = np.array(nodes, dtype=np.float64)
    elements = np.array(elements, dtype=np.int64)
    bnd_set = set(boundary_nodes.tolist())

    # Find boundary edges: edges where both nodes are on the boundary
    # and the edge belongs to exactly one or more elements
    boundary_edges = []
    edge_to_elements: Dict[Tuple[int, int], List[int]] = {}

    for e_idx, tri in enumerate(elements):
        for i in range(3):
            n0, n1 = int(tri[i]), int(tri[(i + 1) % 3])
            if n0 in bnd_set and n1 in bnd_set:
                key = (min(n0, n1), max(n0, n1))
                if key not in edge_to_elements:
                    edge_to_elements[key] = []
                    boundary_edges.append(key)
                edge_to_elements[key].append(e_idx)

    if len(boundary_edges) == 0:
        return nodes, elements

    # Compute outward normal for each boundary node
    # Average the normals of all boundary edges meeting at that node
    node_normals: Dict[int, np.ndarray] = {}
    for n0, n1 in boundary_edges:
        # Edge tangent
        t = nodes[n1] - nodes[n0]
        t_len = np.linalg.norm(t)
        if t_len < 1e-15:
            continue
        t = t / t_len
        # Outward normal: rotate tangent by -90 degrees (points away from interior)
        # We need to determine direction; use the element on this edge
        normal_candidate = np.array([t[1], -t[0]])

        # Check direction: the normal should point AWAY from the interior
        # Find the opposite vertex of the element sharing this edge
        for e_idx in edge_to_elements[(n0, n1)]:
            tri = elements[e_idx]
            # Find the vertex not on this edge
            opp_node = None
            for v in tri:
                if v != n0 and v != n1:
                    opp_node = int(v)
                    break
            if opp_node is not None:
                mid_edge = 0.5 * (nodes[n0] + nodes[n1])
                to_interior = nodes[opp_node] - mid_edge
                # Normal should point AWAY from interior (opposite direction)
                if np.dot(normal_candidate, to_interior) > 0:
                    normal_candidate = -normal_candidate
                break

        # The inward normal (toward interior) for layer insertion
        inward = -normal_candidate

        for n in [n0, n1]:
            if n not in node_normals:
                node_normals[n] = np.zeros(2)
            node_normals[n] += inward

    # Normalize the accumulated normals
    for n in node_normals:
        norm = np.linalg.norm(node_normals[n])
        if norm > 1e-15:
            node_normals[n] /= norm

    # Compute layer distances (cumulative from boundary)
    layer_dist = np.zeros(n_layers)
    for i in range(n_layers):
        layer_dist[i] = first_layer * growth_ratio ** i
    cum_dist = np.cumsum(layer_dist)

    # Create new layer nodes for each boundary node
    # node_layers[orig_node] = [layer_0_new_node, layer_1_new_node, ...]
    node_layers: Dict[int, List[int]] = {}
    new_nodes_list = list(nodes)
    next_id = len(nodes)

    for n in sorted(node_normals.keys()):
        layers = []
        for d in cum_dist:
            new_pos = nodes[n] + d * node_normals[n]
            new_nodes_list.append(new_pos)
            layers.append(next_id)
            next_id += 1
        node_layers[n] = layers

    # Build new elements:
    # For each boundary edge, create layer quads (split into triangles)
    # The original elements touching boundary edges are kept but may
    # need adjustment

    # Identify elements that have a boundary edge
    boundary_elem_set = set()
    for edge, elems in edge_to_elements.items():
        for e in elems:
            boundary_elem_set.add(e)

    # New elements list: start with non-boundary elements
    new_elem_list = []
    for e_idx, tri in enumerate(elements):
        if e_idx not in boundary_elem_set:
            new_elem_list.append(tri.tolist())

    # For boundary elements, we replace them with refined versions
    # Strategy: for each boundary edge, create n_layers quads (each split
    # into 2 triangles), then connect the innermost layer to the opposite vertex

    processed_edges = set()
    for edge_key, elem_indices in edge_to_elements.items():
        if edge_key in processed_edges:
            continue
        processed_edges.add(edge_key)

        n0, n1 = edge_key

        # Skip if either node doesn't have layer info (shouldn't happen)
        if n0 not in node_layers or n1 not in node_layers:
            for e_idx in elem_indices:
                new_elem_list.append(elements[e_idx].tolist())
            continue

        # Create layer elements between boundary and interior
        # Layer 0: n0, n1 (boundary) -> node_layers[n0][0], node_layers[n1][0]
        # Layer i: node_layers[n0][i-1], node_layers[n1][i-1] ->
        #          node_layers[n0][i], node_layers[n1][i]
        prev_a, prev_b = n0, n1
        for layer_i in range(n_layers):
            curr_a = node_layers[n0][layer_i]
            curr_b = node_layers[n1][layer_i]
            # Create 2 triangles for this quad
            new_elem_list.append([prev_a, prev_b, curr_a])
            new_elem_list.append([prev_b, curr_b, curr_a])
            prev_a, prev_b = curr_a, curr_b

        # Connect innermost layer nodes to the opposite vertex of each element
        for e_idx in elem_indices:
            tri = elements[e_idx]
            opp_node = None
            for v in tri:
                if v != n0 and v != n1:
                    opp_node = int(v)
                    break
            if opp_node is not None:
                inner_a = node_layers[n0][n_layers - 1]
                inner_b = node_layers[n1][n_layers - 1]
                new_elem_list.append([inner_a, inner_b, opp_node])

    new_nodes = np.array(new_nodes_list, dtype=np.float64)
    new_elements = np.array(new_elem_list, dtype=np.int64)

    # Orient all elements CCW
    new_elements = _ensure_ccw(new_nodes, new_elements)

    return new_nodes, new_elements


def longest_edge_bisection(nodes: np.ndarray,
                           elements: np.ndarray,
                           marked_elements: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Rivara longest-edge bisection for local refinement.

    Bisect marked elements by splitting their longest edge.
    Propagate to neighbors sharing the bisected edge to maintain
    mesh conformity (no hanging nodes).

    Algorithm:
    1. For each marked element, find its longest edge
    2. Split the longest edge at its midpoint
    3. If a neighbor shares this edge, split that neighbor too
       (bisect from the new midpoint to the opposite vertex)
    4. Repeat propagation until all hanging nodes are resolved

    Parameters
    ----------
    nodes : ndarray, shape (N, 2)
        Node coordinates.
    elements : ndarray, shape (N_elem, 3)
        Tri3 connectivity.
    marked_elements : ndarray
        Indices of elements to refine.

    Returns
    -------
    new_nodes : ndarray, shape (N_new, 2)
        Updated node array with midpoint nodes added.
    new_elements : ndarray, shape (N_elem_new, 3)
        Updated connectivity (refined elements replaced by 2 children,
        original unmarked/unaffected elements preserved).

    Notes
    -----
    A single pass of longest-edge bisection may propagate beyond the
    marked elements. The propagation terminates because each propagation
    step either bisects an edge that is the longest edge of the neighbor
    (no further propagation) or the longest edge of the neighbor is even
    longer (further propagation toward larger elements, which terminates).
    """
    nodes = np.array(nodes, dtype=np.float64)
    elements = np.array(elements, dtype=np.int64)

    # Build edge-to-element adjacency
    edge_to_elems: Dict[Tuple[int, int], List[int]] = {}
    for e_idx in range(len(elements)):
        tri = elements[e_idx]
        for i in range(3):
            n0 = int(tri[i])
            n1 = int(tri[(i + 1) % 3])
            key = (min(n0, n1), max(n0, n1))
            if key not in edge_to_elems:
                edge_to_elems[key] = []
            edge_to_elems[key].append(e_idx)

    # Track new midpoint nodes: edge_key -> midpoint node index
    midpoints: Dict[Tuple[int, int], int] = {}
    new_nodes_list = list(nodes)
    next_node_id = len(nodes)

    def get_midpoint(n0: int, n1: int) -> int:
        """Get or create midpoint node for edge (n0, n1)."""
        nonlocal next_node_id
        key = (min(n0, n1), max(n0, n1))
        if key in midpoints:
            return midpoints[key]
        mid = 0.5 * (np.array(new_nodes_list[n0]) + np.array(new_nodes_list[n1]))
        new_nodes_list.append(mid)
        mid_id = next_node_id
        midpoints[key] = mid_id
        next_node_id += 1
        return mid_id

    # Elements to process (may grow due to propagation)
    to_refine: Set[int] = set(marked_elements.tolist())
    refined: Set[int] = set()  # Already refined elements

    # New elements to add
    children: Dict[int, List[List[int]]] = {}

    # Iterative refinement with propagation
    max_iterations = len(elements) * 3  # safety limit
    iteration = 0

    while to_refine and iteration < max_iterations:
        iteration += 1
        e_idx = to_refine.pop()

        if e_idx in refined:
            continue

        tri = elements[e_idx]
        n0, n1, n2 = int(tri[0]), int(tri[1]), int(tri[2])

        # Find longest edge
        edges = [(n0, n1, n2), (n1, n2, n0), (n2, n0, n1)]
        longest_len = -1.0
        best_edge = None
        for ea, eb, opp in edges:
            length = np.linalg.norm(
                np.array(new_nodes_list[ea]) - np.array(new_nodes_list[eb])
            )
            if length > longest_len:
                longest_len = length
                best_edge = (ea, eb, opp)

        ea, eb, opp = best_edge
        mid = get_midpoint(ea, eb)

        # Split this element into 2 children
        children[e_idx] = [
            [ea, mid, opp],
            [mid, eb, opp],
        ]
        refined.add(e_idx)

        # Check neighbor sharing edge (ea, eb) for conformity
        edge_key = (min(ea, eb), max(ea, eb))
        for neighbor_idx in edge_to_elems.get(edge_key, []):
            if neighbor_idx != e_idx and neighbor_idx not in refined:
                to_refine.add(neighbor_idx)

    # Assemble final element list
    new_elem_list = []
    for e_idx in range(len(elements)):
        if e_idx in children:
            new_elem_list.extend(children[e_idx])
        else:
            new_elem_list.append(elements[e_idx].tolist())

    new_nodes_arr = np.array(new_nodes_list, dtype=np.float64)
    new_elements = np.array(new_elem_list, dtype=np.int64)

    # Ensure CCW orientation
    new_elements = _ensure_ccw(new_nodes_arr, new_elements)

    return new_nodes_arr, new_elements


def _ensure_ccw(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Ensure all triangles have counter-clockwise orientation.

    Parameters
    ----------
    nodes : ndarray, shape (N, 2)
    elements : ndarray, shape (N_elem, 3)

    Returns
    -------
    elements : ndarray, shape (N_elem, 3)
        Reoriented connectivity.
    """
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]
    cross = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
             (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    cw_mask = cross < 0
    elements[cw_mask, 1], elements[cw_mask, 2] = (
        elements[cw_mask, 2].copy(), elements[cw_mask, 1].copy()
    )
    return elements
