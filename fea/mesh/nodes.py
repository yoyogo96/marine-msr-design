"""
Mesh data structures and utilities for the FEA package.

Provides the Mesh dataclass for storing finite element meshes and
utility functions for mesh manipulation (e.g., Tri3 to Tri6 conversion).

Mesh conventions:
    - Node numbering: 0-indexed
    - Element connectivity: counter-clockwise node ordering
    - Coordinate systems: 'cartesian' (x, y) or 'axisymmetric' (r, z)
    - Material zones: integer IDs for multi-material assemblies
    - Boundary tags: string labels for named boundary groups
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Mesh:
    """
    Finite element mesh data structure.

    Attributes
    ----------
    nodes : ndarray, shape (N_nodes, 2)
        Nodal coordinates. For cartesian: (x, y). For axisymmetric: (r, z).
    elements : ndarray, shape (N_elem, n_per_elem)
        Element connectivity array (0-indexed node indices).
        n_per_elem = 3 for Tri3, 6 for Tri6.
    element_type : str
        Element type identifier: 'tri3' or 'tri6'.
    material_ids : ndarray, shape (N_elem,)
        Integer zone IDs assigning each element to a material region.
    boundary_edges : dict
        Mapping from boundary tag (str) to list of edge tuples.
        Each edge is (node_i, node_j) with consistent orientation.
        For Tri6, edges include mid-nodes: (node_i, node_mid, node_j).
    boundary_nodes : dict
        Mapping from boundary tag (str) to ndarray of unique node indices
        on that boundary.
    coord_system : str
        Coordinate system: 'cartesian' or 'axisymmetric'.
    """
    nodes: np.ndarray
    elements: np.ndarray
    element_type: str
    material_ids: np.ndarray
    boundary_edges: Dict[str, List[Tuple[int, ...]]] = field(default_factory=dict)
    boundary_nodes: Dict[str, np.ndarray] = field(default_factory=dict)
    coord_system: str = 'cartesian'

    def __post_init__(self):
        """Validate mesh data after initialization."""
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.elements = np.asarray(self.elements, dtype=np.int64)
        self.material_ids = np.asarray(self.material_ids, dtype=np.int64)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError(
                f"nodes must have shape (N, 2), got {self.nodes.shape}"
            )
        if self.elements.ndim != 2:
            raise ValueError(
                f"elements must be 2D array, got shape {self.elements.shape}"
            )
        if self.element_type == 'tri3' and self.elements.shape[1] != 3:
            raise ValueError(
                f"Tri3 elements must have 3 nodes per element, "
                f"got {self.elements.shape[1]}"
            )
        if self.element_type == 'tri6' and self.elements.shape[1] != 6:
            raise ValueError(
                f"Tri6 elements must have 6 nodes per element, "
                f"got {self.elements.shape[1]}"
            )
        if len(self.material_ids) != len(self.elements):
            raise ValueError(
                f"material_ids length ({len(self.material_ids)}) must match "
                f"number of elements ({len(self.elements)})"
            )
        if self.coord_system not in ('cartesian', 'axisymmetric'):
            raise ValueError(
                f"coord_system must be 'cartesian' or 'axisymmetric', "
                f"got '{self.coord_system}'"
            )
        # Validate node indices are in range
        max_node = np.max(self.elements)
        if max_node >= len(self.nodes):
            raise ValueError(
                f"Element connectivity references node {max_node}, but mesh "
                f"only has {len(self.nodes)} nodes (0-indexed)"
            )
        if np.min(self.elements) < 0:
            raise ValueError("Element connectivity contains negative node indices")

    @property
    def n_nodes(self):
        """Number of nodes in the mesh."""
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        """Number of elements in the mesh."""
        return self.elements.shape[0]

    def element_coords(self, e):
        """
        Get physical coordinates for element e.

        Parameters
        ----------
        e : int
            Element index (0-indexed).

        Returns
        -------
        coords : ndarray, shape (n_per_elem, 2)
            Coordinates of the element's nodes.
        """
        return self.nodes[self.elements[e]]

    def element_area(self, e):
        """
        Compute the area of element e (works for both Tri3 and Tri6).

        For Tri3: exact via cross product.
        For Tri6: uses the corner nodes (linear approximation) since
        the true area requires Jacobian integration for curved elements.

        Parameters
        ----------
        e : int
            Element index (0-indexed).

        Returns
        -------
        area : float
            Area of the element (positive for CCW ordering).
        """
        conn = self.elements[e]
        # Use first 3 nodes (corners) for area computation
        x0, y0 = self.nodes[conn[0]]
        x1, y1 = self.nodes[conn[1]]
        x2, y2 = self.nodes[conn[2]]
        area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        return area


def tri3_to_tri6(mesh):
    """
    Convert a Tri3 mesh to a Tri6 mesh by inserting mid-edge nodes.

    Each edge shared by two elements gets a single mid-edge node (no
    duplicates). Boundary edges and boundary nodes are updated to
    include the new mid-edge nodes.

    Mid-edge node numbering within each element:
        Node 3: midpoint of edge 0-1
        Node 4: midpoint of edge 1-2
        Node 5: midpoint of edge 2-0

    Parameters
    ----------
    mesh : Mesh
        Input Tri3 mesh (element_type must be 'tri3').

    Returns
    -------
    mesh6 : Mesh
        Output Tri6 mesh with mid-edge nodes inserted.

    Raises
    ------
    ValueError
        If input mesh is not Tri3.
    """
    if mesh.element_type != 'tri3':
        raise ValueError(
            f"tri3_to_tri6 requires a 'tri3' mesh, got '{mesh.element_type}'"
        )

    n_nodes_orig = mesh.n_nodes
    n_elem = mesh.n_elements

    # Build edge-to-midnode mapping
    # Edge key: (min_node, max_node) to ensure uniqueness
    edge_to_midnode = {}
    new_nodes_list = []
    next_node_id = n_nodes_orig

    def get_or_create_midnode(n1, n2):
        """Get existing mid-node or create a new one for edge (n1, n2)."""
        nonlocal next_node_id
        edge_key = (min(n1, n2), max(n1, n2))
        if edge_key in edge_to_midnode:
            return edge_to_midnode[edge_key]
        # Create new mid-node at midpoint
        mid_coords = 0.5 * (mesh.nodes[n1] + mesh.nodes[n2])
        new_nodes_list.append(mid_coords)
        mid_id = next_node_id
        edge_to_midnode[edge_key] = mid_id
        next_node_id += 1
        return mid_id

    # Build Tri6 connectivity
    elements6 = np.empty((n_elem, 6), dtype=np.int64)
    for e in range(n_elem):
        n0, n1, n2 = mesh.elements[e]
        m3 = get_or_create_midnode(n0, n1)  # mid of edge 0-1
        m4 = get_or_create_midnode(n1, n2)  # mid of edge 1-2
        m5 = get_or_create_midnode(n2, n0)  # mid of edge 2-0
        elements6[e] = [n0, n1, n2, m3, m4, m5]

    # Assemble new node array
    if new_nodes_list:
        new_nodes_arr = np.array(new_nodes_list, dtype=np.float64)
        nodes6 = np.vstack([mesh.nodes, new_nodes_arr])
    else:
        nodes6 = mesh.nodes.copy()

    # Update boundary edges to include mid-nodes
    boundary_edges6 = {}
    for tag, edges in mesh.boundary_edges.items():
        new_edges = []
        for edge in edges:
            n1, n2 = edge[0], edge[1]
            edge_key = (min(n1, n2), max(n1, n2))
            mid = edge_to_midnode.get(edge_key)
            if mid is not None:
                new_edges.append((n1, mid, n2))
            else:
                # Edge not found in element edges -- keep as-is with warning
                new_edges.append((n1, n2))
        boundary_edges6[tag] = new_edges

    # Update boundary nodes to include mid-edge nodes
    boundary_nodes6 = {}
    for tag, bnodes in mesh.boundary_nodes.items():
        node_set = set(bnodes.tolist())
        # Add mid-nodes for boundary edges
        if tag in boundary_edges6:
            for edge in boundary_edges6[tag]:
                for n in edge:
                    node_set.add(n)
        boundary_nodes6[tag] = np.array(sorted(node_set), dtype=np.int64)

    return Mesh(
        nodes=nodes6,
        elements=elements6,
        element_type='tri6',
        material_ids=mesh.material_ids.copy(),
        boundary_edges=boundary_edges6,
        boundary_nodes=boundary_nodes6,
        coord_system=mesh.coord_system,
    )
