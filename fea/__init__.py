"""
FEA - Finite Element Analysis Package for Marine MSR (40 MWth)

Core infrastructure for thermal-hydraulic, neutronics, and structural
analysis of molten salt reactor components using triangular elements.

Element library:
    - Tri3: 3-node linear triangle (scalar fields: thermal, neutronics)
    - Tri6: 6-node quadratic triangle (2D elasticity, structural)

Coordinate systems:
    - Cartesian (x, y)
    - Axisymmetric (r, z) with 2*pi*r integration

Author: MSR Analysis Team
"""

__version__ = "0.1.0"

from .elements.quadrature import (
    gauss_triangle_1pt,
    gauss_triangle_3pt,
    gauss_triangle_7pt,
    gauss_line_2pt,
)
from .elements.tri3 import (
    shape_functions_tri3,
    shape_gradients_tri3,
    stiffness_scalar_tri3,
    mass_tri3,
    load_vector_tri3,
    boundary_mass_tri3,
    boundary_load_tri3,
)
from .elements.tri6 import (
    shape_functions_tri6,
    shape_gradients_tri6,
    jacobian_tri6,
    B_matrix_tri6,
    D_matrix_plane_strain,
    D_matrix_plane_stress,
    stiffness_elastic_tri6,
    thermal_load_tri6,
    mass_tri6,
)
from .mesh.nodes import Mesh, tri3_to_tri6
from .assembly.sparse_assembler import (
    assemble_global_matrix,
    assemble_global_vector,
    assemble_scalar_stiffness,
    assemble_scalar_mass,
    assemble_scalar_load,
    assemble_elastic_stiffness,
    assemble_thermal_load,
)
from .assembly.boundary_conditions import (
    apply_dirichlet_penalty,
    apply_dirichlet_elimination,
    apply_convective_robin,
    get_boundary_dofs,
)
from .materials.properties import MaterialLibrary
from .solvers.thermal import solve_thermal, ThermalResult
