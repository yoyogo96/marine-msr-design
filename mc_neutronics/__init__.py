"""
Multi-Group Monte Carlo Neutron Transport Package
==================================================

A custom 2-group Monte Carlo neutron transport code for the basic nuclear
design of a 40 MWth Marine Molten Salt Reactor (MSR).

This package implements forward-mode Monte Carlo simulation of neutron
histories in a graphite-moderated, channel-type MSR with FLiBe + UF4
fuel salt. The code uses a 2-group energy structure (fast/thermal with
0.625 eV cutoff) and tracks neutrons through a 3D hexagonal lattice
geometry consisting of cylindrical fuel salt channels in a graphite
moderator matrix, surrounded by a graphite reflector.

Modules
-------
constants
    Physical constants, energy group structure, and nuclear data parameters.
materials
    Material definitions with 2-group macroscopic cross-sections for
    fuel salt (FLiBe + UF4), graphite moderator, graphite reflector,
    and void (vacuum boundary).
geometry
    3D hexagonal lattice geometry with cylindrical core and reflector.
    Point location, distance-to-boundary, and source sampling routines.

Design Basis
------------
- Thermal power: 40 MWth
- Fuel salt: FLiBe + 5 mol% UF4, 12% enriched (HALEU)
- Moderator: IG-110 nuclear graphite
- Core diameter: ~1.245 m, height: ~1.494 m (H/D = 1.2)
- 562 hexagonal fuel channels, 25 mm diameter, 50 mm pitch
- Graphite reflector: 150 mm thick
- Operating temperature: 600-700 C (average 650 C)

References
----------
- ORNL-4541: Molten-Salt Reactor Program Semiannual Progress Report
- ORNL/TM-2005/218: Assessment of Candidate Molten Salt Coolants
- Robertson, R.C., "Conceptual Design Study of a Single-Fluid MSR," 1971
"""

__version__ = "0.1.0"
__author__ = "MSR Design Team"
