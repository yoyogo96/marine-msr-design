"""
Visualization module for FEA results of the 40 MWth Marine MSR.

Generates publication-quality figures of mesh, scalar fields,
deformed shapes, and vector fields using matplotlib.

All plots use Korean-compatible font settings:
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

Standard output: 300 DPI PNG images for inclusion in reports.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import TriMesh
import os

# Font settings for Korean support
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# Material zone names and colors for each level
ZONE_NAMES_L1 = {
    0: 'Homogenized Core',
    1: 'Radial Reflector',
    2: 'Axial Reflector',
    3: 'Vessel (Hastelloy-N)',
}
ZONE_COLORS_L1 = {
    0: '#FF6B6B',  # coral red (core)
    1: '#4ECDC4',  # teal (radial reflector)
    2: '#45B7D1',  # sky blue (axial reflector)
    3: '#96CEB4',  # sage green (vessel)
}

ZONE_NAMES_L23 = {
    0: 'Fuel Salt',
    1: 'Graphite Moderator',
}
ZONE_COLORS_L23 = {
    0: '#FF6B6B',  # coral red (fuel)
    1: '#4ECDC4',  # teal (graphite)
}


def plot_mesh(mesh, ax=None, show_materials=True, title=None, level=1):
    """Plot mesh with triplot, colored by material zone.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, a new figure is created.
    show_materials : bool
        If True, color elements by material zone. Default True.
    title : str or None
        Plot title. If None, auto-generated.
    level : int
        Geometry level (1, 2, or 3) for zone name lookup.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    nodes = mesh.nodes
    # Use corner nodes only for connectivity
    elements = mesh.elements[:, :3]  # Works for both Tri3 and Tri6

    if show_materials:
        zone_names = ZONE_NAMES_L1 if level == 1 else ZONE_NAMES_L23
        zone_colors = ZONE_COLORS_L1 if level == 1 else ZONE_COLORS_L23

        unique_zones = np.unique(mesh.material_ids)
        for zone in unique_zones:
            mask = mesh.material_ids == zone
            zone_elems = elements[mask]
            color = zone_colors.get(int(zone), '#CCCCCC')
            name = zone_names.get(int(zone), f'Zone {zone}')

            triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1],
                                                zone_elems)
            ax.tripcolor(triangulation, np.ones(np.sum(mask)),
                        cmap=matplotlib.colors.ListedColormap([color]),
                        alpha=0.6)
            # Add to legend via proxy artist
            ax.fill([], [], color=color, alpha=0.6, label=name)

        ax.legend(loc='upper right', fontsize=9)

    # Draw mesh edges
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax.triplot(triangulation, 'k-', linewidth=0.2, alpha=0.3)

    if mesh.coord_system == 'axisymmetric':
        ax.set_xlabel('r [m]')
        ax.set_ylabel('z [m]')
    else:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    ax.set_aspect('equal')
    if title is None:
        title = f'FEA Mesh (Level {level}) - {mesh.n_nodes} nodes, {mesh.n_elements} elements'
    ax.set_title(title)

    return fig, ax


def plot_field(mesh, field, ax=None, title=None, cmap='RdYlBu_r',
               label=None, n_levels=20, vmin=None, vmax=None):
    """Contour plot of scalar field on mesh using tricontourf.

    For Tri6 meshes, uses the Tri3 subset (corner nodes) for plotting.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    field : ndarray, shape (N_nodes,)
        Nodal scalar field values.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.
    title : str or None
        Plot title.
    cmap : str
        Colormap name. Default 'RdYlBu_r'.
    label : str or None
        Colorbar label.
    n_levels : int
        Number of contour levels. Default 20.
    vmin : float or None
        Minimum value for color scale.
    vmax : float or None
        Maximum value for color scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    nodes = mesh.nodes
    elements = mesh.elements[:, :3]  # Corner nodes only

    # Only use nodes referenced by elements
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    if vmin is None:
        vmin = np.min(field)
    if vmax is None:
        vmax = np.max(field)

    levels = np.linspace(vmin, vmax, n_levels + 1)
    if np.all(levels == levels[0]):
        levels = np.array([levels[0] - 1, levels[0] + 1])

    tcf = ax.tricontourf(triangulation, field, levels=levels, cmap=cmap,
                          extend='both')
    cb = fig.colorbar(tcf, ax=ax, shrink=0.8)
    if label is not None:
        cb.set_label(label)

    # Draw mesh edges lightly
    ax.triplot(triangulation, 'k-', linewidth=0.1, alpha=0.15)

    if mesh.coord_system == 'axisymmetric':
        ax.set_xlabel('r [m]')
        ax.set_ylabel('z [m]')
    else:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_deformed(mesh, displacement, scale=1.0, ax=None, title=None):
    """Plot deformed mesh shape.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh (Tri6 from structural solver).
    displacement : ndarray, shape (N_nodes, 2)
        Nodal displacements [u_x, u_y].
    scale : float
        Displacement magnification factor. Default 1.0.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.
    title : str or None
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    nodes = mesh.nodes
    elements = mesh.elements[:, :3]

    # Original mesh (gray)
    tri_orig = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax.triplot(tri_orig, 'k-', linewidth=0.3, alpha=0.3, label='Original')

    # Deformed mesh (colored)
    deformed_nodes = nodes.copy()
    deformed_nodes[:, 0] += scale * displacement[:, 0]
    deformed_nodes[:, 1] += scale * displacement[:, 1]

    # Displacement magnitude for coloring
    disp_mag = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)

    tri_def = mtri.Triangulation(deformed_nodes[:, 0], deformed_nodes[:, 1],
                                  elements)
    tcf = ax.tricontourf(tri_def, disp_mag, levels=20, cmap='plasma')
    ax.triplot(tri_def, 'b-', linewidth=0.2, alpha=0.3)

    cb = fig.colorbar(tcf, ax=ax, shrink=0.8)
    cb.set_label('Displacement magnitude [m]')

    if mesh.coord_system == 'axisymmetric':
        ax.set_xlabel('r [m]')
        ax.set_ylabel('z [m]')
    else:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    ax.set_aspect('equal')
    if title is None:
        title = f'Deformed Shape (scale={scale:.0f}x)'
    ax.set_title(title)
    ax.legend(loc='upper right')

    return fig, ax


def plot_vector_field(mesh, vectors, ax=None, title=None, scale=None):
    """Quiver plot of vector field (heat flux, etc.).

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    vectors : ndarray, shape (N_elem, 2)
        Vector field values at element centroids.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.
    title : str or None
        Plot title.
    scale : float or None
        Arrow scale factor. If None, auto-scaled.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    elements = mesh.elements[:, :3]
    # Compute element centroids
    centroids = np.mean(mesh.nodes[elements], axis=1)

    mag = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)

    q = ax.quiver(centroids[:, 0], centroids[:, 1],
                   vectors[:, 0], vectors[:, 1],
                   mag, cmap='hot_r', scale=scale,
                   alpha=0.7, width=0.002)
    cb = fig.colorbar(q, ax=ax, shrink=0.8)
    cb.set_label('Magnitude')

    if mesh.coord_system == 'axisymmetric':
        ax.set_xlabel('r [m]')
        ax.set_ylabel('z [m]')
    else:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)

    return fig, ax


def generate_fea_figures(results, output_dir='results/fea', level=1):
    """Generate all standard FEA figures and save as PNG.

    Generates:
        1. Mesh with material zones
        2. Neutron flux distribution (Level 1 only)
        3. Temperature distribution
        4. Von Mises stress distribution
        5. Displacement magnitude

    Parameters
    ----------
    results : CoupledResult
        Results from CoupledAnalysis.run().
    output_dir : str
        Directory for output PNG files.
    level : int
        Geometry level (1, 2, or 3).
    """
    os.makedirs(output_dir, exist_ok=True)
    dpi = 300

    mesh = results.mesh
    thermal = results.thermal
    structural = results.structural
    neutronics = results.neutronics

    # --- 1. Mesh with material zones ---
    print(f"  Generating mesh figure...")
    fig, ax = plot_mesh(mesh, show_materials=True, level=level)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'mesh_materials.png'), dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)

    # --- 2. Neutron flux (Level 1 only) ---
    if neutronics is not None:
        print(f"  Generating neutron flux figure...")
        fig, ax = plot_field(
            mesh, neutronics.flux,
            title=f'Neutron Flux Distribution (keff = {neutronics.keff:.5f})',
            cmap='inferno',
            label='Normalized Flux [-]',
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'neutron_flux.png'), dpi=dpi,
                    bbox_inches='tight')
        plt.close(fig)

        # Power density (element-centered -> nodal average for plotting)
        print(f"  Generating power density figure...")
        from fea.postprocessing.field_output import nodal_average
        power_nodal = nodal_average(mesh, neutronics.power_density)
        fig, ax = plot_field(
            mesh, power_nodal,
            title='Fission Power Density',
            cmap='hot_r',
            label='Power Density [W/m3]',
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'power_density.png'), dpi=dpi,
                    bbox_inches='tight')
        plt.close(fig)

    # --- 3. Temperature distribution ---
    print(f"  Generating temperature figure...")
    T = thermal.temperature
    fig, ax = plot_field(
        mesh, T,
        title='Temperature Distribution',
        cmap='RdYlBu_r',
        label='Temperature [K]',
    )
    # Add temperature annotations
    T_min_idx = np.argmin(T)
    T_max_idx = np.argmax(T)
    ax.annotate(f'Tmin={T[T_min_idx]:.0f} K',
                xy=(mesh.nodes[T_min_idx, 0], mesh.nodes[T_min_idx, 1]),
                fontsize=8, color='blue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.annotate(f'Tmax={T[T_max_idx]:.0f} K',
                xy=(mesh.nodes[T_max_idx, 0], mesh.nodes[T_max_idx, 1]),
                fontsize=8, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'temperature.png'), dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)

    # --- 4. Von Mises stress ---
    print(f"  Generating von Mises stress figure...")
    from fea.postprocessing.field_output import nodal_average as na
    vm_nodal = na(structural.mesh, structural.von_mises)

    # Map Tri6 nodal values to Tri3 mesh for plotting
    # structural.mesh is Tri6, but field is on Tri6 nodes
    struct_mesh = structural.mesh
    fig, ax = plot_field(
        struct_mesh, vm_nodal / 1e6,
        title='Von Mises Stress Distribution',
        cmap='YlOrRd',
        label='Von Mises Stress [MPa]',
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'von_mises_stress.png'), dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)

    # --- 5. Displacement magnitude ---
    print(f"  Generating displacement figure...")
    disp = structural.displacement
    disp_mag = np.sqrt(disp[:, 0]**2 + disp[:, 1]**2)

    fig, ax = plot_field(
        struct_mesh, disp_mag * 1e6,
        title='Displacement Magnitude',
        cmap='plasma',
        label='Displacement [um]',
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'displacement.png'), dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)

    # --- 6. Deformed shape ---
    print(f"  Generating deformed shape figure...")
    max_disp = np.max(disp_mag)
    # Auto-scale: deformation visible as ~5% of domain size
    domain_size = max(np.max(mesh.nodes[:, 0]) - np.min(mesh.nodes[:, 0]),
                      np.max(mesh.nodes[:, 1]) - np.min(mesh.nodes[:, 1]))
    if max_disp > 0:
        scale = 0.05 * domain_size / max_disp
    else:
        scale = 1.0

    fig, ax = plot_deformed(struct_mesh, disp, scale=scale)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'deformed_shape.png'), dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)

    # --- 7. Heat flux vector field ---
    print(f"  Generating heat flux figure...")
    fig, ax = plot_vector_field(
        mesh, thermal.heat_flux,
        title='Heat Flux Vector Field',
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'heat_flux.png'), dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)

    print(f"  All figures saved to {output_dir}/")
