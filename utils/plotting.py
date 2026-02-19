"""
MSR Design Report - Plotting Utilities
Provides consistent figure formatting for the design report.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Try Korean font, fallback to default
try:
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS Korean
except:
    pass

plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')

# Color palette (colorblind-safe)
COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'danger': '#F44336',
    'success': '#4CAF50',
    'info': '#00BCD4',
    'dark': '#37474F',
    'salt': '#FF5722',
    'graphite': '#607D8B',
    'vessel': '#795548',
    'neutron': '#9C27B0',
    'gamma': '#E91E63',
}


def save_figure(fig, name, tight=True):
    """Save figure to figures directory.

    Args:
        fig: matplotlib Figure object
        name: Base filename (without extension)
        tight: Apply tight_layout before saving (default True)

    Returns:
        str: Absolute path to saved file
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {path}")
    return path


def create_dual_axis_plot(x, y1, y2, xlabel, y1label, y2label, title, filename):
    """Create a plot with two y-axes sharing the x-axis.

    Args:
        x: x-axis data array
        y1: Primary y-axis data array
        y2: Secondary y-axis data array
        xlabel: x-axis label string
        y1label: Primary y-axis label string
        y2label: Secondary y-axis label string
        title: Plot title string
        filename: Base filename for saving (without extension)

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=COLORS['primary'])
    ax1.plot(x, y1, color=COLORS['primary'], linewidth=2)
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])

    ax2 = ax1.twinx()
    ax2.set_ylabel(y2label, color=COLORS['danger'])
    ax2.plot(x, y2, color=COLORS['danger'], linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=COLORS['danger'])

    ax1.set_title(title)
    save_figure(fig, filename)
    return fig


def create_temperature_profile(z, temps_dict, title, filename):
    """Plot axial temperature distributions for multiple components.

    Args:
        z: Axial position array in meters
        temps_dict: Dict mapping label -> temperature array in Kelvin
        title: Plot title string
        filename: Base filename for saving (without extension)

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    line_styles = ['-', '--', '-.', ':']
    color_keys = ['salt', 'graphite', 'vessel', 'primary']

    for i, (label, temps) in enumerate(temps_dict.items()):
        ax.plot(
            z,
            np.array(temps) - 273.15,
            label=label,
            linewidth=2,
            linestyle=line_styles[i % len(line_styles)],
            color=COLORS.get(color_keys[i % len(color_keys)], COLORS['dark'])
        )

    ax.set_xlabel('Axial Position (m)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(title)
    ax.legend()
    save_figure(fig, filename)
    return fig


def create_transient_plot(t, variables_dict, xlabel, ylabel, title, filename, log_x=False):
    """Plot transient time histories for one or more variables.

    Args:
        t: Time array
        variables_dict: Dict mapping label -> values array
        xlabel: x-axis label string
        ylabel: y-axis label string
        title: Plot title string
        filename: Base filename for saving (without extension)
        log_x: Use logarithmic x-axis (default False)

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    for label, values in variables_dict.items():
        ax.plot(t, values, label=label, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_x:
        ax.set_xscale('log')
    ax.legend()
    save_figure(fig, filename)
    return fig


def create_core_cross_section(core_radius, channel_pitch, channel_radius, n_rings, filename):
    """Draw core cross-section showing hexagonal fuel channels in graphite moderator.

    Channels are placed on a hexagonal lattice up to n_rings rings.
    Only channels whose centre falls within 95% of core_radius are drawn.

    Args:
        core_radius: Active core radius in meters
        channel_pitch: Centre-to-centre pitch between channels in meters
        channel_radius: Individual fuel-channel radius in meters
        n_rings: Number of hexagonal rings to populate
        filename: Base filename for saving (without extension)

    Returns:
        tuple: (matplotlib.figure.Figure, int number of channels drawn)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    theta = np.linspace(0, 2 * np.pi, 100)

    # Core boundary
    ax.plot(
        core_radius * np.cos(theta),
        core_radius * np.sin(theta),
        'k-', linewidth=2, label='Core Boundary'
    )

    # Vessel wall (simplified: core_radius + 0.15 m gap)
    vessel_r = core_radius + 0.15
    ax.plot(
        vessel_r * np.cos(theta),
        vessel_r * np.sin(theta),
        color=COLORS['vessel'], linewidth=3, label='Vessel Wall'
    )

    # Build hexagonal channel positions
    channels_x = []
    channels_y = []
    for ring in range(n_rings + 1):
        if ring == 0:
            channels_x.append(0.0)
            channels_y.append(0.0)
        else:
            for i in range(6 * ring):
                angle = 2 * np.pi * i / (6 * ring)
                x = ring * channel_pitch * np.cos(angle)
                y = ring * channel_pitch * np.sin(angle)
                if np.sqrt(x**2 + y**2) <= core_radius * 0.95:
                    channels_x.append(x)
                    channels_y.append(y)

    # Draw each fuel channel
    for cx, cy in zip(channels_x, channels_y):
        circle = plt.Circle((cx, cy), channel_radius, color=COLORS['salt'], alpha=0.6)
        ax.add_patch(circle)

    # Graphite moderator background
    ax.set_facecolor('#E0E0E0')

    ax.set_xlim(-core_radius * 1.3, core_radius * 1.3)
    ax.set_ylim(-core_radius * 1.3, core_radius * 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Core Cross-Section / 노심 횡단면')
    ax.legend(loc='upper right')

    ax.annotate(
        f'Channels: {len(channels_x)}',
        xy=(0.02, 0.02),
        xycoords='axes fraction',
        fontsize=10
    )

    save_figure(fig, filename)
    return fig, len(channels_x)
