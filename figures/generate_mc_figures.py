#!/usr/bin/env python3
"""
Monte Carlo Neutronics Chapter - Publication-Quality Figures
40 MWth Marine MSR Design Report

Generates 5 figures for Chapter 10 (Monte Carlo neutronics analysis):
  1. mc_hex_lattice.png     - Core cross-section (top view, hexagonal lattice)
  2. mc_axial_section.png   - Axial cross-section (side view)
  3. mc_energy_groups.png   - 2-group energy structure diagram
  4. mc_critical_enrichment.png - Critical enrichment bisection search
  5. mc_temperature_coeff.png   - Temperature reactivity coefficient
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import os

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 11

OUTPUT_DIR = '/Users/yoyogo/Documents/claude/msr/figures'


def save_figure(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Figure 1: Core Cross-Section (Top View) - Hex Lattice
# ===================================================================
def figure_hex_lattice():
    print("[1/5] Generating mc_hex_lattice.png ...")

    channel_d = 2.5        # cm
    channel_r = channel_d / 2.0
    pitch = 5.0            # cm flat-to-flat hex pitch
    core_radius = 62.25    # cm
    reflector_thickness = 15.0  # cm
    vessel_thickness = 2.0      # cm (visual only)

    # Generate hex lattice positions using axial coordinates
    positions = []
    # We need enough range to cover the core
    n_max = int(core_radius / pitch) + 2
    for q in range(-n_max, n_max + 1):
        for r in range(-n_max, n_max + 1):
            x = pitch * (q + r * 0.5)
            y = pitch * r * np.sqrt(3) / 2.0
            if np.sqrt(x**2 + y**2) <= core_radius - channel_r:
                positions.append((x, y))

    print(f"    Number of fuel channels: {len(positions)}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    outer_vessel_r = core_radius + reflector_thickness + vessel_thickness
    outer_reflector_r = core_radius + reflector_thickness

    # Vessel wall (outermost dark ring)
    vessel_ring = plt.Circle((0, 0), outer_vessel_r, color='#2c3e50',
                             fill=True, zorder=1)
    ax.add_patch(vessel_ring)

    # Reflector ring
    reflector_ring = plt.Circle((0, 0), outer_reflector_r, color='#95a5a6',
                                fill=True, zorder=2)
    ax.add_patch(reflector_ring)

    # Core region (graphite background)
    core_bg = plt.Circle((0, 0), core_radius, color='#d5d8dc', fill=True,
                         zorder=3)
    ax.add_patch(core_bg)

    # Fuel channels
    fuel_patches = []
    for (x, y) in positions:
        circle = Circle((x, y), channel_r)
        fuel_patches.append(circle)

    fuel_collection = PatchCollection(fuel_patches, facecolor='#e67e22',
                                      edgecolor='#d35400', linewidth=0.3,
                                      zorder=4)
    ax.add_collection(fuel_collection)

    # Labels with arrows
    # Fuel channel label
    label_x, label_y = 45, 68
    target_x, target_y = positions[len(positions) // 3]
    ax.annotate('연료 채널 (FLiBe + UF\u2084)',
                xy=(target_x, target_y), xytext=(label_x, label_y),
                fontsize=10, fontweight='bold', color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#c0392b', alpha=0.9),
                zorder=10)

    # Graphite moderator label
    ax.annotate('흑연 감속재',
                xy=(30, 30), xytext=(-70, 68),
                fontsize=10, fontweight='bold', color='#555555',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#555555', alpha=0.9),
                zorder=10)

    # Reflector label
    ax.annotate('반경방향 반사체 (15 cm)',
                xy=(core_radius + reflector_thickness / 2, 0),
                xytext=(55, -75),
                fontsize=10, fontweight='bold', color='#7f8c8d',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#7f8c8d', alpha=0.9),
                zorder=10)

    # Scale bar (20 cm)
    bar_y = -outer_vessel_r - 8
    bar_x_start = -10
    bar_length = 20
    ax.plot([bar_x_start, bar_x_start + bar_length], [bar_y, bar_y],
            'k-', linewidth=2.5, zorder=10)
    ax.plot([bar_x_start, bar_x_start], [bar_y - 1.5, bar_y + 1.5],
            'k-', linewidth=2.5, zorder=10)
    ax.plot([bar_x_start + bar_length, bar_x_start + bar_length],
            [bar_y - 1.5, bar_y + 1.5], 'k-', linewidth=2.5, zorder=10)
    ax.text(bar_x_start + bar_length / 2, bar_y - 4, '20 cm',
            ha='center', fontsize=10, fontweight='bold', zorder=10)

    # Axis setup
    lim = outer_vessel_r + 12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - 8, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('y (cm)', fontsize=12)
    ax.set_title('그림 10.1 - 노심 수평 단면도 (육각격자 연료배열)',
                 fontsize=14, fontweight='bold', pad=15)

    # Light grid
    ax.grid(True, alpha=0.15, linestyle='--')

    fig.tight_layout()
    save_figure(fig, 'mc_hex_lattice.png')


# ===================================================================
# Figure 2: Axial Cross-Section (Side View)
# ===================================================================
def figure_axial_section():
    print("[2/5] Generating mc_axial_section.png ...")

    core_w = 124.5  # cm (diameter)
    core_h = 149.4  # cm (active height)
    refl_t = 15.0   # cm reflector thickness
    void_t = 20.0   # cm void space above/below
    vessel_t = 3.0   # cm vessel wall thickness

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Total dimensions
    total_w = core_w + 2 * refl_t + 2 * vessel_t
    total_h = core_h + 2 * refl_t + 2 * void_t + 2 * vessel_t

    # Origin at center of core
    # Core region
    core_left = -core_w / 2
    core_bot = -core_h / 2

    # Draw from outside in

    # Vessel wall
    vx = -(core_w / 2 + refl_t + vessel_t)
    vy = -(core_h / 2 + refl_t + void_t + vessel_t)
    vw = total_w
    vh = total_h
    vessel = Rectangle((vx, vy), vw, vh, facecolor='#2c3e50',
                        edgecolor='#1a252f', linewidth=2, zorder=1)
    ax.add_patch(vessel)

    # Void regions (top and bottom, inside vessel)
    inner_w = core_w + 2 * refl_t
    # Top void
    void_top = Rectangle((-inner_w / 2, core_h / 2 + refl_t),
                          inner_w, void_t,
                          facecolor='white', edgecolor='#555', linewidth=0.8,
                          zorder=2)
    ax.add_patch(void_top)
    # Bottom void
    void_bot = Rectangle((-inner_w / 2, -(core_h / 2 + refl_t + void_t)),
                          inner_w, void_t,
                          facecolor='white', edgecolor='#555', linewidth=0.8,
                          zorder=2)
    ax.add_patch(void_bot)

    # Radial reflector (left and right)
    # Left reflector
    refl_left = Rectangle((-(core_w / 2 + refl_t), -(core_h / 2 + refl_t)),
                           refl_t, core_h + 2 * refl_t,
                           facecolor='#aab7b8', edgecolor='#555',
                           linewidth=0.8, zorder=3)
    ax.add_patch(refl_left)
    # Right reflector
    refl_right = Rectangle((core_w / 2, -(core_h / 2 + refl_t)),
                            refl_t, core_h + 2 * refl_t,
                            facecolor='#aab7b8', edgecolor='#555',
                            linewidth=0.8, zorder=3)
    ax.add_patch(refl_right)

    # Axial reflector (top and bottom)
    refl_top = Rectangle((-core_w / 2, core_h / 2),
                          core_w, refl_t,
                          facecolor='#aab7b8', edgecolor='#555',
                          linewidth=0.8, zorder=3)
    ax.add_patch(refl_top)
    refl_bot = Rectangle((-core_w / 2, -(core_h / 2 + refl_t)),
                          core_w, refl_t,
                          facecolor='#aab7b8', edgecolor='#555',
                          linewidth=0.8, zorder=3)
    ax.add_patch(refl_bot)

    # Core region (graphite background)
    core_rect = Rectangle((core_left, core_bot), core_w, core_h,
                           facecolor='#d5d8dc', edgecolor='#555',
                           linewidth=1.0, zorder=4)
    ax.add_patch(core_rect)

    # Fuel channels as vertical stripes in core
    n_stripes = 25
    stripe_spacing = core_w / (n_stripes + 1)
    for i in range(1, n_stripes + 1):
        sx = core_left + i * stripe_spacing
        stripe = Rectangle((sx - 1.0, core_bot + 3), 2.0, core_h - 6,
                            facecolor='#e67e22', edgecolor='none',
                            alpha=0.8, zorder=5)
        ax.add_patch(stripe)

    # Region labels
    ax.text(0, 0, '노심 활성 영역\n(흑연 + 연료채널)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.85, edgecolor='#2c3e50'),
            zorder=8)

    ax.text(0, core_h / 2 + refl_t / 2, '축방향 반사체',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#555', zorder=8)

    ax.text(0, -(core_h / 2 + refl_t / 2), '축방향 반사체',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#555', zorder=8)

    ax.text(-(core_w / 2 + refl_t / 2), 0, '반\n사\n체',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#555', rotation=0, zorder=8)

    ax.text((core_w / 2 + refl_t / 2), 0, '반\n사\n체',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#555', rotation=0, zorder=8)

    ax.text(0, core_h / 2 + refl_t + void_t / 2, '공극 (Void)',
            ha='center', va='center', fontsize=9, color='#777', zorder=8)

    ax.text(0, -(core_h / 2 + refl_t + void_t / 2), '공극 (Void)',
            ha='center', va='center', fontsize=9, color='#777', zorder=8)

    # --- Dimension arrows ---
    arrow_color = '#c0392b'
    dim_fontsize = 9

    # Helper for dimension lines
    def dim_arrow(ax, x1, y1, x2, y2, text, text_offset=(0, 0),
                  ha='center', va='center', rotation=0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='<->', color=arrow_color,
                                    lw=1.3),
                    zorder=9)
        tx = (x1 + x2) / 2 + text_offset[0]
        ty = (y1 + y2) / 2 + text_offset[1]
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=dim_fontsize,
                fontweight='bold', color=arrow_color, rotation=rotation,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.9),
                zorder=10)

    # Core width dimension (bottom)
    dy = -(core_h / 2 + refl_t + void_t + vessel_t + 12)
    dim_arrow(ax, -core_w / 2, dy, core_w / 2, dy,
              f'{core_w} cm')

    # Core height dimension (right side)
    dx = core_w / 2 + refl_t + vessel_t + 15
    dim_arrow(ax, dx, -core_h / 2, dx, core_h / 2,
              f'{core_h} cm', text_offset=(2, 0), rotation=90)

    # Reflector thickness (top)
    ref_y = core_h / 2 + refl_t + void_t + vessel_t + 8
    dim_arrow(ax, core_w / 2, ref_y, core_w / 2 + refl_t, ref_y,
              f'{refl_t} cm', text_offset=(0, 5))

    # Axial reflector thickness (left side)
    ref_x = -(core_w / 2 + refl_t + vessel_t + 15)
    dim_arrow(ax, ref_x, core_h / 2, ref_x, core_h / 2 + refl_t,
              f'{refl_t} cm', text_offset=(-10, 0))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#e67e22', edgecolor='#d35400',
                       label='연료 채널 (FLiBe + UF\u2084)'),
        mpatches.Patch(facecolor='#d5d8dc', edgecolor='#555',
                       label='흑연 감속재'),
        mpatches.Patch(facecolor='#aab7b8', edgecolor='#555',
                       label='반사체 (흑연)'),
        mpatches.Patch(facecolor='white', edgecolor='#555',
                       label='공극 (Void)'),
        mpatches.Patch(facecolor='#2c3e50', edgecolor='#1a252f',
                       label='압력용기'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.95, edgecolor='#ccc')

    # Axis
    margin = 25
    ax.set_xlim(vx - margin, vx + vw + margin + 10)
    ax.set_ylim(vy - margin, vy + vh + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('반경 방향 (cm)', fontsize=12)
    ax.set_ylabel('축 방향 (cm)', fontsize=12)
    ax.set_title('그림 10.2 - 노심 축방향 단면도',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.1, linestyle='--')

    fig.tight_layout()
    save_figure(fig, 'mc_axial_section.png')


# ===================================================================
# Figure 3: 2-Group Energy Structure
# ===================================================================
def figure_energy_groups():
    print("[3/5] Generating mc_energy_groups.png ...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    E = np.logspace(-3, 7.2, 5000)  # eV

    # Maxwellian thermal spectrum at 650 C (923 K)
    kT = 8.617e-5 * 923  # eV
    maxwellian = np.sqrt(E) * np.exp(-E / kT)
    maxwellian = maxwellian / np.max(maxwellian)

    # Fission spectrum (Watt spectrum approximation)
    E_MeV = E / 1e6
    chi = np.exp(-E_MeV / 1.3) * np.sinh(np.sqrt(2.0 * E_MeV))
    chi = np.where(E > 1e4, chi, 0)  # only above 10 keV
    chi = chi / np.max(chi[np.isfinite(chi)])
    chi = np.nan_to_num(chi)

    # Combined spectrum envelope for visual
    combined = maxwellian * 0.85
    # Smooth blend in epithermal region
    blend_mask = (E > 0.5) & (E < 1e5)
    epi = 1.0 / E
    epi_norm = epi / np.max(epi[blend_mask]) * 0.15
    combined = np.where(blend_mask, np.maximum(combined, epi_norm), combined)
    combined = np.where(E > 1e4, np.maximum(combined, chi * 0.7), combined)

    # Background bands
    boundary = 0.625  # eV
    ax.axvspan(1e-3, boundary, alpha=0.15, color='#2980b9', zorder=0,
               label='열중성자군 (E < 0.625 eV)')
    ax.axvspan(boundary, 1.5e7, alpha=0.12, color='#e74c3c', zorder=0,
               label='고속중성자군 (E > 0.625 eV)')

    # Plot spectra
    ax.plot(E, maxwellian, color='#2980b9', linewidth=2.0,
            label='Maxwell 열 스펙트럼 (650\u00b0C)', zorder=3)
    ax.plot(E[E > 1e4], chi[E > 1e4], color='#e74c3c', linewidth=2.0,
            label='핵분열 스펙트럼 (\u03c7)', zorder=3)

    # 1/E epithermal
    epi_mask = (E > 1.0) & (E < 1e5)
    epi_spec = 0.12 / E[epi_mask] * 1e3
    ax.plot(E[epi_mask], epi_spec, color='#8e44ad', linewidth=1.5,
            linestyle='--', label='1/E 감속 스펙트럼', zorder=3, alpha=0.7)

    # Boundary line
    ax.axvline(x=boundary, color='#2c3e50', linewidth=2.5, linestyle='-',
               zorder=5)
    ax.text(boundary, 1.15, '0.625 eV\n(군 경계)',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f',
                      edgecolor='#f39c12', alpha=0.95),
            zorder=6)

    # Mark thermal peak
    E_thermal_peak = 2 * kT  # ~0.065 eV at 650 C -> actually peak of sqrt(E)*exp(-E/kT) is at E = kT/2
    # Peak of E^(1/2)*exp(-E/kT) is at E = kT/2
    E_thermal_peak = kT / 2
    ax.annotate(f'열 피크\n~{E_thermal_peak * 1000:.0f} meV\n(650\u00b0C)',
                xy=(E_thermal_peak, 0.62), xytext=(1e-2, 0.75),
                fontsize=9, fontweight='bold', color='#2980b9',
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#2980b9', alpha=0.9),
                zorder=7)

    # Mark fission peak
    ax.annotate('핵분열 피크\n~2 MeV',
                xy=(2e6, 0.72), xytext=(3e4, 0.85),
                fontsize=9, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#e74c3c', alpha=0.9),
                zorder=7)

    # Group labels
    ax.text(0.01, 0.05, '제1군 (열)', fontsize=13, fontweight='bold',
            color='#2980b9', alpha=0.6, zorder=2,
            transform=ax.transAxes)
    ax.text(0.7, 0.05, '제2군 (고속)', fontsize=13, fontweight='bold',
            color='#e74c3c', alpha=0.6, zorder=2,
            transform=ax.transAxes)

    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1.5e7)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel('에너지 (eV)', fontsize=12)
    ax.set_ylabel('상대 속밀도 (임의 단위)', fontsize=12)
    ax.set_title('그림 10.3 - 2군 에너지 구조',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2, which='both', linestyle='--')

    fig.tight_layout()
    save_figure(fig, 'mc_energy_groups.png')


# ===================================================================
# Figure 4: Critical Enrichment Search (Bisection)
# ===================================================================
def figure_critical_enrichment():
    print("[4/5] Generating mc_critical_enrichment.png ...")

    # Bisection search data: (enrichment %, k_eff, sigma)
    data = [
        (7.50, 1.327, 0.012),
        (5.25, 1.158, 0.013),
        (4.12, 1.049, 0.007),
        (3.56, 0.974, 0.008),
        (3.84, 1.034, 0.010),
        (3.70, 1.012, 0.010),
    ]

    enrich = [d[0] for d in data]
    keff = [d[1] for d in data]
    sigma = [d[2] for d in data]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Criticality line
    ax.axhline(y=1.0, color='#2c3e50', linewidth=2, linestyle='--',
               alpha=0.7, zorder=2, label='임계선 ($k_{eff}$ = 1.0)')

    # Bisection path (light connecting lines in order)
    ax.plot(enrich, keff, color='#bdc3c7', linewidth=1.2, linestyle='-',
            marker='none', zorder=3, alpha=0.7)

    # Data points with error bars
    ax.errorbar(enrich, keff, yerr=sigma, fmt='o', color='#2980b9',
                markersize=9, markeredgecolor='#1a5276',
                markeredgewidth=1.5, capsize=5, capthick=1.5,
                elinewidth=1.5, zorder=5, label='이분법 탐색점')

    # Number each point
    for i, (e, k, s) in enumerate(data):
        offset_x = 0.12
        offset_y = 0.018
        if i == 3:  # below criticality
            offset_y = -0.028
        if i == 0:
            offset_x = -0.15
        ax.text(e + offset_x, k + offset_y, str(i + 1),
                fontsize=10, fontweight='bold', color='#2c3e50',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.15', facecolor='#f0f3f4',
                          edgecolor='#2c3e50', linewidth=1.2),
                zorder=7)

    # Highlight final converged point
    ax.plot(3.70, 1.012, 'o', markersize=16, markerfacecolor='none',
            markeredgecolor='#27ae60', markeredgewidth=2.5, zorder=6)
    ax.annotate('수렴점\n(3.70%, $k_{eff}$=1.012)',
                xy=(3.70, 1.012), xytext=(4.5, 0.95),
                fontsize=10, fontweight='bold', color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#27ae60', alpha=0.95),
                zorder=8)

    # Conceptual design critical enrichment (7.35%)
    ax.plot(7.35, 1.0, 's', markersize=11, color='#e74c3c',
            markeredgecolor='#922b21', markeredgewidth=1.5, zorder=6,
            label='개념설계 임계 농축도 (7.35%)')
    ax.annotate('개념설계\n(7.35%)',
                xy=(7.35, 1.0), xytext=(7.8, 1.08),
                fontsize=9, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#e74c3c', alpha=0.9),
                zorder=8)

    # Shaded region around criticality
    ax.axhspan(0.99, 1.01, color='#27ae60', alpha=0.08, zorder=1)

    ax.set_xlabel('$^{235}$U 농축도 (%)', fontsize=12)
    ax.set_ylabel('$k_{eff}$', fontsize=13)
    ax.set_title('그림 10.4 - 이분법 임계 농축도 탐색 결과',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(3.0, 8.5)
    ax.set_ylim(0.92, 1.40)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle='--')

    fig.tight_layout()
    save_figure(fig, 'mc_critical_enrichment.png')


# ===================================================================
# Figure 5: Temperature Reactivity Coefficient
# ===================================================================
def figure_temperature_coeff():
    print("[5/5] Generating mc_temperature_coeff.png ...")

    # Data points
    T = np.array([600.0, 700.0])
    keff = np.array([1.530, 1.489])
    sigma = np.array([0.008, 0.008])

    # dk/dT in pcm/K
    dk_dT = (keff[1] - keff[0]) / (T[1] - T[0])  # per K
    dk_dT_pcm = dk_dT * 1e5  # pcm/K

    # Linear fit for extended line
    T_fit = np.linspace(550, 750, 100)
    keff_fit = keff[0] + dk_dT * (T_fit - T[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Fit line
    ax.plot(T_fit, keff_fit, color='#e74c3c', linewidth=2.0, linestyle='-',
            zorder=3, label=f'선형 추세선 (dk/dT = {dk_dT_pcm:.1f} pcm/K)')

    # Data points
    ax.errorbar(T, keff, yerr=sigma, fmt='s', color='#2980b9',
                markersize=12, markeredgecolor='#1a5276',
                markeredgewidth=2, capsize=6, capthick=2,
                elinewidth=2, zorder=5,
                label='Monte Carlo 계산 결과')

    # Annotate slope
    mid_T = 650
    mid_k = keff[0] + dk_dT * (mid_T - T[0])
    ax.annotate(
        f'dk/dT = {dk_dT_pcm:.1f} pcm/K\n'
        f'(\u0394k = {(keff[1] - keff[0]):.3f}, \u0394T = 100 K)',
        xy=(mid_T, mid_k), xytext=(mid_T + 35, mid_k + 0.020),
        fontsize=11, fontweight='bold', color='#c0392b',
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8',
                  edgecolor='#c0392b', alpha=0.95),
        zorder=8)

    # Add dashed lines to show delta
    ax.plot([600, 700], [keff[0], keff[0]], 'k--', alpha=0.3, linewidth=1)
    ax.plot([700, 700], [keff[0], keff[1]], 'k--', alpha=0.3, linewidth=1)

    # Temperature labels on points
    for t, k, s in zip(T, keff, sigma):
        ax.text(t, k + 0.015, f'{t:.0f}\u00b0C\n$k_{{eff}}$={k:.3f}\u00b1{s:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#2c3e50', alpha=0.9),
                zorder=7)

    # Negative temperature coefficient indicator
    ax.annotate('', xy=(720, 1.485), xytext=(720, 1.525),
                arrowprops=dict(arrowstyle='->', color='#27ae60',
                                lw=3, alpha=0.5),
                zorder=4)
    ax.text(730, 1.505, '음의 온도\n반응도 계수\n(안전 특성)',
            fontsize=9, color='#27ae60', fontweight='bold',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1',
                      edgecolor='#27ae60', alpha=0.9),
            zorder=7)

    ax.set_xlabel('연료염 온도 (\u00b0C)', fontsize=12)
    ax.set_ylabel('$k_{eff}$', fontsize=13)
    ax.set_title('그림 10.5 - 온도 반응도 계수 (dk/dT)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(550, 760)
    ax.set_ylim(1.45, 1.57)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle='--')

    fig.tight_layout()
    save_figure(fig, 'mc_temperature_coeff.png')


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Monte Carlo Neutronics Figures - 40 MWth Marine MSR")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    figure_hex_lattice()
    figure_axial_section()
    figure_energy_groups()
    figure_critical_enrichment()
    figure_temperature_coeff()

    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
