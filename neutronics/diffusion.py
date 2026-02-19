"""
1-Group Neutron Diffusion Equation Solver in (r, z) Cylindrical Geometry
========================================================================

Solves the steady-state neutron diffusion eigenvalue problem:

    -D nabla^2 phi + Sigma_a phi = (1/k) nu*Sigma_f phi

in (r, z) cylindrical coordinates with azimuthal symmetry (no theta
dependence). The Laplacian in cylindrical coordinates is:

    nabla^2 phi = (1/r) d/dr (r dphi/dr) + d^2 phi/dz^2

Method:
  - Finite difference discretization on an (Nr x Nz) mesh
  - Power iteration for the keff eigenvalue
  - Inner iterations use SOR (Successive Over-Relaxation) to solve
    the fixed-source diffusion equation at each power iteration step
  - Extrapolated boundary conditions: phi = 0 at r_ext = R + 2D,
    z = +-z_ext where z_ext = H/2 + 2D

Boundary Conditions:
  - Symmetry at r = 0:  dphi/dr = 0  (zero gradient)
  - Symmetry at z = 0:  dphi/dz = 0  (zero gradient, solve upper half)
  - Vacuum at r = R_ext: phi = 0
  - Vacuum at z = H_ext/2: phi = 0

The mesh covers r in [0, R_ext] and z in [0, z_ext] (upper half only,
exploiting axial symmetry about the midplane).

Convergence criteria:
  - keff: |k^(n+1) - k^n| < 1e-6
  - Flux: max|phi^(n+1) - phi^n| / max|phi^n| < 1e-5

Sources:
  - Duderstadt & Hamilton, "Nuclear Reactor Analysis", Ch. 5-6
  - Stacey, "Nuclear Reactor Physics", Ch. 5
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CORE_AVG_TEMP, U235_ENRICHMENT


def solve_diffusion(core_radius, core_height, D, sigma_a, nu_sigma_f,
                    Nr=30, Nz=40, max_iter=5000, tol_k=1e-6, tol_flux=1e-5,
                    omega=1.5):
    """Solve the 1-group diffusion eigenvalue problem using power iteration + SOR.

    The domain covers the upper-right quadrant of the (r,z) cross-section:
        r in [0, R_ext], z in [0, z_ext]
    where R_ext = R + 2D (extrapolated radius) and z_ext = H/2 + 2D.

    Args:
        core_radius: Physical core radius in m
        core_height: Physical core height in m
        D: Diffusion coefficient in m
        sigma_a: Macroscopic absorption cross-section in 1/m
        nu_sigma_f: nu times macroscopic fission cross-section in 1/m
        Nr: Number of radial mesh points (default 30)
        Nz: Number of axial mesh points (default 40)
        max_iter: Maximum power iterations (default 5000)
        tol_k: keff convergence tolerance (default 1e-6)
        tol_flux: Flux convergence tolerance (default 1e-5)
        omega: SOR relaxation parameter (default 1.5)

    Returns:
        dict:
            - keff: Effective multiplication factor
            - flux_2d: 2D flux array (Nr x Nz), normalized to max = 1
            - r_mesh: Radial mesh points in m (Nr,)
            - z_mesh: Axial mesh points in m (Nz,)
            - power_2d: Relative power distribution (Nr x Nz)
            - iterations: Number of power iterations performed
            - converged: Boolean convergence flag
    """
    # Extrapolated boundaries
    R_ext = core_radius + 2.0 * D
    z_ext = core_height / 2.0 + 2.0 * D

    # Mesh spacing
    dr = R_ext / (Nr - 1)
    dz = z_ext / (Nz - 1)

    # Mesh coordinates
    r_mesh = np.linspace(0, R_ext, Nr)
    z_mesh = np.linspace(0, z_ext, Nz)

    # Initialize flux with cosine*Bessel guess
    # J0(2.405 * r / R_ext) * cos(pi * z / (2 * z_ext))
    from scipy.special import j0 as bessel_j0
    phi = np.zeros((Nr, Nz))
    for i in range(Nr):
        for j in range(Nz):
            r_val = r_mesh[i]
            z_val = z_mesh[j]
            phi[i, j] = max(bessel_j0(2.405 * r_val / R_ext) *
                           math.cos(math.pi * z_val / (2.0 * z_ext)), 0.0)

    # Normalize
    phi_max = phi.max()
    if phi_max > 0:
        phi /= phi_max

    # Initial keff guess
    k_eff = 1.0

    # Precompute coefficients for the finite difference stencil
    # At each interior point (i,j):
    #   -D [1/r dphi/dr + d2phi/dr2 + d2phi/dz2] + Sigma_a phi = (1/k) nu_Sigma_f phi
    #
    # d2phi/dr2 ~ (phi[i+1,j] - 2*phi[i,j] + phi[i-1,j]) / dr^2
    # 1/r dphi/dr ~ (1/r_i) * (phi[i+1,j] - phi[i-1,j]) / (2*dr)
    # d2phi/dz2 ~ (phi[i,j+1] - 2*phi[i,j] + phi[i,j-1]) / dz^2
    #
    # Combined:
    #   -D * [(1/dr^2 + 1/(2*r_i*dr)) phi[i+1,j]
    #        + (1/dr^2 - 1/(2*r_i*dr)) phi[i-1,j]
    #        + (1/dz^2) phi[i,j+1]
    #        + (1/dz^2) phi[i,j-1]
    #        - (2/dr^2 + 2/dz^2) phi[i,j]]
    #   + Sigma_a phi[i,j] = S[i,j]

    dr2 = dr * dr
    dz2 = dz * dz

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        # --- Fission source ---
        source = nu_sigma_f * phi / k_eff  # (Nr, Nz)

        phi_old = phi.copy()

        # --- SOR inner iteration ---
        # Solve: -D nabla^2 phi + Sigma_a phi = source
        for sor_iter in range(50):  # Inner iterations per power iteration
            phi_prev = phi.copy()

            for i in range(Nr):
                for j in range(Nz):
                    # Boundary: phi = 0 at r = R_ext and z = z_ext
                    if i == Nr - 1 or j == Nz - 1:
                        phi[i, j] = 0.0
                        continue

                    r_i = r_mesh[i]

                    # Neighbor values with boundary conditions
                    # r = 0 symmetry: dphi/dr = 0 -> phi[-1,j] = phi[1,j]
                    if i == 0:
                        # At r=0, use L'Hopital: (1/r)(dphi/dr) -> d2phi/dr2
                        # So Laplacian = 2 * d2phi/dr2 + d2phi/dz2
                        phi_ip = phi[i + 1, j]
                        phi_im = phi[i + 1, j]  # symmetry: phi(-dr) = phi(dr)

                        # Coefficients for r=0 case
                        a_r = D * 2.0 / dr2   # coefficient of phi[i+1]
                        a_l = D * 2.0 / dr2   # coefficient of phi[i-1] (= phi[i+1] by symmetry)
                        a_center_r = D * 4.0 / dr2  # center coefficient from radial
                    else:
                        phi_ip = phi[i + 1, j] if i + 1 < Nr else 0.0
                        phi_im = phi[i - 1, j]

                        a_r = D * (1.0 / dr2 + 1.0 / (2.0 * r_i * dr))
                        a_l = D * (1.0 / dr2 - 1.0 / (2.0 * r_i * dr))
                        a_center_r = D * 2.0 / dr2

                    # z = 0 symmetry: dphi/dz = 0 -> phi[i,-1] = phi[i,1]
                    if j == 0:
                        phi_jp = phi[i, j + 1]
                        phi_jm = phi[i, j + 1]  # symmetry
                    else:
                        phi_jp = phi[i, j + 1] if j + 1 < Nz else 0.0
                        phi_jm = phi[i, j - 1]

                    a_u = D / dz2
                    a_d = D / dz2
                    a_center_z = D * 2.0 / dz2

                    # Total center coefficient
                    a_center = a_center_r + a_center_z + sigma_a

                    # Right-hand side
                    rhs = source[i, j] + a_r * phi_ip + a_l * phi_im + a_u * phi_jp + a_d * phi_jm

                    # SOR update
                    phi_new = rhs / a_center
                    phi[i, j] = (1.0 - omega) * phi[i, j] + omega * phi_new

            # Check inner convergence
            phi_max_val = np.abs(phi).max()
            if phi_max_val > 0:
                inner_err = np.abs(phi - phi_prev).max() / phi_max_val
                if inner_err < 1e-6:
                    break

        # --- Update keff ---
        # k_new = k_old * (integral of nu_Sigma_f * phi_new) / (integral of nu_Sigma_f * phi_old)
        # Using volume-weighted integration in cylindrical coordinates

        # Volume element: 2*pi*r*dr*dz (factor of 2 for full height symmetry)
        # Build volume weight array
        r_weights = r_mesh.copy()
        r_weights[0] = dr / 4.0  # Avoid r=0; use average radius of first cell

        fission_new = np.zeros((Nr, Nz))
        fission_old = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                vol = 2.0 * math.pi * r_weights[i] * dr * dz
                fission_new[i, j] = nu_sigma_f * phi[i, j] * vol
                fission_old[i, j] = nu_sigma_f * phi_old[i, j] * vol

        sum_new = fission_new.sum()
        sum_old = fission_old.sum()

        if sum_old > 0:
            k_new = k_eff * sum_new / sum_old
        else:
            k_new = 1.0

        # Check convergence
        k_err = abs(k_new - k_eff)

        # Normalize flux
        phi_max_val = phi.max()
        if phi_max_val > 0:
            phi /= phi_max_val

        flux_err = np.abs(phi - phi_old / (phi_old.max() if phi_old.max() > 0 else 1.0)).max()

        k_eff = k_new
        n_iter = iteration + 1

        if k_err < tol_k and flux_err < tol_flux:
            converged = True
            break

    # Power distribution proportional to Sigma_f * phi
    power_2d = phi.copy()  # Proportional to fission rate (uniform Sigma_f)

    return {
        'keff': k_eff,
        'flux_2d': phi,
        'r_mesh': r_mesh,
        'z_mesh': z_mesh,
        'power_2d': power_2d,
        'iterations': n_iter,
        'converged': converged,
        'R_ext': R_ext,
        'z_ext': z_ext,
    }


def get_power_profile(result):
    """Extract axial and radial power profiles from diffusion solution.

    Integrates the 2D power distribution over one coordinate to obtain
    the 1D profile in the other:
      - Axial profile: integrate over r (volume-weighted)
      - Radial profile: integrate over z

    Args:
        result: Dict returned by solve_diffusion()

    Returns:
        dict:
            - axial_z: Axial coordinate array (m)
            - axial_power: Axial power profile (normalized to peak = 1)
            - radial_r: Radial coordinate array (m)
            - radial_power: Radial power profile (normalized to peak = 1)
            - axial_peaking: Axial peaking factor (max/average)
            - radial_peaking: Radial peaking factor (max/average)
    """
    flux = result['flux_2d']
    r_mesh = result['r_mesh']
    z_mesh = result['z_mesh']

    Nr = len(r_mesh)
    Nz = len(z_mesh)
    dr = r_mesh[1] - r_mesh[0] if Nr > 1 else 1.0

    # --- Axial power profile: P(z) = integral of phi(r,z) * 2*pi*r*dr ---
    axial_power = np.zeros(Nz)
    for j in range(Nz):
        for i in range(Nr):
            r_val = r_mesh[i]
            if i == 0:
                r_val = dr / 4.0
            axial_power[j] += flux[i, j] * 2.0 * math.pi * r_val * dr

    # Normalize
    ap_max = axial_power.max()
    if ap_max > 0:
        axial_power /= ap_max

    # --- Radial power profile: P(r) = integral of phi(r,z) * dz ---
    dz = z_mesh[1] - z_mesh[0] if Nz > 1 else 1.0
    radial_power = np.zeros(Nr)
    for i in range(Nr):
        for j in range(Nz):
            radial_power[i] += flux[i, j] * dz

    rp_max = radial_power.max()
    if rp_max > 0:
        radial_power /= rp_max

    # --- Peaking factors ---
    # Axial peaking: ratio of maximum to average (excluding zero boundary)
    # Use the physical region (before extrapolation boundary)
    axial_avg = axial_power[axial_power > 0].mean() if np.any(axial_power > 0) else 1.0
    axial_peaking = axial_power.max() / axial_avg if axial_avg > 0 else 1.0

    radial_avg = radial_power[radial_power > 0].mean() if np.any(radial_power > 0) else 1.0
    radial_peaking = radial_power.max() / radial_avg if radial_avg > 0 else 1.0

    return {
        'axial_z': z_mesh,
        'axial_power': axial_power,
        'radial_r': r_mesh,
        'radial_power': radial_power,
        'axial_peaking': axial_peaking,
        'radial_peaking': radial_peaking,
    }


def print_diffusion_results(result, profiles=None):
    """Print formatted diffusion solver results.

    Args:
        result: Dict returned by solve_diffusion()
        profiles: Dict returned by get_power_profile() (computed if None)
    """
    if profiles is None:
        profiles = get_power_profile(result)

    print("=" * 72)
    print("  1-GROUP NEUTRON DIFFUSION SOLUTION")
    print("=" * 72)

    print(f"\n  --- Eigenvalue ---")
    print(f"    keff:                 {result['keff']:12.6f}")
    print(f"    Reactivity:           {(result['keff'] - 1.0) / result['keff'] * 1e5:12.1f} pcm")
    print(f"    Converged:            {'Yes' if result['converged'] else 'No'}")
    print(f"    Iterations:           {result['iterations']:12d}")

    print(f"\n  --- Mesh ---")
    print(f"    Nr x Nz:              {len(result['r_mesh'])} x {len(result['z_mesh'])}")
    print(f"    R_ext:                {result['R_ext']:12.4f} m")
    print(f"    z_ext:                {result['z_ext']:12.4f} m")

    print(f"\n  --- Power Distribution ---")
    print(f"    Axial Peaking:        {profiles['axial_peaking']:12.3f}")
    print(f"    Radial Peaking:       {profiles['radial_peaking']:12.3f}")

    # Print axial profile at a few points
    print(f"\n  --- Axial Power Profile (at core centerline) ---")
    Nz = len(profiles['axial_z'])
    n_print = min(11, Nz)
    indices = np.linspace(0, Nz - 1, n_print, dtype=int)
    print(f"    {'z (m)':>10s}  {'P(z)/P_max':>12s}")
    for j in indices:
        print(f"    {profiles['axial_z'][j]:10.4f}  {profiles['axial_power'][j]:12.4f}")

    # Print radial profile at a few points
    print(f"\n  --- Radial Power Profile (at midplane) ---")
    Nr = len(profiles['radial_r'])
    n_print = min(11, Nr)
    indices = np.linspace(0, Nr - 1, n_print, dtype=int)
    print(f"    {'r (m)':>10s}  {'P(r)/P_max':>12s}")
    for i in indices:
        print(f"    {profiles['radial_r'][i]:10.4f}  {profiles['radial_power'][i]:12.4f}")

    print()


if __name__ == '__main__':
    from neutronics.cross_sections import compute_homogenized_cross_sections
    from neutronics.core_geometry import design_core

    print("Setting up diffusion problem...\n")

    # Get core geometry
    geom = design_core()
    print(f"  Core radius: {geom.core_radius:.4f} m")
    print(f"  Core height: {geom.core_height:.4f} m")

    # Get cross-sections
    xs = compute_homogenized_cross_sections()
    print(f"  D:           {xs['D']:.6f} m")
    print(f"  Sigma_a:     {xs['sigma_a']:.4f} 1/m")
    print(f"  nu*Sigma_f:  {xs['nu_sigma_f']:.4f} 1/m")

    print(f"\nSolving diffusion equation (Nr=30, Nz=40)...")
    result = solve_diffusion(
        core_radius=geom.core_radius,
        core_height=geom.core_height,
        D=xs['D'],
        sigma_a=xs['sigma_a'],
        nu_sigma_f=xs['nu_sigma_f'],
        Nr=30, Nz=40,
    )

    profiles = get_power_profile(result)
    print_diffusion_results(result, profiles)
