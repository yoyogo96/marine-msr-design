"""
keff Eigenvalue Solver via Power Iteration Monte Carlo
=======================================================

High-level driver for the Monte Carlo k-eigenvalue calculation of the
40 MWth Marine MSR.  Wraps the low-level transport engine (particle.py)
and tally infrastructure (tallies.py) into a clean power-iteration loop
with:

  - Configurable batch size, batch count, and inactive generations
  - Shannon entropy monitoring for fission-source convergence
  - Cumulative k_eff statistics with standard-error-of-the-mean
  - Result dataclass with full diagnostics (flux, power, peaking)
  - Progress reporting with optional verbosity control

Algorithm
---------
Standard k-eigenvalue power iteration:

    1.  Initialise N source neutrons uniformly in the fuel salt.
    2.  For each batch b = 1, ..., B:
        a.  Transport all source neutrons -> collect fission bank.
        b.  k_batch = |fission_bank| / N.
        c.  Compute Shannon entropy of fission source distribution.
        d.  Resample fission bank to exactly N sites (normalise).
        e.  If b > B_inactive:  accumulate tallies (flux, power, keff).
        f.  Print progress.
    3.  Assemble EigenvalueResult from accumulated statistics.

The inactive batches allow the fission source distribution to converge
from the initial (spatially uniform) guess toward the true fundamental
mode.  Shannon entropy is monitored to verify convergence.

References
----------
- Lux & Koblinger, "MC Particle Transport Methods," 1991, ch. 10
- Romano & Forget, "The OpenMC MC Code," Ann. Nucl. Energy, 2013
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .constants import (
    N_GROUPS, CHI, NU, ENERGY_PER_FISSION,
    MAT_FUEL_SALT, MAT_GRAPHITE_MOD, MAT_GRAPHITE_REF, MAT_VOID,
    MATERIAL_NAMES,
)
from .materials import Material, create_msr_materials
from .geometry import MSRGeometry
from .tallies import TallySystem, MeshTally, GlobalTally, create_default_tallies
from .particle import (
    run_batch,
    normalise_fission_bank,
    create_initial_source,
    sample_isotropic_direction,
)


# =====================================================================
# Result data class
# =====================================================================
@dataclass
class EigenvalueResult:
    """Container for all results from a k-eigenvalue Monte Carlo run.

    Attributes
    ----------
    keff : float
        Mean k_eff from active batches.
    keff_std : float
        Standard error of the mean k_eff (sigma / sqrt(N_active)).
    keff_history : list of float
        Per-batch k_eff values (all batches, including inactive).
    entropy_history : list of float
        Shannon entropy of the fission source per batch.
    n_active : int
        Number of active (scoring) batches used for statistics.
    n_inactive : int
        Number of inactive (discarded) batches.
    n_batches : int
        Total batches = n_active + n_inactive.
    n_particles : int
        Neutrons per batch.
    leakage_fraction : float
        Fraction of neutrons that leaked out of the geometry.
    neutron_balance : dict
        Detailed neutron balance (leakage, absorption, fission counts).
    flux_data : dict
        Mesh flux data from TallySystem.get_flux() --
        keys: 'mean' [nr, nz, ng], 'std', 'rel_err', 'ci_half'.
    power_data : dict
        Power distribution from TallySystem.get_power_distribution() --
        keys: 'power' [nr, nz] (normalised), 'power_raw', 'std'.
    peaking_factors : dict
        Peaking factor data -- keys: 'radial', 'axial', 'total',
        'radial_profile' [nr], 'axial_profile' [nz].
    mesh_r_edges : np.ndarray
        Radial bin edges used for the mesh tally [cm].
    mesh_z_edges : np.ndarray
        Axial bin edges used for the mesh tally [cm].
    total_time : float
        Wall-clock time for the entire calculation [seconds].
    tally_system : TallySystem or None
        Reference to the full tally system (for post-processing).
    """

    keff: float
    keff_std: float
    keff_history: List[float]
    entropy_history: List[float]
    n_active: int
    n_inactive: int
    n_batches: int
    n_particles: int
    leakage_fraction: float
    neutron_balance: Dict
    flux_data: Dict
    power_data: Dict
    peaking_factors: Dict
    mesh_r_edges: np.ndarray = field(repr=False)
    mesh_z_edges: np.ndarray = field(repr=False)
    total_time: float = 0.0
    tally_system: object = field(default=None, repr=False)

    # ---- Convenience properties ----
    @property
    def keff_ci_95(self):
        """95% confidence interval for k_eff."""
        return (self.keff - 1.96 * self.keff_std,
                self.keff + 1.96 * self.keff_std)

    @property
    def axial_peaking(self) -> float:
        """Axial power peaking factor."""
        return self.peaking_factors.get("axial", 1.0)

    @property
    def radial_peaking(self) -> float:
        """Radial power peaking factor."""
        return self.peaking_factors.get("radial", 1.0)

    @property
    def total_peaking(self) -> float:
        """Total (3D) power peaking factor."""
        return self.peaking_factors.get("total", 1.0)

    @property
    def flux_mean(self) -> np.ndarray:
        """Mean flux array [nr, nz, n_groups]."""
        return self.flux_data.get("mean", np.array([]))

    @property
    def power_distribution(self) -> np.ndarray:
        """Normalised power distribution [nr, nz]."""
        return self.power_data.get("power", np.array([]))

    def summary(self) -> str:
        """Return a formatted summary of the eigenvalue calculation."""
        ci_lo, ci_hi = self.keff_ci_95
        lines = [
            "",
            "=" * 70,
            "  Monte Carlo k-Eigenvalue Calculation Results",
            "=" * 70,
            "",
            f"  k_eff           = {self.keff:.5f} +/- {self.keff_std:.5f}",
            f"  95% CI          = [{ci_lo:.5f}, {ci_hi:.5f}]",
            f"  Reactivity      = {(self.keff - 1.0) / self.keff * 1e5:.0f} pcm",
            "",
            f"  Batches         = {self.n_batches} total "
            f"({self.n_inactive} inactive + {self.n_active} active)",
            f"  Particles/batch = {self.n_particles:,}",
            f"  Total histories = {self.n_batches * self.n_particles:,}",
            "",
            f"  Leakage frac    = {self.leakage_fraction:.4f} "
            f"({self.leakage_fraction * 100:.2f}%)",
            f"  Non-leakage P   = {1.0 - self.leakage_fraction:.4f}",
            "",
            f"  Axial peaking   = {self.axial_peaking:.3f}",
            f"  Radial peaking  = {self.radial_peaking:.3f}",
            f"  Total peaking   = {self.total_peaking:.3f}",
            "",
            f"  Wall time       = {self.total_time:.1f} s",
            f"  Histories/sec   = {self.n_batches * self.n_particles / max(self.total_time, 0.01):,.0f}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the formatted summary to stdout."""
        print(self.summary())


# =====================================================================
# Shannon entropy computation
# =====================================================================
def _compute_shannon_entropy(
    source_bank: List[Dict],
    r_edges: np.ndarray,
    z_edges: np.ndarray,
) -> float:
    """Compute Shannon entropy of the fission source spatial distribution.

    The source positions are binned on a coarse (r, z) mesh and the
    entropy H = -sum(p_i * ln(p_i)) is computed.  A converged fission
    source should yield a roughly constant entropy across batches.

    Parameters
    ----------
    source_bank : list of dict
        Source sites with 'position' key.
    r_edges : np.ndarray
        Radial bin edges [cm].
    z_edges : np.ndarray
        Axial bin edges [cm].

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    if len(source_bank) == 0:
        return 0.0

    positions = np.array([s["position"] for s in source_bank])
    r_vals = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    z_vals = positions[:, 2]

    hist, _, _ = np.histogram2d(r_vals, z_vals, bins=[r_edges, z_edges])
    hist = hist.ravel()
    total = hist.sum()
    if total == 0:
        return 0.0

    probs = hist / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# =====================================================================
# Eigenvalue solver
# =====================================================================
class EigenvalueSolver:
    """keff eigenvalue solver using power iteration with batch Monte Carlo.

    This is the primary high-level interface for running a k-eigenvalue
    calculation on the Marine MSR geometry.  It manages the full workflow:
    source initialisation, batch transport, fission bank normalisation,
    tally accumulation, and result assembly.

    Parameters
    ----------
    geometry : MSRGeometry
        Geometry description (hex-lattice cylindrical MSR).
    materials : dict of {int: Material}
        Material definitions keyed by material ID.
    n_particles : int
        Number of neutrons per batch (default 5000).
    n_batches : int
        Total number of batches, including inactive (default 120).
    n_inactive : int
        Number of initial batches to discard for source convergence
        (default 20).
    mesh_r_bins : int
        Number of radial bins for the flux/power mesh tally (default 20).
    mesh_z_bins : int
        Number of axial bins for the flux/power mesh tally (default 30).
    seed : int or None
        Random seed for reproducibility.  None uses non-deterministic
        seeding.

    Examples
    --------
    >>> from mc_neutronics.geometry import MSRGeometry
    >>> from mc_neutronics.materials import create_msr_materials
    >>> geom = MSRGeometry()
    >>> mats = create_msr_materials()
    >>> solver = EigenvalueSolver(geom, mats, n_particles=2000, n_batches=50)
    >>> result = solver.solve()
    >>> print(f"k_eff = {result.keff:.5f} +/- {result.keff_std:.5f}")
    """

    def __init__(
        self,
        geometry: MSRGeometry,
        materials: Dict[int, Material],
        n_particles: int = 5000,
        n_batches: int = 120,
        n_inactive: int = 20,
        mesh_r_bins: int = 20,
        mesh_z_bins: int = 30,
        seed: Optional[int] = None,
    ):
        self.geometry = geometry
        self.materials = materials
        self.n_particles = n_particles
        self.n_batches = n_batches
        self.n_inactive = n_inactive
        self.n_active = n_batches - n_inactive
        self.mesh_r_bins = mesh_r_bins
        self.mesh_z_bins = mesh_z_bins
        self.rng = np.random.default_rng(seed)

        # Validate
        if self.n_active <= 0:
            raise ValueError(
                f"n_batches ({n_batches}) must be greater than "
                f"n_inactive ({n_inactive})"
            )

        # Build mesh edges
        r_max = geometry.outer_radius
        self.r_edges = np.linspace(0.0, r_max, mesh_r_bins + 1)
        self.z_edges = np.linspace(
            -geometry.core_half_height,
            geometry.core_half_height,
            mesh_z_bins + 1,
        )

        # Coarse mesh for Shannon entropy (fewer bins for robust estimate)
        n_entropy_r = min(8, mesh_r_bins)
        n_entropy_z = min(10, mesh_z_bins)
        self.entropy_r_edges = np.linspace(0.0, r_max, n_entropy_r + 1)
        self.entropy_z_edges = np.linspace(
            -geometry.core_half_height,
            geometry.core_half_height,
            n_entropy_z + 1,
        )

    # -----------------------------------------------------------------
    # Source initialisation
    # -----------------------------------------------------------------
    def _initialize_source(self) -> List[Dict]:
        """Create the initial source bank with n_particles neutrons
        uniformly distributed in the fuel salt.

        Each source neutron has:
          - position: random point in a fuel channel
          - direction: isotropic random unit vector
          - group: sampled from the fission spectrum chi
          - weight: 1.0

        Returns
        -------
        list of dict
            Source bank ready for run_batch().
        """
        return create_initial_source(
            self.n_particles, self.geometry, self.rng
        )

    # -----------------------------------------------------------------
    # Fission bank normalisation
    # -----------------------------------------------------------------
    def _normalize_fission_bank(
        self, fission_bank: List[Dict]
    ) -> List[Dict]:
        """Resample fission bank to exactly n_particles sites.

        If the fission bank has more sites than needed, randomly
        subsample (without replacement if possible, with replacement
        otherwise).  If fewer, duplicate randomly.  This maintains
        a constant population across generations.

        Parameters
        ----------
        fission_bank : list of dict
            Raw fission bank from current batch transport.

        Returns
        -------
        list of dict
            New source bank of length self.n_particles.
        """
        return normalise_fission_bank(
            fission_bank, self.n_particles, self.rng
        )

    # -----------------------------------------------------------------
    # Shannon entropy
    # -----------------------------------------------------------------
    def _compute_entropy(self, source_bank: List[Dict]) -> float:
        """Compute Shannon entropy of the fission source distribution.

        Uses a coarse (r, z) mesh to bin source positions and computes
        H = -sum(p_i ln p_i).  Entropy should stabilise after the
        inactive batches, indicating source convergence.

        Parameters
        ----------
        source_bank : list of dict
            Current fission bank or source bank.

        Returns
        -------
        float
            Shannon entropy in nats.
        """
        return _compute_shannon_entropy(
            source_bank, self.entropy_r_edges, self.entropy_z_edges
        )

    # -----------------------------------------------------------------
    # Main solve loop
    # -----------------------------------------------------------------
    def solve(self, verbose: bool = True) -> EigenvalueResult:
        """Run the k-eigenvalue power iteration calculation.

        Algorithm:
          1. Initialise source uniformly in fuel salt.
          2. For each batch b = 1 .. n_batches:
             a. Create/reset tallies for this batch.
             b. Transport all source neutrons; collect fission bank.
             c. k_batch = |fission_bank| / n_particles.
             d. Compute Shannon entropy.
             e. If b > n_inactive: accumulate tally statistics.
             f. Resample fission bank to n_particles for next batch.
             g. Print progress every 10 batches (or every batch if verbose > 1).
          3. Assemble EigenvalueResult with all statistics.

        Parameters
        ----------
        verbose : bool
            If True, print progress information.  Default True.

        Returns
        -------
        EigenvalueResult
            Complete results including k_eff, flux, power, peaking.
        """
        t_start = time.perf_counter()

        if verbose:
            print("=" * 70)
            print("  Monte Carlo k-Eigenvalue Calculation")
            print("=" * 70)
            print(f"  Particles/batch:  {self.n_particles:,}")
            print(f"  Total batches:    {self.n_batches}")
            print(f"  Inactive batches: {self.n_inactive}")
            print(f"  Active batches:   {self.n_active}")
            print(f"  Mesh:             {self.mesh_r_bins} r x {self.mesh_z_bins} z")
            print(f"  Total histories:  {self.n_particles * self.n_batches:,}")
            print("-" * 70)

        # ---- Create tally system ----
        tallies = TallySystem(
            self.r_edges, self.z_edges,
            n_cells=1, n_groups=N_GROUPS,
        )

        # ---- Initialise source ----
        source_bank = self._initialize_source()

        # ---- Batch loop ----
        keff_history: List[float] = []
        entropy_history: List[float] = []
        cumulative_keff: List[float] = []  # running mean of active keff

        for batch in range(self.n_batches):
            is_active = batch >= self.n_inactive
            batch_label = "active" if is_active else "inactive"

            # Begin batch in tallies (only score during active batches)
            if is_active:
                tallies.begin_batch()

            # --- Transport all neutrons ---
            fission_bank = run_batch(
                source_bank,
                self.geometry,
                self.materials,
                tallies if is_active else None,
                self.rng,
            )

            # --- Batch keff ---
            n_fission = len(fission_bank)
            n_source = len(source_bank)
            keff_batch = n_fission / max(n_source, 1)
            keff_history.append(keff_batch)

            # --- Shannon entropy ---
            entropy = self._compute_entropy(fission_bank)
            entropy_history.append(entropy)

            # --- Record in global tallies ---
            tallies.record_batch_keff(n_fission, n_source)

            # --- End batch for active tallies ---
            if is_active:
                tallies.end_batch()

            # --- Cumulative keff statistics ---
            active_keffs = keff_history[self.n_inactive:]
            if len(active_keffs) > 0:
                cum_mean = float(np.mean(active_keffs))
                if len(active_keffs) >= 2:
                    cum_std = float(
                        np.std(active_keffs, ddof=1)
                        / np.sqrt(len(active_keffs))
                    )
                else:
                    cum_std = 0.0
            else:
                cum_mean = keff_batch
                cum_std = 0.0

            # --- Progress reporting ---
            if verbose:
                # Print every batch for the first few, then every 10
                should_print = (
                    batch < 5
                    or batch == self.n_inactive - 1
                    or batch == self.n_inactive
                    or (batch + 1) % 10 == 0
                    or batch == self.n_batches - 1
                )
                if should_print:
                    if is_active and len(active_keffs) >= 2:
                        print(
                            f"  Batch {batch + 1:4d}/{self.n_batches} "
                            f"({batch_label:8s}): "
                            f"k_batch = {keff_batch:.5f}, "
                            f"k_cum = {cum_mean:.5f} +/- {cum_std:.5f}, "
                            f"H = {entropy:.3f}, "
                            f"fission = {n_fission}"
                        )
                    else:
                        print(
                            f"  Batch {batch + 1:4d}/{self.n_batches} "
                            f"({batch_label:8s}): "
                            f"k_batch = {keff_batch:.5f}, "
                            f"H = {entropy:.3f}, "
                            f"fission = {n_fission}"
                        )

            # --- Normalise fission bank for next generation ---
            if len(fission_bank) == 0:
                print(
                    "  WARNING: Empty fission bank at batch "
                    f"{batch + 1}.  System may be deeply subcritical."
                )
                # Re-initialise from uniform source as fallback
                source_bank = self._initialize_source()
            else:
                source_bank = self._normalize_fission_bank(fission_bank)

        # ================================================================
        # Assemble results
        # ================================================================
        t_end = time.perf_counter()
        total_time = t_end - t_start

        # k_eff statistics from active batches
        keff_data = tallies.get_keff(n_inactive=self.n_inactive)
        keff_mean = keff_data["mean"]
        keff_std = keff_data["std"]

        # Neutron balance
        neutron_balance = tallies.get_neutron_balance()
        leakage_frac = tallies.glob.get_leakage_fraction()

        # Flux and power from mesh tallies
        flux_data = tallies.get_flux()
        power_data = tallies.get_power_distribution()
        peaking_data = tallies.get_peaking_factors()

        result = EigenvalueResult(
            keff=keff_mean,
            keff_std=keff_std,
            keff_history=keff_history,
            entropy_history=entropy_history,
            n_active=self.n_active,
            n_inactive=self.n_inactive,
            n_batches=self.n_batches,
            n_particles=self.n_particles,
            leakage_fraction=leakage_frac,
            neutron_balance=neutron_balance,
            flux_data=flux_data,
            power_data=power_data,
            peaking_factors=peaking_data,
            mesh_r_edges=self.r_edges.copy(),
            mesh_z_edges=self.z_edges.copy(),
            total_time=total_time,
            tally_system=tallies,
        )

        if verbose:
            result.print_summary()

        return result


# =====================================================================
# Convenience functions
# =====================================================================
def quick_eigenvalue(
    n_particles: int = 2000,
    n_batches: int = 50,
    n_inactive: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> EigenvalueResult:
    """Quick MC eigenvalue run with reduced statistics for testing.

    Creates standard MSR geometry and materials, then runs a short
    eigenvalue calculation.  Good for verifying that the transport
    code is working before committing to a full calculation.

    Parameters
    ----------
    n_particles : int
        Neutrons per batch (default 2000).
    n_batches : int
        Total batches (default 50).
    n_inactive : int
        Inactive batches (default 10).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    EigenvalueResult
        Calculation results.

    Examples
    --------
    >>> from mc_neutronics.eigenvalue import quick_eigenvalue
    >>> result = quick_eigenvalue(n_particles=500, n_batches=20, n_inactive=5)
    >>> print(f"k_eff = {result.keff:.4f} +/- {result.keff_std:.4f}")
    """
    geometry = MSRGeometry()
    materials = create_msr_materials()

    solver = EigenvalueSolver(
        geometry=geometry,
        materials=materials,
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=n_inactive,
        mesh_r_bins=15,
        mesh_z_bins=20,
        seed=seed,
    )

    return solver.solve(verbose=verbose)


def production_eigenvalue(
    enrichment: float = 0.12,
    temperature: float = 923.15,
    n_particles: int = 5000,
    n_batches: int = 150,
    n_inactive: int = 30,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> EigenvalueResult:
    """Production-quality MC eigenvalue run with good statistics.

    Uses finer mesh tallies and more batches than the quick version.
    Intended for generating results suitable for publication or design
    comparison.

    Parameters
    ----------
    enrichment : float
        U-235 mass fraction (default 0.12 = 12% HALEU).
    temperature : float
        Core average temperature [K] (default 923.15 K = 650 C).
    n_particles : int
        Neutrons per batch (default 5000).
    n_batches : int
        Total batches (default 150).
    n_inactive : int
        Inactive batches (default 30).
    seed : int or None
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    EigenvalueResult
        Full calculation results with good statistics.
    """
    geometry = MSRGeometry()
    materials = create_msr_materials(enrichment=enrichment,
                                     temperature=temperature)

    solver = EigenvalueSolver(
        geometry=geometry,
        materials=materials,
        n_particles=n_particles,
        n_batches=n_batches,
        n_inactive=n_inactive,
        mesh_r_bins=25,
        mesh_z_bins=40,
        seed=seed,
    )

    return solver.solve(verbose=verbose)


def enrichment_search(
    target_keff: float = 1.0,
    keff_tolerance: float = 0.01,
    enrichment_range: tuple = (0.05, 0.20),
    n_particles: int = 3000,
    n_batches: int = 80,
    n_inactive: int = 15,
    max_iterations: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Search for the enrichment that gives a target k_eff.

    Uses a simple bisection approach:  run MC at two enrichments,
    bracket the target, then bisect until convergence.

    Parameters
    ----------
    target_keff : float
        Target k_eff (default 1.0 for exactly critical).
    keff_tolerance : float
        Convergence tolerance on k_eff (default 0.01).
    enrichment_range : tuple
        (min, max) enrichment to search (default 5-20%).
    n_particles : int
        Neutrons per batch for each MC run.
    n_batches : int
        Total batches per MC run.
    n_inactive : int
        Inactive batches per MC run.
    max_iterations : int
        Maximum bisection iterations.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'enrichment': critical enrichment,
        'keff': k_eff at that enrichment,
        'keff_std': statistical uncertainty,
        'iterations': list of (enrichment, keff, keff_std),
        'converged': bool
    """
    e_lo, e_hi = enrichment_range
    iterations = []

    geometry = MSRGeometry()

    for i in range(max_iterations):
        e_mid = (e_lo + e_hi) / 2.0
        materials = create_msr_materials(enrichment=e_mid)

        if verbose:
            print(f"\n{'='*50}")
            print(f"  Enrichment search iteration {i+1}: "
                  f"e = {e_mid*100:.2f}%")
            print(f"  Range: [{e_lo*100:.2f}%, {e_hi*100:.2f}%]")
            print(f"{'='*50}")

        solver = EigenvalueSolver(
            geometry=geometry,
            materials=materials,
            n_particles=n_particles,
            n_batches=n_batches,
            n_inactive=n_inactive,
            mesh_r_bins=10,
            mesh_z_bins=15,
            seed=seed + i,
        )

        result = solver.solve(verbose=verbose)
        iterations.append((e_mid, result.keff, result.keff_std))

        if verbose:
            print(f"  -> k_eff = {result.keff:.5f} +/- {result.keff_std:.5f}")
            print(f"     target = {target_keff:.5f}, "
                  f"delta = {result.keff - target_keff:+.5f}")

        # Check convergence
        if abs(result.keff - target_keff) < keff_tolerance:
            if verbose:
                print(f"\n  CONVERGED at enrichment = {e_mid*100:.2f}%")
            return {
                "enrichment": e_mid,
                "keff": result.keff,
                "keff_std": result.keff_std,
                "iterations": iterations,
                "converged": True,
            }

        # Bisect
        if result.keff > target_keff:
            e_hi = e_mid
        else:
            e_lo = e_mid

    # Did not converge
    e_final = (e_lo + e_hi) / 2.0
    if verbose:
        print(f"\n  NOT CONVERGED after {max_iterations} iterations.")
        print(f"  Best estimate: enrichment = {e_final*100:.2f}%")

    return {
        "enrichment": e_final,
        "keff": iterations[-1][1] if iterations else 0.0,
        "keff_std": iterations[-1][2] if iterations else 0.0,
        "iterations": iterations,
        "converged": False,
    }


def temperature_coefficient(
    T_low: float = 873.15,
    T_high: float = 973.15,
    enrichment: float = 0.12,
    n_particles: int = 3000,
    n_batches: int = 80,
    n_inactive: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Compute the temperature coefficient of reactivity from MC.

    Runs two MC calculations at different temperatures and estimates
    dk/dT from the difference.

    Parameters
    ----------
    T_low : float
        Lower temperature [K] (default 600 C).
    T_high : float
        Upper temperature [K] (default 700 C).
    enrichment : float
        U-235 mass fraction.
    n_particles : int
        Neutrons per batch.
    n_batches : int
        Total batches.
    n_inactive : int
        Inactive batches.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'dk_dT_pcm_per_K': temperature coefficient in pcm/K,
        'keff_low': k_eff at T_low,
        'keff_high': k_eff at T_high,
        'T_low': T_low, 'T_high': T_high,
        'is_negative': bool (True = inherent safety)
    """
    geometry = MSRGeometry()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Temperature Coefficient Calculation")
        print(f"  T_low = {T_low - 273.15:.0f} C, T_high = {T_high - 273.15:.0f} C")
        print(f"{'='*60}")

    # Low temperature
    if verbose:
        print(f"\n--- Low temperature: {T_low - 273.15:.0f} C ---")
    mats_lo = create_msr_materials(enrichment=enrichment, temperature=T_low)
    solver_lo = EigenvalueSolver(
        geometry, mats_lo,
        n_particles=n_particles, n_batches=n_batches,
        n_inactive=n_inactive, mesh_r_bins=10, mesh_z_bins=15,
        seed=seed,
    )
    result_lo = solver_lo.solve(verbose=verbose)

    # High temperature
    if verbose:
        print(f"\n--- High temperature: {T_high - 273.15:.0f} C ---")
    mats_hi = create_msr_materials(enrichment=enrichment, temperature=T_high)
    solver_hi = EigenvalueSolver(
        geometry, mats_hi,
        n_particles=n_particles, n_batches=n_batches,
        n_inactive=n_inactive, mesh_r_bins=10, mesh_z_bins=15,
        seed=seed + 1,
    )
    result_hi = solver_hi.solve(verbose=verbose)

    # Temperature coefficient
    dT = T_high - T_low
    dk = result_hi.keff - result_lo.keff
    dk_dT = dk / dT  # per K
    dk_dT_pcm = dk_dT * 1e5  # pcm/K

    # Propagate uncertainty
    dk_std = np.sqrt(result_lo.keff_std**2 + result_hi.keff_std**2)
    dk_dT_std_pcm = dk_std / dT * 1e5

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Temperature Coefficient Results")
        print(f"{'='*60}")
        print(f"  k_eff({T_low - 273.15:.0f} C) = "
              f"{result_lo.keff:.5f} +/- {result_lo.keff_std:.5f}")
        print(f"  k_eff({T_high - 273.15:.0f} C) = "
              f"{result_hi.keff:.5f} +/- {result_hi.keff_std:.5f}")
        print(f"  dk/dT = {dk_dT_pcm:.2f} +/- {dk_dT_std_pcm:.2f} pcm/K")
        if dk_dT < 0:
            print(f"  -> NEGATIVE temperature coefficient (inherent safety)")
        else:
            print(f"  -> WARNING: Positive temperature coefficient!")

    return {
        "dk_dT_pcm_per_K": dk_dT_pcm,
        "dk_dT_std_pcm_per_K": dk_dT_std_pcm,
        "keff_low": result_lo.keff,
        "keff_low_std": result_lo.keff_std,
        "keff_high": result_hi.keff,
        "keff_high_std": result_hi.keff_std,
        "T_low": T_low,
        "T_high": T_high,
        "is_negative": dk_dT < 0,
        "result_low": result_lo,
        "result_high": result_hi,
    }


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  eigenvalue.py -- Eigenvalue Solver Self-Test")
    print("=" * 70)

    # Quick run with reduced statistics
    result = quick_eigenvalue(
        n_particles=500,
        n_batches=30,
        n_inactive=8,
        seed=12345,
        verbose=True,
    )

    print(f"\n--- Quick Eigenvalue Result ---")
    print(f"  k_eff = {result.keff:.5f} +/- {result.keff_std:.5f}")
    ci_lo, ci_hi = result.keff_ci_95
    print(f"  95% CI: [{ci_lo:.5f}, {ci_hi:.5f}]")
    print(f"  Leakage fraction: {result.leakage_fraction:.4f}")
    print(f"  Axial peaking: {result.axial_peaking:.3f}")
    print(f"  Radial peaking: {result.radial_peaking:.3f}")
    print(f"  Wall time: {result.total_time:.1f} s")

    # Sanity checks
    assert 0.5 < result.keff < 2.0, \
        f"k_eff = {result.keff} is out of plausible range"
    assert result.keff_std > 0, "Standard error should be positive"
    assert result.total_time > 0, "Time should be positive"
    assert len(result.keff_history) == 30, "Should have 30 batch keff values"
    assert len(result.entropy_history) == 30, "Should have 30 entropy values"
    assert result.leakage_fraction >= 0.0, "Leakage should be non-negative"

    print(f"\n  Entropy convergence check:")
    # Entropy should roughly stabilise after inactive batches
    early_H = np.mean(result.entropy_history[:5])
    late_H = np.mean(result.entropy_history[-5:])
    print(f"    Early entropy (first 5):  {early_H:.3f}")
    print(f"    Late entropy  (last 5):   {late_H:.3f}")

    print(f"\n  Flux and power arrays:")
    print(f"    Flux mean shape: {result.flux_mean.shape}")
    print(f"    Power shape:     {result.power_distribution.shape}")

    print(f"\n  All eigenvalue.py self-tests PASSED.")
    print("=" * 70)
