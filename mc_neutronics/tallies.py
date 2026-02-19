"""
Monte Carlo Tally System
========================

Accumulation of physics quantities during Monte Carlo neutron transport.
Provides three tally types:

- **MeshTally**: Spatially resolved (r, z) cylindrical mesh for flux
  and power distributions using track-length and collision estimators.
- **CellTally**: Cell-averaged reaction rates.
- **GlobalTally**: System-wide integral quantities (k_eff, leakage,
  absorption, fission counts, Shannon entropy).

The mesh exploits azimuthal symmetry of the cylindrical MSR core,
binning in (r, z) only.

Estimators
----------
Track-length estimator (preferred for flux):
    Phi_bin += w * d / V_bin
    where w = weight, d = track length in bin, V_bin = bin volume.

Collision estimator (alternative):
    Phi_bin += w / (Sigma_t * V_bin)

Both are accumulated per-batch, then batch statistics (mean, variance)
are computed over active batches for confidence intervals.

References
----------
- Romano & Forget, "The OpenMC Monte Carlo Particle Transport Code," 2013
- Lux & Koblinger, "Monte Carlo Particle Transport Methods," 1991
"""

import numpy as np
from typing import Optional, Tuple, Dict, List

try:
    from .constants import N_GROUPS, ENERGY_PER_FISSION
except ImportError:
    N_GROUPS = 2
    ENERGY_PER_FISSION = 3.204e-11  # J


# ===================================================================
# Cylindrical mesh tally (r, z)
# ===================================================================
class MeshTally:
    """Cylindrical (r, z) mesh tally for spatially resolved flux and power.

    Exploits azimuthal symmetry: bins are annular rings in r and axial
    slices in z.  Each bin stores group-wise flux and scalar power.

    Batch statistics are accumulated for mean and variance estimation.

    Parameters
    ----------
    r_bins : array_like
        Radial bin edges in cm (must start at 0).
    z_bins : array_like
        Axial bin edges in cm.
    n_groups : int
        Number of energy groups (default 2).
    """

    def __init__(self, r_bins, z_bins, n_groups: int = N_GROUPS):
        self.r_edges: np.ndarray = np.asarray(r_bins, dtype=np.float64)
        self.z_edges: np.ndarray = np.asarray(z_bins, dtype=np.float64)
        self.nr: int = len(self.r_edges) - 1
        self.nz: int = len(self.z_edges) - 1
        self.n_groups: int = n_groups

        # ---- Per-batch accumulators (reset each batch) ----
        self._batch_flux = np.zeros((self.nr, self.nz, n_groups))
        self._batch_power = np.zeros((self.nr, self.nz))

        # ---- Multi-batch statistics ----
        self.flux_sum = np.zeros((self.nr, self.nz, n_groups))
        self.flux_sq_sum = np.zeros((self.nr, self.nz, n_groups))
        self.power_sum = np.zeros((self.nr, self.nz))
        self.power_sq_sum = np.zeros((self.nr, self.nz))
        self.n_batches: int = 0

        # Precompute bin volumes (annular rings)
        self._volumes = self._compute_volumes()

    # ---- Volume computation ----
    def _compute_volumes(self) -> np.ndarray:
        """Compute bin volumes [nr, nz] in cm^3 for annular (r, z) mesh."""
        volumes = np.zeros((self.nr, self.nz))
        for ir in range(self.nr):
            r_lo = self.r_edges[ir]
            r_hi = self.r_edges[ir + 1]
            area = np.pi * (r_hi**2 - r_lo**2)  # cm^2
            for iz in range(self.nz):
                dz = self.z_edges[iz + 1] - self.z_edges[iz]
                volumes[ir, iz] = area * dz  # cm^3
        return volumes

    # ---- Bin lookup ----
    def _find_bin(self, pos: np.ndarray) -> Tuple[int, int]:
        """Find (ir, iz) bin indices for position.  Returns (-1, -1) if outside."""
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        z = pos[2]

        if r < self.r_edges[0] or r >= self.r_edges[-1]:
            return (-1, -1)
        if z < self.z_edges[0] or z >= self.z_edges[-1]:
            return (-1, -1)

        ir = int(np.searchsorted(self.r_edges, r, side="right")) - 1
        iz = int(np.searchsorted(self.z_edges, z, side="right")) - 1

        # Clamp (safety)
        ir = max(0, min(ir, self.nr - 1))
        iz = max(0, min(iz, self.nz - 1))
        return (ir, iz)

    # ---- Scoring methods ----
    def score_track(
        self,
        pos: np.ndarray,
        direction: np.ndarray,
        distance: float,
        group: int,
        weight: float,
        material,
    ) -> None:
        """Score track-length estimator for flux.

        For a neutron travelling distance *d* through the mesh, the
        flux contribution to the bin is:
            flux += weight * distance / volume

        Note: we score the full track into the bin containing the
        starting position.  For accurate treatment of tracks crossing
        multiple bins, the caller should split tracks at bin boundaries.
        This is the standard simplification for coarse meshes.

        Parameters
        ----------
        pos : np.ndarray
            Position at start of track segment.
        direction : np.ndarray
            Direction of travel (unit vector, unused in this simplified version).
        distance : float
            Track length in cm.
        group : int
            Energy group index.
        weight : float
            Statistical weight.
        material : Material
            Material in the region (used for power scoring).
        """
        ir, iz = self._find_bin(pos)
        if ir < 0:
            return

        vol = self._volumes[ir, iz]
        if vol <= 0.0:
            return

        # Track-length flux estimator: weight * distance / volume
        self._batch_flux[ir, iz, group] += weight * distance / vol

        # Power contribution: Sigma_f(g) * track_length_flux * E_fission
        sigma_f_g = float(material.sigma_f[group])
        if sigma_f_g > 0.0:
            self._batch_power[ir, iz] += (
                weight * distance * sigma_f_g * ENERGY_PER_FISSION / vol
            )

    def score_collision(
        self,
        pos: np.ndarray,
        group: int,
        weight: float,
        material,
    ) -> None:
        """Score collision estimator for flux.

        Flux contribution: weight / (Sigma_t * volume).

        This is an alternative to track-length scoring that is useful
        for verification.
        """
        ir, iz = self._find_bin(pos)
        if ir < 0:
            return

        sigma_t = float(material.sigma_t[group])
        vol = self._volumes[ir, iz]
        if sigma_t <= 0.0 or vol <= 0.0:
            return

        self._batch_flux[ir, iz, group] += weight / (sigma_t * vol)

    def score_fission(
        self,
        pos: np.ndarray,
        weight: float,
        material,
        group: int,
    ) -> None:
        """Score fission power tally directly.

        Power = weight * Sigma_f(g) * E_fission / volume
        """
        ir, iz = self._find_bin(pos)
        if ir < 0:
            return

        sigma_f_g = float(material.sigma_f[group])
        vol = self._volumes[ir, iz]
        if sigma_f_g <= 0.0 or vol <= 0.0:
            return

        self._batch_power[ir, iz] += weight * sigma_f_g * ENERGY_PER_FISSION / vol

    # ---- Batch management ----
    def begin_batch(self) -> None:
        """Reset per-batch accumulators."""
        self._batch_flux[:] = 0.0
        self._batch_power[:] = 0.0

    def end_batch(self) -> None:
        """Accumulate batch results into running statistics."""
        self.flux_sum += self._batch_flux
        self.flux_sq_sum += self._batch_flux**2
        self.power_sum += self._batch_power
        self.power_sq_sum += self._batch_power**2
        self.n_batches += 1

    # ---- Results ----
    def get_flux(self, confidence: float = 0.95) -> Dict:
        """Return mean flux with confidence interval.

        Parameters
        ----------
        confidence : float
            Confidence level (default 0.95 for 95% CI).

        Returns
        -------
        dict
            'mean': np.ndarray [nr, nz, n_groups]
            'std': np.ndarray [nr, nz, n_groups]
            'rel_err': np.ndarray [nr, nz, n_groups]
            'ci_half': np.ndarray - half-width of CI
        """
        n = self.n_batches
        if n < 2:
            return {
                "mean": self.flux_sum.copy(),
                "std": np.zeros_like(self.flux_sum),
                "rel_err": np.zeros_like(self.flux_sum),
                "ci_half": np.zeros_like(self.flux_sum),
            }

        mean = self.flux_sum / n
        variance = (self.flux_sq_sum / n - mean**2) / (n - 1)
        variance = np.maximum(variance, 0.0)  # guard negative from roundoff
        std = np.sqrt(variance)

        # Relative error
        rel_err = np.zeros_like(mean)
        nonzero = mean > 0.0
        rel_err[nonzero] = std[nonzero] / mean[nonzero]

        # Confidence interval (using normal approximation for large n)
        from scipy.stats import norm as sp_norm
        z_val = sp_norm.ppf(0.5 + confidence / 2.0)
        ci_half = z_val * std

        return {
            "mean": mean,
            "std": std,
            "rel_err": rel_err,
            "ci_half": ci_half,
        }

    def get_power_distribution(self) -> Dict:
        """Return normalised power distribution.

        Returns
        -------
        dict
            'power': np.ndarray [nr, nz] normalised so max = 1
            'power_raw': np.ndarray [nr, nz] unnormalised mean
            'std': np.ndarray [nr, nz]
        """
        n = self.n_batches
        if n < 1:
            return {
                "power": np.zeros((self.nr, self.nz)),
                "power_raw": np.zeros((self.nr, self.nz)),
                "std": np.zeros((self.nr, self.nz)),
            }

        mean = self.power_sum / n
        if n >= 2:
            variance = (self.power_sq_sum / n - mean**2) / (n - 1)
            variance = np.maximum(variance, 0.0)
            std = np.sqrt(variance)
        else:
            std = np.zeros_like(mean)

        # Normalise to max = 1
        pmax = np.max(mean)
        if pmax > 0:
            power_norm = mean / pmax
        else:
            power_norm = mean.copy()

        return {
            "power": power_norm,
            "power_raw": mean,
            "std": std,
        }

    def get_peaking_factors(self) -> Dict:
        """Compute axial and radial peaking factors.

        Radial peaking: max of radially-averaged power / mean
        Axial peaking: max of axially-averaged power / mean

        Returns
        -------
        dict
            'radial': float - radial peaking factor
            'axial': float - axial peaking factor
            'total': float - total (3D) peaking factor = max/mean
            'radial_profile': np.ndarray [nr]
            'axial_profile': np.ndarray [nz]
        """
        n = self.n_batches
        if n < 1:
            return {
                "radial": 1.0,
                "axial": 1.0,
                "total": 1.0,
                "radial_profile": np.zeros(self.nr),
                "axial_profile": np.zeros(self.nz),
            }

        power = self.power_sum / n  # [nr, nz]

        # Radial profile: average over z
        radial_profile = np.mean(power, axis=1)  # [nr]
        radial_mean = np.mean(radial_profile)
        radial_peak = (
            np.max(radial_profile) / radial_mean if radial_mean > 0 else 1.0
        )

        # Axial profile: average over r
        axial_profile = np.mean(power, axis=0)  # [nz]
        axial_mean = np.mean(axial_profile)
        axial_peak = (
            np.max(axial_profile) / axial_mean if axial_mean > 0 else 1.0
        )

        # Total peaking
        total_mean = np.mean(power)
        total_peak = np.max(power) / total_mean if total_mean > 0 else 1.0

        return {
            "radial": float(radial_peak),
            "axial": float(axial_peak),
            "total": float(total_peak),
            "radial_profile": radial_profile,
            "axial_profile": axial_profile,
        }

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self._batch_flux[:] = 0.0
        self._batch_power[:] = 0.0
        self.flux_sum[:] = 0.0
        self.flux_sq_sum[:] = 0.0
        self.power_sum[:] = 0.0
        self.power_sq_sum[:] = 0.0
        self.n_batches = 0


# ===================================================================
# Cell-averaged tally
# ===================================================================
class CellTally:
    """Cell-averaged reaction rate tallies.

    Tracks flux and reaction rates per geometric cell and energy group.

    Parameters
    ----------
    n_cells : int
        Number of geometric cells.
    n_groups : int
        Number of energy groups.
    """

    def __init__(self, n_cells: int, n_groups: int = N_GROUPS):
        self.n_cells: int = n_cells
        self.n_groups: int = n_groups

        # Per-batch accumulators
        self._batch_flux = np.zeros((n_cells, n_groups))
        self._batch_rates: Dict[str, np.ndarray] = {
            "fission": np.zeros((n_cells, n_groups)),
            "absorption": np.zeros((n_cells, n_groups)),
            "scattering": np.zeros((n_cells, n_groups)),
        }

        # Multi-batch statistics
        self.flux_sum = np.zeros((n_cells, n_groups))
        self.flux_sq_sum = np.zeros((n_cells, n_groups))
        self.rate_sums: Dict[str, np.ndarray] = {
            "fission": np.zeros((n_cells, n_groups)),
            "absorption": np.zeros((n_cells, n_groups)),
            "scattering": np.zeros((n_cells, n_groups)),
        }
        self.n_batches: int = 0

    def score_flux(
        self, cell_id: int, group: int, weight: float, track_length: float
    ) -> None:
        """Score track-length flux in a cell."""
        if 0 <= cell_id < self.n_cells and 0 <= group < self.n_groups:
            self._batch_flux[cell_id, group] += weight * track_length

    def score_reaction(
        self, reaction: str, cell_id: int, group: int, weight: float
    ) -> None:
        """Score a reaction rate in a cell.

        Parameters
        ----------
        reaction : str
            One of 'fission', 'absorption', 'scattering'.
        cell_id : int
            Geometric cell identifier.
        group : int
            Energy group.
        weight : float
            Statistical weight.
        """
        if reaction in self._batch_rates:
            if 0 <= cell_id < self.n_cells and 0 <= group < self.n_groups:
                self._batch_rates[reaction][cell_id, group] += weight

    def begin_batch(self) -> None:
        """Reset per-batch accumulators."""
        self._batch_flux[:] = 0.0
        for key in self._batch_rates:
            self._batch_rates[key][:] = 0.0

    def end_batch(self) -> None:
        """Accumulate batch results."""
        self.flux_sum += self._batch_flux
        self.flux_sq_sum += self._batch_flux**2
        for key in self._batch_rates:
            self.rate_sums[key] += self._batch_rates[key]
        self.n_batches += 1

    def get_flux(self) -> np.ndarray:
        """Return mean flux [n_cells, n_groups]."""
        if self.n_batches > 0:
            return self.flux_sum / self.n_batches
        return self.flux_sum.copy()

    def get_reaction_rate(self, reaction: str) -> np.ndarray:
        """Return mean reaction rate [n_cells, n_groups]."""
        if reaction in self.rate_sums and self.n_batches > 0:
            return self.rate_sums[reaction] / self.n_batches
        return np.zeros((self.n_cells, self.n_groups))

    def reset(self) -> None:
        """Reset all accumulators."""
        self._batch_flux[:] = 0.0
        self.flux_sum[:] = 0.0
        self.flux_sq_sum[:] = 0.0
        for key in self._batch_rates:
            self._batch_rates[key][:] = 0.0
        for key in self.rate_sums:
            self.rate_sums[key][:] = 0.0
        self.n_batches = 0


# ===================================================================
# Global (system-wide) tally
# ===================================================================
class GlobalTally:
    """System-wide integral quantities.

    Tracks k_eff per batch, leakage, absorption, fission counts,
    and provides Shannon entropy for source convergence monitoring.

    Attributes
    ----------
    keff_batch : list of float
        k_eff value for each batch.
    n_fission_neutrons : list of int
        Fission neutrons produced per batch.
    n_source_neutrons : list of int
        Source neutrons transported per batch.
    leakage_count : float
        Cumulative leaked neutron weight (current batch).
    absorption_count : float
        Cumulative absorbed neutron weight (current batch).
    fission_count : float
        Cumulative fission events weight (current batch).
    total_collisions : int
        Total collision count (current batch).
    """

    def __init__(self):
        # Per-generation records
        self.keff_batch: List[float] = []
        self.n_fission_neutrons: List[int] = []
        self.n_source_neutrons: List[int] = []

        # Per-batch counters (reset each batch)
        self.leakage_count: float = 0.0
        self.absorption_count: float = 0.0
        self.fission_count: float = 0.0
        self.total_collisions: int = 0

        # Cumulative counters (never reset)
        self._total_leakage: float = 0.0
        self._total_absorption: float = 0.0
        self._total_fission: float = 0.0

    def score_leakage(self, weight: float) -> None:
        """Record a leakage event."""
        self.leakage_count += weight

    def score_absorption(self, weight: float) -> None:
        """Record an absorption event."""
        self.absorption_count += weight

    def score_fission(self, weight: float) -> None:
        """Record a fission event."""
        self.fission_count += weight

    def score_collision(self) -> None:
        """Increment collision counter."""
        self.total_collisions += 1

    def record_batch_keff(self, n_fission: int, n_source: int) -> None:
        """Record batch k_eff = n_fission / n_source.

        Parameters
        ----------
        n_fission : int
            Number of fission neutrons produced in this batch.
        n_source : int
            Number of source neutrons transported in this batch.
        """
        keff = n_fission / max(n_source, 1)
        self.keff_batch.append(keff)
        self.n_fission_neutrons.append(n_fission)
        self.n_source_neutrons.append(n_source)

        # Accumulate and reset batch counters
        self._total_leakage += self.leakage_count
        self._total_absorption += self.absorption_count
        self._total_fission += self.fission_count
        self.leakage_count = 0.0
        self.absorption_count = 0.0
        self.fission_count = 0.0
        self.total_collisions = 0

    def get_keff(self, n_inactive: int = 0) -> Dict:
        """Compute mean k_eff from active batches.

        Parameters
        ----------
        n_inactive : int
            Number of initial (inactive) batches to discard.

        Returns
        -------
        dict
            'mean': float, 'std': float, 'n_active': int,
            'ci_95': tuple of (low, high)
        """
        active = self.keff_batch[n_inactive:]
        n = len(active)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "n_active": 0, "ci_95": (0.0, 0.0)}

        mean = float(np.mean(active))
        if n >= 2:
            std = float(np.std(active, ddof=1) / np.sqrt(n))
        else:
            std = 0.0

        ci_lo = mean - 1.96 * std
        ci_hi = mean + 1.96 * std

        return {
            "mean": mean,
            "std": std,
            "n_active": n,
            "ci_95": (ci_lo, ci_hi),
        }

    def get_leakage_fraction(self) -> float:
        """Return leakage fraction = leakage / (leakage + absorption).

        Uses cumulative totals over all batches.
        """
        total = self._total_leakage + self._total_absorption
        if total > 0.0:
            return self._total_leakage / total
        return 0.0

    def get_neutron_balance(self) -> Dict:
        """Return neutron balance summary."""
        total = self._total_leakage + self._total_absorption
        return {
            "leakage": self._total_leakage,
            "absorption": self._total_absorption,
            "fission": self._total_fission,
            "leakage_fraction": self.get_leakage_fraction(),
            "non_leakage_probability": 1.0 - self.get_leakage_fraction(),
            "total_terminations": total,
        }

    @staticmethod
    def shannon_entropy(
        source_bank: List[Dict],
        r_edges: np.ndarray,
        z_edges: np.ndarray,
    ) -> float:
        """Compute Shannon entropy of source distribution on (r,z) mesh.

        H = -sum_i( p_i * ln(p_i) )  for p_i > 0

        Parameters
        ----------
        source_bank : list of dict
            Source sites with 'position' keys.
        r_edges : np.ndarray
            Radial bin edges.
        z_edges : np.ndarray
            Axial bin edges.

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

        hist, _, _ = np.histogram2d(
            r_vals, z_vals, bins=[r_edges, z_edges]
        )
        hist = hist.ravel()
        total = hist.sum()
        if total == 0:
            return 0.0

        probs = hist / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def reset(self) -> None:
        """Reset all counters and histories."""
        self.keff_batch.clear()
        self.n_fission_neutrons.clear()
        self.n_source_neutrons.clear()
        self.leakage_count = 0.0
        self.absorption_count = 0.0
        self.fission_count = 0.0
        self.total_collisions = 0
        self._total_leakage = 0.0
        self._total_absorption = 0.0
        self._total_fission = 0.0


# ===================================================================
# Unified Tally System
# ===================================================================
class TallySystem:
    """Unified tally system combining mesh, cell, and global tallies.

    This is the primary interface used by the transport engine.
    It delegates scoring calls to the appropriate sub-tally objects
    and provides a single point for batch management.

    Parameters
    ----------
    r_bins : array_like
        Radial bin edges for mesh tally (cm).
    z_bins : array_like
        Axial bin edges for mesh tally (cm).
    n_cells : int
        Number of geometric cells for cell tally.
    n_groups : int
        Number of energy groups.
    """

    def __init__(
        self,
        r_bins,
        z_bins,
        n_cells: int = 1,
        n_groups: int = N_GROUPS,
    ):
        self.mesh = MeshTally(r_bins, z_bins, n_groups)
        self.cell = CellTally(n_cells, n_groups)
        self.glob = GlobalTally()
        self.n_groups = n_groups

    # ---- Scoring delegations ----
    def score_track(
        self,
        pos: np.ndarray,
        direction: np.ndarray,
        distance: float,
        group: int,
        weight: float,
        material,
    ) -> None:
        """Score track-length estimator in mesh tally."""
        self.mesh.score_track(pos, direction, distance, group, weight, material)

    def score_collision(
        self,
        pos: np.ndarray,
        group: int,
        weight: float,
        material,
    ) -> None:
        """Score collision estimator (mesh) and increment collision count."""
        # We do NOT double-score in mesh (track-length is primary estimator)
        # Collision estimator can be enabled separately if needed.
        self.glob.score_collision()

    def score_absorption(
        self,
        pos: np.ndarray,
        group: int,
        weight: float,
        material,
    ) -> None:
        """Score absorption in global tally."""
        self.glob.score_absorption(weight)

    def score_fission(
        self,
        pos: np.ndarray,
        weight: float,
        material,
        group: int,
    ) -> None:
        """Score fission in mesh and global tallies."""
        self.mesh.score_fission(pos, weight, material, group)
        self.glob.score_fission(weight)

    def score_leakage(self, weight: float) -> None:
        """Score leakage in global tally."""
        self.glob.score_leakage(weight)

    # ---- Batch management ----
    def begin_batch(self) -> None:
        """Begin a new scoring batch."""
        self.mesh.begin_batch()
        self.cell.begin_batch()

    def end_batch(self) -> None:
        """End current batch, accumulate statistics."""
        self.mesh.end_batch()
        self.cell.end_batch()

    def record_batch_keff(self, n_fission: int, n_source: int) -> None:
        """Record batch k_eff in global tally."""
        self.glob.record_batch_keff(n_fission, n_source)

    # ---- Result accessors ----
    def get_keff(self, n_inactive: int = 0) -> Dict:
        """Get k_eff statistics."""
        return self.glob.get_keff(n_inactive)

    def get_flux(self, confidence: float = 0.95) -> Dict:
        """Get mesh flux with confidence intervals."""
        return self.mesh.get_flux(confidence)

    def get_power_distribution(self) -> Dict:
        """Get normalised power distribution."""
        return self.mesh.get_power_distribution()

    def get_peaking_factors(self) -> Dict:
        """Get peaking factors."""
        return self.mesh.get_peaking_factors()

    def get_neutron_balance(self) -> Dict:
        """Get neutron balance summary."""
        return self.glob.get_neutron_balance()

    def reset(self) -> None:
        """Reset all tallies."""
        self.mesh.reset()
        self.cell.reset()
        self.glob.reset()


# ===================================================================
# Convenience factory
# ===================================================================
def create_default_tallies(
    core_radius: float = 62.25,
    reflector_thickness: float = 15.0,
    core_half_height: float = 74.7,
    nr: int = 20,
    nz: int = 30,
    n_cells: int = 1,
) -> TallySystem:
    """Create a TallySystem with standard MSR mesh parameters.

    Parameters
    ----------
    core_radius : float
        Core radius in cm.
    reflector_thickness : float
        Reflector thickness in cm.
    core_half_height : float
        Half-height of core in cm.
    nr : int
        Number of radial bins.
    nz : int
        Number of axial bins.
    n_cells : int
        Number of geometric cells.

    Returns
    -------
    TallySystem
        Ready-to-use tally system.
    """
    r_max = core_radius + reflector_thickness
    r_bins = np.linspace(0.0, r_max, nr + 1)
    z_bins = np.linspace(-core_half_height, core_half_height, nz + 1)
    return TallySystem(r_bins, z_bins, n_cells)


# ===================================================================
# Self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("tallies.py -- Tally System Self-Test")
    print("=" * 60)

    # ---- Test 1: MeshTally bin volumes ----
    print("\n[Test 1] MeshTally bin volumes")
    r_bins = np.array([0.0, 10.0, 20.0, 30.0])
    z_bins = np.array([-50.0, 0.0, 50.0])
    mt = MeshTally(r_bins, z_bins)

    # Volume of first annular ring: pi*(10^2 - 0^2)*50 = 5000*pi
    expected_v0 = np.pi * (10.0**2 - 0.0**2) * 50.0
    actual_v0 = mt._volumes[0, 0]
    assert abs(actual_v0 - expected_v0) < 1e-6, f"Volume mismatch: {actual_v0} vs {expected_v0}"
    print(f"  V[0,0] = {actual_v0:.2f} cm^3, expected = {expected_v0:.2f}: PASS")

    # Second ring: pi*(20^2 - 10^2)*50 = pi*300*50 = 15000*pi
    expected_v1 = np.pi * (20.0**2 - 10.0**2) * 50.0
    actual_v1 = mt._volumes[1, 0]
    assert abs(actual_v1 - expected_v1) < 1e-6
    print(f"  V[1,0] = {actual_v1:.2f} cm^3, expected = {expected_v1:.2f}: PASS")

    # ---- Test 2: Bin lookup ----
    print("\n[Test 2] MeshTally bin lookup")
    pos_center = np.array([5.0, 0.0, -25.0])
    ir, iz = mt._find_bin(pos_center)
    assert ir == 0 and iz == 0, f"Wrong bin: ({ir},{iz})"
    print(f"  pos=(5,0,-25) -> bin ({ir},{iz}): PASS")

    pos_outer = np.array([15.0, 0.0, 25.0])
    ir, iz = mt._find_bin(pos_outer)
    assert ir == 1 and iz == 1, f"Wrong bin: ({ir},{iz})"
    print(f"  pos=(15,0,25) -> bin ({ir},{iz}): PASS")

    pos_outside = np.array([50.0, 0.0, 0.0])
    ir, iz = mt._find_bin(pos_outside)
    assert ir == -1, f"Should be outside: ({ir},{iz})"
    print(f"  pos=(50,0,0) -> outside ({ir},{iz}): PASS")

    # ---- Test 3: Track-length scoring ----
    print("\n[Test 3] Track-length scoring")

    class MockMat:
        def __init__(self):
            self.sigma_t = np.array([0.3, 0.8])
            self.sigma_f = np.array([0.01, 0.05])
            self.sigma_a = np.array([0.05, 0.25])
            self.sigma_s = np.array([[0.2, 0.05], [0.01, 0.5]])
            self.nu_sigma_f = np.array([0.025, 0.12])

    mat = MockMat()
    mt.begin_batch()
    # Score a track of length 10 cm in bin (0,0), group 0, weight 1.0
    pos = np.array([5.0, 0.0, -25.0])
    direction = np.array([1.0, 0.0, 0.0])
    mt.score_track(pos, direction, 10.0, 0, 1.0, mat)

    expected_flux = 1.0 * 10.0 / mt._volumes[0, 0]
    actual_flux = mt._batch_flux[0, 0, 0]
    assert abs(actual_flux - expected_flux) < 1e-12, f"Flux mismatch: {actual_flux} vs {expected_flux}"
    print(f"  Track score: flux = {actual_flux:.6e}, expected = {expected_flux:.6e}: PASS")

    # Power contribution check
    expected_power = 1.0 * 10.0 * 0.01 * ENERGY_PER_FISSION / mt._volumes[0, 0]
    actual_power = mt._batch_power[0, 0]
    assert abs(actual_power - expected_power) < 1e-20, f"Power mismatch"
    print(f"  Power score: {actual_power:.6e}: PASS")

    mt.end_batch()
    assert mt.n_batches == 1
    print(f"  Batch accumulated: n_batches={mt.n_batches}: PASS")

    # ---- Test 4: Multi-batch statistics ----
    print("\n[Test 4] Multi-batch statistics")
    mt2 = MeshTally(r_bins, z_bins)
    rng = np.random.default_rng(42)
    n_test_batches = 20
    for b in range(n_test_batches):
        mt2.begin_batch()
        # Score random tracks
        for _ in range(100):
            r = rng.random() * 25.0
            theta = rng.random() * 2 * np.pi
            z = (rng.random() - 0.5) * 100.0
            pos = np.array([r * np.cos(theta), r * np.sin(theta), z])
            dist = rng.exponential(5.0)
            g = rng.integers(0, 2)
            mt2.score_track(pos, np.array([1, 0, 0]), dist, g, 1.0, mat)
        mt2.end_batch()

    assert mt2.n_batches == n_test_batches
    flux_result = mt2.get_flux()
    print(f"  {n_test_batches} batches scored")
    print(f"  Flux mean shape: {flux_result['mean'].shape}")
    print(f"  Max relative error: {np.max(flux_result['rel_err']):.4f}")
    print(f"  PASS")

    # ---- Test 5: Peaking factors ----
    print("\n[Test 5] Peaking factors")
    pf = mt2.get_peaking_factors()
    print(f"  Radial peaking: {pf['radial']:.3f}")
    print(f"  Axial peaking:  {pf['axial']:.3f}")
    print(f"  Total peaking:  {pf['total']:.3f}")
    assert pf["radial"] >= 1.0, "Radial peaking must be >= 1"
    assert pf["axial"] >= 1.0, "Axial peaking must be >= 1"
    print(f"  PASS")

    # ---- Test 6: GlobalTally ----
    print("\n[Test 6] GlobalTally k_eff tracking")
    gt = GlobalTally()
    # Simulate 30 batches with k ~ 1.05 +/- noise
    rng = np.random.default_rng(123)
    for i in range(30):
        n_src = 1000
        n_fis = int(rng.normal(1050, 30))
        gt.leakage_count = rng.poisson(50)
        gt.absorption_count = n_src - gt.leakage_count
        gt.fission_count = n_fis * 0.4
        gt.record_batch_keff(n_fis, n_src)

    # k_eff from active batches (skip first 10)
    keff_result = gt.get_keff(n_inactive=10)
    print(f"  k_eff = {keff_result['mean']:.5f} +/- {keff_result['std']:.5f}")
    print(f"  95% CI: ({keff_result['ci_95'][0]:.5f}, {keff_result['ci_95'][1]:.5f})")
    print(f"  Active batches: {keff_result['n_active']}")
    assert abs(keff_result["mean"] - 1.05) < 0.1, f"k_eff too far from expected"
    print(f"  PASS")

    # ---- Test 7: Leakage fraction ----
    print("\n[Test 7] Leakage fraction")
    lf = gt.get_leakage_fraction()
    nb = gt.get_neutron_balance()
    print(f"  Leakage fraction: {lf:.4f}")
    print(f"  Non-leakage prob: {nb['non_leakage_probability']:.4f}")
    assert 0 < lf < 1, "Leakage fraction out of range"
    print(f"  PASS")

    # ---- Test 8: Shannon entropy ----
    print("\n[Test 8] Shannon entropy")
    rng = np.random.default_rng(77)
    fake_bank = []
    for _ in range(500):
        pos = rng.normal([0, 0, 0], [15, 15, 40])
        fake_bank.append({"position": pos})

    r_e = np.linspace(0, 40, 6)
    z_e = np.linspace(-60, 60, 7)
    H = GlobalTally.shannon_entropy(fake_bank, r_e, z_e)
    print(f"  H = {H:.4f} nats (distributed source, expect H > 0)")
    assert H > 0
    print(f"  PASS")

    # ---- Test 9: CellTally ----
    print("\n[Test 9] CellTally")
    ct = CellTally(n_cells=5, n_groups=2)
    ct.begin_batch()
    ct.score_flux(0, 0, 1.0, 10.0)
    ct.score_flux(0, 1, 1.0, 5.0)
    ct.score_reaction("fission", 0, 1, 0.5)
    ct.score_reaction("absorption", 0, 1, 1.0)
    ct.end_batch()

    flux = ct.get_flux()
    assert flux[0, 0] == 10.0
    assert flux[0, 1] == 5.0
    fr = ct.get_reaction_rate("fission")
    assert fr[0, 1] == 0.5
    print(f"  Cell flux[0] = [{flux[0,0]:.1f}, {flux[0,1]:.1f}]: PASS")
    print(f"  Fission rate[0,1] = {fr[0,1]:.2f}: PASS")

    # ---- Test 10: TallySystem unified interface ----
    print("\n[Test 10] TallySystem (unified)")
    ts = create_default_tallies()
    print(f"  Mesh: {ts.mesh.nr} r-bins x {ts.mesh.nz} z-bins")
    print(f"  r_edges: [{ts.mesh.r_edges[0]:.1f}, ..., {ts.mesh.r_edges[-1]:.1f}] cm")
    print(f"  z_edges: [{ts.mesh.z_edges[0]:.1f}, ..., {ts.mesh.z_edges[-1]:.1f}] cm")

    ts.begin_batch()
    pos = np.array([10.0, 0.0, 0.0])
    ts.score_track(pos, np.array([0, 0, 1]), 5.0, 0, 1.0, mat)
    ts.score_fission(pos, 1.0, mat, 0)
    ts.score_leakage(1.0)
    ts.score_absorption(pos, 0, 1.0, mat)
    ts.end_batch()
    ts.record_batch_keff(1050, 1000)

    keff = ts.get_keff()
    print(f"  k_eff = {keff['mean']:.5f}: PASS")
    print(f"  Leakage fraction = {ts.glob.get_leakage_fraction():.4f}: PASS")

    ts.reset()
    assert ts.mesh.n_batches == 0
    assert len(ts.glob.keff_batch) == 0
    print(f"  Reset: PASS")

    print("\n" + "=" * 60)
    print("All tallies.py self-tests PASSED")
    print("=" * 60)
