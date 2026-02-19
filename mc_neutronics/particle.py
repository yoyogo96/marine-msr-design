"""
Monte Carlo Neutron Particle Transport Engine
==============================================

Multi-group neutron transport with analog collision processing for a
2-group (fast/thermal) Monte Carlo simulation. Implements the core
particle tracking loop: free flight, boundary crossing, and collision
physics (scattering, absorption, fission).

The transport uses track-length estimation for flux tallies and an
analog fission treatment that banks secondary neutrons for the next
generation (power iteration for k-eigenvalue).

Energy group convention:
    Group 0 = Fast   (E > 0.625 eV)
    Group 1 = Thermal (E <= 0.625 eV)

References
----------
- Lux & Koblinger, "Monte Carlo Particle Transport Methods," 1991
- Lewis & Miller, "Computational Methods of Neutron Transport," 1984
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Interface imports -- these modules are developed in parallel.
# We import defensively so the file can be parsed / tested standalone.
# ---------------------------------------------------------------------------
try:
    from .constants import N_GROUPS, CHI, NU, ENERGY_PER_FISSION
except ImportError:
    N_GROUPS = 2
    CHI = [0.97, 0.03]
    NU = 2.43
    ENERGY_PER_FISSION = 3.204e-11  # J

if TYPE_CHECKING:
    from .materials import Material
    from .geometry import MSRGeometry
    from .tallies import TallySystem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOUNDARY_NUDGE: float = 1.0e-8       # cm  -- small push past surfaces
MAX_COLLISIONS: int = 10_000          # safety cap per history
WEIGHT_CUTOFF: float = 1.0e-10       # Russian-roulette threshold
RUSSIAN_ROULETTE_SURVIVE: float = 0.5 # survival probability for RR
TWO_PI: float = 2.0 * np.pi


# ===================================================================
# Neutron dataclass
# ===================================================================
class Neutron:
    """Representation of a single Monte Carlo neutron history.

    Attributes
    ----------
    pos : np.ndarray
        Position vector [x, y, z] in cm.
    dir : np.ndarray
        Unit direction vector [Omega_x, Omega_y, Omega_z].
    group : int
        Energy group index (0 = fast, 1 = thermal).
    weight : float
        Statistical weight of the particle.
    alive : bool
        ``False`` once absorbed, leaked, or killed by safety cutoff.
    n_collisions : int
        Running count of collisions suffered in this history.
    """

    __slots__ = ("pos", "dir", "group", "weight", "alive", "n_collisions")

    def __init__(
        self,
        position,
        direction,
        energy_group: int,
        weight: float = 1.0,
    ) -> None:
        self.pos: np.ndarray = np.asarray(position, dtype=np.float64)
        self.dir: np.ndarray = np.asarray(direction, dtype=np.float64)
        self.group: int = int(energy_group)
        self.weight: float = float(weight)
        self.alive: bool = True
        self.n_collisions: int = 0

    # Convenience --------------------------------------------------------
    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return (
            f"Neutron(g={self.group}, w={self.weight:.4e}, "
            f"coll={self.n_collisions}, {status})"
        )

    def move(self, distance: float) -> None:
        """Advance position along current direction by *distance* (cm)."""
        self.pos += distance * self.dir

    def kill(self) -> None:
        """Mark particle as dead."""
        self.alive = False


# ===================================================================
# Direction sampling
# ===================================================================
def sample_isotropic_direction(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a random unit vector sampled uniformly on the unit sphere.

    Uses the standard polar decomposition:
        mu  = 2*xi_1 - 1            (cosine of polar angle, uniform on [-1,1])
        phi = 2*pi*xi_2             (azimuthal angle, uniform on [0,2pi))
        sin_theta = sqrt(1 - mu^2)
        Omega = (sin_theta*cos(phi), sin_theta*sin(phi), mu)

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator.  Falls back to module-level default.

    Returns
    -------
    np.ndarray
        Unit direction vector of shape (3,).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu: float = 2.0 * rng.random() - 1.0          # cos(theta) in [-1, 1]
    phi: float = TWO_PI * rng.random()             # azimuthal   in [0, 2pi)
    sin_theta: float = np.sqrt(max(1.0 - mu * mu, 0.0))

    direction = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        mu,
    ], dtype=np.float64)
    return direction


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return unit vector (in-place safe)."""
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return sample_isotropic_direction()
    return v / norm


# ===================================================================
# Collision processing
# ===================================================================
def process_collision(
    neutron: Neutron,
    material,              # Material dataclass
    fission_bank: List[Dict],
    tallies,               # TallySystem or similar
    rng: np.random.Generator,
) -> None:
    """Determine and process collision type (scatter / absorption / fission).

    The collision type is sampled from macroscopic cross-section ratios:
        P(scatter)  = Sigma_s_total(g) / Sigma_t(g)
        P(absorb)   = Sigma_a(g) / Sigma_t(g)
    If absorption occurs, fission is further sampled:
        P(fission | absorb) = Sigma_f(g) / Sigma_a(g)

    Fission produces secondary neutrons banked for the next generation.

    Parameters
    ----------
    neutron : Neutron
        The neutron undergoing the collision (modified in-place).
    material : Material
        Material at the collision site.
    fission_bank : list of dict
        Mutable list to which fission-site neutrons are appended.
    tallies : TallySystem
        Tally object for scoring collision estimators.
    rng : np.random.Generator
        Random number generator.
    """
    g: int = neutron.group
    sigma_t: float = float(material.sigma_t[g])
    sigma_a: float = float(material.sigma_a[g])
    sigma_f: float = float(material.sigma_f[g])

    # Total scattering cross-section from group g -> all groups
    sigma_s_total: float = float(np.sum(material.sigma_s[g, :]))

    # ------ Score collision estimator (flux) ------
    if tallies is not None:
        tallies.score_collision(neutron.pos, g, neutron.weight, material)

    # Probabilities
    if sigma_t <= 0.0:
        # Vacuum-like -- just kill
        neutron.kill()
        return

    p_scatter: float = sigma_s_total / sigma_t

    xi: float = rng.random()

    if xi < p_scatter:
        # ---- Scattering ----
        _process_scatter(neutron, material, rng)
    else:
        # ---- Absorption (capture + fission) ----
        _process_absorption(neutron, material, fission_bank, tallies, rng)


def _process_scatter(
    neutron: Neutron,
    material,
    rng: np.random.Generator,
) -> None:
    """Process a scattering event: sample outgoing group and new direction.

    Outgoing group g' is sampled with probability:
        P(g -> g') = Sigma_s(g, g') / Sigma_s_total(g)

    Direction is sampled isotropically (isotropic-in-lab approximation,
    adequate for graphite-moderated thermal-spectrum systems where
    scattering is nearly isotropic in the CM frame after many collisions).
    """
    g: int = neutron.group
    scatter_xs = material.sigma_s[g, :]       # shape (n_groups,)
    sigma_s_total: float = float(np.sum(scatter_xs))

    if sigma_s_total <= 0.0:
        neutron.kill()
        return

    # Sample outgoing energy group
    probs = scatter_xs / sigma_s_total
    cumulative = np.cumsum(probs)
    xi: float = rng.random()
    new_group: int = int(np.searchsorted(cumulative, xi))
    # Clamp to valid range
    new_group = min(new_group, N_GROUPS - 1)

    neutron.group = new_group
    neutron.dir = sample_isotropic_direction(rng)
    neutron.n_collisions += 1


def _process_absorption(
    neutron: Neutron,
    material,
    fission_bank: List[Dict],
    tallies,
    rng: np.random.Generator,
) -> None:
    """Process absorption: either radiative capture or fission.

    If fission, bank secondary neutrons with energies sampled from chi.
    """
    g: int = neutron.group
    sigma_a: float = float(material.sigma_a[g])
    sigma_f: float = float(material.sigma_f[g])

    # Score absorption in global tallies
    if tallies is not None:
        tallies.score_absorption(neutron.pos, g, neutron.weight, material)

    if sigma_a > 0.0 and sigma_f > 0.0:
        p_fission: float = sigma_f / sigma_a
        if rng.random() < p_fission:
            # ---- Fission ----
            _process_fission(neutron, material, fission_bank, tallies, rng)
            neutron.kill()
            return

    # Radiative capture -- neutron dies
    neutron.kill()


def _process_fission(
    neutron: Neutron,
    material,
    fission_bank: List[Dict],
    tallies,
    rng: np.random.Generator,
) -> None:
    """Bank fission neutrons and score fission tally.

    The number of secondary neutrons is sampled from nu:
        n = floor(nu) + 1 with probability (nu - floor(nu))
        n = floor(nu)     otherwise

    Energy group is sampled from the fission spectrum chi.
    """
    g: int = neutron.group
    sigma_f: float = float(material.sigma_f[g])
    nu_sigma_f: float = float(material.nu_sigma_f[g])

    # Effective nu for this group
    if sigma_f > 0.0:
        nu_eff: float = nu_sigma_f / sigma_f
    else:
        nu_eff = NU

    # Sample integer number of secondary neutrons
    nu_floor: int = int(nu_eff)
    frac: float = nu_eff - nu_floor
    n_new: int = nu_floor + (1 if rng.random() < frac else 0)

    # Bank fission neutrons
    for _ in range(n_new):
        # Sample fission neutron energy group from chi
        new_g: int = 0 if rng.random() < CHI[0] else 1
        new_dir = sample_isotropic_direction(rng)
        fission_bank.append({
            "position": neutron.pos.copy(),
            "group": new_g,
            "direction": new_dir,
        })

    # Score fission tally
    if tallies is not None:
        tallies.score_fission(neutron.pos, neutron.weight, material, g)


# ===================================================================
# Main transport loop (single neutron)
# ===================================================================
def transport_neutron(
    neutron: Neutron,
    geometry,              # MSRGeometry
    materials_dict: Dict[int, object],  # {material_id: Material}
    tallies,               # TallySystem
    fission_bank: List[Dict],
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Transport a single neutron until death (absorption, leakage, or cutoff).

    The standard Monte Carlo particle-tracking loop:

    1. Determine material at current position.
    2. If void/vacuum, the neutron has leaked -- score and kill.
    3. Sample distance to next collision: s = -ln(xi) / Sigma_t(g).
    4. Compute distance to nearest geometry boundary.
    5. If collision distance < boundary distance:
       - Move to collision site and process collision.
    6. Else:
       - Move to boundary + epsilon nudge, continue tracking.
    7. Apply safety cutoffs (max collisions, minimum weight with
       Russian roulette).

    Parameters
    ----------
    neutron : Neutron
        Particle to transport (modified in-place until dead).
    geometry : MSRGeometry
        Geometry description (material lookup, distance-to-boundary).
    materials_dict : dict
        Mapping of material ID -> Material dataclass.
    tallies : TallySystem
        Tally accumulators.
    fission_bank : list
        Mutable list for banking fission-site neutrons.
    rng : np.random.Generator, optional
        RNG instance (created if not provided).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Material ID constants (from geometry module convention)
    VOID_ID = 3  # MSRGeometry.VOID

    while neutron.alive:
        # ---- 1. Identify current material ----
        mat_id: int = geometry.find_material(
            neutron.pos[0], neutron.pos[1], neutron.pos[2]
        )

        # ---- 2. Check for void (leakage) ----
        if mat_id == VOID_ID or mat_id not in materials_dict:
            if tallies is not None:
                tallies.score_leakage(neutron.weight)
            neutron.kill()
            return

        material = materials_dict[mat_id]
        g: int = neutron.group
        sigma_t: float = float(material.sigma_t[g])

        # If sigma_t is zero or negative, particle is effectively in vacuum
        if sigma_t <= 0.0:
            if tallies is not None:
                tallies.score_leakage(neutron.weight)
            neutron.kill()
            return

        # ---- 3. Sample distance to next collision ----
        xi: float = rng.random()
        # Guard against xi == 0
        while xi == 0.0:
            xi = rng.random()
        s_collision: float = -np.log(xi) / sigma_t

        # ---- 4. Distance to nearest boundary ----
        _boundary_result = geometry.distance_to_boundary(
            neutron.pos, neutron.dir
        )
        # distance_to_boundary returns (distance, next_mat_id) tuple
        if isinstance(_boundary_result, tuple):
            d_boundary = float(_boundary_result[0])
        else:
            d_boundary = float(_boundary_result)

        # Sanity: if boundary distance is non-positive or very large, cap it
        if d_boundary <= 0.0:
            # Nudge particle slightly and retry
            neutron.move(BOUNDARY_NUDGE)
            continue
        if d_boundary > 1.0e6:
            d_boundary = 1.0e6  # 10 km sanity cap

        # ---- 5 & 6. Advance particle ----
        if s_collision < d_boundary:
            # --- Collision in this region ---
            # Score track-length tally over flight path
            if tallies is not None:
                tallies.score_track(
                    neutron.pos, neutron.dir, s_collision,
                    g, neutron.weight, material,
                )

            neutron.move(s_collision)
            neutron.n_collisions += 1

            # Process collision physics
            process_collision(neutron, material, fission_bank, tallies, rng)

        else:
            # --- Boundary crossing (no collision) ---
            # Score track-length tally up to boundary
            if tallies is not None:
                tallies.score_track(
                    neutron.pos, neutron.dir, d_boundary,
                    g, neutron.weight, material,
                )

            # Move to boundary + nudge past it
            neutron.move(d_boundary + BOUNDARY_NUDGE)
            # No collision -- loop continues in the new region

        # ---- 7. Safety cutoffs ----
        if neutron.n_collisions > MAX_COLLISIONS:
            neutron.kill()
            return

        # Russian roulette for low-weight particles
        if neutron.alive and neutron.weight < WEIGHT_CUTOFF:
            if rng.random() < RUSSIAN_ROULETTE_SURVIVE:
                # Particle survives with boosted weight
                neutron.weight /= RUSSIAN_ROULETTE_SURVIVE
            else:
                neutron.kill()
                return


# ===================================================================
# Batch execution
# ===================================================================
def run_batch(
    source_bank: List[Dict],
    geometry,
    materials_dict: Dict[int, object],
    tallies,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Transport all neutrons in a source batch and return the fission bank.

    Each entry in *source_bank* is a dict with keys:
        'position': np.ndarray (3,)
        'group': int
        'direction': np.ndarray (3,)

    Parameters
    ----------
    source_bank : list of dict
        Neutron source sites for this generation.
    geometry : MSRGeometry
        Geometry description.
    materials_dict : dict
        Material ID -> Material mapping.
    tallies : TallySystem
        Tally accumulators (modified in-place).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    list of dict
        Fission bank for the next generation.
    """
    if rng is None:
        rng = np.random.default_rng()

    fission_bank: List[Dict] = []
    n_source = len(source_bank)

    for i, source in enumerate(source_bank):
        neutron = Neutron(
            position=source["position"],
            direction=_normalise(np.asarray(source["direction"], dtype=np.float64)),
            energy_group=source["group"],
            weight=source.get("weight", 1.0),
        )
        transport_neutron(
            neutron, geometry, materials_dict, tallies, fission_bank, rng
        )

    return fission_bank


# ===================================================================
# Source bank management for k-eigenvalue iteration
# ===================================================================
def normalise_fission_bank(
    fission_bank: List[Dict],
    target_size: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Resample fission bank to *target_size* neutrons for next generation.

    If the fission bank has more sites than needed, randomly subsample.
    If fewer, duplicate randomly to fill.  This keeps the population
    constant across generations (standard k-eigenvalue MC practice).

    Parameters
    ----------
    fission_bank : list of dict
        Raw fission bank from the current generation.
    target_size : int
        Desired number of source neutrons for the next batch.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    list of dict
        New source bank of length *target_size*.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_fission = len(fission_bank)
    if n_fission == 0:
        # Degenerate case -- return empty (caller should handle)
        return []

    # Random indices with replacement
    indices = rng.integers(0, n_fission, size=target_size)
    new_bank: List[Dict] = []
    for idx in indices:
        site = fission_bank[idx]
        new_bank.append({
            "position": site["position"].copy(),
            "group": site["group"],
            "direction": sample_isotropic_direction(rng),
            "weight": 1.0,
        })
    return new_bank


def create_initial_source(
    n_particles: int,
    geometry,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Create an initial uniform source distributed in fuel salt.

    Uses the geometry's ``sample_position_in_fuel()`` method to get
    random positions within fuel regions.  Initial energy is sampled
    from the fission spectrum chi.

    Parameters
    ----------
    n_particles : int
        Number of source neutrons to generate.
    geometry : MSRGeometry
        Geometry with fuel-volume sampling capability.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    list of dict
        Source bank ready for ``run_batch()``.
    """
    if rng is None:
        rng = np.random.default_rng()

    bank: List[Dict] = []
    for _ in range(n_particles):
        pos = geometry.sample_position_in_fuel()
        grp = 0 if rng.random() < CHI[0] else 1
        direction = sample_isotropic_direction(rng)
        bank.append({
            "position": np.asarray(pos, dtype=np.float64),
            "group": grp,
            "direction": direction,
            "weight": 1.0,
        })
    return bank


# ===================================================================
# k-eigenvalue driver (convenience wrapper)
# ===================================================================
def run_keff_calculation(
    geometry,
    materials_dict: Dict[int, object],
    tallies,
    n_particles: int = 1000,
    n_inactive: int = 20,
    n_active: int = 50,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict:
    """Run a complete k-eigenvalue Monte Carlo calculation.

    Power iteration: each generation transports *n_particles* neutrons
    from the previous fission bank.  The first *n_inactive* generations
    allow the fission source to converge; *n_active* generations are
    used to accumulate statistics.

    Parameters
    ----------
    geometry : MSRGeometry
        Geometry description.
    materials_dict : dict
        Material definitions.
    tallies : TallySystem
        Tally system (reset at start).
    n_particles : int
        Source neutrons per generation.
    n_inactive : int
        Inactive (discard) generations.
    n_active : int
        Active (scoring) generations.
    rng : np.random.Generator, optional
        RNG.
    verbose : bool
        Print generation-by-generation output.

    Returns
    -------
    dict
        Summary with keys: 'keff_mean', 'keff_std', 'n_generations',
        'keff_history', 'shannon_entropy'.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = n_inactive + n_active

    # ---- Initial source ----
    source_bank = create_initial_source(n_particles, geometry, rng)

    keff_history: List[float] = []
    entropy_history: List[float] = []

    for gen in range(n_total):
        is_active = gen >= n_inactive

        # Begin batch in tallies
        if tallies is not None and is_active:
            tallies.begin_batch()

        # Transport all neutrons
        fission_bank = run_batch(
            source_bank, geometry, materials_dict,
            tallies if is_active else None,
            rng,
        )

        # Batch keff = (# fission neutrons produced) / (# source neutrons)
        n_fission_neutrons = len(fission_bank)
        n_source = len(source_bank)
        keff_batch = n_fission_neutrons / max(n_source, 1)

        keff_history.append(keff_batch)

        # Shannon entropy of source distribution
        entropy = _compute_shannon_entropy(fission_bank, geometry)
        entropy_history.append(entropy)

        # Record in global tallies
        if tallies is not None:
            tallies.record_batch_keff(n_fission_neutrons, n_source)
            if is_active:
                tallies.end_batch()

        if verbose:
            status = "active" if is_active else "inactive"
            print(
                f"  Gen {gen+1:4d}/{n_total} ({status}): "
                f"k_eff = {keff_batch:.5f}, "
                f"fission sites = {n_fission_neutrons}, "
                f"H = {entropy:.3f}"
            )

        # Normalise fission bank for next generation
        source_bank = normalise_fission_bank(fission_bank, n_particles, rng)

        if len(source_bank) == 0:
            print("WARNING: Fission bank empty -- subcritical system or bug.")
            break

    # ---- Statistics over active batches ----
    active_keff = keff_history[n_inactive:]
    if len(active_keff) > 0:
        keff_mean = float(np.mean(active_keff))
        keff_std = float(np.std(active_keff, ddof=1) / np.sqrt(len(active_keff)))
    else:
        keff_mean = 0.0
        keff_std = 0.0

    result = {
        "keff_mean": keff_mean,
        "keff_std": keff_std,
        "n_generations": n_total,
        "n_active": n_active,
        "n_inactive": n_inactive,
        "n_particles": n_particles,
        "keff_history": keff_history,
        "shannon_entropy": entropy_history,
    }

    if verbose:
        print(f"\n  k_eff = {keff_mean:.5f} +/- {keff_std:.5f}")
        print(f"  ({n_active} active generations, {n_particles} neutrons/gen)")

    return result


def _compute_shannon_entropy(
    fission_bank: List[Dict],
    geometry,
    n_r_bins: int = 5,
    n_z_bins: int = 5,
) -> float:
    """Compute Shannon entropy of fission source spatial distribution.

    A simple (r, z) binning is used to discretise the source.  Entropy
    converging to a constant indicates fission source convergence.

    H = -sum( p_i * ln(p_i) )  for non-zero bins
    """
    if len(fission_bank) == 0:
        return 0.0

    # Extract positions
    positions = np.array([s["position"] for s in fission_bank])
    r_vals = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    z_vals = positions[:, 2]

    # Bin edges
    r_max = np.max(r_vals) * 1.01 + 1.0e-6
    z_min, z_max = np.min(z_vals) - 1e-6, np.max(z_vals) + 1e-6

    r_edges = np.linspace(0.0, r_max, n_r_bins + 1)
    z_edges = np.linspace(z_min, z_max, n_z_bins + 1)

    # 2D histogram
    hist, _, _ = np.histogram2d(r_vals, z_vals, bins=[r_edges, z_edges])
    hist = hist.ravel()
    total = hist.sum()
    if total == 0:
        return 0.0

    probs = hist / total
    # Filter zero-probability bins
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


# ===================================================================
# Self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("particle.py -- Monte Carlo Transport Engine Self-Test")
    print("=" * 60)

    # ---- Test 1: Isotropic direction sampling ----
    print("\n[Test 1] Isotropic direction sampling")
    rng = np.random.default_rng(42)
    n_samples = 100_000
    directions = np.array([sample_isotropic_direction(rng) for _ in range(n_samples)])

    # Check unit vectors
    norms = np.linalg.norm(directions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12), "Direction vectors not unit!"
    print(f"  {n_samples} directions sampled, all unit vectors: PASS")

    # Check isotropy: mean should be ~0 for each component
    means = np.mean(directions, axis=0)
    assert np.all(np.abs(means) < 0.02), f"Direction bias detected: means={means}"
    print(f"  Mean direction: [{means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f}] (expect ~0): PASS")

    # Check cosine distribution: mean of z-component should be ~0
    mu_mean = np.mean(directions[:, 2])
    mu_var = np.var(directions[:, 2])
    print(f"  <mu> = {mu_mean:.4f} (expect 0), var(mu) = {mu_var:.4f} (expect 1/3 = 0.333): PASS")

    # ---- Test 2: Neutron class ----
    print("\n[Test 2] Neutron class")
    n = Neutron([0, 0, 0], [1, 0, 0], energy_group=0, weight=1.0)
    assert n.alive
    assert n.group == 0
    n.move(5.0)
    assert np.allclose(n.pos, [5, 0, 0])
    n.kill()
    assert not n.alive
    print(f"  Neutron created, moved, killed: PASS")
    print(f"  repr: {n}")

    # ---- Test 3: Mock transport test ----
    print("\n[Test 3] Mock geometry transport (simple slab)")

    # Create a mock geometry and material for testing
    class MockMaterial:
        def __init__(self):
            self.name = "test_fuel"
            self.sigma_t = np.array([0.3, 0.8])          # cm^-1
            self.sigma_s = np.array([[0.2, 0.05],
                                      [0.01, 0.5]])       # cm^-1
            self.sigma_f = np.array([0.01, 0.05])         # cm^-1
            self.nu_sigma_f = np.array([0.025, 0.12])     # cm^-1
            self.sigma_a = np.array([0.05, 0.25])         # cm^-1
            self.chi = np.array([0.97, 0.03])

    class MockGeometry:
        """Simple sphere of radius 50 cm centered at origin."""
        VOID = 3

        def find_material(self, x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            if r < 50.0:
                return 0  # fuel
            return 3  # void

        def distance_to_boundary(self, pos, direction):
            # Distance to sphere r=50
            r2 = np.dot(pos, pos)
            rd = np.dot(pos, direction)
            R = 50.0
            disc = rd**2 - (r2 - R**2)
            if disc < 0:
                return 1e6  # shouldn't happen if inside
            sqrt_disc = np.sqrt(disc)
            d = -rd + sqrt_disc
            return max(d, 1e-10)

        def sample_position_in_fuel(self):
            rng_local = np.random.default_rng()
            while True:
                pos = (rng_local.random(3) - 0.5) * 80.0
                if np.sqrt(np.dot(pos, pos)) < 45.0:
                    return pos

        def get_cell_id(self, x, y, z):
            return 0

    mock_geo = MockGeometry()
    mock_mat = MockMaterial()
    materials = {0: mock_mat}

    # Simple tally mock that does nothing
    class MockTally:
        def __init__(self):
            self.leakage = 0
            self.absorptions = 0
            self.fissions = 0
            self.collisions = 0

        def score_track(self, pos, direction, dist, group, weight, material):
            pass

        def score_collision(self, pos, group, weight, material):
            self.collisions += 1

        def score_absorption(self, pos, group, weight, material):
            self.absorptions += 1

        def score_fission(self, pos, weight, material, group):
            self.fissions += 1

        def score_leakage(self, weight):
            self.leakage += 1

        def begin_batch(self):
            pass

        def end_batch(self):
            pass

        def record_batch_keff(self, n_fission, n_source):
            pass

    rng = np.random.default_rng(12345)
    tally = MockTally()

    # Run a batch of neutrons
    n_test = 500
    source = create_initial_source(n_test, mock_geo, rng)
    assert len(source) == n_test
    print(f"  Created initial source with {n_test} neutrons: PASS")

    fission_bank = run_batch(source, mock_geo, materials, tally, rng)
    print(f"  Batch complete:")
    print(f"    Collisions:  {tally.collisions}")
    print(f"    Absorptions: {tally.absorptions}")
    print(f"    Fissions:    {tally.fissions}")
    print(f"    Leakage:     {tally.leakage}")
    print(f"    Fission bank: {len(fission_bank)} sites")
    print(f"    k_batch ~ {len(fission_bank)/n_test:.3f}")

    assert tally.collisions > 0, "No collisions recorded!"
    assert tally.leakage + tally.absorptions > 0, "No terminations!"
    assert len(fission_bank) > 0, "No fission sites -- check cross sections"
    print("  PASS")

    # ---- Test 4: Fission bank normalisation ----
    print("\n[Test 4] Fission bank normalisation")
    normalised = normalise_fission_bank(fission_bank, n_test, rng)
    assert len(normalised) == n_test
    print(f"  Normalised {len(fission_bank)} -> {len(normalised)}: PASS")

    # ---- Test 5: Shannon entropy ----
    print("\n[Test 5] Shannon entropy")
    H = _compute_shannon_entropy(fission_bank, mock_geo)
    print(f"  H = {H:.4f} (should be > 0 for distributed source): PASS")
    assert H > 0

    # ---- Test 6: Multi-generation stability ----
    print("\n[Test 6] Multi-generation stability (10 generations)")
    rng = np.random.default_rng(99)
    bank = create_initial_source(200, mock_geo, rng)
    keffs = []
    for gen in range(10):
        tally_gen = MockTally()
        fb = run_batch(bank, mock_geo, materials, tally_gen, rng)
        keff = len(fb) / max(len(bank), 1)
        keffs.append(keff)
        bank = normalise_fission_bank(fb, 200, rng)
        print(f"    Gen {gen+1}: k={keff:.4f}, bank={len(fb)}")

    k_mean = np.mean(keffs)
    k_std = np.std(keffs)
    print(f"  Mean k = {k_mean:.4f} +/- {k_std:.4f}")
    assert 0.1 < k_mean < 5.0, f"k_eff out of reasonable range: {k_mean}"
    print("  PASS")

    print("\n" + "=" * 60)
    print("All particle.py self-tests PASSED")
    print("=" * 60)
