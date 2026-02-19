"""
3D Hexagonal Lattice Geometry for MSR Monte Carlo Transport
=============================================================

Defines the geometry for a channel-type graphite-moderated MSR with:
  - Cylindrical fuel salt channels in a hexagonal lattice
  - Graphite moderator matrix surrounding the channels
  - Cylindrical core region
  - Annular graphite reflector
  - Vacuum boundary outside the reflector

Coordinate System
-----------------
- Origin at the center of the core (axially and radially)
- x, y: radial coordinates [cm]
- z: axial coordinate [cm], z=0 at core midplane
- All internal calculations in CENTIMETERS (nuclear convention)

Hexagonal Lattice
-----------------
The hex lattice uses a flat-to-flat pitch (distance between parallel
flat faces of adjacent hexagons). For a hexagonal cell with flat-to-flat
distance P:
  - Side length: a = P / sqrt(3)
  - Area per cell: A = (sqrt(3)/2) * P^2
  - Vertex-to-vertex: 2a = 2P / sqrt(3)

The lattice is oriented with flats on top and bottom (pointy-top would
require rotating the basis vectors by 30 degrees).

Geometry Hierarchy
------------------
1. Outer boundary: Cylinder of radius R_core + R_reflector, height H_core
2. Reflector: Annular region R_core < r < R_core + R_reflector
3. Core: Cylinder of radius R_core, height H_core
4. Within core: hex lattice of fuel channels (radius R_channel)
5. Each hex cell: circular fuel channel + graphite matrix
"""

import math
from typing import Tuple, List, Optional

import numpy as np

from .constants import (
    MAT_VOID, MAT_FUEL_SALT, MAT_GRAPHITE_MOD, MAT_GRAPHITE_REF,
    DISTANCE_EPSILON, MATERIAL_NAMES,
)


# =============================================================================
# GEOMETRY CONSTANTS (from config.py, converted to cm)
# =============================================================================

# Core dimensions (config values in meters, convert to cm)
_CORE_RADIUS_M = 0.6225          # m (core_diameter / 2 ~ 1.245 / 2)
_CORE_HEIGHT_M = 1.494           # m (H/D = 1.2 * 1.245)
_CHANNEL_RADIUS_M = 0.0125       # m (channel_diameter / 2 = 25 mm / 2)
_CHANNEL_PITCH_M = 0.05          # m (50 mm hex pitch)
_REFLECTOR_THICKNESS_M = 0.15    # m (150 mm graphite reflector)

# Convert to centimeters for nuclear calculations
DEFAULT_CORE_RADIUS = _CORE_RADIUS_M * 100.0        # cm
DEFAULT_CORE_HEIGHT = _CORE_HEIGHT_M * 100.0         # cm
DEFAULT_CHANNEL_RADIUS = _CHANNEL_RADIUS_M * 100.0   # cm
DEFAULT_CHANNEL_PITCH = _CHANNEL_PITCH_M * 100.0     # cm
DEFAULT_REFLECTOR_THICKNESS = _REFLECTOR_THICKNESS_M * 100.0  # cm
DEFAULT_N_CHANNELS = 562


# =============================================================================
# HELPER: HEXAGONAL GEOMETRY
# =============================================================================

def _hex_axial_to_cartesian(q: int, r: int, pitch: float) -> Tuple[float, float]:
    """Convert axial hex coordinates (q, r) to Cartesian (x, y).

    Uses flat-top hexagon orientation where the hex grid basis vectors are:
      e_q = (pitch, 0)
      e_r = (pitch/2, pitch * sqrt(3)/2)

    Parameters
    ----------
    q, r : int
        Axial hexagonal coordinates.
    pitch : float
        Flat-to-flat distance [cm].

    Returns
    -------
    tuple of float
        (x, y) Cartesian coordinates of hex center [cm].
    """
    x = pitch * (q + 0.5 * r)
    y = pitch * (math.sqrt(3.0) / 2.0) * r
    return x, y


def _point_in_hexagon(px: float, py: float,
                      cx: float, cy: float,
                      pitch: float) -> bool:
    """Test if point (px, py) is inside a flat-top regular hexagon.

    For a flat-top hexagon centered at (cx, cy) with flat-to-flat
    distance equal to pitch, the half-width is pitch/2 and the
    half-height is pitch/sqrt(3).

    The hexagon boundary is defined by three pairs of parallel lines.
    A point is inside if it satisfies all three constraints.

    Parameters
    ----------
    px, py : float
        Point coordinates [cm].
    cx, cy : float
        Hexagon center coordinates [cm].
    pitch : float
        Flat-to-flat distance [cm].

    Returns
    -------
    bool
        True if point is inside the hexagon.
    """
    # Translate to hexagon-centered coordinates
    dx = abs(px - cx)
    dy = abs(py - cy)

    # Half flat-to-flat distance
    h = pitch / 2.0
    # Side length
    a = pitch / math.sqrt(3.0)

    # Three constraints for flat-top hexagon:
    # 1. |dx| <= a (within vertical extent = side length)
    # 2. |dy| <= h (within horizontal extent = half-pitch)
    # 3. The sloped edges: dy + dx * sqrt(3)/2 ... wait
    #
    # For a flat-top hex (flats on left/right):
    #   The hex has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    #   from center, at distance a from center.
    #   Actually, let's use the standard approach:
    #
    # For flat-top hexagon with flat-to-flat = pitch:
    #   half_w = pitch / 2          (half the flat-to-flat, in x-direction)
    #   half_h = pitch / sqrt(3)    (half vertex-to-vertex, in y-direction... no)
    #
    # Let me be precise. With flat-top orientation:
    #   - Flats are horizontal (top and bottom)
    #   - Vertices point left and right
    #   - Flat-to-flat distance = pitch (vertical distance between top/bottom flats)
    #   - half_flat = pitch / 2
    #   - side_length = pitch / sqrt(3)
    #   - vertex-to-vertex = 2 * side_length = 2 * pitch / sqrt(3)
    #
    # Constraints (flat-top, flats horizontal):
    #   |dy| <= pitch / 2
    #   |dx| <= pitch / sqrt(3)
    #   |dy| + |dx| * sqrt(3) / 2 <= pitch / 2... no
    #
    # Cleaner: a flat-top hex with "radius" (center-to-vertex) = R
    #   R = pitch / sqrt(3)
    #   The three constraint pairs:
    #     |dy| <= R * sqrt(3)/2 = pitch/2  ... (horizontal flats)
    #     |dx| * sqrt(3)/2 + |dy| / 2 <= R * sqrt(3)/2
    #       simplify: |dx| * sqrt(3) + |dy| <= pitch
    #
    # Actually the simplest correct formulation for flat-top hex:
    #   condition 1: |dy| <= pitch / 2
    #   condition 2: |dy| + |dx| * sqrt(3) <= pitch

    if dy > h:
        return False
    if dy + dx * math.sqrt(3.0) > pitch:
        return False
    return True


def _distance_to_hexagon_walls(px: float, py: float,
                               dx: float, dy: float,
                               cx: float, cy: float,
                               pitch: float) -> float:
    """Distance from point to nearest hexagon wall along direction.

    Computes the distance along the ray (px,py) + t*(dx,dy) to the
    boundary of a flat-top hexagon centered at (cx,cy) with flat-to-flat
    distance = pitch.

    The hexagon boundary consists of 6 line segments. We compute the
    intersection with each edge and return the minimum positive distance.

    Parameters
    ----------
    px, py : float
        Point coordinates [cm].
    dx, dy : float
        Direction vector (x, y components, need not be normalized in xy).
    cx, cy : float
        Hexagon center [cm].
    pitch : float
        Flat-to-flat distance [cm].

    Returns
    -------
    float
        Distance to nearest hexagon wall [cm]. Returns inf if no intersection.
    """
    # Translate to hex-centered coordinates
    rx = px - cx
    ry = py - cy

    h = pitch / 2.0
    sqrt3 = math.sqrt(3.0)

    min_dist = float('inf')

    # The 6 edges of a flat-top hexagon can be expressed as:
    #   n . r = h_offset
    # where n is the outward normal and h_offset is the signed distance.
    #
    # For flat-top hex with flats horizontal, the 6 normals are:
    #   (0, +1), (0, -1)                           -> top/bottom flats
    #   (+sqrt3/2, +1/2), (+sqrt3/2, -1/2)         -> upper-right, lower-right
    #   (-sqrt3/2, +1/2), (-sqrt3/2, -1/2)         -> upper-left, lower-left
    #
    # Each face is at signed distance h from center along its normal.

    normals = [
        (0.0, 1.0),
        (0.0, -1.0),
        (sqrt3 / 2.0, 0.5),
        (sqrt3 / 2.0, -0.5),
        (-sqrt3 / 2.0, 0.5),
        (-sqrt3 / 2.0, -0.5),
    ]

    for nx, ny in normals:
        # Face equation: nx*x + ny*y = h
        # Ray: x = rx + t*dx, y = ry + t*dy
        # t = (h - nx*rx - ny*ry) / (nx*dx + ny*dy)
        denom = nx * dx + ny * dy
        if abs(denom) < 1e-15:
            continue  # parallel to this face
        t = (h - nx * rx - ny * ry) / denom
        if t > DISTANCE_EPSILON:
            min_dist = min(min_dist, t)

    return min_dist


# =============================================================================
# MSR GEOMETRY CLASS
# =============================================================================

class MSRGeometry:
    """3D hexagonal-lattice cylindrical MSR geometry.

    The geometry consists of:
    1. A cylindrical core of fuel channels in a hex lattice graphite matrix
    2. An annular graphite reflector surrounding the core
    3. Vacuum boundary outside

    All dimensions are stored and computed in centimeters.

    Parameters
    ----------
    core_radius : float
        Core radius [cm]. Default from config.
    core_height : float
        Core full height [cm]. Default from config.
    channel_radius : float
        Fuel channel radius [cm]. Default from config.
    channel_pitch : float
        Hex lattice flat-to-flat pitch [cm]. Default from config.
    reflector_thickness : float
        Radial reflector thickness [cm]. Default from config.
    n_channels : int
        Expected number of fuel channels (for validation). Default from config.
    """

    def __init__(self,
                 core_radius: float = DEFAULT_CORE_RADIUS,
                 core_height: float = DEFAULT_CORE_HEIGHT,
                 channel_radius: float = DEFAULT_CHANNEL_RADIUS,
                 channel_pitch: float = DEFAULT_CHANNEL_PITCH,
                 reflector_thickness: float = DEFAULT_REFLECTOR_THICKNESS,
                 n_channels: int = DEFAULT_N_CHANNELS):

        self.core_radius = core_radius
        self.core_height = core_height
        self.core_half_height = core_height / 2.0
        self.channel_radius = channel_radius
        self.channel_pitch = channel_pitch
        self.reflector_thickness = reflector_thickness
        self.outer_radius = core_radius + reflector_thickness

        # Axial reflector extends above and below the core
        self.axial_half_height = self.core_half_height + reflector_thickness

        # Pre-compute hex lattice channel centers
        self._channel_centers = self._generate_hex_lattice()
        self.n_channels = len(self._channel_centers)

        # Build KD-tree-like acceleration: store centers as numpy array
        if self.n_channels > 0:
            self._centers_array = np.array(self._channel_centers)  # (N, 2)
        else:
            self._centers_array = np.empty((0, 2))

        # Precompute channel area and volume for diagnostics
        self.channel_area = math.pi * channel_radius**2  # cm^2
        self.cell_area = (math.sqrt(3.0) / 2.0) * channel_pitch**2  # cm^2
        self.fuel_fraction = self.channel_area / self.cell_area
        self.graphite_fraction = 1.0 - self.fuel_fraction

        # Validation
        if n_channels > 0 and abs(self.n_channels - n_channels) > 20:
            print(f"  WARNING: Expected ~{n_channels} channels, "
                  f"generated {self.n_channels}")

    def _generate_hex_lattice(self) -> List[Tuple[float, float]]:
        """Generate hex lattice channel center positions within the core.

        Creates a hexagonal grid using axial coordinates (q, r) and
        keeps only channels whose centers fall within the core radius.

        Returns
        -------
        list of (float, float)
            Channel center (x, y) positions [cm].
        """
        centers = []
        pitch = self.channel_pitch
        R = self.core_radius

        # Maximum q, r indices needed to cover the core
        # The hex grid extends roughly R/pitch in each direction
        max_index = int(math.ceil(R / pitch)) + 2

        for q in range(-max_index, max_index + 1):
            for r in range(-max_index, max_index + 1):
                x, y = _hex_axial_to_cartesian(q, r, pitch)
                # Check if channel center is within core radius
                # Leave margin so the full channel fits inside
                dist = math.sqrt(x * x + y * y)
                if dist + self.channel_radius <= R:
                    centers.append((x, y))

        return centers

    def find_material(self, x: float, y: float, z: float) -> int:
        """Determine the material at a given point.

        The lookup follows a hierarchy:
        1. Outside outer cylinder (core + reflector) -> VOID
        2. Above/below the core (|z| > H/2) -> VOID
        3. In reflector annulus (core_radius < r < outer_radius) -> REFLECTOR
        4. In core region, find nearest hex cell:
           a. Within a channel radius of any channel center -> FUEL
           b. Otherwise -> GRAPHITE MODERATOR

        Parameters
        ----------
        x, y, z : float
            Position coordinates [cm].

        Returns
        -------
        int
            Material ID (MAT_VOID, MAT_FUEL_SALT, MAT_GRAPHITE_MOD,
            or MAT_GRAPHITE_REF).
        """
        # Check axial bounds (core + axial reflector + void)
        abs_z = abs(z)
        if abs_z > self.axial_half_height:
            return MAT_VOID

        if abs_z > self.core_half_height:
            # In axial reflector region: graphite above/below core
            # Check radial extent - reflector extends to outer_radius
            r2_check = x * x + y * y
            if r2_check > self.outer_radius * self.outer_radius:
                return MAT_VOID
            return MAT_GRAPHITE_REF

        # Radial distance from axis
        r2 = x * x + y * y
        r = math.sqrt(r2)

        # Outside outer boundary
        if r > self.outer_radius:
            return MAT_VOID

        # In reflector annulus
        if r > self.core_radius:
            return MAT_GRAPHITE_REF

        # In core region: check if inside a fuel channel
        # Find nearest channel center using vectorized distance computation
        if self.n_channels == 0:
            return MAT_GRAPHITE_MOD

        # Vectorized nearest-neighbor search
        dx = self._centers_array[:, 0] - x
        dy = self._centers_array[:, 1] - y
        dist2 = dx * dx + dy * dy
        min_dist2 = np.min(dist2)

        if min_dist2 <= self.channel_radius**2:
            return MAT_FUEL_SALT
        else:
            return MAT_GRAPHITE_MOD

    def find_nearest_channel(self, x: float, y: float) -> Tuple[int, float, float, float]:
        """Find the nearest fuel channel to a point.

        Parameters
        ----------
        x, y : float
            Position in the xy-plane [cm].

        Returns
        -------
        tuple
            (channel_index, center_x, center_y, distance) where distance
            is from (x,y) to the channel center [cm].
        """
        if self.n_channels == 0:
            return -1, 0.0, 0.0, float('inf')

        dx = self._centers_array[:, 0] - x
        dy = self._centers_array[:, 1] - y
        dist2 = dx * dx + dy * dy
        idx = int(np.argmin(dist2))
        cx, cy = self._channel_centers[idx]
        dist = math.sqrt(dist2[idx])
        return idx, cx, cy, dist

    def distance_to_boundary(self, pos: np.ndarray,
                             direction: np.ndarray) -> Tuple[float, int]:
        """Compute distance to next material boundary along a ray.

        Checks intersections with:
        1. Top and bottom planes (z = +/-H/2)
        2. Core cylinder (r = core_radius)
        3. Outer cylinder (r = outer_radius)
        4. Nearest channel wall (circle)
        5. Nearest hex cell walls (6 planes)

        Returns the minimum positive distance and the material on the
        far side of that boundary.

        Parameters
        ----------
        pos : ndarray, shape (3,)
            Current position [x, y, z] in cm.
        direction : ndarray, shape (3,)
            Unit direction vector [dx, dy, dz].

        Returns
        -------
        tuple
            (distance [cm], next_material_id).
        """
        x, y, z = pos[0], pos[1], pos[2]
        dx, dy, dz = direction[0], direction[1], direction[2]

        min_dist = float('inf')
        next_mat = MAT_VOID  # default: escaping

        current_mat = self.find_material(x, y, z)

        # --- Axial planes ---
        if abs(dz) > 1e-15:
            # Determine which axial planes are relevant based on current position
            abs_z = abs(z)

            if abs_z <= self.core_half_height:
                # Inside core: next axial boundary is core top/bottom -> axial reflector
                t_top = (self.core_half_height - z) / dz
                if t_top > DISTANCE_EPSILON and t_top < min_dist:
                    min_dist = t_top
                    next_mat = MAT_GRAPHITE_REF

                t_bot = (-self.core_half_height - z) / dz
                if t_bot > DISTANCE_EPSILON and t_bot < min_dist:
                    min_dist = t_bot
                    next_mat = MAT_GRAPHITE_REF

            else:
                # In axial reflector: check both axial reflector outer plane and core re-entry
                # Outer axial plane (exit to void)
                if z > 0:
                    t_out = (self.axial_half_height - z) / dz
                    t_core = (self.core_half_height - z) / dz  # re-entry into core (negative z direction)
                else:
                    t_out = (-self.axial_half_height - z) / dz
                    t_core = (-self.core_half_height - z) / dz  # re-entry into core (positive z direction)

                if t_out > DISTANCE_EPSILON and t_out < min_dist:
                    min_dist = t_out
                    next_mat = MAT_VOID

                if t_core > DISTANCE_EPSILON and t_core < min_dist:
                    # Re-entering core: need to determine which material
                    new_x = x + t_core * dx
                    new_y = y + t_core * dy
                    new_r2 = new_x * new_x + new_y * new_y
                    if new_r2 > self.core_radius * self.core_radius:
                        next_mat = MAT_GRAPHITE_REF  # radial reflector region
                    else:
                        # Probe the core material at re-entry point
                        new_z = z + t_core * dz
                        probe_mat = self.find_material(
                            new_x + DISTANCE_EPSILON * dx,
                            new_y + DISTANCE_EPSILON * dy,
                            new_z + DISTANCE_EPSILON * dz,
                        )
                        next_mat = probe_mat if probe_mat != MAT_VOID else MAT_GRAPHITE_MOD
                    min_dist = t_core

        # --- Radial distance in xy plane ---
        r2 = x * x + y * y
        r = math.sqrt(r2)
        dxy2 = dx * dx + dy * dy

        # Helper: distance to cylinder of radius R from inside/outside
        def _dist_to_cylinder(R: float) -> float:
            """Intersection of ray with cylinder r = R (infinite in z)."""
            # Solve: (x + t*dx)^2 + (y + t*dy)^2 = R^2
            # a*t^2 + b*t + c = 0
            a = dxy2
            if a < 1e-30:
                return float('inf')  # moving purely axially
            b = 2.0 * (x * dx + y * dy)
            c = r2 - R * R
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0:
                return float('inf')
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            # Return smallest positive root
            if t1 > DISTANCE_EPSILON:
                return t1
            if t2 > DISTANCE_EPSILON:
                return t2
            return float('inf')

        # --- Core cylinder boundary ---
        t_core = _dist_to_cylinder(self.core_radius)
        if t_core < min_dist:
            # Determine what's on the other side
            # Check if we're crossing from inside core to reflector or vice versa
            new_x = x + t_core * dx
            new_y = y + t_core * dy
            new_r = math.sqrt(new_x**2 + new_y**2)
            if new_r >= self.core_radius - DISTANCE_EPSILON:
                # Moving outward: core -> reflector
                if current_mat in (MAT_FUEL_SALT, MAT_GRAPHITE_MOD):
                    min_dist = t_core
                    next_mat = MAT_GRAPHITE_REF
                # Moving inward: reflector -> core
                elif current_mat == MAT_GRAPHITE_REF:
                    min_dist = t_core
                    # Determine if entering fuel or graphite
                    new_z = z + t_core * dz
                    probe_mat = self.find_material(
                        new_x + DISTANCE_EPSILON * dx,
                        new_y + DISTANCE_EPSILON * dy,
                        new_z
                    )
                    next_mat = probe_mat

        # --- Outer cylinder boundary ---
        t_outer = _dist_to_cylinder(self.outer_radius)
        if t_outer < min_dist:
            min_dist = t_outer
            next_mat = MAT_VOID

        # --- Channel walls (only relevant inside the core) ---
        if current_mat in (MAT_FUEL_SALT, MAT_GRAPHITE_MOD) and self.n_channels > 0:
            idx, cx, cy, dist_to_center = self.find_nearest_channel(x, y)

            if current_mat == MAT_FUEL_SALT:
                # Inside a channel: distance to channel wall (circle)
                t_chan = self._dist_to_circle(x, y, dx, dy, cx, cy,
                                             self.channel_radius, inside=True)
                if DISTANCE_EPSILON < t_chan < min_dist:
                    min_dist = t_chan
                    next_mat = MAT_GRAPHITE_MOD

            elif current_mat == MAT_GRAPHITE_MOD:
                # In graphite between channels
                # Check distance to nearest channel (entering fuel)
                t_into_chan = self._dist_to_circle(x, y, dx, dy, cx, cy,
                                                  self.channel_radius, inside=False)
                if DISTANCE_EPSILON < t_into_chan < min_dist:
                    min_dist = t_into_chan
                    next_mat = MAT_FUEL_SALT

                # Also check adjacent channels that the ray might hit
                # Check the ~6 nearest channels beyond the closest
                if self.n_channels > 1:
                    ddx = self._centers_array[:, 0] - x
                    ddy = self._centers_array[:, 1] - y
                    dist2_all = ddx * ddx + ddy * ddy
                    # Sort by distance and check the nearest ~7
                    nearest_idx = np.argpartition(dist2_all,
                                                  min(7, self.n_channels - 1))
                    for ki in range(min(7, self.n_channels)):
                        ci = nearest_idx[ki]
                        if ci == idx:
                            continue
                        ccx, ccy = self._channel_centers[ci]
                        t_adj = self._dist_to_circle(x, y, dx, dy, ccx, ccy,
                                                     self.channel_radius, inside=False)
                        if DISTANCE_EPSILON < t_adj < min_dist:
                            min_dist = t_adj
                            next_mat = MAT_FUEL_SALT

                # Check hex cell boundary (transition to adjacent cell)
                t_hex = _distance_to_hexagon_walls(x, y, dx, dy, cx, cy,
                                                   self.channel_pitch)
                if DISTANCE_EPSILON < t_hex < min_dist:
                    # Crossing hex boundary stays in graphite (or enters another
                    # channel which we check above), so material doesn't change
                    # but we update the cell for tallying purposes.
                    # Don't update min_dist for same-material boundaries
                    # unless we need it for cell tracking.
                    pass

        return min_dist, next_mat

    @staticmethod
    def _dist_to_circle(px: float, py: float,
                        dx: float, dy: float,
                        cx: float, cy: float,
                        R: float,
                        inside: bool) -> float:
        """Distance from point to circle boundary along a 2D ray.

        Solves: |(p + t*d) - c|^2 = R^2

        Parameters
        ----------
        px, py : float
            Point position [cm].
        dx, dy : float
            Direction (xy components).
        cx, cy : float
            Circle center [cm].
        R : float
            Circle radius [cm].
        inside : bool
            If True, point is inside the circle (return exit distance).
            If False, point is outside (return entry distance).

        Returns
        -------
        float
            Distance to circle boundary [cm]. Returns inf if no intersection.
        """
        # Translate to circle-centered coordinates
        rx = px - cx
        ry = py - cy

        a = dx * dx + dy * dy
        if a < 1e-30:
            return float('inf')
        b = 2.0 * (rx * dx + ry * dy)
        c = rx * rx + ry * ry - R * R

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return float('inf')

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        if inside:
            # Inside circle: want the exit point (larger positive root)
            if t2 > DISTANCE_EPSILON:
                return t2
            if t1 > DISTANCE_EPSILON:
                return t1
            return float('inf')
        else:
            # Outside circle: want the entry point (smaller positive root)
            if t1 > DISTANCE_EPSILON:
                return t1
            if t2 > DISTANCE_EPSILON:
                return t2
            return float('inf')

    def sample_position_in_fuel(self, rng: Optional[np.random.Generator] = None
                                ) -> np.ndarray:
        """Sample a uniformly random position within the fuel salt.

        Strategy: pick a random channel, then sample a uniform random
        point within that cylindrical channel.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator. If None, creates a default one.

        Returns
        -------
        ndarray, shape (3,)
            Random position [x, y, z] in cm within fuel salt.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.n_channels == 0:
            raise RuntimeError("No fuel channels in geometry")

        # Pick a random channel
        chan_idx = rng.integers(0, self.n_channels)
        cx, cy = self._channel_centers[chan_idx]

        # Uniform random point in circle of radius channel_radius
        # Using rejection-free method: r = R * sqrt(U), theta = 2*pi*V
        u = rng.random()
        v = rng.random()
        r = self.channel_radius * math.sqrt(u)
        theta = 2.0 * math.pi * v
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)

        # Uniform random z in [-H/2, +H/2]
        z = self.core_half_height * (2.0 * rng.random() - 1.0)

        return np.array([x, y, z])

    def sample_isotropic_direction(self, rng: Optional[np.random.Generator] = None
                                   ) -> np.ndarray:
        """Sample an isotropic unit direction vector.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        ndarray, shape (3,)
            Unit direction vector [dx, dy, dz].
        """
        if rng is None:
            rng = np.random.default_rng()

        # Uniform on unit sphere: cos(theta) uniform in [-1, 1], phi in [0, 2pi)
        mu = 2.0 * rng.random() - 1.0  # cos(theta)
        phi = 2.0 * math.pi * rng.random()
        sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))

        dx = sin_theta * math.cos(phi)
        dy = sin_theta * math.sin(phi)
        dz = mu

        return np.array([dx, dy, dz])

    def get_cell_id(self, x: float, y: float, z: float) -> int:
        """Return a unique cell identifier for tallying.

        Cell ID encoding:
        - 0: Void
        - 1 to N_channels: Fuel channel index
        - N_channels + 1: Graphite moderator (bulk)
        - N_channels + 2: Graphite reflector

        Parameters
        ----------
        x, y, z : float
            Position [cm].

        Returns
        -------
        int
            Unique cell identifier.
        """
        mat = self.find_material(x, y, z)

        if mat == MAT_VOID:
            return 0
        elif mat == MAT_GRAPHITE_REF:
            return self.n_channels + 2
        elif mat == MAT_GRAPHITE_MOD:
            return self.n_channels + 1
        elif mat == MAT_FUEL_SALT:
            idx, _, _, _ = self.find_nearest_channel(x, y)
            return idx + 1  # 1-indexed
        return 0

    def get_fuel_volume(self) -> float:
        """Total fuel salt volume [cm^3].

        Returns
        -------
        float
            Total fuel volume = N_channels * pi * R_channel^2 * H_core [cm^3].
        """
        return self.n_channels * self.channel_area * self.core_height

    def get_moderator_volume(self) -> float:
        """Total graphite moderator volume in core [cm^3].

        Returns
        -------
        float
            Total moderator volume = core_volume - fuel_volume [cm^3].
        """
        core_vol = math.pi * self.core_radius**2 * self.core_height
        return core_vol - self.get_fuel_volume()

    def get_reflector_volume(self) -> float:
        """Total reflector volume [cm^3].

        Returns
        -------
        float
            Annular reflector volume [cm^3].
        """
        return (math.pi * (self.outer_radius**2 - self.core_radius**2)
                * self.core_height)

    def summary(self) -> str:
        """Return a formatted summary of the geometry."""
        fuel_vol = self.get_fuel_volume()
        mod_vol = self.get_moderator_volume()
        ref_vol = self.get_reflector_volume()
        core_vol = math.pi * self.core_radius**2 * self.core_height

        lines = [
            "MSR Geometry Summary",
            "=" * 50,
            f"  Core radius:           {self.core_radius:10.2f} cm "
            f"({self.core_radius/100:.4f} m)",
            f"  Core height:           {self.core_height:10.2f} cm "
            f"({self.core_height/100:.4f} m)",
            f"  Core volume:           {core_vol:10.0f} cm^3 "
            f"({core_vol/1e6:.4f} m^3)",
            f"  Outer radius:          {self.outer_radius:10.2f} cm "
            f"({self.outer_radius/100:.4f} m)",
            f"",
            f"  Channel radius:        {self.channel_radius:10.2f} cm",
            f"  Channel pitch:         {self.channel_pitch:10.2f} cm",
            f"  Number of channels:    {self.n_channels:10d}",
            f"  Fuel fraction (cell):  {self.fuel_fraction:10.4f} "
            f"({self.fuel_fraction*100:.1f}%)",
            f"  Graphite fraction:     {self.graphite_fraction:10.4f} "
            f"({self.graphite_fraction*100:.1f}%)",
            f"",
            f"  Fuel volume:           {fuel_vol:10.0f} cm^3 "
            f"({fuel_vol/1e6:.4f} m^3)",
            f"  Moderator volume:      {mod_vol:10.0f} cm^3 "
            f"({mod_vol/1e6:.4f} m^3)",
            f"  Reflector volume:      {ref_vol:10.0f} cm^3 "
            f"({ref_vol/1e6:.4f} m^3)",
            f"  Reflector thickness:   {self.reflector_thickness:10.2f} cm",
            f"  Axial reflector:       {self.reflector_thickness:10.1f} cm graphite (top + bottom)",
            f"  Total height:          {2 * self.axial_half_height:10.1f} cm",
        ]
        return "\n".join(lines)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("  MSR Geometry Verification")
    print("=" * 60)

    # Create geometry with default parameters
    geom = MSRGeometry()
    print(f"\n{geom.summary()}")

    # --- Verify channel count ---
    print(f"\n--- Channel Count ---")
    print(f"  Generated channels:  {geom.n_channels}")
    print(f"  Expected (config):   {DEFAULT_N_CHANNELS}")
    diff_pct = abs(geom.n_channels - DEFAULT_N_CHANNELS) / DEFAULT_N_CHANNELS * 100
    print(f"  Difference:          {diff_pct:.1f}%")

    # --- Verify fuel fraction ---
    print(f"\n--- Volume Fractions ---")
    fuel_vol = geom.get_fuel_volume()
    mod_vol = geom.get_moderator_volume()
    core_vol = math.pi * geom.core_radius**2 * geom.core_height
    actual_fuel_frac = fuel_vol / core_vol
    print(f"  Fuel fraction (actual):    {actual_fuel_frac:.4f} "
          f"(target: 0.2300)")
    print(f"  Graphite fraction (actual): {1-actual_fuel_frac:.4f} "
          f"(target: 0.7700)")

    # --- Test material lookup at known positions ---
    print(f"\n--- Material Lookup Tests ---")
    test_points = [
        (0.0, 0.0, 0.0, "Core center"),
        (geom.core_radius + 5.0, 0.0, 0.0, "In reflector"),
        (geom.outer_radius + 1.0, 0.0, 0.0, "Outside (void)"),
        (0.0, 0.0, geom.core_half_height + 1.0, "Above core (void)"),
    ]

    # Add a point at a known channel center
    if geom.n_channels > 0:
        cx, cy = geom._channel_centers[0]
        test_points.append((cx, cy, 0.0, f"Channel 0 center ({cx:.2f},{cy:.2f})"))
        # Point between channels (should be graphite)
        if geom.n_channels > 1:
            cx2, cy2 = geom._channel_centers[1]
            mx = (cx + cx2) / 2.0
            my = (cy + cy2) / 2.0
            test_points.append((mx, my, 0.0, f"Between channels ({mx:.2f},{my:.2f})"))

    for x, y, z, desc in test_points:
        mat = geom.find_material(x, y, z)
        print(f"  ({x:8.2f}, {y:8.2f}, {z:8.2f}) -> "
              f"{MATERIAL_NAMES.get(mat, '???'):20s}  [{desc}]")

    # --- Test source sampling ---
    print(f"\n--- Source Sampling ---")
    rng = np.random.default_rng(42)
    n_samples = 10000
    n_in_fuel = 0
    for _ in range(n_samples):
        pos = geom.sample_position_in_fuel(rng)
        mat = geom.find_material(pos[0], pos[1], pos[2])
        if mat == MAT_FUEL_SALT:
            n_in_fuel += 1
    print(f"  Sampled {n_samples} source positions")
    print(f"  In fuel: {n_in_fuel}/{n_samples} ({n_in_fuel/n_samples*100:.1f}%)")
    if n_in_fuel == n_samples:
        print(f"  -> All samples correctly in fuel: PASS")
    else:
        print(f"  -> WARNING: {n_samples - n_in_fuel} samples NOT in fuel!")

    # --- Test distance-to-boundary ---
    print(f"\n--- Distance-to-Boundary Tests ---")

    # From core center, moving radially outward (+x)
    pos = np.array([0.0, 0.0, 0.0])
    dir_x = np.array([1.0, 0.0, 0.0])
    dist, next_mat = geom.distance_to_boundary(pos, dir_x)
    print(f"  From origin, +x direction:")
    print(f"    Distance: {dist:.4f} cm, next material: {MATERIAL_NAMES.get(next_mat, '???')}")

    # From inside a channel, moving radially
    if geom.n_channels > 0:
        cx, cy = geom._channel_centers[0]
        pos_chan = np.array([cx, cy, 0.0])
        dist_ch, next_ch = geom.distance_to_boundary(pos_chan, dir_x)
        print(f"  From channel 0 center ({cx:.2f},{cy:.2f}), +x direction:")
        print(f"    Distance: {dist_ch:.4f} cm, next material: "
              f"{MATERIAL_NAMES.get(next_ch, '???')}")
        print(f"    Expected ~channel_radius = {geom.channel_radius:.4f} cm")

    # From core center, moving up (+z)
    dir_z = np.array([0.0, 0.0, 1.0])
    dist_z, next_z = geom.distance_to_boundary(pos, dir_z)
    print(f"  From origin, +z direction:")
    print(f"    Distance: {dist_z:.4f} cm, next material: {MATERIAL_NAMES.get(next_z, '???')}")
    print(f"    Expected ~core_half_height = {geom.core_half_height:.4f} cm")

    # --- Performance benchmark ---
    print(f"\n--- Performance Benchmark ---")
    n_lookups = 100000
    positions = np.column_stack([
        rng.uniform(-geom.core_radius, geom.core_radius, n_lookups),
        rng.uniform(-geom.core_radius, geom.core_radius, n_lookups),
        rng.uniform(-geom.core_half_height, geom.core_half_height, n_lookups),
    ])

    t0 = time.perf_counter()
    counts = {MAT_VOID: 0, MAT_FUEL_SALT: 0, MAT_GRAPHITE_MOD: 0, MAT_GRAPHITE_REF: 0}
    for i in range(n_lookups):
        mat = geom.find_material(positions[i, 0], positions[i, 1], positions[i, 2])
        counts[mat] += 1
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rate = n_lookups / elapsed
    print(f"  {n_lookups:,} material lookups in {elapsed:.3f} s ({rate:,.0f} lookups/s)")
    print(f"  Material distribution (uniform random in core bounding box):")
    for mat_id, count in sorted(counts.items()):
        frac = count / n_lookups * 100
        print(f"    {MATERIAL_NAMES.get(mat_id, '???'):20s}: {count:6d} ({frac:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  Geometry verification complete.")
    print(f"{'='*60}")
