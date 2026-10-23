"""

Lightcone geometry: distance-redshift conversion, snapshot shells, randomised box replication and sky projection.

Conventions: the observer sits at the origin of the lightcone frame and looks along +z. Lengths are comoving Mpc and
angles are radians. Sky coordinates are spherical about the line of sight, so that the field centre is (RA, Dec) = (0, 0):
RA = atan2(x, z) and Dec = arcsin(y / d), where d is the comoving distance.

Based on the Simba lightcone code of Christopher Lovell (https://github.com/christopherlovell/simba_lightcone_generation).

"""

# default libraries
from dataclasses import dataclass

# other packages
import numpy as np
from astropy.cosmology import FLRW
from scipy.integrate import cumulative_simpson
from scipy.optimize import brentq

FIELD_SHAPES = ("square", "circle")
RANDOMISE_MODES = ("none", "snapshot", "replica")
SPEED_OF_LIGHT_KMS = 299792.458


class DistanceRedshiftTable:
    """
    Tabulated line-of-sight comoving distance D(z) and its inverse, for fast vectorised conversion.

    Parameters
    ----------
    cosmology: astropy.cosmology.FLRW
        The cosmology of the simulation.
    z_max: float
        Highest redshift the table must cover.
    n_points: int
        Number of tabulated redshifts.
    """

    def __init__(self, cosmology: FLRW, z_max: float, n_points: int = 100_001) -> None:
        self.z_max = z_max
        self._z = np.linspace(0.0, z_max, n_points)
        hubble_distance = cosmology.hubble_distance.to_value("Mpc")
        self._d = hubble_distance * cumulative_simpson(cosmology.inv_efunc(self._z), x=self._z, initial=0.0)

    def distance(self, z: float | np.ndarray) -> np.ndarray:
        """
        Comoving distance (Mpc) at redshift z.
        """
        if np.any(np.asarray(z) > self.z_max):
            raise ValueError(f"Redshift beyond the tabulated range (z_max = {self.z_max}).")
        return np.interp(z, self._z, self._d)

    def redshift(self, d: float | np.ndarray) -> np.ndarray:
        """
        Redshift at comoving distance d (Mpc).
        """
        if np.any(np.asarray(d) > self._d[-1]):
            raise ValueError(f"Distance beyond the tabulated range (z_max = {self.z_max}).")
        return np.interp(d, self._d, self._z)


def field_half_angle(area: float, field_shape: str) -> float:
    """
    Half-angle (radians) of a field with the given solid angle.

    For a square field this is the half-width in RA and Dec, so the field is |RA|, |Dec| <= half_angle, which covers
    4 * half_angle * sin(half_angle) sr. For a circular field it is the radius, covering 2 pi (1 - cos(half_angle)) sr.

    Parameters
    ----------
    area: float
        Solid angle of the field (deg^2).
    field_shape: str
        ``square`` or ``circle``.
    """
    if field_shape not in FIELD_SHAPES:
        raise ValueError(f"field_shape must be one of {FIELD_SHAPES}, not '{field_shape}'.")
    solid_angle = area * (np.pi / 180.0) ** 2
    if not 0.0 < solid_angle < 2.0 * np.pi:
        raise ValueError(f"Field area must be between 0 and a hemisphere ({2 * np.pi * (180 / np.pi) ** 2:.0f} deg^2).")

    if field_shape == "circle":
        return float(np.arccos(1.0 - solid_angle / (2.0 * np.pi)))
    return brentq(lambda hw: 4.0 * hw * np.sin(hw) - solid_angle, 0.0, np.pi / 2)


def max_off_axis_angle(half_angle: float, field_shape: str) -> float:
    """
    Largest angle (radians) between the line of sight and any direction in the field (the corners, for a square).
    """
    if field_shape == "circle":
        return half_angle
    return float(np.arccos(np.cos(half_angle) ** 2))


@dataclass(frozen=True, slots=True)
class Shell:
    """
    Comoving-distance shell of the lightcone filled from one snapshot.

    Attributes
    ----------
    snapshot_index: int
        Index of the snapshot in the redshift-sorted list.
    d_min, d_max: float
        Shell edges (comoving Mpc).
    """

    snapshot_index: int
    d_min: float
    d_max: float


def snapshot_shells(snapshot_distances: np.ndarray, d_min: float, d_max: float) -> list[Shell]:
    """
    Splits [d_min, d_max] into shells, assigning each distance to its nearest snapshot. Shell edges therefore sit
    midway between consecutive snapshot distances, and the first (last) snapshot also fills everything nearer (further).

    Parameters
    ----------
    snapshot_distances: ndarray
        Comoving distances of the snapshot redshifts, strictly increasing.
    d_min, d_max: float
        Distance range of the lightcone.

    Returns
    -------
    shells: list[Shell]
        Non-empty shells, in increasing distance.
    """
    snapshot_distances = np.asarray(snapshot_distances, dtype=np.float64)
    if np.any(np.diff(snapshot_distances) <= 0):
        raise ValueError("Snapshot distances must be strictly increasing (duplicate snapshot redshifts?).")

    midpoints = 0.5 * (snapshot_distances[1:] + snapshot_distances[:-1])
    lower = np.concatenate([[-np.inf], midpoints])
    upper = np.concatenate([midpoints, [np.inf]])

    shells = []
    for i, (lo, hi) in enumerate(zip(np.maximum(lower, d_min), np.minimum(upper, d_max))):
        if hi > lo:
            shells.append(Shell(snapshot_index=i, d_min=float(lo), d_max=float(hi)))
    return shells


@dataclass(frozen=True, slots=True)
class BoxTransform:
    """
    Symmetry operation of the periodic box: an axis permutation with reflections, then a periodic shift.

    The transformed position of a point p is ``(flips * p[permutation] + shift) mod boxsize``.
    """

    permutation: np.ndarray
    flips: np.ndarray
    shift: np.ndarray

    def positions(self, pos: np.ndarray, boxsize: float) -> np.ndarray:
        """
        Applies the transform to (N, 3) positions, returning positions in [0, boxsize).
        """
        return np.mod(self.flips * pos[:, self.permutation] + self.shift, boxsize)

    def vectors(self, vec: np.ndarray) -> np.ndarray:
        """
        Applies the rotation part of the transform to (N, 3) vectors such as velocities.
        """
        return self.flips * vec[:, self.permutation]


IDENTITY_TRANSFORM = BoxTransform(permutation=np.arange(3), flips=np.ones(3), shift=np.zeros(3))


def random_transform(rng: np.random.Generator, boxsize: float) -> BoxTransform:
    """
    Draws one of the 48 symmetry operations of the cube with a uniform random shift.
    """
    return BoxTransform(
        permutation=rng.permutation(3),
        flips=rng.choice([-1.0, 1.0], size=3),
        shift=rng.uniform(0.0, boxsize, size=3),
    )


def replica_cells(boxsize: float, d_min: float, d_max: float, theta_max: float) -> np.ndarray:
    """
    Integer offsets of the box replicas that can overlap the part of the cone (half-angle theta_max about +z)
    between distances d_min and d_max. Replica n spans n * boxsize + [0, boxsize)^3.

    Returns
    -------
    cells: ndarray
        (M, 3) integer offsets.
    """
    transverse = d_max * np.sin(theta_max)
    lo = np.floor(np.array([-transverse, -transverse, d_min * np.cos(theta_max)]) / boxsize).astype(np.int64)
    hi = np.floor(np.array([transverse, transverse, d_max]) / boxsize).astype(np.int64)

    axes = [np.arange(lo[k], hi[k] + 1) for k in range(3)]
    cells = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # cull with the bounding sphere of each replica
    centres = (cells + 0.5) * boxsize
    radius = 0.5 * np.sqrt(3.0) * boxsize
    dist = np.linalg.norm(centres, axis=1)
    in_shell = (dist - radius <= d_max) & (dist + radius >= d_min)
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.arccos(np.clip(centres[:, 2] / dist, -1.0, 1.0))
        slack = np.arcsin(np.clip(radius / dist, 0.0, 1.0))
    in_cone = (dist <= radius) | (angle <= theta_max + slack)
    return cells[in_shell & in_cone]


def sky_coordinates(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sky coordinates of lightcone-frame positions q (N, 3).

    Returns
    -------
    ra, dec: ndarray
        Radians, about the field centre on the +z axis.
    d: ndarray
        Comoving distance.
    """
    d = np.linalg.norm(q, axis=1)
    ra = np.arctan2(q[:, 0], q[:, 2])
    dec = np.arcsin(np.clip(q[:, 1] / np.where(d > 0, d, 1.0), -1.0, 1.0))
    return ra, dec, d


def in_field(q: np.ndarray, ra: np.ndarray, dec: np.ndarray, half_angle: float, field_shape: str) -> np.ndarray:
    """
    Boolean mask of positions q (N, 3) whose sky coordinates fall inside the field.
    """
    if field_shape == "circle":
        d = np.linalg.norm(q, axis=1)
        return q[:, 2] >= d * np.cos(half_angle)
    return (np.abs(ra) <= half_angle) & (np.abs(dec) <= half_angle)


def select_in_replica(
    pos: np.ndarray,
    cell: np.ndarray,
    boxsize: float,
    shell: Shell,
    half_angle: float,
    field_shape: str,
    theta_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Selects the objects of one box replica that lie inside the field and the shell.

    Parameters
    ----------
    pos: ndarray
        (N, 3) transformed positions in [0, boxsize).
    cell: ndarray
        Integer offset of the replica.
    shell: Shell
        The distance shell being filled.
    theta_max: float
        Largest off-axis angle of the field (see max_off_axis_angle), used for a cheap pre-cut.

    Returns
    -------
    idx: ndarray
        Indices into pos of the selected objects.
    ra, dec, d: ndarray
        Their sky coordinates (radians) and comoving distances.
    q: ndarray
        Their (M, 3) lightcone-frame positions.
    """
    q = pos + cell * boxsize

    # cheap pre-cut on the line-of-sight coordinate before computing norms and angles
    candidate = np.flatnonzero((q[:, 2] >= shell.d_min * np.cos(theta_max)) & (q[:, 2] <= shell.d_max))
    q = q[candidate]

    ra, dec, d = sky_coordinates(q)
    keep = (d >= shell.d_min) & (d < shell.d_max) & in_field(q, ra, dec, half_angle, field_shape)
    return candidate[keep], ra[keep], dec[keep], d[keep], q[keep]


def observed_redshift(z_cos: np.ndarray, q: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    Redshift including the line-of-sight peculiar velocity: 1 + z_obs = (1 + z_cos)(1 + v_los / c).

    Parameters
    ----------
    z_cos: ndarray
        Cosmological redshifts.
    q: ndarray
        (N, 3) lightcone-frame positions.
    vel: ndarray
        (N, 3) peculiar velocities (km/s) in the lightcone frame.
    """
    v_los = np.einsum("ij,ij->i", q, vel) / np.linalg.norm(q, axis=1)
    return (1.0 + z_cos) * (1.0 + v_los / SPEED_OF_LIGHT_KMS) - 1.0
