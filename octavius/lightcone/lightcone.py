"""

Lightcone construction from a set of Octavius catalogues at different redshifts, and a reader for the output.

The lightcone is filled shell by shell: each comoving-distance shell takes its objects from the snapshot nearest in
distance, using randomly transformed periodic replicas of the box to cover the full field and depth. Only the
catalogues are needed; no raw snapshot is read.

NOTE: for devs, please give user-facing APIs the classic numpy docstring style.

"""

# default libraries
from datetime import UTC, datetime
from pathlib import Path

# other packages
import h5py
import numpy as np
import astropy.units as u
from astropy.cosmology import w0waCDM

# internal imports
from ..utils.loader import load_catalogue, GroupCollection
from ..version import __version__
from ..log import get_logger
from .geometry import (
    FIELD_SHAPES,
    RANDOMISE_MODES,
    IDENTITY_TRANSFORM,
    DistanceRedshiftTable,
    BoxTransform,
    Shell,
    field_half_angle,
    max_off_axis_angle,
    snapshot_shells,
    random_transform,
    replica_cells,
    select_in_replica,
    observed_redshift,
)

logger = get_logger()

LIGHTCONE_VERSION = 1
COSMOLOGY_FIELDS = ("h", "omega_matter", "omega_lambda", "w_0", "w_a")

DEFAULT_COLUMNS = {  # (position, velocity) used to place each group type
    "galaxies": ("com_pos_baryon", "com_vel_baryon"),
    "haloes": ("minpot_pos", "com_vel_total"),
}

# per-object columns every lightcone has: name -> (unit, description)
LIGHTCONE_COLUMNS = {
    "ra": ("deg", "Right ascension relative to the field centre (on the line of sight)."),
    "dec": ("deg", "Declination relative to the field centre (on the line of sight)."),
    "comoving_distance": ("Mpc", "Line-of-sight comoving distance from the observer."),
    "z_cos": ("", "Cosmological redshift at the comoving distance."),
    "z_obs": ("", "Observed redshift, including the line-of-sight peculiar velocity."),
    "snapshot_index": ("", "Index into the shells datasets of the source catalogue."),
    "group_index": ("", "Row of the object in its source catalogue."),
    "replica": ("", "Integer offset (in box lengths) of the box replica the object was taken from."),
}


def build_lightcone(
    catalogue_paths: list[Path],
    output_path: Path,
    area: float,
    z_min: float,
    z_max: float,
    group_type: str = "galaxies",
    field_shape: str = "square",
    properties: list[str] | None = None,
    n_lightcones: int = 1,
    seed: int | None = None,
    redshift_space: bool = False,
    randomise: str = "snapshot",
    position_column: str | None = None,
    velocity_column: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Builds lightcones of galaxies or haloes from Octavius catalogues at a range of redshifts.

    Each comoving-distance shell of the lightcone is filled from the snapshot nearest to it in distance, so shell edges
    sit midway between the snapshot distances. The box is periodically replicated to cover the field, so the field may
    be wider than the box; each replica is a randomly rotated, reflected and shifted copy (see ``randomise``).

    Parameters
    ----------
    catalogue_paths: list[pathlib.Path]
        Catalogues of one simulation, in any order. Their redshifts set the shells.
    output_path: pathlib.Path
        Output HDF5 file. With several lightcones, ``_000``, ``_001``, ... is appended to the stem.
    area: float
        Solid angle of the field (deg^2).
    z_min, z_max: float
        Redshift range, in cosmological redshift.
    group_type: str
        ``galaxies`` or ``haloes``. Default: ``galaxies``
    field_shape: str
        ``square`` (|RA|, |Dec| <= half-width) or ``circle`` (radius about the line of sight). Default: ``square``
    properties: list[str], optional
        Catalogue datasets to copy into the lightcone (e.g. ``["mass_star", "sfr"]``), with their units.
    n_lightcones: int
        Number of independent lightcones. Default: 1
    seed: int, optional
        Seed for the random box transforms; the seed used is stored in the output either way.
    redshift_space: bool
        Also compute ``z_obs``, which includes the line-of-sight peculiar velocity. Default: ``False``
    randomise: str
        How often to draw a new random box transform: ``snapshot`` (once per shell; structures stay continuous within a
        shell, but repeat across the field when it is wider than the box), ``replica`` (every box replica; fewer
        repeated structures, but discontinuities at replica faces) or ``none`` (one transform for the whole lightcone).
        Default: ``snapshot``
    position_column, velocity_column: str, optional
        Catalogue datasets placing each object. Defaults: ``com_pos_baryon``/``com_vel_baryon`` for galaxies and
        ``minpot_pos``/``com_vel_total`` for haloes.
    overwrite: bool
        Replace existing output files. Default: ``False``

    Returns
    -------
    lightcone_paths: list[pathlib.Path]
        The lightcone files written.
    """
    if group_type not in DEFAULT_COLUMNS:
        raise ValueError(f"group_type must be one of {list(DEFAULT_COLUMNS)}, not '{group_type}'.")
    if field_shape not in FIELD_SHAPES:
        raise ValueError(f"field_shape must be one of {FIELD_SHAPES}, not '{field_shape}'.")
    if randomise not in RANDOMISE_MODES:
        raise ValueError(f"randomise must be one of {RANDOMISE_MODES}, not '{randomise}'.")
    if not 0.0 <= z_min < z_max:
        raise ValueError(f"Need 0 <= z_min < z_max (got z_min = {z_min}, z_max = {z_max}).")
    if n_lightcones < 1:
        raise ValueError("Need at least one lightcone.")

    properties = list(properties or [])
    clashes = set(properties) & set(LIGHTCONE_COLUMNS)
    if clashes:
        raise ValueError(f"Properties {sorted(clashes)} clash with lightcone column names.")
    position_column = position_column or DEFAULT_COLUMNS[group_type][0]
    velocity_column = velocity_column or DEFAULT_COLUMNS[group_type][1]

    output_path = Path(output_path)
    if n_lightcones == 1:
        lightcone_paths = [output_path]
    else:
        lightcone_paths = [
            output_path.with_name(f"{output_path.stem}_{i:03d}{output_path.suffix}") for i in range(n_lightcones)
        ]
    existing = [p for p in lightcone_paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Lightcone files already exist (pass overwrite=True to replace): {existing}")

    # snapshots, sorted by redshift, and the shared cosmology
    catalogue_paths = [Path(p) for p in catalogue_paths]
    snapshots = sorted((_read_snapshot_header(p) for p in catalogue_paths), key=lambda s: s["redshift"])
    snapshot_redshifts = np.array([s["redshift"] for s in snapshots])
    sim = _check_consistent(snapshots)
    boxsize = sim["boxsize"]

    cosmology = w0waCDM(
        H0=100.0 * sim["h"], Om0=sim["omega_matter"], Ode0=sim["omega_lambda"], w0=sim["w_0"], wa=sim["w_a"]
    )
    table = DistanceRedshiftTable(cosmology=cosmology, z_max=1.01 * max(z_max, snapshot_redshifts.max()) + 0.01)
    shells = snapshot_shells(
        snapshot_distances=table.distance(snapshot_redshifts), d_min=table.distance(z_min), d_max=table.distance(z_max)
    )
    if z_max > snapshot_redshifts.max():
        logger.warning(
            f"z_max = {z_max} is beyond the highest snapshot (z = {snapshot_redshifts.max():.3f}), which fills the rest of the lightcone."
        )

    half_angle = field_half_angle(area=area, field_shape=field_shape)
    theta_max = max_off_axis_angle(half_angle=half_angle, field_shape=field_shape)

    seed_sequence = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in seed_sequence.spawn(n_lightcones)]
    whole_transforms = [random_transform(rng, boxsize) for rng in rngs] if randomise == "none" else None

    logger.info(
        f"Building {n_lightcones} {field_shape} lightcone(s) of {group_type}: {area} deg^2, {z_min} < z < {z_max}, "
        f"{len(shells)} shells from {len(snapshots)} catalogues, box {boxsize:.1f} cMpc."
    )

    selections: list[list[dict]] = [[] for _ in range(n_lightcones)]
    property_attrs: dict[str, dict] = {}

    for shell in shells:
        snapshot = snapshots[shell.snapshot_index]
        cells = replica_cells(boxsize=boxsize, d_min=shell.d_min, d_max=shell.d_max, theta_max=theta_max)
        logger.info(
            f"Shell z = {table.redshift(shell.d_min):.3f}-{table.redshift(shell.d_max):.3f} from {snapshot['path'].name} "
            f"(z = {snapshot['redshift']:.3f}): {len(cells)} box replicas."
        )

        with load_catalogue(catalogue_path=snapshot["path"]) as catalogue:
            collection = catalogue.galaxies if group_type == "galaxies" else catalogue.haloes
            if collection is None or len(collection) == 0:
                logger.warning(f"No {group_type} in {snapshot['path'].name}; its shell is empty.")
                continue

            _check_columns(collection, [position_column] + ([velocity_column] if redshift_space else []) + properties)
            pos = collection.get_dataset(position_column, to_units="Mpc")  # comoving
            vel = collection.get_dataset(velocity_column, to_units="km/s") if redshift_space else None
            props = {name: collection.get_dataset(name) for name in properties}
            for name in properties:
                property_attrs.setdefault(name, collection.get_attributes(name))

        for i, rng in enumerate(rngs):
            if randomise == "snapshot":
                transform = random_transform(rng, boxsize)
            elif randomise == "none":
                transform = whole_transforms[i]
            else:
                transform = IDENTITY_TRANSFORM  # redrawn per replica below

            pos_t = transform.positions(pos, boxsize)
            for cell in cells:
                if randomise == "replica":
                    transform = random_transform(rng, boxsize)
                    pos_t = transform.positions(pos, boxsize)
                selection = _select(
                    pos_t=pos_t,
                    vel=vel,
                    props=props,
                    transform=transform,
                    cell=cell,
                    boxsize=boxsize,
                    shell=shell,
                    half_angle=half_angle,
                    field_shape=field_shape,
                    theta_max=theta_max,
                    table=table,
                )
                if selection is not None:
                    selections[i].append(selection)

    header = {
        "lightcone_format_version": LIGHTCONE_VERSION,
        "octavius_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "group_type": group_type,
        "field_shape": field_shape,
        "area": area,
        "half_angle": np.degrees(half_angle),
        "z_min": z_min,
        "z_max": z_max,
        "redshift_space": redshift_space,
        "randomise": randomise,
        "seed": str(seed_sequence.entropy),
        "position_column": position_column,
        "velocity_column": velocity_column if redshift_space else "",
        "boxsize": boxsize,
        **{name: sim[name] for name in COSMOLOGY_FIELDS},
    }
    shell_data = {
        "catalogue_path": np.array([str(s["path"].resolve()) for s in snapshots], dtype=h5py.string_dtype()),
        "snapshot_redshift": snapshot_redshifts,
        "d_min": _per_snapshot(shells, len(snapshots), "d_min"),
        "d_max": _per_snapshot(shells, len(snapshots), "d_max"),
    }

    for i, path in enumerate(lightcone_paths):
        header["lightcone_number"] = i
        n_objects = _write_lightcone(
            path=path,
            selections=selections[i],
            header=header,
            shell_data=shell_data,
            properties=properties,
            property_attrs=property_attrs,
            redshift_space=redshift_space,
        )
        logger.info(f"Wrote {n_objects:,} {group_type} to {path}.")

    return lightcone_paths


def _read_snapshot_header(path: Path) -> dict:
    """
    Reads the redshift, box size (cMpc) and cosmology of a catalogue.
    """
    with load_catalogue(catalogue_path=path) as catalogue:
        snapshot = {
            "path": path,
            "redshift": catalogue.redshift,
            "boxsize": catalogue.sim_info("boxsize", to_units="Mpc"),
        }
        snapshot.update({name: catalogue.sim_info(name) for name in COSMOLOGY_FIELDS})
    return snapshot


def _check_consistent(snapshots: list[dict]) -> dict:
    """
    Checks the catalogues share one box and cosmology, and that no two share a redshift; returns the shared values.
    """
    first = snapshots[0]
    for snapshot in snapshots[1:]:
        for name in ("boxsize",) + COSMOLOGY_FIELDS:
            if not np.isclose(snapshot[name], first[name], rtol=1e-6, atol=1e-10):
                raise ValueError(
                    f"Catalogues disagree on {name}: {first[name]} ({first['path'].name}) vs {snapshot[name]} ({snapshot['path'].name})."
                )
    redshifts = [s["redshift"] for s in snapshots]
    if len(set(redshifts)) != len(redshifts):
        raise ValueError("Two catalogues have the same redshift.")
    return first


def _check_columns(collection: GroupCollection, names: list[str]) -> None:
    """
    Raises a KeyError naming any requested datasets missing from the catalogue.
    """
    missing = [name for name in names if name not in collection._dataset_paths]
    if missing:
        raise KeyError(
            f"Datasets {missing} are not in the catalogue; see GroupCollection.describe() for those available."
        )


def _select(
    pos_t: np.ndarray,
    vel: np.ndarray | None,
    props: dict[str, np.ndarray],
    transform: BoxTransform,
    cell: np.ndarray,
    boxsize: float,
    shell: Shell,
    half_angle: float,
    field_shape: str,
    theta_max: float,
    table: DistanceRedshiftTable,
) -> dict | None:
    """
    Selects the objects of one replica and returns their lightcone columns, or None if there are none.
    """
    idx, ra, dec, d, q = select_in_replica(
        pos=pos_t,
        cell=cell,
        boxsize=boxsize,
        shell=shell,
        half_angle=half_angle,
        field_shape=field_shape,
        theta_max=theta_max,
    )
    if len(idx) == 0:
        return None

    z_cos = table.redshift(d)
    selection = {
        "ra": np.degrees(ra),
        "dec": np.degrees(dec),
        "comoving_distance": d,
        "z_cos": z_cos,
        "snapshot_index": np.full(len(idx), shell.snapshot_index, dtype=np.int64),
        "group_index": idx.astype(np.int64),
        "replica": np.broadcast_to(cell, (len(idx), 3)).astype(np.int64),
    }
    if vel is not None:
        selection["z_obs"] = observed_redshift(z_cos=z_cos, q=q, vel=transform.vectors(vel[idx]))
    for name, values in props.items():
        selection[name] = values[idx]
    return selection


def _per_snapshot(shells: list[Shell], n_snapshots: int, field: str) -> np.ndarray:
    """
    Shell edges indexed by snapshot; NaN for snapshots that fill no shell.
    """
    values = np.full(n_snapshots, np.nan)
    for shell in shells:
        values[shell.snapshot_index] = getattr(shell, field)
    return values


def _write_lightcone(
    path: Path,
    selections: list[dict],
    header: dict,
    shell_data: dict,
    properties: list[str],
    property_attrs: dict[str, dict],
    redshift_space: bool,
) -> int:
    """
    Writes one lightcone to HDF5, sorted by comoving distance; returns the number of objects.
    """
    columns = [c for c in LIGHTCONE_COLUMNS if redshift_space or c != "z_obs"]
    data = {}
    for name in columns + properties:
        if selections:
            data[name] = np.concatenate([s[name] for s in selections])
        else:
            data[name] = np.zeros((0, 3) if name == "replica" else 0)
    order = np.argsort(data["comoving_distance"], kind="stable")

    with h5py.File(path, "w") as f:
        header_group = f.create_group("header")
        for key, value in header.items():
            header_group.attrs[key] = value
        header_group.attrs["n_objects"] = len(order)

        shells_group = f.create_group("shells")
        for key, value in shell_data.items():
            shells_group.create_dataset(key, data=value)
        shells_group["d_min"].attrs["unit"] = shells_group["d_max"].attrs["unit"] = "Mpc"

        objects = f.create_group("objects")
        for name in columns:
            dataset = objects.create_dataset(name, data=data[name][order])
            unit, description = LIGHTCONE_COLUMNS[name]
            dataset.attrs["unit"] = unit
            dataset.attrs["a_exp"] = 0
            dataset.attrs["description"] = description
        for name in properties:
            dataset = objects.create_dataset(name, data=data[name][order])
            for key, value in property_attrs.get(name, {}).items():
                dataset.attrs[key] = value

    return len(order)


def load_lightcone(lightcone_path: Path) -> LightconeCatalogue:
    """
    Load a lightcone written by build_lightcone, returning a LightconeCatalogue object.

    Parameters
    ----------
    lightcone_path: pathlib.Path
        pathlib.Path object pointing to a lightcone file.

    Returns
    -------
    lightcone: LightconeCatalogue
        The loaded lightcone interface.
    """
    with h5py.File(lightcone_path, "r") as f:
        file_version = f["header"].attrs.get("lightcone_format_version", 0)
        if file_version > LIGHTCONE_VERSION:
            raise ValueError(
                f"Lightcone version {file_version} is newer than this version of Octavius can read (up to {LIGHTCONE_VERSION}); please update."
            )
    return LightconeCatalogue(lightcone_path=Path(lightcone_path))


class LightconeCatalogue:
    """
    An object-oriented interface for a lightcone file.

    Attributes
    ----------
    header: dict
        Construction parameters and cosmology (area, z range, seed, ...).
    catalogue_paths: list[pathlib.Path]
        Source catalogues, indexed by the ``snapshot_index`` column.
    snapshot_redshifts: ndarray
        Redshifts of the source catalogues.
    n_objects: int
        Number of objects in the lightcone.

    Methods
    -------
    get_dataset(name, mask, to_units, to_physical)
        Read a per-object dataset with optional unit/physical conversion.
    get_units(name)
        Get the units of a dataset.
    keys()
        Lists available dataset names.
    """

    def __init__(self, lightcone_path: Path) -> None:

        self._file: h5py.File = h5py.File(lightcone_path, "r")
        self._path = lightcone_path

        self.header: dict = dict(self._file["header"].attrs)
        self.catalogue_paths: list[Path] = [
            Path(p.decode() if isinstance(p, bytes) else p) for p in self._file["shells/catalogue_path"][:]
        ]
        self.snapshot_redshifts: np.ndarray = self._file["shells/snapshot_redshift"][:]
        self.n_objects: int = int(self.header["n_objects"])
        self._objects = self._file["objects"]

    def keys(self) -> list[str]:
        """
        Lists the available per-object datasets.
        """
        return list(self._objects.keys())

    def get_units(self, name: str) -> str:
        """
        Returns the units of a dataset as a string.
        """
        return self._objects[name].attrs["unit"]

    def get_dataset(
        self,
        name: str,
        mask: np.ndarray | None = None,
        to_units: str | None = None,
        to_physical: bool = False,
    ) -> np.ndarray:
        """
        Retrieves a per-object dataset and applies any requested conversions.

        Parameters
        ----------
        name: str
            The name of the dataset.
        mask: ndarray, optional
            Boolean mask or indices to apply to the dataset.
        to_units: str, optional
            Target units to convert the dataset to.
        to_physical: bool, optional
            Convert comoving properties to physical, using the scale factor of each object's source snapshot.

        Returns
        -------
        data: ndarray
            Array with requested conversions applied.
        """
        dataset = self._objects[name]
        data = dataset[:]
        snapshot_index = self._objects["snapshot_index"][:]
        if mask is not None:
            data = data[mask]
            snapshot_index = snapshot_index[mask]

        if to_physical and dataset.attrs.get("a_exp", 0) != 0:
            scale_factor = 1.0 / (1.0 + self.snapshot_redshifts[snapshot_index])
            data = data * scale_factor.reshape((-1,) + (1,) * (data.ndim - 1)) ** dataset.attrs["a_exp"]

        if to_units is not None:
            data = data * u.Unit(dataset.attrs["unit"]).to(u.Unit(to_units))

        return data

    def close(self) -> None:
        """
        Manually closes the lightcone HDF5 file.
        """
        self._file.close()

    def __enter__(self) -> LightconeCatalogue:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_objects

    def __repr__(self) -> str:
        h = self.header
        return (
            f'LightconeCatalogue "{self._path.name}" | {h["n_objects"]} {h["group_type"]} | {h["field_shape"]} '
            f"{h['area']} deg^2 | {h['z_min']} < z < {h['z_max']}"
        )
