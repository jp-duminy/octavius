"""

Tests for lightcone construction: geometry helpers, and lightcones built from synthetic catalogues of uniform points.

"""

# default libraries
from pathlib import Path

# testing
import pytest

# other packages
import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# internal imports
from octavius.lightcone import build_lightcone, load_lightcone
from octavius.lightcone.geometry import (
    DistanceRedshiftTable,
    field_half_angle,
    max_off_axis_angle,
    snapshot_shells,
    observed_redshift,
)
from octavius.utils.loader import load_catalogue

COSMOLOGY = {"h": 0.68, "omega_matter": 0.3, "omega_lambda": 0.7, "w_0": -1.0, "w_a": 0.0}
BOXSIZE_MPC = 100.0
N_POINTS = 20_000


def write_catalogue(
    path: Path, redshift: float, seed: int, boxsize_mpc: float = BOXSIZE_MPC, velocity: float = 0.0, **cosmology
) -> Path:
    """
    Writes a minimal Octavius catalogue of galaxies uniformly distributed in the box.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, boxsize_mpc * 1000.0, size=(N_POINTS, 3))  # ckpc
    vel = rng.normal(0.0, velocity, size=(N_POINTS, 3)) if velocity > 0 else np.zeros((N_POINTS, 3))
    columns = {
        "com_pos_baryon": (pos, "kpc", 1),
        "com_vel_baryon": (vel, "km/s", 0),
        "mass_star": (10 ** rng.uniform(8, 11, N_POINTS), "Msun", 0),
        "radius_half_star": (rng.uniform(1, 10, N_POINTS), "kpc", 1),
    }
    header = {"redshift": redshift, "scale_factor": 1.0 / (1.0 + redshift), "boxsize": boxsize_mpc * 1000.0}
    header.update(COSMOLOGY | cosmology)

    with h5py.File(path, "w") as f:
        metadata = f.create_group("metadata")
        metadata.attrs.update(
            {"catalogue_format_version": 1, "simulation_type": "TEST", "octavius_version": "test", "timestamp": "now"}
        )
        metadata.create_group("config_parameters").attrs["halo_id_source"] = "SNAPSHOT"
        h = f.create_group("header")
        for name, value in header.items():
            dataset = h.create_dataset(name, data=value)
            dataset.attrs.update(
                {"unit": "kpc" if name == "boxsize" else "", "a_exp": int(name == "boxsize"), "description": name}
            )
        galaxies = f.create_group("galaxy_data")
        galaxies.create_dataset("GalID", data=np.arange(N_POINTS))
        galaxies.create_group("membership")
        core = galaxies.create_group("properties/core")
        for name, (data, unit, a_exp) in columns.items():
            dataset = core.create_dataset(name, data=data)
            dataset.attrs.update({"unit": unit, "a_exp": a_exp, "description": name})
    return path


@pytest.fixture(scope="module")
def catalogues(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    tmp_dir = tmp_path_factory.mktemp("lightcone_catalogues")
    return [write_catalogue(tmp_dir / f"cat_{i}.hdf5", redshift=z, seed=i) for i, z in enumerate([1.0, 0.25, 0.5])]


def flat_cosmology() -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=100 * COSMOLOGY["h"], Om0=COSMOLOGY["omega_matter"])


# geometry


def test_distance_table_matches_astropy():
    cosmology = flat_cosmology()
    table = DistanceRedshiftTable(cosmology=cosmology, z_max=5.0)
    z = np.array([0.0, 0.01, 0.3, 1.0, 2.7, 5.0])
    assert np.allclose(table.distance(z), cosmology.comoving_distance(z).to_value("Mpc"), rtol=1e-6, atol=1e-6)
    assert np.allclose(table.redshift(table.distance(z)), z, atol=1e-6)
    with pytest.raises(ValueError):
        table.distance(6.0)


@pytest.mark.parametrize("area", [0.01, 1.0, 100.0, 5000.0])
def test_field_half_angle(area: float):
    solid_angle = area * (np.pi / 180) ** 2
    hw = field_half_angle(area, "square")
    assert np.isclose(4 * hw * np.sin(hw), solid_angle)
    radius = field_half_angle(area, "circle")
    assert np.isclose(2 * np.pi * (1 - np.cos(radius)), solid_angle)
    assert max_off_axis_angle(hw, "square") > hw
    if area <= 1.0:  # small fields reduce to the flat-sky values
        assert np.isclose(np.degrees(hw), np.sqrt(area) / 2, rtol=1e-4)
    with pytest.raises(ValueError):
        field_half_angle(area, "hexagon")


def test_field_half_angle_rejects_more_than_a_hemisphere():
    with pytest.raises(ValueError):
        field_half_angle(30000.0, "square")


def test_snapshot_shells_are_contiguous_and_nearest():
    distances = np.array([100.0, 300.0, 400.0, 1000.0])
    shells = snapshot_shells(distances, d_min=50.0, d_max=800.0)
    assert [s.snapshot_index for s in shells] == [0, 1, 2, 3]
    assert shells[0].d_min == 50.0 and shells[-1].d_max == 800.0
    for a, b in zip(shells[:-1], shells[1:]):
        assert a.d_max == b.d_min
    assert np.isclose(shells[1].d_min, 200.0) and np.isclose(shells[1].d_max, 350.0)

    # a range inside one snapshot's shell gives a single shell
    assert [s.snapshot_index for s in snapshot_shells(distances, d_min=310.0, d_max=340.0)] == [1]
    with pytest.raises(ValueError):
        snapshot_shells(np.array([100.0, 100.0]), 0.0, 200.0)


def test_observed_redshift():
    q = np.array([[0.0, 0.0, 1000.0], [0.0, 1000.0, 0.0]])
    vel = np.array([[0.0, 0.0, 300.0], [0.0, 0.0, 300.0]])
    z_obs = observed_redshift(np.array([0.5, 0.5]), q, vel)
    assert np.isclose(z_obs[0], 1.5 * (1 + 300.0 / 299792.458) - 1)
    assert np.isclose(z_obs[1], 0.5)  # transverse velocity does not shift the redshift


# lightcones


@pytest.mark.parametrize("field_shape", ["square", "circle"])
@pytest.mark.parametrize("randomise", ["snapshot", "replica", "none"])
def test_lightcone_geometry_and_counts(catalogues: list[Path], tmp_path: Path, field_shape: str, randomise: str):
    area, z_min, z_max = 4.0, 0.1, 1.2  # wider than the box beyond z ~ 0.75, so lateral tiling is exercised
    (path,) = build_lightcone(
        catalogue_paths=catalogues,
        output_path=tmp_path / "lc.hdf5",
        area=area,
        z_min=z_min,
        z_max=z_max,
        field_shape=field_shape,
        randomise=randomise,
        seed=1,
    )

    with load_lightcone(path) as lc:
        ra, dec, d, z = (lc.get_dataset(n) for n in ("ra", "dec", "comoving_distance", "z_cos"))
        snapshot_index, group_index, replica = (lc.get_dataset(n) for n in ("snapshot_index", "group_index", "replica"))
        snapshot_z = lc.snapshot_redshifts
        hw = lc.header["half_angle"]

    # catalogues are sorted by redshift
    assert np.all(np.diff(snapshot_z) > 0)

    # inside the field and the redshift range, sorted by distance
    if field_shape == "square":
        assert np.all(np.abs(ra) <= hw) and np.all(np.abs(dec) <= hw)
    else:
        cos_sep = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
        assert np.all(cos_sep >= np.cos(np.radians(hw)) - 1e-12)
    assert np.all((z >= z_min - 1e-9) & (z <= z_max + 1e-9))
    assert np.all(np.diff(d) >= 0)

    # every object comes from the snapshot nearest in distance
    table = DistanceRedshiftTable(flat_cosmology(), z_max=2.0)
    snapshot_d = table.distance(snapshot_z)
    nearest = np.argmin(np.abs(d[:, None] - snapshot_d[None, :]), axis=1)
    assert np.array_equal(nearest, snapshot_index)

    # each object appears at most once per replica
    keys = np.column_stack([snapshot_index, group_index, replica])
    assert len(np.unique(keys, axis=0)) == len(keys)

    # the count matches the mean density times the lightcone volume
    solid_angle = area * (np.pi / 180) ** 2
    volume = solid_angle / 3 * (table.distance(z_max) ** 3 - table.distance(z_min) ** 3)
    expected = N_POINTS / BOXSIZE_MPC**3 * volume
    assert abs(len(d) - expected) < 5 * np.sqrt(expected)


def test_lightcone_reproducible(catalogues: list[Path], tmp_path: Path):
    kwargs = {"catalogue_paths": catalogues, "area": 1.0, "z_min": 0.2, "z_max": 0.8}
    (a,) = build_lightcone(output_path=tmp_path / "a.hdf5", seed=7, **kwargs)
    (b,) = build_lightcone(output_path=tmp_path / "b.hdf5", seed=7, **kwargs)
    (c,) = build_lightcone(output_path=tmp_path / "c.hdf5", seed=8, **kwargs)
    with load_lightcone(a) as la, load_lightcone(b) as lb, load_lightcone(c) as lc:
        assert la.header["seed"] == "7"
        assert np.array_equal(la.get_dataset("ra"), lb.get_dataset("ra"))
        assert len(la) != len(lc) or not np.array_equal(la.get_dataset("ra"), lc.get_dataset("ra"))


def test_multiple_lightcones_differ(catalogues: list[Path], tmp_path: Path):
    paths = build_lightcone(
        catalogue_paths=catalogues,
        output_path=tmp_path / "lc.hdf5",
        area=1.0,
        z_min=0.2,
        z_max=0.8,
        n_lightcones=3,
        seed=3,
    )
    assert [p.name for p in paths] == ["lc_000.hdf5", "lc_001.hdf5", "lc_002.hdf5"]
    group_indices = []
    for i, path in enumerate(paths):
        with load_lightcone(path) as lc:
            assert lc.header["lightcone_number"] == i
            group_indices.append(lc.get_dataset("group_index"))
    assert not np.array_equal(group_indices[0], group_indices[1])


def test_properties_and_physical_conversion(catalogues: list[Path], tmp_path: Path):
    (path,) = build_lightcone(
        catalogue_paths=catalogues,
        output_path=tmp_path / "lc.hdf5",
        area=1.0,
        z_min=0.2,
        z_max=1.2,
        properties=["mass_star", "radius_half_star"],
        seed=2,
    )
    with load_lightcone(path) as lc:
        assert {"mass_star", "radius_half_star"} <= set(lc.keys())
        assert lc.get_units("mass_star") == "Msun"
        snapshot_index, group_index = lc.get_dataset("snapshot_index"), lc.get_dataset("group_index")
        radius_physical = lc.get_dataset("radius_half_star", to_physical=True, to_units="pc")
        mass = lc.get_dataset("mass_star")
        catalogue_paths = lc.catalogue_paths

    for i, cat_path in enumerate(catalogue_paths):
        rows = snapshot_index == i
        if not rows.any():
            continue
        with load_catalogue(cat_path) as cat:
            expected_mass = cat.galaxies.get_dataset("mass_star")[group_index[rows]]
            expected_radius = cat.galaxies.get_dataset("radius_half_star", to_physical=True, to_units="pc")[
                group_index[rows]
            ]
        assert np.array_equal(mass[rows], expected_mass)
        assert np.allclose(radius_physical[rows], expected_radius)


def test_redshift_space(tmp_path: Path):
    velocity = 500.0
    cats = [
        write_catalogue(tmp_path / f"v_{i}.hdf5", redshift=z, seed=i, velocity=velocity)
        for i, z in enumerate([0.3, 0.6])
    ]
    (path,) = build_lightcone(
        catalogue_paths=cats,
        output_path=tmp_path / "lc.hdf5",
        area=1.0,
        z_min=0.2,
        z_max=0.7,
        redshift_space=True,
        seed=4,
    )
    with load_lightcone(path) as lc:
        z_cos, z_obs = lc.get_dataset("z_cos"), lc.get_dataset("z_obs")
    dz = (z_obs - z_cos) / (1 + z_cos) * 299792.458  # line-of-sight velocity, km/s
    assert np.isclose(np.std(dz), velocity, rtol=0.05)
    assert abs(np.mean(dz)) < 5 * velocity / np.sqrt(len(dz))


def test_lightcone_errors(catalogues: list[Path], tmp_path: Path):
    kwargs = {
        "catalogue_paths": catalogues,
        "output_path": tmp_path / "lc.hdf5",
        "area": 1.0,
        "z_min": 0.2,
        "z_max": 0.8,
    }
    build_lightcone(**kwargs)
    with pytest.raises(FileExistsError):
        build_lightcone(**kwargs)
    build_lightcone(overwrite=True, **kwargs)

    with pytest.raises(KeyError):
        build_lightcone(**(kwargs | {"properties": ["not_a_dataset"]}), overwrite=True)
    with pytest.raises(ValueError):
        build_lightcone(**(kwargs | {"properties": ["ra"]}), overwrite=True)
    with pytest.raises(ValueError):
        build_lightcone(**(kwargs | {"z_min": 1.0, "z_max": 0.5}), overwrite=True)

    other = write_catalogue(tmp_path / "other.hdf5", redshift=2.0, seed=9, h=0.7)
    with pytest.raises(ValueError, match="disagree on h"):
        build_lightcone(**(kwargs | {"catalogue_paths": catalogues + [other]}), overwrite=True)


def test_lightcone_cli(catalogues: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from octavius.run_octavius import main, parse_args

    args = parse_args(["lightcone", *map(str, catalogues), "--area", "1", "--z-min", "0.2", "--z-max", "0.8"])
    assert args.catalogues == catalogues and args.group_type == "galaxies" and args.shape == "square"

    output = tmp_path / "cli.hdf5"
    argv = ["octavius", "lightcone", *map(str, catalogues), "-a", "1", "--z-min", "0.2", "--z-max", "0.8"]
    argv += ["-o", str(output), "--shape", "circle", "-p", "mass_star", "--redshift-space", "--seed", "5", "-n", "2"]
    monkeypatch.setattr("sys.argv", argv)
    main()
    for i in range(2):
        with load_lightcone(tmp_path / f"cli_{i:03d}.hdf5") as lc:
            assert lc.header["field_shape"] == "circle" and lc.header["seed"] == "5"
            assert {"mass_star", "z_obs"} <= set(lc.keys())
