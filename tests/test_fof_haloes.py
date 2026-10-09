"""

Tests the periodic 3D friends-of-friends halo finder against a brute-force reference (scipy k-d tree pairs ->
connected components), including haloes straddling the periodic boundary and slab boundaries.

"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from octavius.data_management import OctaviusConstants
from octavius.data_management.conventions import OctaviusConfig
from octavius.data_management.snapshot_readers import build_reader
from octavius.external_halo_sources.halo_data_structures import SnapshotHaloSource, build_halo_source
from octavius.galaxy_finding.fof_halo_algorithm import find_fof_haloes
from octavius.run_octavius import analyse_snapshot
from octavius.utils.generate_snapshots import generate_simba_snapshot, generate_swift_snapshot
from tests.validation.output_validation import validate_group_counts, validate_halo_membership

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"

SEED = 8812391
BOXSIZE = 100.0
LINKING_LENGTH = 1.5


def make_clustered_positions(rng: np.random.Generator, n_background: int, n_clusters: int) -> np.ndarray:
    """
    Uniform background plus Gaussian clusters, some centred on the box edges/corners to test periodicity.
    """
    centres = rng.uniform(0, BOXSIZE, size=(n_clusters, 3))
    centres[0] = [0.0, 50.0, 50.0]
    centres[1] = [BOXSIZE, BOXSIZE, 0.0]
    clusters = [c + rng.normal(0, 1.5, size=(rng.integers(20, 400), 3)) for c in centres]
    background = rng.uniform(0, BOXSIZE, size=(n_background, 3))

    return np.mod(np.concatenate([background, *clusters]), BOXSIZE)


def reference_groups(pos: np.ndarray, min_members: int) -> np.ndarray:
    """
    Brute-force FOF; returns component labels with groups below min_members set to -1 (labels arbitrary).
    """
    pairs = cKDTree(pos, boxsize=BOXSIZE).query_pairs(r=LINKING_LENGTH, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pos), len(pos)))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    labels[sizes[labels] < min_members] = -1

    return labels


def assert_same_partition(labels_a: np.ndarray, labels_b: np.ndarray) -> None:
    """
    Asserts two labellings define the same groups (up to relabelling), with -1 meaning ungrouped in both.
    """
    assert np.array_equal(labels_a == -1, labels_b == -1)
    grouped = labels_a != -1
    pairs = np.unique(np.stack([labels_a[grouped], labels_b[grouped]], axis=1), axis=0)
    assert len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1]))


@pytest.mark.parametrize("n_slabs", [1, 3, 16, 1000, None])  # 1000 is clipped to the thinnest allowed slabs
@pytest.mark.parametrize("min_members", [1, 20])
def test_fof_matches_brute_force(n_slabs: int | None, min_members: int) -> None:
    rng = np.random.default_rng(SEED)
    pos = make_clustered_positions(rng, n_background=20000, n_clusters=40)

    halo_ids = find_fof_haloes(
        dm_pos=pos, boxsize=BOXSIZE, linking_length=LINKING_LENGTH, min_members=min_members, n_slabs=n_slabs
    )["dm"]

    assert_same_partition(halo_ids, reference_groups(pos, min_members))


def test_halo_ids_ordered_by_size() -> None:
    rng = np.random.default_rng(SEED)
    pos = make_clustered_positions(rng, n_background=5000, n_clusters=20)

    halo_ids = find_fof_haloes(dm_pos=pos, boxsize=BOXSIZE, linking_length=LINKING_LENGTH, min_members=20)["dm"]
    sizes = np.bincount(halo_ids[halo_ids >= 0])

    assert sizes.min() >= 20
    assert np.all(np.diff(sizes) <= 0)


def test_baryons_attach_to_nearest_dm() -> None:
    rng = np.random.default_rng(SEED)
    dm_pos = make_clustered_positions(rng, n_background=5000, n_clusters=20)
    gas_pos = make_clustered_positions(rng, n_background=3000, n_clusters=20) - 0.3  # some fall below 0: wrapped

    halo_ids = find_fof_haloes(
        dm_pos=dm_pos, boxsize=BOXSIZE, linking_length=LINKING_LENGTH, baryon_pos={"gas": gas_pos}, min_members=20
    )

    dist, nearest = cKDTree(dm_pos, boxsize=BOXSIZE).query(
        np.mod(gas_pos, BOXSIZE), distance_upper_bound=LINKING_LENGTH
    )
    expected = np.full(len(gas_pos), -1, dtype=np.int64)
    found = np.isfinite(dist)
    expected[found] = halo_ids["dm"][nearest[found]]

    assert np.array_equal(halo_ids["gas"], expected)


def test_rejects_oversized_linking_length() -> None:
    with pytest.raises(ValueError):
        find_fof_haloes(dm_pos=np.zeros((10, 3)), boxsize=BOXSIZE, linking_length=BOXSIZE / 2)


@pytest.mark.parametrize("sim_type", ["SIMBA", "SWIFT-KIARA"])
def test_fof_source_recovers_snapshot_haloes(sim_type: str, tmp_path: Path) -> None:
    """
    The synthetic snapshots place tight haloes amongst uniform interlopers, so FOF should recover the snapshot's own
    haloes exactly (up to relabelling) for every ptype. An interloper may legitimately land within a linking length of
    a halo and be attached, so those are only required to be rare.
    """
    snapshot_path = tmp_path / "snapshot.hdf5"
    if sim_type == "SIMBA":
        generate_simba_snapshot(path=snapshot_path)
    else:
        generate_swift_snapshot(path=snapshot_path, model="KIARA")

    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type=sim_type,
        snapshot_path=snapshot_path,
        output_dir=tmp_path,
        cores_per_rank=1,
        halo_id_source="FOF",
        photometry_table_path=None,
        min_dm_per_halo=10,
    )
    reader = build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)
    fof = build_halo_source(config=config, reader=reader).read_halo_ids(ptypes=reader.available_ptypes)
    snapshot = SnapshotHaloSource(reader=reader).read_halo_ids(ptypes=reader.available_ptypes)

    assert fof.n_field_haloes == snapshot.n_field_haloes
    for ptype in reader.available_ptypes:
        bound = snapshot.field_ids[ptype] >= 0
        assert_same_partition(fof.field_ids[ptype][bound], snapshot.field_ids[ptype][bound])
        assert np.sum(fof.field_ids[ptype][~bound] >= 0) <= 0.05 * np.sum(~bound)


@pytest.mark.parametrize("without_mpi", [False, True])
def test_pipeline_runs_with_fof_haloes(without_mpi: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if without_mpi:  # exercises the serial path taken when mpi4py is not installed
        monkeypatch.setattr("octavius.run_octavius.get_mpi_communicator", lambda: None)

    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)

    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type="SIMBA",
        snapshot_path=snapshot_path,
        output_dir=tmp_path,
        cores_per_rank=1,
        halo_id_source="FOF",
        photometry_table_path=None,
        stages={
            "find_galaxies": True,
            "properties_core": True,
            "properties_ptype_specific": True,
            "properties_local_environment": True,
            "photometry": False,
        },
        min_dm_per_halo=10,
        min_stars_per_galaxy=2,
        b=1.5,
        velocity_factor=5,
        compress_catalogue=False,
    )

    with h5py.File(analyse_snapshot(config=config), "r") as catalogue:
        assert len(catalogue["halo_data"]["properties/core/n_dm"]) == 3
        validate_halo_membership(f=catalogue)
        validate_group_counts(f=catalogue, group_data="halo_data")


def test_fof_source_only_attaches_listed_ptypes(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)

    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type="SIMBA",
        snapshot_path=snapshot_path,
        output_dir=tmp_path,
        cores_per_rank=1,
        halo_id_source="FOF",
        photometry_table_path=None,
        min_dm_per_halo=10,
        halo_attach_ptypes=["star"],
    )
    reader = build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)
    halo_ids = build_halo_source(config=config, reader=reader).read_halo_ids(ptypes=reader.available_ptypes).field_ids

    assert np.any(halo_ids["star"] >= 0)
    for ptype in ("gas", "bh"):
        assert len(halo_ids[ptype]) == reader.particle_counts[ptype]
        assert np.all(halo_ids[ptype] == -1)


def test_config_rejects_unknown_attach_ptypes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="halo_attach_ptypes"):
        OctaviusConfig.from_yaml(
            config_path=CONFIG_PATH,
            snapshot_path=tmp_path / "snapshot.hdf5",
            output_dir=tmp_path,
            photometry_table_path=None,
            halo_attach_ptypes=["gas", "dm"],
        )


def strip_halo_ids(snapshot_path: Path, reader_halo_id_name: str) -> None:
    """
    Removes the synthetic snapshot's own halo IDs, as for a snapshot written without them.
    """
    with h5py.File(snapshot_path, "r+") as f:
        for group in f.values():
            if isinstance(group, h5py.Group) and reader_halo_id_name in group:
                del group[reader_halo_id_name]


def fof_config(snapshot_path: Path, tmp_path: Path, sim_type: str, **overrides) -> OctaviusConfig:
    return OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type=sim_type,
        snapshot_path=snapshot_path,
        output_dir=tmp_path,
        cores_per_rank=1,
        photometry_table_path=None,
        **{"halo_id_source": "FOF", "min_dm_per_halo": 10, **overrides},
    )


def make_snapshot(sim_type: str, path: Path) -> None:
    if sim_type == "SIMBA":
        generate_simba_snapshot(path=path)
    else:
        generate_swift_snapshot(path=path, model="KIARA")


@pytest.mark.parametrize("sim_type", ["SIMBA", "SWIFT-KIARA"])
def test_written_halo_ids_read_back_as_snapshot_haloes(sim_type: str, tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.hdf5"
    make_snapshot(sim_type, snapshot_path)
    config = fof_config(snapshot_path, tmp_path, sim_type, write_halo_ids=True)
    reader = build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)
    strip_halo_ids(snapshot_path, reader.id_map["HaloID"])

    fof = build_halo_source(config=config, reader=reader).read_halo_ids(ptypes=reader.available_ptypes)
    slabs = {ptype: slice(0, reader.particle_counts[ptype]) for ptype in reader.available_ptypes}
    assert reader.write_halo_ids(
        field_ids=fof.field_ids, slabs=slabs, comm=None, attributes={"octavius_halo_finder": "FOF"}
    )

    snapshot = SnapshotHaloSource(reader=reader).read_halo_ids(ptypes=reader.available_ptypes)
    assert snapshot.n_field_haloes == fof.n_field_haloes
    for ptype in reader.available_ptypes:
        assert np.array_equal(snapshot.field_ids[ptype], fof.field_ids[ptype])

    # Octavius's own halo IDs may be replaced, e.g. by a rerun with another linking length
    assert reader.write_halo_ids(
        field_ids=fof.field_ids, slabs=slabs, comm=None, attributes={"octavius_halo_finder": "FOF"}
    )


def test_halo_ids_from_elsewhere_are_not_overwritten(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)
    config = fof_config(snapshot_path, tmp_path, "SIMBA", write_halo_ids=True)
    reader = build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)
    before = SnapshotHaloSource(reader=reader).read_halo_ids(ptypes=reader.available_ptypes).field_ids

    field_ids = {ptype: np.full(len(ids), -1, dtype=np.int64) for ptype, ids in before.items()}
    slabs = {ptype: slice(0, len(ids)) for ptype, ids in before.items()}
    assert not reader.write_halo_ids(
        field_ids=field_ids, slabs=slabs, comm=None, attributes={"octavius_halo_finder": "FOF"}
    )

    after = SnapshotHaloSource(reader=reader).read_halo_ids(ptypes=reader.available_ptypes).field_ids
    for ptype in before:
        assert np.array_equal(before[ptype], after[ptype])


@pytest.mark.parametrize("has_halo_ids", [True, False])
def test_snap_or_fof_picks_source(has_halo_ids: bool, tmp_path: Path) -> None:
    from octavius.external_halo_sources.fof import FOFHaloSource

    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_swift_snapshot(path=snapshot_path, model="KIARA")
    config = fof_config(snapshot_path, tmp_path, "SWIFT-KIARA", halo_id_source="SNAP_OR_FOF")
    reader = build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)
    if not has_halo_ids:
        strip_halo_ids(snapshot_path, reader.id_map["HaloID"])

    source = build_halo_source(config=config, reader=reader)

    assert type(source) is (SnapshotHaloSource if has_halo_ids else FOFHaloSource)


def test_write_halo_ids_needs_fof(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="write_halo_ids"):
        fof_config(tmp_path / "snapshot.hdf5", tmp_path, "SIMBA", halo_id_source="SNAPSHOT", write_halo_ids=True)


@pytest.mark.parametrize("without_mpi", [False, True])
def test_pipeline_reuses_written_halo_ids(without_mpi: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The intended workflow: the first SNAP_OR_FOF run finds FOF haloes and writes them to the snapshot, the second
    reads them from the snapshot and produces the same catalogue.
    """
    if without_mpi:
        monkeypatch.setattr("octavius.run_octavius.get_mpi_communicator", lambda: None)

    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)
    strip_halo_ids(snapshot_path, "HaloID")

    catalogues = []
    for run in ("first", "second"):
        output_dir = tmp_path / run
        output_dir.mkdir()
        config = fof_config(
            snapshot_path,
            output_dir,
            "SIMBA",
            halo_id_source="SNAP_OR_FOF",
            write_halo_ids=True,
            stages={
                "find_galaxies": True,
                "properties_core": True,
                "properties_ptype_specific": False,
                "properties_local_environment": False,
                "photometry": False,
            },
            min_stars_per_galaxy=2,
            b=1.5,
            velocity_factor=5,
            compress_catalogue=False,
        )
        with h5py.File(analyse_snapshot(config=config), "r") as catalogue:
            catalogues.append({k: catalogue["halo_data/properties/core"][k][...] for k in ("n_dm", "mass_total")})

        with h5py.File(snapshot_path, "r") as f:
            assert f["PartType1/HaloID"].attrs["octavius_halo_finder"] == "FOF"

    assert len(catalogues[0]["n_dm"]) == 3
    for key in catalogues[0]:
        assert np.array_equal(catalogues[0][key], catalogues[1][key])


@pytest.fixture
def octavius_warnings(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """
    The OCTAVIUS logger does not propagate, so attach caplog's handler to it directly.
    """
    logger = logging.getLogger("OCTAVIUS")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="OCTAVIUS")
    yield caplog
    logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(("halo_b", "warns"), [(0.2, False), (1.8, True)])  # 1.8 links the whole box into one halo
def test_warns_when_largest_halo_percolates(
    halo_b: float, warns: bool, tmp_path: Path, octavius_warnings: pytest.LogCaptureFixture
) -> None:
    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)
    config = fof_config(snapshot_path, tmp_path, "SIMBA", halo_b=halo_b)
    reader = build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)

    build_halo_source(config=config, reader=reader).read_halo_ids(ptypes=reader.available_ptypes)

    assert any("largest halo holds" in r.getMessage() for r in octavius_warnings.records) == warns


def test_min_dm_per_halo_to_store_validation(tmp_path: Path) -> None:
    config = fof_config(tmp_path / "snapshot.hdf5", tmp_path, "SIMBA")
    assert config.min_dm_per_halo_to_store == config.min_dm_per_halo

    with pytest.raises(ValueError, match="min_dm_per_halo_to_store"):
        fof_config(tmp_path / "snapshot.hdf5", tmp_path, "SIMBA", min_dm_per_halo_to_store=11)


def test_stored_small_haloes_allow_lower_threshold_later(tmp_path: Path) -> None:
    """
    Haloes down to min_dm_per_halo_to_store are written to the snapshot but left out of that run's catalogue; a later
    run reading them from the snapshot can then use a lower min_dm_per_halo.
    """
    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)
    strip_halo_ids(snapshot_path, "HaloID")

    # the background varies between synthetic snapshots, so plant a 2-particle group: move one isolated DM particle
    # next to another
    with h5py.File(snapshot_path, "r+") as f:
        pos = f["PartType1/Coordinates"]
        boxsize = f["Header"].attrs["BoxSize"]
        isolated = np.flatnonzero(find_fof_haloes(dm_pos=pos[...], boxsize=boxsize, b=0.6, min_members=2)["dm"] == -1)
        pos[isolated[1]] = pos[isolated[0]] + 1e-3

    n_haloes = []
    for run, min_dm in (("first", 10), ("second", 2)):
        output_dir = tmp_path / run
        output_dir.mkdir()
        config = fof_config(
            snapshot_path,
            output_dir,
            "SIMBA",
            halo_id_source="SNAP_OR_FOF",
            write_halo_ids=True,
            halo_b=0.6,
            min_dm_per_halo=min_dm,
            min_dm_per_halo_to_store=2,
            compress_catalogue=False,
            stages={
                "find_galaxies": False,
                "properties_core": True,
                "properties_ptype_specific": False,
                "properties_local_environment": False,
                "photometry": False,
            },
        )
        with h5py.File(analyse_snapshot(config=config), "r") as catalogue:
            n_haloes.append(len(catalogue["halo_data/properties/core/n_dm"]))

    with h5py.File(snapshot_path, "r") as f:
        assert f["PartType1/HaloID"].attrs["min_dm_per_halo"] == 2
        stored = np.bincount(f["PartType1/HaloID"][...])[1:]  # 1-indexed, 0 is no halo

    assert stored.min() == 2  # the planted group was stored
    assert n_haloes == [np.sum(stored >= 10), len(stored)]
