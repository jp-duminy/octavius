"""

Tests snapshot reader quirks: SWIFT's negative star formation rates and per-reader derived columns.

"""

from pathlib import Path

import h5py
import numpy as np

from octavius.data_management import OctaviusConstants
from octavius.data_management.conventions import OctaviusConfig
from octavius.data_management.snapshot_readers import build_reader
from octavius.utils.generate_snapshots import generate_simba_snapshot, generate_swift_snapshot

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"


def make_reader(snapshot_path: Path, sim_type: str):
    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type=sim_type,
        snapshot_path=snapshot_path,
        output_dir=snapshot_path.parent,
        cores_per_rank=1,
        photometry_table_path=None,
    )
    return build_reader(snapshot_path=snapshot_path, constants=OctaviusConstants(), config=config)


def read_all(reader, ptype: str, dataset: str) -> np.ndarray:
    """
    Reads a whole dataset through the pipeline read path (one rank, every particle kept).
    """
    n = reader.particle_counts[ptype]
    reader.slabs, reader.masks, reader.maps, reader.comm = (
        {ptype: slice(0, n)},
        {ptype: np.ones(n, dtype=bool)},
        None,
        None,
    )
    return reader.read_dataset(ptype=ptype, dataset=dataset)


def test_swift_negative_sfrs_are_zeroed(tmp_path: Path) -> None:
    """
    SWIFT stores minus the scale factor of last star formation for gas which has stopped forming stars; these must read
    as zero SFR, while genuine SFRs are unchanged.
    """
    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_swift_snapshot(path=snapshot_path, model="KIARA")
    with h5py.File(snapshot_path, "r+") as f:
        raw = f["PartType0/StarFormationRates"]
        sfr = raw[...]
        sfr[::3] = -np.linspace(0.1, 0.9, len(sfr[::3]))  # scale factors of last star formation
        raw[...] = sfr

    reader = make_reader(snapshot_path, "SWIFT-KIARA")
    read = read_all(reader, "gas", "sfr")

    assert np.all(read[::3] == 0)
    positive = sfr > 0
    assert np.any(positive)
    assert np.allclose(read[positive] / sfr[positive], read[positive][0] / sfr[positive][0])  # only a unit change


def test_derived_columns_are_per_reader(tmp_path: Path) -> None:
    simba_path, swift_path = tmp_path / "simba.hdf5", tmp_path / "swift.hdf5"
    generate_simba_snapshot(path=simba_path)
    generate_swift_snapshot(path=swift_path, model="KIARA")

    simba = make_reader(simba_path, "SIMBA")
    swift = make_reader(swift_path, "SWIFT-KIARA")

    assert "sfr" in swift.derived_columns and "sfr" not in simba.derived_columns
    assert "mass_HI" in simba.derived_columns and "mass_HI" not in swift.derived_columns  # KIARA stores HI masses
