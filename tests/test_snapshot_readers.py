"""

Tests snapshot reader quirks: SWIFT's negative star formation rates and per-reader derived columns.

"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from octavius.data_management import OctaviusConstants
from octavius.data_management.conventions import OctaviusConfig
from octavius.data_management.snapshot_readers import build_reader
from octavius.utils.generate_snapshots import generate_simba_snapshot, generate_swift_snapshot, generate_tng_snapshot

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"


def make_reader(snapshot_path: Path, sim_type: str):
    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        halo_id_source="SUBFIND" if sim_type == "TNG" else "SNAPSHOT",
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


def test_element_fractions(tmp_path: Path) -> None:
    """
    Oxygen is column 4 of the element mass fractions in SIMBA, SWIFT and TNG (8 in SIMBA-C); check fraction_O reads it.

    AI-generated: claude-opus-5-5
    """
    simba_path, swift_path, tng_path = tmp_path / "simba.hdf5", tmp_path / "swift.hdf5", tmp_path / "tng.hdf5"
    generate_simba_snapshot(path=simba_path)
    generate_swift_snapshot(path=swift_path, model="KIARA")
    generate_tng_snapshot(path=tng_path, subfind_path=tmp_path / "subfind.hdf5")

    for snapshot_path, sim_type, element_dataset in [
        (simba_path, "SIMBA", "Metallicity"),
        (swift_path, "SWIFT-KIARA", "ElementMassFractions"),
        (tng_path, "TNG", "GFM_Metals"),
    ]:
        reader = make_reader(snapshot_path, sim_type)
        assert "O" in reader.available_elements

        with h5py.File(snapshot_path, "r") as f:
            raw_oxygen = f["PartType0"][element_dataset][:, 4]

        np.testing.assert_allclose(read_all(reader, "gas", "fraction_O"), raw_oxygen)

    # SIMBA-C stores the Chem5 elements, with oxygen in column 8
    simba_c = make_reader(simba_path, "SIMBA-C")
    with h5py.File(simba_path, "r") as f:
        raw_oxygen = f["PartType0/Metallicity"][:, 8]

    np.testing.assert_allclose(read_all(simba_c, "gas", "fraction_O"), raw_oxygen)


def test_missing_elements(tmp_path: Path) -> None:
    """
    Requesting an element the snapshot does not have should raise an error listing the elements it does have.

    AI-generated: claude-opus-5-5
    """
    simba_path, swift_path = tmp_path / "simba.hdf5", tmp_path / "swift.hdf5"
    generate_simba_snapshot(path=simba_path)
    generate_swift_snapshot(path=swift_path, model="KIARA")

    # zinc is not among the 11 SWIFT elements
    swift = make_reader(swift_path, "SWIFT-KIARA")
    with pytest.raises(KeyError, match="Zn is not among the elements.*Elements available: H, He, C"):
        read_all(swift, "gas", "fraction_Zn")

    # the mock snapshot has 11 Metallicity columns but SIMBA-C expects the full Chem5 list, so iron is missing
    simba_c = make_reader(simba_path, "SIMBA-C")
    assert simba_c.available_elements[-1] == "Ne"
    with pytest.raises(KeyError, match="Fe is not among the elements"):
        read_all(simba_c, "gas", "fraction_Fe")
