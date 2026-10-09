"""

Tests combined (multi-ptype) centres of mass, which set galaxy centres and so every galaxy radius and kinematic
quantity.

"""

from pathlib import Path

import h5py
import numpy as np

from octavius.aggregate_properties.properties_core import _combine_centre_of_mass
from octavius.data_management.conventions import OctaviusConfig
from octavius.run_octavius import analyse_snapshot
from octavius.utils.generate_snapshots import generate_simba_snapshot

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"
BOXSIZE = 100.0


class FakeStore(dict):
    """
    The parts of a GroupStore that _combine_centre_of_mass uses.
    """

    @property
    def n_groups(self) -> int:
        return len(self["mass_gas"])


def test_combined_centre_of_mass() -> None:
    """
    Group 0 sits mid-box; group 1 straddles the periodic boundary; group 2 has no gas (the first ptype); group 3 has no
    particles at all.
    """
    store = FakeStore(
        {
            "mass_gas": np.array([1.0, 3.0, 0.0, 0.0]),
            "mass_star": np.array([3.0, 1.0, 2.0, 0.0]),
            "_pos_gas": np.array([[40.0, 50.0, 50.0], [99.0, 50.0, 50.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "_pos_star": np.array([[44.0, 50.0, 50.0], [3.0, 50.0, 50.0], [70.0, 20.0, 10.0], [0.0, 0.0, 0.0]]),
            "_vel_gas": np.zeros((4, 3)),
            "_vel_star": np.zeros((4, 3)),
        }
    )

    com = _combine_centre_of_mass(
        group_store=store,
        combined_mass=store["mass_gas"] + store["mass_star"],
        collective_name="baryon",
        constituent_ptypes=["gas", "star"],
        boxsize=BOXSIZE,
    )["com_pos_baryon"]

    assert np.allclose(com[0], [43.0, 50.0, 50.0])
    assert np.allclose(com[1], [0.0, 50.0, 50.0])  # (3 * -1 + 1 * 3) / 4 = 0 from the boundary
    assert np.allclose(com[2], [70.0, 20.0, 10.0])
    assert np.all(np.isnan(com[3]))


def test_galaxy_centres_lie_within_galaxies(tmp_path: Path) -> None:
    """
    The synthetic haloes are tight clumps, so each galaxy's baryonic centre must be close to its halo's centre, and its
    stellar half-mass radius small.
    """
    snapshot_path = tmp_path / "snapshot.hdf5"
    generate_simba_snapshot(path=snapshot_path)
    config = OctaviusConfig.from_yaml(
        config_path=CONFIG_PATH,
        simulation_type="SIMBA",
        snapshot_path=snapshot_path,
        output_dir=tmp_path,
        cores_per_rank=1,
        halo_id_source="SNAPSHOT",
        photometry_table_path=None,
        stages={
            "find_galaxies": True,
            "properties_core": True,
            "properties_ptype_specific": False,
            "properties_local_environment": False,
            "photometry": False,
        },
        min_dm_per_halo=0,
        min_stars_per_galaxy=2,
        b=1.5,
        velocity_factor=5,
        compress_catalogue=False,
    )

    with h5py.File(analyse_snapshot(config=config), "r") as f:
        boxsize = f["header/boxsize"][()]
        galaxy_com = f["galaxy_data/properties/core/com_pos_baryon"][...]
        galaxy_halo = f["galaxy_data/membership/field_halo_index"][...]
        halo_centre = f["halo_data/properties/core/minpot_pos"][...]
        r_half = f["galaxy_data/properties/core/radius_half_mass_star"][...]
        halo_extent = f["halo_data/properties/core/radius_max_total"][...]

    offset = galaxy_com - halo_centre[galaxy_halo]
    offset -= boxsize * np.round(offset / boxsize)
    distance = np.sqrt(np.sum(offset**2, axis=1))

    assert len(distance) > 0
    assert np.all(distance < halo_extent[galaxy_halo])
    assert np.all(r_half < halo_extent[galaxy_halo])
