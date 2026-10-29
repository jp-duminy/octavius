"""

Tests whether the catalogue output is self-consistent and physical. This is to catch corrupted/obviously incorrect fields; to check whether the numbers are reasonable and behave as expected, please refer to validation/validation_suite.py which has methods for inspecting the fields and contrasting with a reference catalogue.

"""

from pathlib import Path
import numpy as np
import h5py

from tests.validation.output_validation import (
    check_for_nans,
    validate_galaxy_mapping,
    validate_subhalo_mapping,
    validate_galaxy_membership,
    validate_group_counts,
    validate_halo_membership,
    validate_mass_budget,
)
from octavius import load_catalogue, build_analyser, OctaviusConfig

CONFIG_PATH = Path(__file__).parent.parent / "octavius" / "config.yaml"


def test_nans(mock_catalogue: h5py.File) -> None:
    """
    Verifies NaN fields specified in validation/validation_columns.py
    """
    check_for_nans(f=mock_catalogue)


def test_galaxy_mapping(mock_catalogue: h5py.File) -> None:
    """
    Verifies galaxy-halo mapping is self-consistent.
    """
    validate_galaxy_mapping(f=mock_catalogue)


def test_galaxy_membership(mock_catalogue: h5py.File) -> None:
    """
    Verifies galaxy-particle mapping is self-consistent.
    """
    validate_galaxy_membership(f=mock_catalogue)


def test_galaxy_data(mock_catalogue: h5py.File) -> None:
    """
    Tests galaxy_data is self consistent in counts and mass budgets.
    """
    validate_group_counts(f=mock_catalogue, group_data="galaxy_data")


def test_halo_mapping(mock_catalogue: h5py.File) -> None:
    """
    Verifies halo-halo mapping is self-consistent.
    """
    validate_subhalo_mapping(f=mock_catalogue)


def test_halo_membership(mock_catalogue: h5py.File) -> None:
    """
    Verifies halo-particle mapping is self-consistent.
    """
    validate_halo_membership(f=mock_catalogue)


def test_halo_data(mock_catalogue: h5py.File) -> None:
    """
    Tests halo_data is self consistent in counts and mass budgets.
    """
    validate_group_counts(f=mock_catalogue, group_data="halo_data")


def test_masses(mock_catalogue: h5py.File) -> None:
    """
    Verifies individual & combined halo/galaxy mass budgets.
    """
    validate_mass_budget(f=mock_catalogue)


def test_loader(mock_catalogue: h5py.File) -> None:
    """
    Tests whether the loader and OctaviusCatalogue work.
    """
    cat = load_catalogue(Path(mock_catalogue.filename))
    assert len(cat.haloes) > 0
    assert len(cat.haloes.keys()) > 0

    cat.haloes.describe()

    is_comoving = cat.haloes.is_comoving("mass_star")
    assert isinstance(is_comoving, np.bool_)

    units = cat.haloes.get_units("mass_star")
    assert isinstance(units, str)

    mass = cat.haloes.get_dataset("mass_star", to_units="Msun", verbose=True)
    assert len(mass) == cat.n_haloes

    mass_star, n_star = cat.haloes.get_datasets(["mass_star", "n_star"], to_physical=True)
    assert isinstance(mass_star, np.ndarray)
    assert isinstance(n_star, np.ndarray)

    particle_indices = cat.haloes.get_particle_indices("dm", group_index=0)
    assert len(particle_indices) > 0

    galaxies = cat.haloes.get_galaxies(halo_index=0)
    assert isinstance(galaxies, np.ndarray)

    subhaloes = cat.haloes.get_subhaloes(halo_index=0)
    assert isinstance(subhaloes, np.ndarray)

    field_indices = cat.haloes.get_membership("field_halo_index", group_index=0)
    assert isinstance(field_indices, np.int_)

    boxsize = cat.sim_info("boxsize")
    assert isinstance(boxsize, float)

    cat.close()


def test_get_particle_data(mock_catalogue: h5py.File) -> None:
    """
    Tests whether the analyser returns the right particle values for groups in the catalogue.

    AI-generated: claude-opus-5-5
    """
    cat = load_catalogue(Path(mock_catalogue.filename))
    analyser = build_analyser(catalogue=cat)  # uses the config stored in the catalogue

    # one galaxy: compare against reading the whole snapshot dataset and indexing it ourselves
    star_mass = analyser.get_particle_data(name="mass", ptype="star", group_indices=0)
    all_star_mass = analyser._reader.read_full_dataset(ptype="star", dataset="mass")
    snapshot_indices = cat.galaxies.get_particle_indices(ptype="star", group_index=0)
    np.testing.assert_allclose(star_mass, all_star_mass[snapshot_indices])

    # the summed particle masses should match the catalogue stellar mass
    catalogue_star_mass = cat.galaxies.get_dataset("mass_star", to_units="Msun", to_physical=True)
    star_mass_kg = analyser.get_particle_data(name="mass", ptype="star", group_indices=0, to_units="kg")
    np.testing.assert_allclose(star_mass_kg.sum(), catalogue_star_mass[0] * 1.98841e30, rtol=1e-4)

    # several groups, including a repeated one (groups can share particles)
    values, offsets = analyser.get_particle_data(name="mass", ptype="star", group_indices=[0, 0])
    np.testing.assert_array_equal(offsets, [0, len(star_mass), 2 * len(star_mass)])
    np.testing.assert_allclose(values[offsets[1] : offsets[2]], star_mass)

    cat.close()


def test_config_from_catalogue(mock_catalogue: h5py.File) -> None:
    """
    Tests whether the config stored in the catalogue is rebuilt as it was given to the pipeline (see conftest.py).

    AI-generated: claude-opus-5-5
    """
    config = OctaviusConfig.from_catalogue(mock_catalogue.filename)
    assert config.simulation_type == mock_catalogue["metadata"].attrs["simulation_type"]
    assert config.min_stars_per_galaxy == 2
    assert config.b == 1.5
    assert config.subhalo_override is True
    assert config.compress_catalogue is False
    assert config.stages["photometry"] is True
    assert isinstance(config.virial_factors, list)
    assert isinstance(config.snapshot_path, Path)

    # a photometry table which cannot be found turns off photometry
    config = OctaviusConfig.from_catalogue(mock_catalogue.filename, photometry_table_path="/does/not/exist.hdf5")
    assert config.stages["photometry"] is False
