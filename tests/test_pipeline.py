"""

Tests whether the pipeline produces the expected top-level fields.

"""

import h5py
from astropy import units as u

from octavius.data_management import load_internals
from octavius.data_management.conventions import OctaviusConfig

from .conftest import INTERNALS_PATH, CONFIG_PATH, PHOTOMETRY_TABLE_PATH


def test_pipeline(mock_catalogue: h5py.File) -> None:
    """
    Verifies the pipeline produces output.
    """
    assert "halo_data" in mock_catalogue
    assert "membership" in mock_catalogue["halo_data"]
    assert "properties" in mock_catalogue["halo_data"]
    assert "core" in mock_catalogue["halo_data/properties"]
    assert "particle_specific" in mock_catalogue["halo_data/properties"]

    assert "galaxy_data" in mock_catalogue
    assert "membership" in mock_catalogue["galaxy_data"]
    assert "properties" in mock_catalogue["galaxy_data"]
    assert "core" in mock_catalogue["galaxy_data/properties"]
    assert "particle_specific" in mock_catalogue["galaxy_data/properties"]
    assert "environment" in mock_catalogue["galaxy_data/properties"]


def test_output_units() -> None:
    """
    Tests whether the columns stored in internals.yaml have units which actually parse.
    """
    config = OctaviusConfig.from_yaml(
        CONFIG_PATH,
        photometry_table_path=PHOTOMETRY_TABLE_PATH,  # omitting photometry table post-init will crash
        bands=["v"],
    )
    internals = load_internals(INTERNALS_PATH, config)

    for metadata in internals.header_fields.values():
        u.Unit(metadata.unit)

    for metadata in internals.output_columns.values():
        u.Unit(metadata.unit)

    for group in internals.membership_columns.values():
        for metadata in group.values():
            u.Unit(metadata.unit)
