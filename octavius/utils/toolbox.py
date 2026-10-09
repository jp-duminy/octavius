"""

Convenience methods.

"""

# default libraries
from pathlib import Path
import subprocess
import shutil
from dataclasses import replace

# internal imports
from ..data_management import OctaviusConfig
from .generate_snapshots import generate_swift_snapshot
from ..log import get_logger

logger = get_logger()
config_path = Path(__file__).parent.parent.parent / "config.yaml"


def repack_catalogue(
    catalogue_path: Path,
    compression_level: int = 4,  #  https://en.wikipedia.org/wiki/Gzip fastest is 4
) -> None:
    """
    Calls h5repack with with subprocess to apply gzip compression to a catalogue. The compressed
    catalogue is named the same as the original.

    Parameters
    ----------
    catalogue_path : pathlib.Path
        Path to the catalogue you would like to compress.
    compression_level : int
        Desired gzip compression level (1-9)
    """
    logger.info(f"Compressing catalogue to gzip compression level {compression_level}.")
    h5repack_binary = shutil.which("h5repack")
    if h5repack_binary is None:
        raise FileNotFoundError("h5repack is not available on your system, to recompress files please install it.")

    temp_path = catalogue_path.with_name(f"{catalogue_path.stem}_packed.hdf5")

    # during testing, if cuillin was having a bad day it would stop compressing and leave a corrupted file, motivating this
    try:
        subprocess.run(
            ["h5repack", "-f", f"GZIP={compression_level}", str(catalogue_path), str(temp_path)],
            check=True,
        )
        temp_path.replace(catalogue_path)
    except Exception:  # if h5repack fails, delete the corrupted file
        temp_path.unlink(missing_ok=True)
        raise

    logger.info("Successfully compressed catalogue.")


def generate_test_catalogue(
    output_dir: Path,
) -> Path:
    """
    Generates a tiny synthetic Octavius catalogue for testing purposes. This catalogue is produced from a procedurally-generated snapshot filled with junk data; this function can therefore be used to assess the catalogue HDF5 file layout and whether the pipeline runs. It produces its own synthetic junk config file too.

    Parameters
    ----------
    output_dir: pathlib.Path
        Where you would like to output the test catalogue.

    Returns
    -------
    catalogue_path: pathlib.Path
        A path object pointing to the test catalogue.
    """
    from ..run_octavius import analyse_snapshot  # avoid circular import

    snapshot_path = output_dir / "test_snap.hdf5"

    generate_swift_snapshot(path=snapshot_path)

    config = OctaviusConfig.from_yaml(config_path=config_path)
    config = replace(
        config,
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        simulation_type="SWIFT-KIARA",
        halo_id_source="SNAPSHOT",
        min_dm_per_halo=0,
        min_stars_per_galaxy=2,
        min_gas_per_galaxy=0,
        b=1.5,
        velocity_factor=5,
        keep_logs=False,
        terminal_output_level="DEBUG",
    )

    catalogue_path = analyse_snapshot(config=config)
    snapshot_path.unlink()

    logger.info(f"Successfully created a synthetic catalogue at {catalogue_path}.")

    return catalogue_path
