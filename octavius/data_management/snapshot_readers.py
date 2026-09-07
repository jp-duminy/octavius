"""

Relies on the ABC classes defined in snapshot_readers_base.py.

The reader building dictionary is at the bottom of the file.

h5py-backed, MPI-native raw snapshot readers. Parse raw simulation output into Octavius analysis, converting
snapshot-specific terminology into an agnostic interface for the data structures.

This relies on the abstract base class SnapshotReader. In practice, the format-specific differences require some
bespoke treatments here and there with overrides and such.

"""

# type checking
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .conventions import OctaviusConstants, OctaviusConfig

# default libraries
from pathlib import Path

# other packages
import h5py
import numpy as np

# internal imports
from .parallel_reading import redistribute_data, split_slab
from .physics import (
    TNGConstants,
    calculate_temperature,
    calculate_tng_x_neutral,
)
from .conventions import (
    DTYPES,
)
from .snapshot_readers_base import SnapshotReader, SwiftReader, GadgetReader
from ..log import get_logger

logger = get_logger()


def build_reader(snapshot_path: Path, constants: OctaviusConstants, config: OctaviusConfig) -> SnapshotReader:
    """
    Builds a SnapshotReader class depending on what was specified in the config. Returns:

    - SnapshotReader: a bespoke reader class with all generic methods.
    """
    sim_type = config.simulation_type.upper()  # autocapitalise for user convenience
    reader_class = READER_MAP.get(sim_type)

    if reader_class is None:
        raise ValueError(
            f"Unsupported simulation type: '{sim_type}', currently support {', '.join(sorted(READER_MAP))}"
        )

    logger.info(f"Using {sim_type} reader.")
    reader = reader_class(snapshot_path=snapshot_path, constants=constants, n_io_chunks=config.n_io_chunks)

    return reader


class KiaraReader(SwiftReader):
    """
    SWIFT-KIARA snapshot reader.
    """

    dataset_map = {
        **SwiftReader.dataset_map,
        "mass_HI": "AtomicHydrogenMasses",
        "mass_H2": "MolecularHydrogenMasses",
        "dust_mass": "DustMasses",
    }


class EagleReader(SwiftReader):  # NOTE: currently identical to Kiara, but maintained separately to be safe
    """
    SWIFT-EAGLE snapshot reader.
    """

    dataset_map = {
        **SwiftReader.dataset_map,
        "mass_HI": "AtomicHydrogenMasses",
        "mass_H2": "MolecularHydrogenMasses",
    }


class ColibreReader(SwiftReader):
    """
    SWIFT-COLIBRE snapshot reader.
    """

    dataset_map = {
        **SwiftReader.dataset_map,
        "dust_mass_fractions": "TotalDustMassFractions",
        "species_HI": "SpeciesFractions",
        "species_H2": "SpeciesFractions",
    }

    column_indices = {**SwiftReader.column_indices, "species_HI": 1, "species_H2": 7}

    def __init__(self, snapshot_path: Path, constants: OctaviusConstants, n_io_chunks: int) -> None:

        super().__init__(snapshot_path, constants, n_io_chunks)
        self.derived_columns: dict[str, Callable] = {
            "mass_HI": self._derive_mass_HI,
            "mass_H2": self._derive_mass_H2,
            "dust_mass": self._derive_dust_mass,
        }

    def _derive_dust_mass(self, ptype: str = "gas") -> np.ndarray:
        """
        Derives dust mass from fraction stored in snapshot.
        """
        total_mass = self._read_raw(ptype=ptype, dataset="mass")
        dust_fraction = self._read_raw(ptype=ptype, dataset="dust_mass_fractions")

        dust_mass = dust_fraction * total_mass

        return dust_mass

    def _derive_mass_HI(self, ptype: str = "gas") -> np.ndarray:
        """
        Derive HI fraction from species fractions and XH.
        """
        species_HI = self._read_raw(ptype=ptype, dataset="species_HI")
        gas_mass = self._read_raw(ptype=ptype, dataset="mass")

        # derive XH
        helium_fraction = self._read_raw(ptype=ptype, dataset="helium_fraction")
        metallicity = self._read_raw(ptype=ptype, dataset="metallicity")
        fHI = (1.0 - helium_fraction - metallicity) * species_HI
        np.clip(fHI, a_min=0.0, a_max=1.0, out=fHI)

        mass_H1 = fHI * gas_mass

        return mass_H1

    def _derive_mass_H2(self, ptype: str = "gas") -> np.ndarray:
        """
        Derive H2 fraction from species fractions and XH.
        """
        species_H2 = self._read_raw(ptype=ptype, dataset="species_H2")
        gas_mass = self._read_raw(ptype=ptype, dataset="mass")

        # derive XH
        helium_fraction = self._read_raw(ptype=ptype, dataset="helium_fraction")
        metallicity = self._read_raw(ptype=ptype, dataset="metallicity")

        fH2 = (1.0 - helium_fraction - metallicity) * 2.0 * species_H2  # diatomic
        np.clip(fH2, a_min=0.0, a_max=1.0, out=fH2)
        mass_H2 = fH2 * gas_mass

        return mass_H2


class SimbaReader(GadgetReader):
    """
    SIMBA (GIZMO) snapshot reader; assumes default units. Works on gadget framework from inherited
    conventions.
    """

    dataset_map = {
        **GadgetReader.dataset_map,
        "H2_fraction": "FractionH2",
        "sfr": "StarFormationRate",
        "age": "StellarFormationTime",  # NOTE: we compute age from formationtime, but using "age" is for reader agnosticity
        "metallicity": "Metallicity",
        "helium_fraction": "Metallicity",  # helium fraction is metallicity[:, 1] (metallicity is nx11 array)
        "dust_mass": "Dust_Masses",
        "smoothing_length": "SmoothingLength",
        "potential": "Potential",
    }

    id_map = {
        **GadgetReader.id_map,
        "HaloID": "HaloID",
    }

    column_indices = {
        "helium_fraction": 1,  # slice of 2D datasets
        "metallicity": 0,
    }

    def __init__(self, snapshot_path: Path, constants: OctaviusConstants, n_io_chunks: int) -> None:

        super().__init__(snapshot_path, constants, n_io_chunks)
        self.derived_columns["mass_HI"] = self._derive_mass_HI
        self.derived_columns["mass_H2"] = self._derive_mass_H2

    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
        """
        Reads snapshot-sourced HaloIDs. GIZMO uses 0 as the sentinel value; we map to Octavius's -1.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        halo_id_name = self.id_map["HaloID"]  # equivalent for SIMBA but best practice to use the dict

        if slab.start is None:  # SnapshotHaloSource calls this without slab arg
            slab = slice(0, self.particle_counts[ptype])

        slab_length = slab.stop - slab.start

        with h5py.File(self.snapshot_path, "r") as f:
            halo_hdf5_dataset = f[hdf5_group][halo_id_name]
            raw_halo_ids = np.empty(shape=slab_length, dtype=halo_hdf5_dataset.dtype)

            for chunk in split_slab(slab, self.n_io_chunks):
                offset = chunk.start - slab.start
                chunk_length = chunk.stop - chunk.start
                raw_halo_ids[offset : offset + chunk_length] = halo_hdf5_dataset[chunk]

            raw_halo_ids = raw_halo_ids.astype(
                DTYPES.get("HaloID", np.int64), copy=False
            )  # change dtype here otherwise you get int overflow

        raw_halo_ids -= 1  # shift IDs left to compensate with Octavius sentinel

        return raw_halo_ids

    def _derive_mass_HI(self, ptype: str = "gas") -> np.ndarray:
        """
        Converts the NeutralHydrogenAbundance (nHI/nH) to fHI (fraction of mass which is hydrogen)
        """
        neutral_fraction = self._read_raw(ptype=ptype, dataset="HI_abundance")
        helium_fraction = self._read_raw(ptype=ptype, dataset="helium_fraction")
        metallicity = self._read_raw(ptype=ptype, dataset="metallicity")
        gas_mass = self._read_raw(ptype=ptype, dataset="mass")

        fHI = (1.0 - helium_fraction - metallicity) * neutral_fraction
        np.clip(fHI, a_min=0.0, a_max=1.0, out=fHI)
        mass_HI = fHI * gas_mass

        return mass_HI

    def _derive_mass_H2(self, ptype: str = "gas") -> np.ndarray:
        """
        Converts the FractionH2 (mH2/mH) to H2 mass fraction of total particle mass.
        """
        molecular_fraction = self._read_raw(ptype=ptype, dataset="H2_fraction")
        helium_fraction = self._read_raw(ptype=ptype, dataset="helium_fraction")
        metallicity = self._read_raw(ptype=ptype, dataset="metallicity")
        gas_mass = self._read_raw(ptype=ptype, dataset="mass")

        fH2 = (1.0 - helium_fraction - metallicity) * molecular_fraction
        np.clip(fH2, a_min=0.0, a_max=1.0, out=fH2)
        mass_H2 = fH2 * gas_mass

        return mass_H2


class TNGReader(GadgetReader):
    """
    TNG snapshot reader; assumes default GADGET units.
    """

    dataset_map = {
        **GadgetReader.dataset_map,
        "sfr": "StarFormationRate",
        "age": "GFM_StellarFormationTime",
        "metallicity": "GFM_Metallicity",
        "helium_fraction": "GFM_Metals",
        "smoothing_length": "SubfindHsml",
        "potential": "Potential",
    }

    id_map = {
        **GadgetReader.id_map,
        "HaloID": "",
    }

    column_indices = {
        "helium_fraction": 1,  # slice of 2D datasets
    }

    def __init__(self, snapshot_path: Path, constants: OctaviusConstants, n_io_chunks: int) -> None:

        super().__init__(snapshot_path, constants, n_io_chunks)
        self.tng_constants = TNGConstants()
        self._hydrogen_cache: tuple[np.ndarray, np.ndarray] | None = None
        self.derived_columns["mass_H2"] = self._derive_mass_H2
        self.derived_columns["mass_HI"] = self._derive_mass_HI

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Overrides read_dataset() for DM masses.
        """
        if ptype == "dm" and dataset == "mass":
            return self._derive_dm_mass()

        return super().read_dataset(ptype, dataset)

    def has_dataset(self, ptype: str, dataset: str) -> bool:
        """
        Overrides has_dataset() for DM masses.
        """
        if ptype == "dm" and dataset == "mass":
            return True
        return super().has_dataset(ptype, dataset)

    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
        """
        Do not exist in TNG snapshots.
        """
        ptype, slab = ptype, slab
        raise ValueError("Snapshot halo IDs do not exist in TNG snapshots.")

    def read_requested_columns(
        self,
        ptype: str,
        datasets: list[str],
        sorted_snapshot_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Reimplemented to clear the hydrogen cache.
        """
        self.subset_indices = sorted_snapshot_indices
        result = {dataset: self.read_dataset(ptype, dataset) for dataset in datasets}
        self._hydrogen_cache = None  # clear the hydrogen cache

        return result

    def _compute_hydrogen_masses(self, ptype: str = "gas") -> tuple[np.ndarray, np.ndarray]:
        """
        Derives mass_HI and mass_H2 using Blitz & Rosolowsky (2006) & Stevens et al. (2019). Since
        this is rather expensive in requesting a whopping 8 datasets, we use a cache.
        """
        if self._hydrogen_cache is not None:
            return self._hydrogen_cache

        neutral_fraction = self._read_raw(ptype=ptype, dataset="HI_abundance")  # TODO: change name convention
        electron_abundance = self._read_raw(ptype=ptype, dataset="electron_abundance")
        rho = self._read_raw(ptype=ptype, dataset="rho")
        internal_energy = self._read_raw(ptype=ptype, dataset="internal_energy")
        sfr = self._read_raw(ptype=ptype, dataset="sfr")
        helium_fraction = self._read_raw(ptype=ptype, dataset="helium_fraction")
        metallicity = self._read_raw(ptype=ptype, dataset="metallicity")
        gas_mass = self._read_raw(ptype=ptype, dataset="mass")
        hydrogen_fraction = 1.0 - helium_fraction - metallicity

        x_neutral = calculate_tng_x_neutral(
            internal_energy=internal_energy,
            neutral_fraction=neutral_fraction,
            rho=rho,
            sfr=sfr,
            hydrogen_fraction=hydrogen_fraction,
            constants=self.constants,
            tng_constants=self.tng_constants,
        )

        temperature = calculate_temperature(
            internal_energy=internal_energy,
            electron_abundance=electron_abundance,
            helium_fraction=helium_fraction,
            constants=self.constants,
        )

        nH = rho * hydrogen_fraction / self.constants.PROTON_MASS_G
        n_total = nH * (1.0 + electron_abundance + helium_fraction / (4.0 * hydrogen_fraction))
        thermal_pressure = n_total * temperature
        R_mol = (
            thermal_pressure / self.tng_constants.BLITZ_P0
        ) ** self.tng_constants.BLITZ_ALPHA  # blitz P0 bakes in kB

        fHI = hydrogen_fraction * x_neutral / (1 + R_mol)
        mass_HI = fHI * gas_mass
        fH2 = hydrogen_fraction * x_neutral * (R_mol / (1 + R_mol))
        mass_H2 = fH2 * gas_mass

        self._hydrogen_cache = (mass_HI, mass_H2)
        return self._hydrogen_cache

    def _derive_mass_HI(self, ptype: str = "gas") -> np.ndarray:
        """
        Derives fHI using Blitz & Rosolowsky (2006) & Stevens et al. (2019)
        """
        return self._compute_hydrogen_masses(ptype=ptype)[0]

    def _derive_mass_H2(self, ptype: str = "gas") -> np.ndarray:
        """
        Derives fH2 using Blitz & Rosolowsky (2006) & Stevens (2019).
        """
        return self._compute_hydrogen_masses(ptype=ptype)[1]

    def _derive_dm_mass(self) -> np.ndarray:
        """
        Derives the DM mass from header mass table.
        """
        logger.debug("Reading 'dm' masses from header mass table.")
        with h5py.File(self.snapshot_path, "r") as f:
            dm_mass_raw = f["Header"].attrs["MassTable"][1]

        dm_mass = dm_mass_raw * self.unit_conversions["mass"]

        if self.subset_indices is not None:
            return np.full(shape=len(self.subset_indices), fill_value=dm_mass, dtype=np.float64)

        slab = self.slabs["dm"]
        slab_length = slab.stop - slab.start
        dm_masses = np.full(slab_length, dm_mass, dtype=np.float64)
        dm_masses = dm_masses[self.masks["dm"]]

        if self.maps is not None:
            dm_masses = redistribute_data(local_data=dm_masses, redistribution_map=self.maps["dm"], comm=self.comm)

        return dm_masses


READER_MAP: dict[str, type[SnapshotReader]] = {
    "SIMBA": SimbaReader,
    "TNG": TNGReader,
    "SWIFT-KIARA": KiaraReader,
    "SWIFT-EAGLE": EagleReader,
    "SWIFT-COLIBRE": ColibreReader,
}
