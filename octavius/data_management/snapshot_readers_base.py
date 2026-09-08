"""

File for the SnapshotReader base classes. These should only ever be inherited, not instantiated.

The base classes define the machinery for how data is loaded, so this is where you will find the
implementations of the MPI methods defined in parallel_reads.py; this allows the supported simulation formats
to be easily extended by inheriting a reader and having all the MPI trickery handled for you.

"""

# type checking
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .conventions import OctaviusConstants
    from .parallel_reading import RedistributionMap
    from mpi4py.MPI import Comm

# default libraries
from pathlib import Path
from abc import ABC, abstractmethod
from functools import cached_property

# other packages
import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
import astropy.units as u

# internal imports
from .parallel_reading import redistribute_data, split_slab, generate_read_plan
from .physics import (
    derive_stellar_age,
    calculate_temperature,
    derive_simulation_attributes,
)
from .conventions import (
    DTYPES,
    CODE_UNITS,
    SimulationAttributes,
    gadget_unit_conversion_factor,
)

from ..log import get_logger

logger = get_logger()


class SnapshotReader(ABC):
    """
    Abstract base class for snapshot readers; must be inherited by any reader class.
    """

    ptype_map: dict[str, str] = NotImplemented  # for ptype names
    dataset_map: dict[str, str] = NotImplemented  # for ptype datasets
    dataset_map_overrides: dict[tuple[str, str], str] = (
        NotImplemented  # where common datasets have different ptype-specific names
    )
    column_indices: dict[str, int] = NotImplemented  # 2D arrays which need indexing
    id_map: dict[str, str] = NotImplemented  # for halo IDs and particle IDs
    particle_counts: dict[str, int] = NotImplemented
    derived_columns: dict[str, Callable] = {}

    def __init__(self, snapshot_path: Path, constants: OctaviusConstants, n_io_chunks: int) -> None:

        self.snapshot_path = snapshot_path
        self.constants = constants
        self.n_io_chunks = n_io_chunks  # set from config
        self.global_indices: dict[str, np.ndarray] | None = None  # instantiate to None so serial path works
        self.maps: dict[str, np.ndarray] | None = None
        self.subset_indices: np.ndarray | None = None
        self.simulation_attributes: SimulationAttributes | None = None

        self.inverse_ptype_map = {v: k for k, v in self.ptype_map.items()}
        self.read_header()  # should set SimulationAttributes & particle_counts on self

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Reads a particle dataset from the HDF5 file, returning it as an ndarray converted into
        internal code units and appropriate dtype, masked to the rank's allocation.
        """
        if not self.has_dataset(ptype, dataset):
            raise KeyError(f"{dataset} either not available or not found for {ptype}.")

        logger.debug(f"Loading '{dataset}' for {ptype}.")

        if dataset in self.derived_columns:
            return self.derived_columns[dataset](ptype)  # passes it to a bespoke function

        return self._read_raw(ptype, dataset)

    def read_requested_columns(
        self,
        ptype: str,
        datasets: list[str],
        sorted_snapshot_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Reads requested datasets for a stage.
        """
        self.subset_indices = sorted_snapshot_indices
        result = {dataset: self.read_dataset(ptype, dataset) for dataset in datasets}

        return result

    def _read_raw(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Reads a HDF5 dataset from the file, handling dtypes, unit conversions, and masks.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map_overrides.get((ptype, dataset), self.dataset_map[dataset])

        # serial mini-slice reads for analyser
        if self.subset_indices is not None:
            read_plan = generate_read_plan(idx_sorted=self.subset_indices)

            with h5py.File(self.snapshot_path, "r") as f:
                hdf5_dataset = f[hdf5_group][hdf5_name]
                conversion_factor = self._unit_conversion_factor(dataset=dataset, hdf5_dataset=hdf5_dataset)
                n_particles = len(self.subset_indices)

                # allocate output
                if dataset in self.column_indices:
                    values = np.empty(n_particles, dtype=hdf5_dataset.dtype)
                else:
                    full_shape = (n_particles,) + hdf5_dataset.shape[1:]
                    values = np.empty(full_shape, dtype=hdf5_dataset.dtype)

                # iterate over the slices and read them into the output
                position = 0
                for read_slice, mask in read_plan:
                    if dataset in self.column_indices:  # 2D columns
                        chunk = hdf5_dataset[read_slice, self.column_indices[dataset]]
                    else:
                        chunk = hdf5_dataset[read_slice]

                    if mask is not None:
                        chunk = chunk[mask]

                    values[position : position + len(chunk)] = chunk
                    position += len(chunk)  # analogous to offsets array

            # close file, convert units and return (no MPI)
            output = self._convert_units(dataset=dataset, values=values, conversion_factor=conversion_factor)

            return output

        # pipeline read path (optional MPI, chunked/slab reads with redistribution)
        slab = self.slabs[ptype]
        slab_length = slab.stop - slab.start

        with h5py.File(self.snapshot_path, "r") as f:
            hdf5_dataset = f[hdf5_group][hdf5_name]
            conversion_factor = self._unit_conversion_factor(dataset=dataset, hdf5_dataset=hdf5_dataset)

            if dataset in self.column_indices:  # for metallicity columns
                col_idx = self.column_indices[dataset]
                raw_array = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk, col_idx]

            else:  # read flat for all others
                full_shape = (slab_length,) + hdf5_dataset.shape[
                    1:
                ]  # this returns () if flat and tuple addition handles it (shape needed for 3d columns)
                raw_array = np.empty(full_shape, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    raw_array[offset : offset + chunk_length] = hdf5_dataset[chunk]

            filtered_array = raw_array[self.masks[ptype]]

        # close file, convert units, return
        output = self._convert_units(dataset=dataset, values=filtered_array, conversion_factor=conversion_factor)

        if self.maps is not None:
            result = redistribute_data(local_data=output, redistribution_map=self.maps[ptype], comm=self.comm)
        else:
            result = output

        return result

    def set_maps(
        self,
        slabs: dict[str, slice],
        masks: dict[str, np.ndarray],
        maps: dict[str, RedistributionMap],
        comm: Comm | None,
    ) -> None:
        """
        Sets the per-rank slabs (which part of the dataset it reads); the masks (for the haloes which belong to it);
        the maps (to know where other haloes belong); and the COMM_WORLD object (for MPI)
        """
        self.slabs = slabs
        self.masks = masks
        self.maps = maps
        self.comm = comm
        self.global_indices: dict[str, np.ndarray] = {}

        assert slabs.keys() == masks.keys()

        for ptype in sorted(
            masks
        ):  # must be sorted so ranks iterate in same order, otherwise rank desync crashes can occur
            slab = slabs[ptype]
            global_indices = np.arange(slab.start, slab.stop, dtype=np.int64)[masks[ptype]]
            self.global_indices[ptype] = redistribute_data(
                local_data=global_indices, redistribution_map=maps[ptype], comm=comm
            )

    def has_dataset(self, ptype: str, dataset: str) -> bool:
        """
        Checks whether a dataset exists in the snapshot.
        """
        if dataset in self.derived_columns:  # trying to support this would be tricky, it will raise an error anyway
            return True

        hdf5_group = self.inverse_ptype_map[ptype]
        hdf5_name = self.dataset_map_overrides.get((ptype, dataset), self.dataset_map.get(dataset))

        logger.debug(f"Checking for raw snapshot dataset '{hdf5_name}' ('{dataset}') ")

        if hdf5_name is None:  # if not defined in the dataset maps
            logger.warning(f"{dataset} is not defined to the reader class.")
            return False

        with h5py.File(self.snapshot_path, "r") as f:
            return hdf5_name in f.get(hdf5_group, {})

    @cached_property
    def available_ptypes(self) -> list[str]:
        """
        Finds which ptypes are available in the raw snapshot (using internal names)
        """
        with h5py.File(self.snapshot_path) as f:
            return [pt for raw, pt in self.ptype_map.items() if raw in f and len(f[raw]) > 0]

    def read_particle_ids(self, ptype: str) -> np.ndarray:
        """
        Reads the snapshot particle IDs for the specified ptype. Returns an array of IDs.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        pid_name = self.id_map["particle_id"]

        with h5py.File(self.snapshot_path, "r") as f:
            hdf5_dataset = f[hdf5_group][pid_name]

            if self.maps is None:
                total_length = hdf5_dataset.shape[0]
                particle_ids = np.empty(total_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slice(0, total_length), self.n_io_chunks):
                    offset = chunk.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                result = particle_ids

            else:
                slab = self.slabs[ptype]
                slab_length = slab.stop - slab.start
                particle_ids = np.empty(slab_length, dtype=hdf5_dataset.dtype)

                for chunk in split_slab(slab, self.n_io_chunks):
                    offset = chunk.start - slab.start
                    chunk_length = chunk.stop - chunk.start
                    particle_ids[offset : offset + chunk_length] = hdf5_dataset[chunk]

                filtered_particle_ids = particle_ids[self.masks[ptype]]
                result = redistribute_data(
                    local_data=filtered_particle_ids, redistribution_map=self.maps[ptype], comm=self.comm
                )

        return result.astype(DTYPES.get("particle_id", np.int64))

    def _convert_units(self, dataset: str, values: np.ndarray, conversion_factor: float) -> np.ndarray:
        """
        Converts units and dtype from raw snapshot to Octavius internal.
        """
        values = values.astype(DTYPES.get(dataset, np.float64), copy=False)  # cast before conversion for precision

        if dataset == "age":  # age needs cosmology calls so has a unique branch
            return derive_stellar_age(
                formation_time=values,
                time_gyr=self.simulation_attributes.time_gyr,
                cosmology=self.simulation_attributes.cosmology,
            )

        if conversion_factor != 1.0:
            values *= conversion_factor

        return values

    @abstractmethod
    def read_header(self) -> SimulationAttributes:
        """
        Read header attributes and, where necessary, convert units; must create a SimulationAttributes dataclass.
        """
        ...

    @abstractmethod
    def _unit_conversion_factor(self, dataset: str, hdf5_dataset: h5py.Dataset) -> float:
        """
        Snapshot-specific unit conversion factors.
        """
        ...

    @abstractmethod
    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
        """
        Reads snapshot-assigned HaloIDs and maps them to a continuous 0-indexed array with a sentinel value of -1.
        """
        ...


class SwiftReader(SnapshotReader):
    """
    Base SWIFT snapshot reader; should not be directly instantiated, only inherited.
    """

    ptype_map: dict[str, str] = {  # invariant for SWIFT snaps
        "PartType0": "gas",
        "PartType1": "dm",
        "PartType4": "star",
        "PartType5": "bh",
    }

    dataset_map: dict[str, str] = {  # SWIFT-invariant names (subset)
        "pos": "Coordinates",
        "vel": "Velocities",
        "mass": "Masses",
        "potential": "Potentials",
        "internal_energy": "InternalEnergies",
        "rho": "Densities",
        "sfr": "StarFormationRates",
        "age": "BirthScaleFactors",
        "metallicity": "MetalMassFractions",
        "temperature": "Temperatures",
        "helium_fraction": "ElementMassFractions",
        "bhmass": "SubgridMasses",
        "bhmdot": "AccretionRates",
        "smoothing_length": "SmoothingLengths",
    }

    dataset_map_overrides: dict[tuple[str, str], str] = {  # global overrides (dynamical vs subgrid bh mass)
        ("bh", "mass"): "DynamicalMasses",
    }

    id_map: dict[str, str] = {  # also invariant for SWIFT
        "particle_id": "ParticleIDs",
        "HaloID": "FOFGroupIDs",
    }

    column_indices = {
        "helium_fraction": 1,  # slice of 2D datasets
    }

    def read_header(self) -> SimulationAttributes:
        """
        Parses header attributes into a dataclass (does derived quantities too). SWIFT can do simulations with evolving dark energy, so we use flat w0wa cdm cosmology, which reduces to flat lambda cdm when wa = 0; SWIFT also splits the header into cosmology and header fields.
        """
        with h5py.File(self.snapshot_path, "r") as f:
            cosmo = f["Cosmology"].attrs
            header = f["Header"].attrs

            boxsize_vec = header["BoxSize"]  # usually stored as (x, y, z) in Mpc comoving
            boxsize_raw = boxsize_vec[0]
            assert np.allclose(boxsize_vec, boxsize_raw), "Octavius does not presently support non-cubic boxes."

            unit_length_cgs = f["Units"].attrs["Unit length in cgs (U_L)"].item()
            boxsize_cgs = boxsize_raw * unit_length_cgs
            boxsize_kpc = boxsize_cgs / u.kpc.to(u.cm)

            n_star = header["NumPart_Total"][4]
            n_gas = header["NumPart_Total"][0]

            self.n_particles_total = header["NumPart_Total"]
            self.particle_counts = {
                ptype_name: int(self.n_particles_total[int(hdf5_key[-1])])
                for hdf5_key, ptype_name in self.ptype_map.items()
                if ptype_name in self.available_ptypes
            }

            h = cosmo["h"].item()
            a = cosmo["Scale-factor"].item()
            w_0 = cosmo["w_0"].item()
            w_a = cosmo["w_a"].item()
            T_cmb_0 = cosmo["T_CMB_0 [K]"].item()  # I am not sure if this is strictly necessary but I put it in anyway
            redshift = cosmo["Redshift"].item()
            omega_matter = cosmo["Omega_m"].item()
            omega_lambda = cosmo["Omega_lambda"].item()

        flat_w0wa_cdm = Flatw0waCDM(
            H0=100 * h, Om0=omega_matter, Tcmb0=T_cmb_0, w0=w_0, wa=w_a
        )  # reduces to lambdacdm if wa=0

        self.simulation_attributes = derive_simulation_attributes(
            cosmology=flat_w0wa_cdm,
            h=h,
            a=a,
            w_0=w_0,
            w_a=w_a,
            redshift=redshift,
            omega_matter=omega_matter,
            omega_lambda=omega_lambda,
            boxsize=boxsize_kpc,
            n_star=n_star,
            n_gas=n_gas,
            constants=self.constants,
        )

        return self.simulation_attributes

    def _unit_conversion_factor(self, dataset: str, hdf5_dataset: h5py.Dataset):
        """
        SWIFT units, parsed from dataset attrs.
        """
        a_exp = hdf5_dataset.attrs["a-scale exponent"]
        h_exp = hdf5_dataset.attrs["h-scale exponent"]
        cgs_factor = hdf5_dataset.attrs["Conversion factor to CGS (not including cosmological corrections)"]

        target_units = CODE_UNITS[dataset]
        target_cgs_units = (1.0 * target_units.unit).cgs.value
        a_correction = self.simulation_attributes.scale_factor ** (a_exp - target_units.a_exponent)
        h_correction = self.simulation_attributes.h**h_exp  # code units do not carry h

        unit_factor = (a_correction * h_correction) * (cgs_factor / target_cgs_units)

        return unit_factor

    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
        """
        Reads (placeholder) FOFGroupIDs as HaloIDs if the external doesn't exist. SWIFT sentinel value is the uint32 max.
        """
        hdf5_group = self.inverse_ptype_map[ptype]
        halo_id_name = self.id_map["HaloID"]

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

            raw_halo_ids = raw_halo_ids.astype(DTYPES.get("HaloID", np.int64), copy=False)

        sentinel_mask = raw_halo_ids == 2147483647  # uint32 max value (as for why they do this? I have no idea)
        raw_halo_ids -= 1  # SWIFT is also 1-indexed
        raw_halo_ids[sentinel_mask] = -1

        return raw_halo_ids


class GadgetReader(SnapshotReader):
    """
    Parses GADGET-style snapshots. Should only be inherited, never directly instantiated.
    """

    ptype_map = {
        "PartType0": "gas",
        "PartType1": "dm",
        "PartType4": "star",
        "PartType5": "bh",
    }
    dataset_map = {
        "pos": "Coordinates",
        "vel": "Velocities",
        "mass": "Masses",
        "HI_abundance": "NeutralHydrogenAbundance",
        "internal_energy": "InternalEnergy",
        "electron_abundance": "ElectronAbundance",
        "rho": "Density",
        "bhmass": "BH_Mass",
        "bhmdot": "BH_Mdot",
    }

    id_map = {
        "particle_id": "ParticleIDs",
    }

    dataset_map_overrides: dict[tuple[str, str], str] = {}

    def __init__(self, snapshot_path: Path, constants: OctaviusConstants, n_io_chunks: int) -> None:

        super().__init__(snapshot_path, constants, n_io_chunks)
        self.unit_conversions = {
            dataset: gadget_unit_conversion_factor(
                dataset, self.simulation_attributes.h, self.simulation_attributes.scale_factor
            )
            for dataset in self.dataset_map
            if dataset in DTYPES
        }
        self.derived_columns: dict[str, Callable] = {
            "temperature": self._derive_temperature,
        }

    def read_header(self) -> SimulationAttributes:
        """
        Parses header attributes into a dataclass (does derived quantities too); assumes FlatLambdaCDM
        """
        with h5py.File(self.snapshot_path, "r") as f:
            header = f["Header"].attrs

            h = header["HubbleParam"]
            boxsize = header["BoxSize"] / h
            omega_matter = header["Omega0"]
            omega_lambda = header["OmegaLambda"]
            a = header["Time"]
            redshift = header["Redshift"]
            n_star, n_gas = header["NumPart_Total"][4], header["NumPart_Total"][0]

            self.n_particles_total = header["NumPart_Total"]
            self.particle_counts = {
                ptype_name: int(self.n_particles_total[int(hdf5_key[-1])])
                for hdf5_key, ptype_name in self.ptype_map.items()
                if ptype_name in self.available_ptypes
            }

        flat_lambda_cdm = FlatLambdaCDM(H0=100 * h, Om0=omega_matter)  # always flatlambdacdm for GADGET

        self.simulation_attributes = derive_simulation_attributes(
            cosmology=flat_lambda_cdm,
            h=h,
            a=a,
            w_0=-1,  # always flatlambdacdm
            w_a=0,
            redshift=redshift,
            omega_matter=omega_matter,
            omega_lambda=omega_lambda,
            boxsize=boxsize,
            n_star=n_star,
            n_gas=n_gas,
            constants=self.constants,
        )

        return self.simulation_attributes

    def _unit_conversion_factor(self, dataset: str, hdf5_dataset: h5py.Dataset) -> float:
        """
        Calls unit conversion factors from Gizmo documentation.
        """
        hdf5_dataset = hdf5_dataset  # unused
        conversion_factor = self.unit_conversions.get(dataset, 1.0)

        return conversion_factor

    def _derive_temperature(self, ptype: str = "gas") -> np.ndarray:
        """
        Reads data to calculate temperature according to method described
        in http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html
        """
        assert ptype == "gas", f"Temperature is configured to be computed from gas, not {ptype}."

        internal_energy = self._read_raw(ptype, "internal_energy")

        try:
            electron_abundance = self._read_raw(ptype, "electron_abundance")
        except KeyError:
            electron_abundance = np.ones(shape=len(internal_energy))

        helium_fraction = self._read_raw(ptype, "helium_fraction")

        temperature = calculate_temperature(
            internal_energy=internal_energy,
            electron_abundance=electron_abundance,
            helium_fraction=helium_fraction,
            constants=self.constants,
        )

        return temperature
