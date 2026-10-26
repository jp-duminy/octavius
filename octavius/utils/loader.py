"""

Catalogue loader for interfacing with Octavius catalogues in a user-friendly, object-oriented manner.

NOTE: for devs, please give user-facing APIs the classic numpy docstring style.

"""

# default libraries
from pathlib import Path

# other packages
import h5py
import numpy as np
import astropy.units as u

# internal imports
from ..version import CATALOGUE_VERSION


def load_catalogue(
    catalogue_path: Path | str,
) -> OctaviusCatalogue:
    """
    Load an Octavius catalogue, returning an OctaviusCatalogue object which contains an
    object-oriented interface for working with Octavius catalogues.

    Parameters
    ----------
    catalogue_path: pathlib.Path or str
        pathlib.Path object pointing to an Octavius catalogue.

    Returns
    -------
    catalogue: OctaviusCatalogue
        The loaded catalogue interface.
    """
    catalogue_path = Path(catalogue_path)  # coerce to Path object

    with h5py.File(catalogue_path, "r") as f:
        file_version = f["metadata"].attrs.get("catalogue_format_version", 0)
        if file_version > CATALOGUE_VERSION:
            raise ValueError(
                f"Catalogue version {file_version} is new and not supported by the version of Octavius currently installed, which can read up to version {CATALOGUE_VERSION}; please update."
            )

    return OctaviusCatalogue(catalogue_path=catalogue_path)


class OctaviusCatalogue:
    """
    An object-oriented interface for an Octavius catalogue.

    Attributes
    ----------

    galaxies: GroupCollection
        Object-oriented interface for accessing galaxy data.
    haloes: GroupCollection
        Object-oriented interface for accessing halo data.
    sim_info: SimInfo
        Object-oriented interface for accessing simulation cosmology/header info.

    simulation_type: str
        The type of simulation which the catalogue was generated from.
    halo_id_source: str
        Which external source the HaloIDs were derived from.
    octavius_version: str
        The version of Octavius which generated the catalogue.
    timestamp: str
        When (UTC) the catalogue was generated.
    redshift: float
        Convenience accessor to the redshift of the simulation snapshot (also in sim_info).
    scale_factor: float
        Convenience accessor to the scale factor of the simulation snapshot (also in sim_info).
    boxsize_comoving: float
        Convenience accessor to the boxsize (ckpc) of the simulation (also in sim_info).
    """

    def __init__(self, catalogue_path: Path) -> None:

        self._file: h5py.File = h5py.File(catalogue_path, "r")
        self._cat: Path = catalogue_path

        self._read_header()

        self.galaxies: GroupCollection | None = (
            GroupCollection(self._file["galaxy_data"], parent=self, scale_factor=self.scale_factor)
            if "galaxy_data" in self._file
            else None
        )
        self.haloes: GroupCollection | None = (
            GroupCollection(self._file["halo_data"], parent=self, scale_factor=self.scale_factor)
            if "halo_data" in self._file
            else None
        )

        self.n_haloes = len(self.haloes) if self.haloes else 0
        self.n_galaxies = len(self.galaxies) if self.galaxies else 0

    def _read_header(self) -> None:
        """
        Stores metadata and the more prominent cosmological attributes on the catalogue
        class (those which users may want to access directly), and initialises the object-oriented
        header cosmology reader on the class for easy access.
        """
        # simulation attributes
        header = self._file["header"]
        self.redshift: float = float(header["redshift"][()])  # common
        self.scale_factor: float = float(header["scale_factor"][()])
        self.boxsize_comoving: float = float(header["boxsize"][()])

        self.sim_info = SimInfo(header=header)

        # metadata
        metadata = self._file["metadata"]
        self.simulation_type: str = metadata.attrs["simulation_type"]
        self.octavius_version: str = metadata.attrs["octavius_version"]
        self.timestamp: str = metadata.attrs["timestamp"]
        self.halo_id_source: str = metadata["config_parameters"].attrs["halo_id_source"]

    def close(self) -> None:
        """
        Manually closes the catalogue HDF5 file.
        """
        self._file.close()

    def __enter__(self) -> OctaviusCatalogue:
        """
        Controls behaviour of "with load(catalogue.hdf5) as cat:" (opens the catalogue).
        """
        return self

    def __exit__(self, *args) -> None:  # python itself passes 3 arguments to the exit method so just ignore them
        """
        Controls behaviour of "with load(catalogue.hdf5) as cat:" (closes the catalogue).
        """
        self.close()

    def __repr__(self) -> str:
        """
        Controls what you see when you print a catalogue interface.
        """
        return f'OctaviusCatalogue "{self._cat.name}" | z = {self.redshift:.3f} | {self.halo_id_source} | boxsize = {self.boxsize_comoving:.2f} kpc | {self.n_haloes} haloes | {self.n_galaxies} galaxies'


class GroupCollection:
    """
    Object-oriented interface for accessing group-level data from the catalogue.

    Methods
    -------

    describe()
        Print available property datasets and membership columns.
    get_dataset(name, mask, to_units, to_physical, verbose)
        Read a single property dataset with optional unit/physical conversion.
    get_datasets(names, mask, to_units, to_physical, verbose)
        Wrapper around get_dataset for reading multiple property datasets.
    get_units(name)
        Get the code units of a dataset as stored in the catalogue.
    is_comoving(name)
        Returns whether a dataset in the catalogue is stored in comoving coordinates.
    get_attributes(name)
        Get the HDF5 attributes (unit, a_exp, description) of a dataset.
    get_membership(name, group_index, verbose)
        Read a single membership dataset with the option of retrieving it for a single group.
    get_galaxies(halo_index)
        Retrieve positional indices into the galaxy data of the galaxies belonging to a halo (halo collections only).
    get_particle_indices(ptype, group_index)
        Retrieve positional indices into the raw simulation snapshot for the particles of a group.
    keys()
        Lists available dataset names.
    """

    def __init__(self, data: h5py.Group, parent: OctaviusCatalogue, scale_factor: float) -> None:

        self._data = data
        self._parent = parent
        self._dataset_paths = _build_dataset_lookup(group=data["properties"])
        self._scale_factor = scale_factor

        if "GalID" in data:  # cleaner way of doing this would be welcome
            self._n_groups = data["GalID"].shape[0]
            self._group_type = "galaxy"
        elif "HaloID" in data:
            self._n_groups = data["HaloID"].shape[0]
            self._group_type = "halo"
        else:
            raise ValueError("No ID column found; check whether catalogue is corrupted?")

    def __repr__(self) -> str:
        """
        Controls behaviour of print(GroupCollection)
        """
        return f"<GroupCollection of type {self._group_type} | {self._n_groups} entries>"

    def __len__(self) -> int:
        """
        Controls behaviour of len(GroupCollection)
        """
        return self._n_groups

    def keys(self) -> list:
        """
        Returns
        -------

        keys: a list of the dataset keys by which the catalogue group collection methods can be keyed.
        """
        return list(self._dataset_paths.keys())

    def describe(self) -> None:
        """
        Prints all datasets available in the catalogue group collection, to save you having to inspect the HDF5 file.
        """
        print(f"Available Datasets ({len(self._dataset_paths)} available):\n")
        for short_name, path in sorted(self._dataset_paths.items()):
            dataset = self._data[f"properties/{path}"]
            unit = dataset.attrs["unit"]
            a_exp = dataset.attrs["a_exp"]
            desc = dataset.attrs["description"]
            print(f"  {short_name:<30s} | [{unit}] | [a_exp: {a_exp}] | {desc}")

        membership = self._data["membership"]
        print(f"\nMembership columns ({len(membership)} datasets)\n")
        for name in sorted(membership.keys()):
            dataset = membership[name]
            desc = dataset.attrs["description"]
            print(f"  {name} | {desc}")

    def get_dataset(
        self,
        name: str,
        mask: np.ndarray | None = None,
        to_units: str | None = None,
        to_physical: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Retrieves a dataset from the properties section of the catalogue and applies any
        requested conversions.

        Parameters
        ----------
        name: str
            The catalogue name of the dataset.
        mask: ndarray, optional
            Boolean mask to apply to the dataset.
        to_units: str, optional
            Target units to convert the dataset to (must be aligned with the ordering of 'names').
        to_physical: bool, optional
            Apply conversion to physical (default: raw, which may be comoving).
        verbose: bool, optional
            Prints the catalogue description of the dataset.

        Returns
        -------
        data: ndarray
            Array with requested conversions applied.
        """
        dataset = self._data[f"properties/{self._dataset_paths[name]}"]
        data = dataset[:]  # read in full dataset (not a problem for catalogue-level quantities)

        if mask is not None:
            data = data[mask]

        if to_physical:
            a_exp = dataset.attrs["a_exp"]
            data *= self._scale_factor**a_exp

        if to_units is not None:
            conversion_factor = u.Unit(dataset.attrs["unit"]).to(u.Unit(to_units))
            data *= conversion_factor

        if verbose:
            print(f"{name}: {dataset.attrs['description']} | dtype: {data.dtype} | {len(data)} entries.")

            if to_units is not None:
                print(f"Applied conversion factor of {conversion_factor:.3e} ({dataset.attrs['unit']} -> {to_units}).")

        return data

    def get_units(self, name: str) -> str:
        """
        Returns the code units of a dataset in the catalogue.

        Parameters
        ----------
        name: str
            The catalogue name of the dataset.

        Returns
        -------
        unit: str
            The units, as a string.
        """
        dataset = self._data[f"properties/{self._dataset_paths[name]}"]
        return dataset.attrs["unit"]

    def is_comoving(self, name: str) -> bool:
        """
        Returns ``True`` if a dataset is stored in comoving coordinates.

        Parameters
        ----------
        name: str
            The catalogue name of the dataset.

        Returns
        -------
        comoving: bool
            Whether the dataset is comoving (has a scale factor exponent baked in).
        """
        dataset = self._data[f"properties/{self._dataset_paths[name]}"]
        return dataset.attrs["a_exp"] != 0

    def get_attributes(self, name: str) -> dict:
        """
        Returns the HDF5 attributes (unit, a_exp, description) of a dataset in the catalogue.

        Parameters
        ----------
        name: str
            The catalogue name of the dataset.

        Returns
        -------
        attributes: dict
            The dataset attributes.
        """
        dataset = self._data[f"properties/{self._dataset_paths[name]}"]
        return dict(dataset.attrs)

    def get_datasets(
        self,
        names: list[str],
        mask: np.ndarray | None = None,
        to_units: list[str] | None = None,
        to_physical: bool = False,
        verbose: bool = False,
    ) -> list[np.ndarray]:
        """
        Retrieves multiple datasets from the properties section of the catalogue and
        applies any requested conversions.

        Parameters
        ----------
        names: list[str]
            The catalogue names of the datasets.
        mask: ndarray, optional
            Boolean mask to apply to the dataset.
        to_units: list[str], optional
            List of target units to convert each dataset to (must be aligned with the ordering of 'names').
        to_physical: bool, optional
            Apply conversion to physical (default: raw, which may be comoving).
        verbose: bool, optional
            Prints the catalogue description of the each dataset.

        Returns
        -------
        data_list: list[np.ndarray]
            List of ndarrays with requested conversions applied in the order specified by names.
        """
        if to_units is not None and len(to_units) != len(names):
            raise ValueError(
                f"Received {len(to_units)} unit inputs and {len(names)} name inputs; if specifying units, please specify them for all datasets and ensure the units list is aligned to the names list."
            )

        data_list = []

        for i, name in enumerate(names):
            data = self.get_dataset(
                name=name,
                mask=mask,
                to_units=to_units[i] if to_units else None,
                to_physical=to_physical,
                verbose=verbose,
            )

            data_list.append(data)

        return data_list

    def get_membership(
        self,
        name: str,
        group_index: int | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Retrieves a membership dataset from the membership section of the catalogue; can optionally
        retrieve members for the specified group_index. Membership datasets are not necessarily (n_groups)
        length, and therefore a mask argument is not provided to avoid unintended behaviour; it is recommended to
        instead load the dataset using this method and then mask appropriately.

        Parameters
        ----------
        name: str
            The name of the membership dataset.
        group_index: int, optional
            Index for a specific group (if you want the membership of a specific group).
        verbose: bool, optional
            Optionally prints the description of the membership dataset.

        Returns
        -------
        data: ndarray
            The array corresponding to the requested membership dataset.
        """
        dataset = self._data[f"membership/{name}"]

        if group_index is not None:
            data = dataset[group_index]
        else:
            data = dataset[:]

        if verbose:
            print(f"{name}: {dataset.attrs['description']} | dtype: {data.dtype}")

        return data

    def get_galaxies(
        self,
        halo_index: int,
    ) -> np.ndarray:
        """
        Retrieves the indices into the catalogue ``galaxy_data`` corresponding to the galaxies which belong to the
        halo at halo_index; will raise an error if used on the galaxies collection.

        Parameters
        ----------
        halo_index: int
            The HaloID (positional index into halo_data) of the halo for which members must be retrieved.

        Returns
        -------
        galaxies: np.ndarray
            An array of indices into galaxy_data to retrieve the galaxies' properties.
        """
        if self._group_type == "galaxy":
            raise ValueError("get_galaxies() is only valid on the halo collection.")

        if halo_index < 0 or halo_index >= self._n_groups:
            raise IndexError(f"HaloID {halo_index} is out of bound: {self._n_groups} haloes exist.")

        offsets = self._data["membership/galaxy_offsets"]
        indices = self._data["membership/galaxy_indices"]

        return indices[offsets[halo_index] : offsets[halo_index + 1]]

    def get_subhaloes(
        self,
        halo_index: int,
    ) -> np.ndarray:
        """
        Retrieves the indices into the catalogue ``halo_data`` corresponding to all subhaloes which belong to the
        halo at halo_index (entire hierarchy); will raise an error if used on the galaxies collection.

        Parameters
        ----------
        halo_index: int
            The HaloID (positional index into halo_data) of the halo for which members must be retrieved.

        Returns
        -------
        subhaloes: np.ndarray
            An array of indices into halo_data to retrieve the subhaloes' properties.
        """
        if self._group_type == "galaxy":
            raise ValueError("get_subhaloes() is only valid on the halo collection.")

        if halo_index < 0 or halo_index >= self._n_groups:
            raise IndexError(f"HaloID {halo_index} is out of bound: {self._n_groups} haloes exist.")

        offsets = self._data["membership/subhalo_offsets"]
        indices = self._data["membership/subhalo_indices"]

        return indices[offsets[halo_index] : offsets[halo_index + 1]]

    def get_particle_indices(
        self,
        ptype: str,
        group_index: int,
    ) -> np.ndarray:
        """
        Retrieves positional indices into the raw snapshot corresponding to the particles in the requested group.

        Parameters
        ----------

        ptype: str
            The particle type to retrieve snapshot indices for. Keyed by Octavius ptype names: star, gas, dm, bh
        group_index: int
            The GroupID (positional index) to retrieve particles for.

        Returns
        -------

        indices: np.ndarray
            Positional indices into the raw snapshot for the particles of type 'ptype' belonging to the group at group_index.
        """
        valid_ptypes = ["star", "gas", "bh", "dm"]

        if ptype not in valid_ptypes:
            raise ValueError(f"{ptype} is not in the catalogue ptypes: {valid_ptypes}")

        if group_index < 0 or group_index >= self._n_groups:
            raise IndexError(f"Index {group_index} is out of bound: {self._n_groups} groups exist.")

        offsets = self._data[f"membership/{ptype}_offsets"]
        indices = self._data[f"membership/{ptype}_indices"]

        return indices[offsets[group_index] : offsets[group_index + 1]]


class SimInfo:
    """
    Object-oriented interface for simulation snapshot information (mainly cosmology).

    Methods
    -------

    __call__(name, to_units, physical, verbose)
        Return the float value of a parameter with optional unit/physical conversion (usage: SimInfo("parameter", ...))
    keys()
        Lists the available parameter names.
    unit(name)
        Retrieve the unit string of a parameter.
    description(name)
        Retrieve the description of a parameter.
    a_exp(name)
        Retrieve the scale factor exponent of a parameter.
    """

    def __init__(self, header: h5py.Group) -> None:

        self._header = header
        self._scale_factor: float = float(header["scale_factor"][()])

    def __call__(self, name: str, to_units: str | None = None, physical: bool = False, verbose: bool = False) -> float:
        """
        Returns a float corresponding to the parameter at "name".

        Parameters
        ----------
        name: str
            The name of the parameter to access.
        to_units: str
            The desired units to convert to.
        physical: bool, optional
            Whether to convert to physical (by default, returns raw value which can be comoving).
        verbose: bool, optional
            Prints the description of the parameter being accessed.

        Returns
        -------
        value: float
            The value of the cosmological parameter with the requested conversions applied.
        """
        dataset = self._header[name]
        value: float = float(dataset[()])

        if physical:  # order of operations
            value *= self._scale_factor ** dataset.attrs["a_exp"]

        if to_units:
            conversion_factor: float = u.Unit(dataset.attrs["unit"]).to(u.Unit(to_units))
            value *= conversion_factor

        if verbose:
            desc: str = dataset.attrs["description"]
            print(f"{name}: {desc}")

        return value

    def keys(self) -> list[str]:
        """
        Lists available cosmological quantities.
        """
        return list(self._header.keys())

    def unit(self, name: str) -> str:
        """
        Returns the units of the parameter at "name" as a string.
        """
        return self._header[name].attrs["unit"]

    def description(self, name: str) -> str:
        """
        Returns the description of the parameter at "name" as a string.
        """
        return self._header[name].attrs["description"]

    def a_exp(self, name: str) -> int:
        """
        Returns the scale factor exponent of the parameter at "name" as an integer.
        """
        return int(self._header[name].attrs["a_exp"])


def _build_dataset_lookup(group: h5py.Group) -> dict[str, str]:
    """
    Builds a lookup so users can just key property names without worrying about the hdf5 file layout (properties/core, properties/ptype_specific, etc.). Returns:

    - dataset_paths: a dict mapping the short catalogue name to the HDF5 group-prefixed name.
    """
    dataset_paths: dict[str, str] = {}

    for name in group:
        subgroup = group[name]
        items = [(f"{name}/{k}", subgroup[k]) for k in subgroup]

        for path, obj in items:
            if not isinstance(obj, h5py.Dataset):
                continue

            short_name = path.rsplit("/", 1)[
                -1
            ]  # users expect to be able to key the dataset directly so remove group prefixes

            dataset_paths[short_name] = path

    return dataset_paths
