"""

Octavius internal data structures for the pipeline.


This is a modularised version of the old DataManager, its functionality divided amongst smaller, specialised
classes and dataclasses. These include:

- ParticleStore: dict-like container for particle-level data
- GroupStore: dict-like container for group-level data

Generally, you will aggregate quantities from the particle stores into a group store. The two are bound through the
membership columns, which can be accessed through get_particle_csr(ptype).

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_management import Internals
    from ..external_halo_sources import HaloAssignments, SubhaloInformation
    from .snapshot_readers import SnapshotReader

# default libraries
from dataclasses import dataclass

# others
import numpy as np

# internal imports
from .conventions import DTYPES, SimulationAttributes, OctaviusConstants
from .csr import (
    build_group_csr,
    propagate_membership_csr,
)
from ..log import get_logger

logger = get_logger()


class ParticleStore:
    """
    Stores dictionaries of properties for one particle type.
    """

    __slots__ = ("columns", "n_particles", "ptype", "is_baryonic")  # fixed slots

    def __init__(self, ptype: str, n_particles: int, is_baryonic: bool):

        self.ptype = ptype
        self.n_particles = n_particles
        self.is_baryonic = is_baryonic
        self.columns: dict[str, np.ndarray] = {}  # O(1) lookup on a lightweight np array (preconverted units)

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Use ParticleStore["key"] to access array.
        """
        return self.columns[key]

    def __setitem__(self, key: str, array: np.ndarray) -> None:
        """
        Use ParticleStore["key"] = array to add/modify an entry.
        """
        assert array.shape[0] == self.n_particles
        self.columns[key] = array

    def __contains__(self, key: str) -> bool:
        """
        Controls {"key" in ParticleStore} behaviour
        """
        return key in self.columns

    def __len__(self) -> int:
        """
        Allows you to use len() on the ParticleStore to find the length of its arrays, avoiding using something (I usually used mass) as a proxy.
        """
        return self.n_particles

    def release(self, *names: str) -> None:  # chose *names as an alternative to names: list[str] for readability
        """
        Call ParticleStore.release("key1", "key2") to delete references to no-longer needed columns (like drop from old datamanager).

        Internally, this is effectively the same as the del method.
        """
        for name in names:
            self.columns.pop(name, None)


def build_particle_stores(
    reader: SnapshotReader,
    internals: Internals,
    halo_assignments: HaloAssignments,
    process_ptypes: dict[str, bool],
) -> dict[str, ParticleStore]:
    """
    Constructs basic particle stores containing mass, position and velocity.
    """
    available = reader.available_ptypes
    requested = [pt for pt in available if process_ptypes.get(pt, True)]

    particles: dict[str, ParticleStore] = {}

    for ptype in requested:
        halo_ids = halo_assignments.field_ids[ptype]
        store = ParticleStore(ptype=ptype, n_particles=len(halo_ids), is_baryonic=ptype in internals.baryonic_ptypes)
        store["HaloID"] = halo_ids
        if halo_assignments.sub_ids is not None:
            store["SubhaloID"] = halo_assignments.sub_ids[ptype]

        particles[ptype] = store

    return particles


class GroupStore:
    """
    Effectively the same idea as the ParticleStore class, but storing group-level information.
    """

    def __init__(
        self, group_ids: np.ndarray, group_key: int, kind: str, original_ids: np.ndarray | None = None
    ):  # original_ids is for external halo readers

        self.group_ids = group_ids
        self.n_groups = len(group_ids)
        self.columns: dict[str, np.ndarray] = {}
        self.csr_membership: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.group_key = group_key
        self.kind = kind

        max_id = group_ids.max() if self.n_groups > 0 else 0
        if self.n_groups == 0:
            logger.debug(f"Empty GroupStore for {group_key}")

        self.id_to_idx = np.full(shape=max_id + 1, fill_value=-1, dtype=DTYPES["particle_id"])
        self.id_to_idx[group_ids] = np.arange(self.n_groups, dtype=DTYPES["csr_offsets"])

        if original_ids is not None:
            self.columns["original_id"] = original_ids

    @property
    def ptypes(self) -> list[str]:
        """
        List of ptypes which this group actually contains.
        """
        return list(self.csr_membership.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Use GroupStore["key"] to access array.
        """
        return self.columns[key]

    def __setitem__(self, key: str, array: np.ndarray) -> None:
        """
        Use GroupStore["key"] = array to add/modify an entry.
        """
        assert array.shape[0] == self.n_groups
        self.columns[key] = array

    def write_batch(self, results: dict[str, np.ndarray], suffix: str = "") -> None:
        """
        Writes the data from a results dictionary into the GroupStore.
        """
        for column_name, column_data in results.items():
            store_key = f"{column_name}_{suffix}" if suffix else column_name

            if store_key in self.columns:
                raise KeyError(f"Column '{store_key}' already exists in GroupStore.")

            self.columns[store_key] = column_data

    def __contains__(self, key: str) -> bool:
        """
        Controls {"key" in GroupStore} behaviour
        """
        return key in self.columns

    def get_indexer(self, group_id: np.ndarray) -> np.ndarray:
        """
        Returns the corresponding index array from the group ID array (vectorised).
        """
        id_to_idx = np.full(len(group_id), -1, dtype=DTYPES["csr_offsets"])
        valid = (group_id >= 0) & (group_id < len(self.id_to_idx))
        id_to_idx[valid] = self.id_to_idx[
            group_id[valid]
        ]  # mask valid indices (-1, the sentinel, is the last array element)

        return id_to_idx

    def get_particle_csr(self, ptype: str) -> tuple[np.ndarray, np.ndarray]:
        """
        For indexing particle arrays in the CSR format. Returns a tuple of:

        - offsets: CSR offsets
        - idx_sorted: indices into ParticleStore aligned to GroupStore order
        """
        return self.csr_membership[ptype]


def build_galaxy_store(
    particles: dict[str, ParticleStore],
    baryonic_ptypes: list[str],
    group_kind: str,
    galaxy_key: str = "GalID",  # NOTE: this doesn't need to be an argument but I prefer it for explicit purposes
) -> GroupStore:
    """
    Constructs the galaxy GroupStore.
    """
    ids = [particles[ptype][galaxy_key] for ptype in particles]
    unique_ids = np.unique(np.concatenate(ids))

    unique_ids = unique_ids[unique_ids != -1]

    store = GroupStore(group_ids=unique_ids, group_key=galaxy_key, kind=group_kind)

    for ptype in baryonic_ptypes:
        offsets, sorted_indices = build_group_csr(
            group_idx=store.get_indexer(group_id=particles[ptype][galaxy_key]), n_groups=store.n_groups
        )
        store.csr_membership[ptype] = (offsets, sorted_indices)

    return store


def build_halo_store(
    particles: dict[str, ParticleStore],
    halo_key: str = "HaloID",
    subhalo_key: str | None = None,
    group_kind: str = "halo",
    subhalo_info: SubhaloInformation | None = None,
    original_halo_ids: np.ndarray | None = None,
) -> GroupStore:
    """
    Constructs a halo GroupStore.
    """
    all_halo_ids = [particles[ptype][halo_key] for ptype in particles]
    unique_hids = np.unique(np.concatenate(all_halo_ids))
    unique_hids = unique_hids[unique_hids != -1]
    n_haloes = len(unique_hids)

    if subhalo_info is not None:
        n_subhaloes = len(subhalo_info.depth)

        field_to_row = np.full(shape=(unique_hids.max() + 1), fill_value=-1, dtype=np.int64)
        field_to_row[unique_hids] = np.arange(n_haloes)

        combined_ids = np.concatenate([unique_hids, subhalo_info.global_index], dtype=np.int64)
        store = GroupStore(group_ids=combined_ids, group_key=halo_key, kind=group_kind)

        for ptype in particles:
            halo_ids = particles[ptype][halo_key]
            sub_ids = particles[ptype][subhalo_key]
            group_idx = np.where(halo_ids == -1, -1, field_to_row[halo_ids])

            mask = sub_ids != -1
            group_idx[mask] = sub_ids[mask] + n_haloes
            offsets, sorted_indices = build_group_csr(group_idx=group_idx, n_groups=store.n_groups)
            store.csr_membership[ptype] = (offsets, sorted_indices)

        parent_rows = np.full(n_haloes + n_subhaloes, -1, dtype=np.int64)
        depth_1_mask = subhalo_info.depth == 1
        deeper_mask = subhalo_info.depth > 1
        parent_rows[n_haloes:][depth_1_mask] = field_to_row[subhalo_info.host_field_ids[depth_1_mask]]
        parent_rows[n_haloes:][deeper_mask] = subhalo_info.parent_index[deeper_mask] + n_haloes

        for ptype in particles:
            exclusive_offsets, exclusive_sorted = store.csr_membership[ptype]
            inclusive_offsets, inclusive_sorted = propagate_membership_csr(
                offsets=exclusive_offsets,
                sorted_indices=exclusive_sorted,
                parent_ids=parent_rows,
                n_groups=store.n_groups,
            )
            store.csr_membership[ptype] = (inclusive_offsets, inclusive_sorted)

        store["parent_halo_index"] = parent_rows
        store["depth"] = np.concatenate([np.zeros(n_haloes, dtype=np.int64), subhalo_info.depth])

    else:
        store = GroupStore(group_ids=unique_hids, group_key=halo_key, kind=group_kind)
        for ptype in particles:
            offsets, sorted_indices = build_group_csr(
                group_idx=store.get_indexer(group_id=particles[ptype][halo_key]), n_groups=store.n_groups
            )
            store.csr_membership[ptype] = (offsets, sorted_indices)

        store["parent_halo_index"] = np.full(
            n_haloes, -1, dtype=np.int64
        )  # NOTE: I know this is inefficient but otherwise tests break (catalogue inconsistency)
        store["depth"] = np.zeros(n_haloes, dtype=np.int64)

    field_originals = (
        original_halo_ids[unique_hids]
        if original_halo_ids is not None
        else np.full(shape=n_haloes, fill_value=-1, dtype=np.int64)
    )
    if subhalo_info is not None:
        sub_originals = (
            subhalo_info.original_sub_ids
            if subhalo_info.original_sub_ids is not None
            else np.full(shape=n_subhaloes, fill_value=-1, dtype=np.int64)
        )
        store["original_ids"] = np.concatenate([field_originals, sub_originals])
    else:
        store["original_ids"] = field_originals

    return store


@dataclass(slots=True)  # no frozen=True as this is inherently supposed to be mutable
class SimulationData:
    """
    Object containing simulation data ready for analysis:

    - hdf5 groups converted to np.ndarrays in code units with correct datatype
    - group IDs (at instantiation, HIDs)
    - Simulation-specific attributes (boxsize, cosmological parameters, etc).
    """

    simulation: SimulationAttributes
    constants: OctaviusConstants
    particles: dict[str, ParticleStore]
    groups: dict[str, GroupStore]

    @property
    def available_ptypes(self) -> list[str]:
        """
        Returns a list of available particle types (total), in Octavius-internal convention.
        """
        return list(self.particles.keys())

    @property
    def available_baryonic_ptypes(self) -> list[str]:
        """
        Returns a list of available particle types (baryonic), in Octavius-internal convention.
        """
        return [pt for pt, store in self.particles.items() if store.is_baryonic]
