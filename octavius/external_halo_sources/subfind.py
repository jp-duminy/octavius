"""

For parsing SUBFIND assignments from catalogues and the corresponding reordering of groups in the snapshot,
which applies to TNG (at the moment).

"""

# type checking (semantic)
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ..data_management import SnapshotReader
    from mpi4py.MPI import Comm

# default libraries
from pathlib import Path
from functools import cached_property

# other packages
import h5py
import numpy as np
from numba import njit

# internal imports
from .halo_data_structures import HaloSource, HaloAssignments, SubhaloInformation, distribute_ids, compute_depths
from ..log import get_logger

logger = get_logger()


class SubfindCatalogue(NamedTuple):
    """
    Container for SUBFIND catalogue fields.
    """

    group_len_ptype: np.ndarray  # (n, 6), one for each ptype (includes extra ptypes)
    group_first_sub: np.ndarray  # index into subhalo table of central subhalo in the fof group
    group_n_subs: np.ndarray  # number of subhaloes in the fof group
    sub_len_ptype: np.ndarray  # (n, 6), number of particles in the subhalo for each type
    sub_gr_nr: np.ndarray  # index into subhalo table of FOF host of the subhalo
    sub_parent: np.ndarray  # local index into subhalo table of subhalo parent


class SubfindHaloSource(HaloSource):
    """
    Parser for SUBFIND halo assignments.
    """

    def __init__(self, catalogue_path: Path, reader: SnapshotReader) -> None:

        super().__init__(reader=reader)
        self.catalogue_path = catalogue_path
        # need to key particle types in the subfind catalogue
        self.ptype_column_map: dict[str, int] = {
            name: int(group.replace("PartType", "")) for group, name in self.reader.ptype_map.items()
        }

    @cached_property
    def _catalogue(self) -> SubfindCatalogue:
        """
        Cache the SUBFIND catalogue arrays on the object.
        """
        with h5py.File(self.catalogue_path, "r") as f:
            group = f["Group"]
            group_len_ptype = group["GroupLenType"][:].astype(np.int64)
            group_first_sub = group["GroupFirstSub"][:].astype(np.int64)
            group_n_subs = group["GroupNsubs"][:].astype(np.int64)

            subhalo = f["Subhalo"]
            sub_len_ptype = subhalo["SubhaloLenType"][:].astype(np.int64)
            sub_gr_nr = subhalo["SubhaloGrNr"][:].astype(np.int64)
            sub_parent = subhalo["SubhaloParent"][:].astype(np.int64)

        catalogue = SubfindCatalogue(
            group_len_ptype=group_len_ptype,
            group_first_sub=group_first_sub,
            group_n_subs=group_n_subs,
            sub_len_ptype=sub_len_ptype,
            sub_gr_nr=sub_gr_nr,
            sub_parent=sub_parent,
        )

        return catalogue

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Reads particles their halo IDs, assigning them based on the indexing of the SUBFIND catalogue.
        """
        field_ids: dict[str, np.ndarray] = {}
        sub_ids: dict[str, np.ndarray] = {}

        # SUBFIND catalogue treats all as subhaloes, so need the sub mask
        global_parents = local_to_global_parent(
            sub_parent=self._catalogue.sub_parent,
            group_first_sub=self._catalogue.group_first_sub,
            sub_gr_number=self._catalogue.sub_gr_nr,
        )
        depths = compute_depths(parent_ids=global_parents)

        # map from global index to index after sub masking
        sub_mask = depths >= 1
        sub_lookup = np.full(len(depths), fill_value=-1, dtype=np.int64)
        sub_lookup[sub_mask] = np.arange(sub_mask.sum(), dtype=np.int64)

        n_field_haloes = len(self._catalogue.group_first_sub)
        original_field_ids = np.arange(n_field_haloes, dtype=np.int64)

        for ptype in ptypes:
            ptype_int = self.ptype_column_map[ptype]
            group_counts = self._catalogue.group_len_ptype[:, ptype_int]  # number of particles in each group for ptype
            sub_counts = self._catalogue.sub_len_ptype[:, ptype_int]  # number of subhaloes in each ptype
            n_particles = self.reader.particle_counts[ptype]

            ptype_field_ids, raw_sub_ids = reconstruct_membership(
                group_counts=group_counts,
                sub_counts=sub_counts,
                group_first_sub=self._catalogue.group_first_sub,
                group_n_subs=self._catalogue.group_n_subs,
                n_particles=n_particles,
            )

            valid = raw_sub_ids >= 0
            remapped_sub_ids = np.full_like(raw_sub_ids, -1)
            remapped_sub_ids[valid] = sub_lookup[raw_sub_ids[valid]]

            field_ids[ptype] = ptype_field_ids
            sub_ids[ptype] = remapped_sub_ids

        n_subhaloes = np.sum(depths >= 1)
        logger.info(f"SUBFIND: {n_field_haloes} field haloes, {n_subhaloes} subhaloes.")

        for ptype, ids in field_ids.items():
            n_assigned = np.sum(ids != -1)
            logger.info(f"  {ptype}: {n_assigned:,} / {len(ids):,} particles assigned to haloes.")

        assignments = HaloAssignments(
            field_ids=field_ids,
            sub_ids=sub_ids,
            n_field_haloes=n_field_haloes,
            original_field_ids=original_field_ids,
        )

        return assignments

    def read_subhalo_info(self) -> SubhaloInformation:
        """
        Read subhalo properties (also dependent on snapshot indexing) from SUBFIND catalogue.
        """
        n_bound = np.sum(self._catalogue.sub_len_ptype, axis=1)  # avoid ptype axis
        global_parents = local_to_global_parent(
            sub_parent=self._catalogue.sub_parent,
            group_first_sub=self._catalogue.group_first_sub,
            sub_gr_number=self._catalogue.sub_gr_nr,
        )
        depths = compute_depths(parent_ids=global_parents)
        sub_mask = depths >= 1  # from compute_depths, depth 0 is a field

        # build lookup for only subhaloes
        sub_lookup = np.full(len(depths), fill_value=-1, dtype=np.int64)
        sub_lookup[sub_mask] = np.arange(sub_mask.sum(), dtype=np.int64)

        parent_index = np.where(
            global_parents[sub_mask] < 0,
            -1,
            sub_lookup[global_parents[sub_mask]],
        )

        n_subhaloes = np.sum(sub_mask)
        original_sub_ids = np.arange(len(depths), dtype=np.int64)[sub_mask]

        subhalo_info = SubhaloInformation(
            host_field_ids=self._catalogue.sub_gr_nr[sub_mask],
            parent_index=parent_index,
            depth=depths[sub_mask],
            n_bound=n_bound[sub_mask],
            global_index=np.arange(n_subhaloes, dtype=np.int64),
            original_sub_ids=original_sub_ids,
        )

        return subhalo_info

    def distribute_field_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Wrapper around distribute_ids() for field halo IDs.
        """
        return distribute_ids(
            slabs=slabs,
            particle_counts=self.reader.particle_counts,
            ptypes=sorted(self.reader.available_ptypes),
            comm=comm,
            global_ids=global_ids,
        )

    def distribute_sub_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_subhalo_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Wrapper around distribute_ids() for subhalo IDs.
        """
        return distribute_ids(
            slabs=slabs,
            particle_counts=self.reader.particle_counts,
            ptypes=sorted(self.reader.available_ptypes),
            comm=comm,
            global_ids=global_subhalo_ids,
        )


@njit(cache=True)
def reconstruct_membership(
    group_counts: np.ndarray,
    sub_counts: np.ndarray,
    group_first_sub: np.ndarray,
    group_n_subs: np.ndarray,
    n_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs the field and sub ID of a specified ptype. Returns:

    - field_ids: global field IDs
    - sub_ids: local sub IDs
    """
    # allocations
    field_ids = np.full(shape=n_particles, fill_value=-1, dtype=np.int64)
    sub_ids = np.full(shape=n_particles, fill_value=-1, dtype=np.int64)

    position = 0  # global position
    # NOTE: the subfind catalogue uses the same format as us for membership so this is just offset slicing
    for group_idx in range(len(group_counts)):
        group_length = group_counts[group_idx]
        field_ids[position : position + group_length] = group_idx

        first_sub = group_first_sub[group_idx]  # and then slicing within each subhalo table
        if first_sub >= 0:  # group_n_subs and first_sub (the central) replicate offsets too
            sub_position = position
            for sub_idx in range(first_sub, first_sub + group_n_subs[group_idx]):
                sub_length = sub_counts[sub_idx]
                sub_ids[sub_position : sub_position + sub_length] = sub_idx
                sub_position += sub_length

        position += group_length  # increment global position

    return field_ids, sub_ids


@njit(cache=True)
def local_to_global_parent(
    sub_parent: np.ndarray,
    group_first_sub: np.ndarray,
    sub_gr_number: np.ndarray,
) -> np.ndarray:
    """
    Converts the local SubhaloParent array to a global parent array for the membership propagation
    steps. Returns:

    - global_parents: mapping from index in the subhalo group to its global FOF parent
    """
    n_subhaloes = len(sub_parent)
    global_parents = np.full(shape=n_subhaloes, fill_value=-1, dtype=np.int64)

    for s in range(n_subhaloes):
        g = sub_gr_number[s]
        central = group_first_sub[g]
        if central < 0:  # no subhaloes
            continue

        local_idx = s - central  # positional index of this subhalo in the global array
        local_parent = sub_parent[s]

        if local_parent == local_idx:  # central within group (so a field)
            global_parents[s] = -1

        elif local_parent == 0 and local_idx != 0:  # lone field
            global_parents[s] = -1

        else:
            global_parents[s] = central + local_parent

    return global_parents
