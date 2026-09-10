"""

Parser for AHF: Amiga's Halo Finder.

AHF paper: https://iopscience.iop.org/article/10.1088/0067-0049/182/2/608

A common problem encountered when parsing AHF is its unique catalogue storage format (not .hdf5),
and the fact its IDs are enormous and approach the int overflow boundary.

NOTE: the AHF parser uses np.loadtxt with a numba parser on the resulting array. The call to
loadtxt is hard-coded to the structure of an AHF catalogue, and the structure of these catalogues
 is somewhat finicky/not too user-friendly. Therefore the parser is quite exposed to any changes
 AHF makes to how they store their information. If catalogues take a long time to parse, a better
 parsing method would perhaps be welcome.

"""

# type checking (semantic)
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ..data_management import SnapshotReader
    from mpi4py.MPI import Comm

# default libraries
from pathlib import Path
from functools import (
    cached_property,
)  # for avoiding rereading files across methods but also not holding too much in __init__

# other packages
import numpy as np
from numba import njit

# internal imports
from .halo_data_structures import (
    HaloAssignments,
    SubhaloInformation,
    HaloSource,
    distribute_ids,
    compute_depths,
    apply_lookup,
)
from ..log import get_logger

logger = get_logger()


class AHFCatalogue(NamedTuple):  # for code readability
    """
    Container for AHF catalogue info.
    """

    parent_indices: np.ndarray
    depths: np.ndarray
    n_particles: np.ndarray
    field_lookup: np.ndarray
    sub_lookup: np.ndarray
    field_of: np.ndarray
    original_field_ids: np.ndarray
    original_sub_ids: np.ndarray


class AHFHaloSource(HaloSource):
    """
    AHF Amiga Halo Finder parser with object-oriented interface. Methods:

    - read_halo_ids: returns HaloAssignments object
    - read_subhalo_ids: returns SubhaloInformation object
    - distribute_raw_halo_ids: distributes slab-based HaloID info from rank 0 to other ranks
    - distribute_raw_subhalo_ids: distributes slab-based SubhaloID info from rank 0 to other ranks
    """

    def __init__(self, haloes_path: Path, particles_path: Path, reader: SnapshotReader) -> None:

        super().__init__(reader=reader)
        self.haloes_path = haloes_path
        self.particles_path = particles_path

    @cached_property  # cached_property allows AHFHaloSource to parse catalogues once while meeting the inheritance requirements of a HaloSource class
    def _haloes_catalogue(self) -> AHFCatalogue:
        """
        Parses and stores AHF_halos file information, deriving raw ahf ids, parent indices,
        subhalo depths, and lookup arrays.
        """
        raw_ahf_ids, raw_host_ids, n_particles = parse_ahf_haloes(self.haloes_path)

        # you now need to remap the comically-large AHF ids (use indices instead)
        parent_indices = remap_ahf_ids(
            ahf_ids=raw_ahf_ids, raw_host_ids=raw_host_ids
        )  # this uses searchsorted so no need for the contiguous ID helper

        field_of = compute_field_index(parent_ids=parent_indices)
        depths = compute_depths(parent_ids=parent_indices)

        is_field = parent_indices == -1

        field_lookup = np.full(len(raw_ahf_ids), fill_value=-1, dtype=np.int64)
        field_lookup[is_field] = np.arange(is_field.sum(), dtype=np.int64)

        sub_lookup = np.full(len(raw_ahf_ids), fill_value=-1, dtype=np.int64)
        sub_lookup[~is_field] = np.arange((~is_field).sum(), dtype=np.int64)

        # rediscover raw IDs for progenitor matching
        field_mask = field_lookup != -1
        original_field_ids = np.empty(shape=field_mask.sum(), dtype=np.int64)  # int64 is very important here
        original_field_ids[field_lookup[field_mask]] = raw_ahf_ids[field_mask]

        sub_mask = sub_lookup != -1
        original_sub_ids = np.empty(shape=sub_mask.sum(), dtype=np.int64)  # int64 is very important here
        original_sub_ids[sub_lookup[sub_mask]] = raw_ahf_ids[sub_mask]

        catalogue = AHFCatalogue(
            parent_indices=parent_indices,
            depths=depths,
            n_particles=n_particles,
            field_lookup=field_lookup,
            sub_lookup=sub_lookup,
            field_of=field_of,
            original_field_ids=original_field_ids,
            original_sub_ids=original_sub_ids,
        )

        return catalogue

    @cached_property
    def _particles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Parses AHF_particles and deduplicates inclusive membership.
        Returns (unique_particle_ids, deepest_catalogue_indices), sorted by unique_pids.
        """
        particles_array = np.loadtxt(self.particles_path, skiprows=1, dtype=np.int64)
        depths = self._haloes_catalogue.depths

        pids, halo_indices = parse_ahf_particles(ahf_particle_array=particles_array, n_haloes=len(depths))

        return deduplicate_ahf_particles(pids=pids, halo_indices=halo_indices, depths=depths)

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Interfaces with the provided SnapshotReader and parses the .AHF_halos/AHF_particles files
        to match snapshot particle IDs with their AHF equivalents and produce the final assignments
        made by AHF. Returns:

        - HaloAssignments dataclass.
        """
        catalogue = self._haloes_catalogue
        unique_pids, deepest_catalogue_indices = self._particles

        halo_assignments: dict[str, np.ndarray] = {}
        subhalo_assignments: dict[str, np.ndarray] = {}

        for ptype in ptypes:
            snapshot_pids = self.reader.read_particle_ids(ptype=ptype)

            positional_hids, positional_subhids = match_ahf_particle_ids(
                snapshot_pids=snapshot_pids,
                unique_ahf_pids=unique_pids,
                field_of=catalogue.field_of,
                deepest_catalogue_indices=deepest_catalogue_indices,
                depths=catalogue.depths,
            )

            halo_assignments[ptype] = apply_lookup(ids=positional_hids, lookup=catalogue.field_lookup)
            subhalo_assignments[ptype] = apply_lookup(ids=positional_subhids, lookup=catalogue.sub_lookup)

        n_total_haloes = np.sum(catalogue.depths == 0)

        sub_info = self.read_subhalo_info()
        for ptype, sub_ids in subhalo_assignments.items():
            in_sub = sub_ids != -1
            assert np.array_equal(sub_info.host_field_ids[sub_ids[in_sub]], halo_assignments[ptype][in_sub]), (
                f"{ptype}: particle HaloID disagrees with its subhalo's host tree."
            )

        n_subhaloes = np.sum(catalogue.depths > 0)
        logger.info(f"AHF: {n_total_haloes:,} field haloes | {n_subhaloes:,} subhaloes.")

        for ptype, ids in halo_assignments.items():
            n_assigned = np.sum(ids != -1)
            logger.info(f"  {ptype}: {n_assigned:,} / {len(ids):,} particles assigned to haloes.")

        return HaloAssignments(
            field_ids=halo_assignments,
            n_field_haloes=n_total_haloes,
            sub_ids=subhalo_assignments,
            original_field_ids=catalogue.original_field_ids,
        )

    def read_subhalo_info(self) -> SubhaloInformation:
        """
        Uses the supplied AHF catalogues to map out hierarchy and parent pointers for subhaloes. Returns:

        - SubhaloInformation dataclass
        """
        catalogue = self._haloes_catalogue

        sub_mask = catalogue.depths > 0

        # parents may be field haloes (depth-1 subs) or other subhaloes (deeper): remap each namespace
        sub_parents = catalogue.parent_indices[sub_mask]
        parent_is_field = catalogue.depths[sub_parents] == 0

        parent_index = np.where(parent_is_field, -1, catalogue.sub_lookup[sub_parents])
        host_halo_ids = catalogue.field_lookup[catalogue.field_of[sub_mask]]

        return SubhaloInformation(
            host_field_ids=host_halo_ids,
            parent_index=parent_index,
            global_index=np.arange(sub_mask.sum(), dtype=np.int64),
            depth=catalogue.depths[sub_mask],
            n_bound=catalogue.n_particles[sub_mask],
            original_sub_ids=catalogue.original_sub_ids,
        )

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


def parse_ahf_haloes(ahf_haloes_path: Path) -> tuple[np.ndarray, ...]:
    """
    Parses a .AHF_haloes file, returning a tuple of (ahf_ids, raw_host_ids, n_particles) (all raw AHF data).
    """
    #  col0=ID, col1=hostHalo, col4=n_particles; includes a header; tab-delimited
    ahf_ids, raw_host_ids, n_particles = np.loadtxt(
        fname=ahf_haloes_path, dtype=np.int64, usecols=[0, 1, 4], skiprows=1, delimiter="\t", unpack=True
    )

    return ahf_ids, raw_host_ids, n_particles


@njit(cache=True)
def parse_ahf_particles(ahf_particle_array: np.ndarray, n_haloes: int) -> tuple[np.ndarray, ...]:
    """
    Iterates on an .AHF_particles file which has been converted into an (n, 2) array by
    np.loadtxt, returning a tuple of (pids, halo_ids).
    """
    n_particles = len(ahf_particle_array) - n_haloes

    pids = np.empty(n_particles, dtype=np.int64)
    halo_indices = np.empty(n_particles, dtype=np.int64)

    current_halo = -1
    write_idx = 0
    # these appear in a columnar structure where for each halo you have:
    # initial n_particles | HaloID row
    # followed by rows of PID | PartType till next halo

    for row_idx in range(len(ahf_particle_array)):
        if (
            ahf_particle_array[row_idx, 1] > 5
        ):  # maximum PartType is 5 in standard convention; use this as a trick to figure out whether we are looking at a halo ID
            current_halo += 1  # NOTE: AHF haloes also have comically large IDs (6 quintillion or so)

        else:
            pids[write_idx] = ahf_particle_array[row_idx, 0]
            halo_indices[write_idx] = current_halo
            write_idx += 1

    return pids, halo_indices


def deduplicate_ahf_particles(
    pids: np.ndarray,
    halo_indices: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolves the duplicate appearances of particles in AHF haloes by assigning them to
    their deepest membership. Returns:

    - sorted_pids: unique particle IDs
    - sorted_halo_indices: corresponding deepest catalogue halo index
    """
    particle_depths = depths[halo_indices]

    sort_order = np.lexsort(
        (particle_depths, pids)
    )  # NOTE: lexsort sorts by second key first (so pids then depths, giving you depth-first pid appearance)

    sorted_pids = pids[sort_order]
    sorted_halo_indices = halo_indices[sort_order]

    # sort is depth-first; therefore, the final appearance of a particle is its deepest assignment
    last_appearance = np.empty(shape=len(sorted_pids), dtype=np.bool_)
    last_appearance[-1] = True
    last_appearance[:-1] = sorted_pids[:-1] != sorted_pids[1:]

    return sorted_pids[last_appearance], sorted_halo_indices[last_appearance]


def match_ahf_particle_ids(
    snapshot_pids: np.ndarray,
    unique_ahf_pids: np.ndarray,
    field_of: np.ndarray,
    deepest_catalogue_indices: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matches AHF particle ids to their counterparts in the raw snapshot, returning an array of HaloIDs and SubhaloIDs.

    - aligned_field_ids: field halo IDs of particles, aligned with snapshot_pids
    - aligned_sub_ids: subhalo IDs of particles, aligned with snapshot_pids
    """
    # find where particle IDs match
    insertion_idx = np.searchsorted(unique_ahf_pids, snapshot_pids)  # match snapshot to AHF, not the other way around
    insertion_idx = np.minimum(
        insertion_idx, len(unique_ahf_pids) - 1
    )  # prevent OOB when max(snapshot_pids) > max(ahf_pids)
    matched = unique_ahf_pids[insertion_idx] == snapshot_pids  # indices into AHF particle IDs which match

    aligned_field_ids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)
    aligned_sub_ids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)

    matched_insertion = insertion_idx[matched]
    deepest = deepest_catalogue_indices[matched_insertion]

    aligned_field_ids[matched] = field_of[deepest]  # assign field IDs via the field halo of their deepest subhalo
    aligned_sub_ids[matched] = np.where(depths[deepest] > 0, deepest, -1)  # subhalo ID is the ID of the deepest subhalo

    return aligned_field_ids, aligned_sub_ids


def remap_ahf_ids(ahf_ids: np.ndarray, raw_host_ids: np.ndarray) -> np.ndarray:
    """
    Map the comically-large AHF IDs to positional indices (necessary otherwise you will allocate an
    unfathomably large array of several exobytes).
    """
    sort_order = np.argsort(ahf_ids, stable=True)
    sorted_ids = ahf_ids[sort_order]
    is_field = raw_host_ids == 0
    parent_indices = np.full(len(ahf_ids), fill_value=-1, dtype=np.int64)

    insertion_idx = np.searchsorted(
        sorted_ids, raw_host_ids[~is_field]
    )  # position within the sorted array gives ascending unique sensible IDs
    parent_indices[~is_field] = sort_order[insertion_idx]

    return parent_indices


@njit(cache=True)
def compute_field_index(parent_ids: np.ndarray) -> np.ndarray:
    """
    Follows the same logic as compute_depths but instead returns the index of the field halo.
    """
    n_haloes = len(parent_ids)
    field_index = np.empty(n_haloes, dtype=np.int64)

    for halo_idx in range(n_haloes):
        current = halo_idx
        while parent_ids[current] != -1:
            current = parent_ids[current]
        field_index[halo_idx] = current

    return field_index
