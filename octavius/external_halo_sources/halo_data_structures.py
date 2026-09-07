"""

Internal agnostic halo source infrastructure, for passing to the likewise-agnostic pipeline.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import SnapshotReader, OctaviusConfig
    from mpi4py.MPI import Comm

# default libraries
from dataclasses import dataclass
from abc import ABC, abstractmethod

# other packages
import numpy as np
from numba import njit

# internal imports
from ..data_management import generate_slabs
from ..log import get_logger

logger = get_logger()


@dataclass(slots=True, frozen=True)
class HaloAssignments:
    """
    Dictionaries of per-ptype HaloID/SubhaloID assignments aligned with the original snapshot.
    These use the internal -1 sentinel. Contains:

    - field_ids: field halo IDs
    - n_field_haloes: the total number of haloes (and subhaloes) present
    - sub_ids: subhalo IDs
    - original_field_ids: original field halo IDs (from external finder)
    """

    field_ids: dict[str, np.ndarray]
    n_field_haloes: int
    sub_ids: dict[str, np.ndarray] | None = None
    original_field_ids: np.ndarray | None = None


@dataclass(slots=True, frozen=True)
class SubhaloInformation:
    """
    Basic subhalo information.

    - host_field_ids: top-level HaloID
    - parent_index: immediate parent subhalo index
    - depth: the level of nesting, always >=1
    - n_bound: the number of bound particles (inclusive)
    - original_sub_ids: the original finder IDs
    """

    host_field_ids: np.ndarray  # top-level HaloID
    parent_index: np.ndarray  # immediate parent subhalo index
    depth: np.ndarray  # >= 1
    global_index: np.ndarray
    n_bound: np.ndarray
    original_sub_ids: np.ndarray | None = None


def build_halo_source(config: OctaviusConfig, reader: SnapshotReader) -> HaloSource:
    """
    Construct the (Sub)HaloID parser based on what was requested in the config. Returns:

    - HaloSource corresponding to the config source.
    """
    id_source = config.halo_id_source.upper()  # autocapitalise for user convenience

    if id_source == "SNAPSHOT":
        logger.info("Using snapshot-assigned HaloIDs.")
        return SnapshotHaloSource(reader=reader)

    elif id_source == "AHF":
        from .ahf import AHFHaloSource  # I had to stick this in here to avoid a circular import

        prefix = config.halo_catalogue_path  # renamed for explicitness
        logger.info("Using AHF-assigned HaloIDs.")
        logger.info(f"Finding AHF catalogues at {prefix}")
        return AHFHaloSource(
            haloes_path=prefix.with_suffix(".AHF_halos"),
            particles_path=prefix.with_suffix(".AHF_particles"),
            reader=reader,
        )

    elif id_source == "HBT-HERONS":
        from .hbt_herons import HeronsHaloSource

        logger.info("Using HBT-HERONS halo IDs.")
        logger.info(f"Finding HBT-HERONS catalogue at {config.halo_catalogue_path}")

        return HeronsHaloSource(catalogue_path=config.halo_catalogue_path, reader=reader)

    elif id_source == "SUBFIND":
        from .subfind import SubfindHaloSource

        assert config.simulation_type == "TNG", f"{config.simulation_type} not supported with SUBFIND."

        logger.info("Using SUBFIND halo assignments.")
        logger.info(f"Finding SUBFIND catalogue at {config.halo_catalogue_path}")

        return SubfindHaloSource(catalogue_path=config.halo_catalogue_path, reader=reader)

    else:
        raise ValueError(f"Unknown halo catalogue source {id_source}, please check config?")


def build_contiguous_id_lookup(ids: np.ndarray) -> np.ndarray:
    """
    Thin wrapper around np.searchsorted which returns an array of the indices which would remap a (sub)halo_id array to be contiguous
    """
    unique_ids = np.unique(ids[ids != -1])
    lookup = np.full(shape=unique_ids.max() + 1, fill_value=-1, dtype=np.int64)
    lookup[unique_ids] = np.arange(len(unique_ids), dtype=np.int64)

    return lookup


class HaloSource(ABC):
    """
    Abstract base class for external halo catalogues. Should not be instantiated
    directly, but always inherited.
    """

    def __init__(self, reader: SnapshotReader) -> None:

        self.reader = reader

    @abstractmethod
    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Reads halo IDs (particle-level) of snapshot particles based on the conventions and quirks of the source
        implementation; returns a HaloAssignments dataclass.
        """
        ...

    @abstractmethod
    def read_subhalo_info(self) -> SubhaloInformation | None:
        """
        Reads subhalo-specific information for constructing hierarchies, if available; returns either a SubhaloInformation
        dataclass or None if the catalogue does not contain hierarchies.
        """
        ...

    @abstractmethod
    def distribute_field_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Distributes particle field halo IDs.
        """
        ...

    @abstractmethod
    def distribute_sub_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_subhalo_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Distributes particle subhalo IDs.
        """
        ...


class SnapshotHaloSource(HaloSource):
    """
    Reads snapshot-assigned halo IDs (no external catalogue).
    Currently assumes snapshots do not contain subhalo information, only FOF groups.
    """

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Simply iterates over ptypes with the reader to produce HaloAssignments.
        """
        halo_ids: dict[str, np.ndarray] = {}

        for ptype in ptypes:
            halo_ids[ptype] = self.reader.read_halo_ids(ptype=ptype)  # these are already contiguous

        max_id = max(int(ids.max()) for ids in halo_ids.values() if len(ids) > 0)
        n_total_haloes = max_id + 1 if max_id >= 0 else 0  # derived, not read (could read from the actual field)

        return HaloAssignments(
            field_ids=halo_ids,
            n_field_haloes=n_total_haloes,
            sub_ids=None,
            original_field_ids=None,
        )

    def distribute_field_ids(
        self, slabs: dict[str, slice], comm: Comm | None, global_ids: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Iterates over ptypes to produce slabs of HaloIDs per-ptype.
        """
        raw_ids: dict[str, np.ndarray] = {}

        for ptype in sorted(self.reader.available_ptypes):
            raw_ids[ptype] = self.reader.read_halo_ids(ptype=ptype, slab=slabs[ptype])  # these are already contiguous

        return raw_ids

    def read_subhalo_info(self) -> SubhaloInformation | None:
        """
        No subhaloes on SnapshotHaloSource.
        """
        return None

    def distribute_sub_ids(
        self,
        slabs: dict[str, slice],
        comm: Comm | None,
        global_subhalo_ids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        No subhaloes on SnapshotHaloSource.
        """
        return None


def distribute_ids(
    slabs: dict[str, slice],
    particle_counts: dict[str, int],
    ptypes: list[str],
    comm: Comm | None,
    global_ids: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Distributes rank 0's raw halo ID information to the other ranks so they know
    the HaloIDs of the particles on their slab. Returns:

    - local_halo_ids: dict keyed per-ptype containing the rank's HaloID arrays
    """
    if comm is None or comm.size == 1:  # serial / mpiexec -n 1 guard
        return {ptype: global_ids[ptype][slabs[ptype]].copy() for ptype in ptypes}

    rank = comm.Get_rank()
    size = comm.Get_size()
    local_halo_ids: dict[str, np.ndarray] = {}

    if rank == 0:
        all_slabs = {
            dest: generate_slabs(rank=dest, n_ranks=size, particle_counts=particle_counts) for dest in range(size)
        }

    # NOTE: done with send/receive as MPI-3 comm.scatter is capped to < int32 elements (https://github.com/PyLops/pylops-mpi/issues/115)
    for ptype_index, ptype in enumerate(ptypes):  # ptypes needs to be sorted to avoid MPI desync
        slab = slabs[ptype]
        slab_length = slab.stop - slab.start

        if rank == 0:
            local_halo_ids[ptype] = global_ids[ptype][slab].copy()  # .copy() instead of sending to itself
            # sequentially send IDs to other ranks
            for dest in range(1, size):
                comm.Send(global_ids[ptype][all_slabs[dest][ptype]], dest=dest, tag=ptype_index)

        else:
            memory_block = np.empty(slab_length, dtype=np.int64)
            comm.Recv(memory_block, source=0, tag=ptype_index)  # ranks wait to receive their local IDs
            local_halo_ids[ptype] = memory_block  # memory block is now filled with global IDs from comm.Send

    return local_halo_ids


@njit(cache=True)
def compute_depths(parent_ids: np.ndarray, max_allowed_depth: int = 15) -> np.ndarray:
    """
    Uses the parent_id array to return a corresponding depths array where depth=1 means its immediate parent is the field halo.
    """
    n_subhaloes = len(parent_ids)
    depths = np.empty(n_subhaloes, dtype=np.int64)

    for idx in range(n_subhaloes):
        current = idx
        depth = 0

        while parent_ids[current] >= 0:
            current = parent_ids[current]
            depth += 1

            if depth > max_allowed_depth:
                raise ValueError("Subhalo depth loop has entered a recursion (cyclic relationship somewhere in data?)")

        depths[idx] = depth

    return depths


def apply_lookup(ids: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """
    Small helper to use the global id lookup with the ptype id arrays.
    """
    result = np.full_like(ids, fill_value=-1)
    matched = ids != -1  # there will be unmatched particles and [-1] on an ndarray picks last element, so must mask
    result[matched] = lookup[ids[matched]]
    return result
