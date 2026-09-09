"""

I/O functions for reading in parallel from the same .hdf5 snapshot file.

The way this works is: consider a dataset of length n, and we are running with a configuration
of r ranks. The parallelism strategy is to bin haloes across ranks by using an empirical law
to estimate their computational weight. Rank 0 runs this algorithm and knows which haloes, and
therefore, which particles, go where. Rank 0 will broadcasts an array called halo_to_rank to the other
ranks, which tells them which HaloID belongs to which rank.

The initial read of a dataset is embarrassingly parallel: each rank owns n/r particles in the read. We
must then get the particles from the initial read onto the ranks which own them according to their HaloIDs.
redistribute_data() uses comm.Alltoallv to reshuffle the particles: ranks go from owning n/r raw particles
to their owned particles. Since haloes are self-contained units of work, the rest of the pipeline then also
becomes embarrassingly parallel (no MPI in the computations).

The existence of the n_io_chunks in the config is because when I wrote this my cluster was on death's doorstep
and struggling on simultaneous big snapshot reads; moving it to a Python loop fixed things.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_structures import ParticleStore
    from .conventions import OctaviusConfig
    from ..external_halo_sources import HaloAssignments, SubhaloInformation
    from mpi4py.MPI import Comm

# default libraries
from dataclasses import dataclass, replace

# other packages
import numpy as np

# internal imports
from ..log import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class RedistributionMap:
    """
    Contains global information for where the particles read by each rank need to go:

    - send_order: rank-sorted order for arrays to mask their particles from (np.argsort(owner))
    - send_counts: the number of particles each rank must send
    - rec_counts: the number of particles each rank must receive
    """

    send_order: np.ndarray
    send_counts: np.ndarray
    rec_counts: np.ndarray


def redistribute_data(
    local_data: np.ndarray,
    redistribution_map: RedistributionMap,
    comm: Comm,
) -> np.ndarray:
    """
    Consumes the RedistributionMap and redistributes particle-level data from ranks' dataset slabs to the other ranks based on the global rank-halo allocation (from generate_rank_halo_assignments()). Returns:

    - received_data: ndarray of the rank's owned data across the slabs
    """
    if comm is None:  # early return for serial case
        return local_data[redistribution_map.send_order]

    from mpi4py.util.dtlib import from_numpy_dtype  # cannot import mpi4py at module-level for serial compatibility

    ordered_data = local_data[redistribution_map.send_order]

    if comm.size == 1:
        return ordered_data

    # displacements for MPI to know where to write memory to
    send_displacements = np.zeros(shape=comm.size, dtype=np.int64)
    send_displacements[1:] = np.cumsum(redistribution_map.send_counts[:-1])
    rec_displacements = np.zeros(comm.size, dtype=np.int64)
    rec_displacements[1:] = np.cumsum(redistribution_map.rec_counts[:-1])

    n_cols = (
        ordered_data.shape[1] if ordered_data.ndim == 2 else 1
    )  # for 3D columns (pos, vel); slightly inelegant but len() crashes on empty data

    send_counts = redistribution_map.send_counts * n_cols
    rec_counts = redistribution_map.rec_counts * n_cols
    send_displacements = send_displacements * n_cols
    rec_displacements = rec_displacements * n_cols

    received_data = np.empty(rec_counts.sum(), dtype=ordered_data.dtype)  # MPI malloc
    ordered_data = ordered_data.ravel()  # .ravel returns a view into memory where possible, .flatten always copies
    mpi_dtype = from_numpy_dtype(
        dtype=ordered_data.dtype
    )  # mpi4py requires its own dtype (and thankfully provides a helper)

    comm.Alltoallv(
        [ordered_data, send_counts, send_displacements, mpi_dtype],
        [received_data, rec_counts, rec_displacements, mpi_dtype],
    )

    if local_data.ndim > 1:
        return np.reshape(
            received_data, shape=(-1, local_data.shape[1])
        )  # -1 means np infers the number of rows from received_data
    return received_data


def build_redistribution_map(
    halo_to_rank: np.ndarray,
    slab_halo_ids: np.ndarray,
    comm: Comm | None,
) -> tuple[RedistributionMap, np.ndarray]:
    """
    Constructs the global RedistributionMap object for a slab of a dataset. Returns:

    - redistribution_map: a RedistributionMap object
    - mask: mask for the slab to filter sentinel/below min_dm_per_halo haloes
    """
    in_halo = (
        slab_halo_ids != -1
    )  # even though rank_assignments masks sentinel particles you still read the dataset (so ignore them again)
    mask = np.zeros(len(slab_halo_ids), dtype=bool)
    mask[in_halo] = (
        halo_to_rank[slab_halo_ids[in_halo]] != -1
    )  # then haloes < min_dm_per_halo also have -1 in the halo_to_rank array
    owner_ranks = halo_to_rank[slab_halo_ids[mask]]

    if comm is None:  # early return for serial case
        send_order = np.arange(
            mask.sum(), dtype=np.int64
        )  # match dataclass signature (redundant work but the alternative is painful)
        send_counts = np.array([mask.sum()], dtype=np.int64)
        rec_counts = send_counts.copy()
        return RedistributionMap(send_order=send_order, send_counts=send_counts, rec_counts=rec_counts), mask

    send_order = np.argsort(owner_ranks, stable=True)
    send_counts = np.bincount(
        owner_ranks, minlength=comm.size
    )  # minlength arg is for the case where a rank owns no particles in the slab
    rec_counts = np.empty_like(send_counts)  # MPI malloc

    comm.Alltoall(sendbuf=send_counts, recvbuf=rec_counts)  # Alltoall is high-performance (not alltoall)

    redistribution_map = RedistributionMap(
        send_order=send_order,
        send_counts=send_counts,
        rec_counts=rec_counts,
    )

    return redistribution_map, mask


def generate_rank_halo_assignments(
    halo_assignments: HaloAssignments,
    config: OctaviusConfig,
    n_ranks: int,
) -> np.ndarray:
    """
    Employs a weighted binning algorithm to balance the computational load across ranks by quantifying haloes' computational weight according to the empirically-tuned constants in the config. Masks out unassigned particles (sentinels) and those belonging to haloes below min_dm_per_halo. Returns:

    - halo_to_rank: an (n_haloes) array where halo_to_rank[i] = rank which halo i is assigned to.
    """
    logger.info(f"Computing halo assignments for {n_ranks} ranks.")
    ptype_counts = {}

    for ptype, halo_ids in halo_assignments.field_ids.items():
        valid = halo_ids[halo_ids != -1]  # masks valid HaloIDs here
        ptype_counts[ptype] = np.bincount(
            valid, minlength=halo_assignments.n_field_haloes
        )  # same logic as sum_per_group in aggregate_helpers.py

    haloes_exist = sum(ptype_counts.values()) > 0
    valid_halo_mask = (
        haloes_exist & (ptype_counts["dm"] >= config.min_dm_per_halo) if "dm" in ptype_counts else haloes_exist
    )  # masks min_dm_per_halo here
    all_valid_hids = np.flatnonzero(valid_halo_mask)

    if all_valid_hids.size == 0:  # guard against no-halo snapshots (high-z)
        logger.warning("No valid halo IDs!")
        return (
            np.full(shape=halo_assignments.n_field_haloes, fill_value=-1, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.zeros(n_ranks, dtype=np.float64),
        )

    n_valid_haloes = halo_assignments.n_field_haloes  # at this point the reader has remapped HaloIDs to 0-indexed

    halo_to_rank = np.full(shape=n_valid_haloes, fill_value=-1, dtype=np.int64)

    halo_weights, total_counts = compute_halo_weights(
        ptype_counts=ptype_counts,
        valid_halo_indices=all_valid_hids,
        stages=config.stages,
    )

    # bin according to computational weight: sort haloes by size descending then sequentially assign to rank with lightest load
    weight_order = np.argsort(halo_weights)[::-1]  # TODO: move to argsort(descending=True) in numpy 2.5.0
    rank_loads = np.zeros(n_ranks)

    # the actual binning algorithm, which is naturally sequential (not performance-heavy (yet))
    for idx in weight_order:
        lightest = int(
            np.argmin(rank_loads)  # rank with least computational weight currently
        )
        halo_to_rank[all_valid_hids[idx]] = lightest
        rank_loads[lightest] += halo_weights[idx]

    return halo_to_rank, halo_weights, total_counts, rank_loads


def compute_halo_weights(
    ptype_counts: dict[str, np.ndarray],
    valid_halo_indices: np.ndarray,
    stages: dict[str, bool],
) -> np.ndarray:
    """
    Computes the per-halo computational weight for the binning algorithm according to the empirical formula, depending on which stages are enabled.
    """
    zeros = np.zeros(len(valid_halo_indices), dtype=np.float64)  # to avoid the boilerplate
    star_counts = ptype_counts.get("star", zeros)[valid_halo_indices]
    gas_counts = ptype_counts.get("gas", zeros)[valid_halo_indices]
    dm_counts = ptype_counts.get("dm", zeros)[valid_halo_indices]
    bh_counts = ptype_counts.get("bh", zeros)[valid_halo_indices]

    weights = np.zeros(
        len(valid_halo_indices), dtype=np.float64
    )  # weights are equally zero if these stages were not to run somehow, which works

    # NOTE: these power laws are all empirical, I tuned them for my dissertation
    if stages.get("find_galaxies", False):
        weights += star_counts**1.2 + gas_counts

    if any(
        stages.get(s, False) for s in ("properties_core", "properties_ptype_specific", "properties_local_environment")
    ):
        weights += star_counts + gas_counts + dm_counts

    if stages.get("photometry", False):
        weights += (star_counts + gas_counts) ** 1.1

    total_counts = star_counts + gas_counts + dm_counts + bh_counts

    return weights, total_counts


def generate_slabs(
    rank: int,
    n_ranks: int,
    particle_counts: dict[str, int],
) -> dict[str, slice]:
    """
    Generates per-ptype slabs, which are slices of [this_rank:next_rank] particles to be read from the raw snapshot. Returns:

    slabs: dict, keyed by ptype to give a slab
    """
    slabs: dict[str, slice] = {}

    for ptype, n_particles in particle_counts.items():
        remainder = n_particles % n_ranks  # need to account for the remainder in division
        base_allocation = n_particles // n_ranks

        particles_per_rank = base_allocation + 1 if rank < remainder else base_allocation

        start = rank * base_allocation + min(rank, remainder)
        end = start + particles_per_rank
        slabs[ptype] = slice(start, end)

    return slabs


def split_slab(slab: slice, n_io_chunks: int) -> list[slice]:
    """
    Splits a dataset slab into smaller chunks. Returns:

    - split_slabs: list of equally-divided slices according to n_io_chunks
    """
    starts = np.linspace(start=slab.start, stop=slab.stop, num=(n_io_chunks + 1), dtype=np.int64)
    split_slabs = [slice(int(starts[i]), int(starts[i + 1])) for i in range(n_io_chunks)]

    return split_slabs


def generate_read_plan(idx_sorted: np.ndarray, gap_threshold: int = 64) -> list[tuple[slice, np.ndarray | None]]:
    """
    Given a list of idx_sorted, returns slices for contiguous reads from a HDF5 dataset.
    gap_threshold controls the threshold beyond which groups of sorted indices are merged into a contiguous read
    then masked (for reducing the number of filesystem reads). Returns:

    - read_plan: a list containing tuples of slices and the corresponding masks.
    """
    if len(idx_sorted) == 0:  # early return guard
        return []  # match type annotation

    gaps = np.diff(idx_sorted)  # length of n-1
    breaks = np.where(gaps > gap_threshold)[0] + 1  # add 1 to compensate
    groups = np.split(idx_sorted, breaks)

    read_plan: list[tuple[slice, np.ndarray | None]] = []

    for group in groups:
        read_slice = slice(group[0], group[-1] + 1)
        slice_length = read_slice.stop - read_slice.start

        if slice_length == len(group):  # no mask needed
            read_plan.append((read_slice, None))

        else:
            mask = np.zeros(shape=slice_length, dtype=bool)
            mask[group - group[0]] = True  # offsets relative to beginning of slice
            read_plan.append((read_slice, mask))

    return read_plan


def assign_local_subhaloes(
    particles: dict[str, ParticleStore],
    subhalo_info: SubhaloInformation | None,
) -> SubhaloInformation:
    """
    Returns the SubhaloInformation object, but masked to the rank's particle allocation.
    """
    if subhalo_info is None:
        return None

    if len(subhalo_info.host_field_ids) == 0:  # edge case: subhalo finder only finds field haloes
        return None

    all_hids = [particles[ptype]["HaloID"] for ptype in particles]
    non_empty = [hids for hids in all_hids if len(hids) > 0]

    if not non_empty:
        return None

    max_hid = max(int(hids.max()) for hids in non_empty)
    present_on_rank = np.zeros(
        shape=(max(max_hid, subhalo_info.host_field_ids.max()) + 1), dtype=bool
    )  # max HaloID on rank and max host halo ID can be different due to field haloes with no substructure (sizing it to either/or introduced this edge case and dropped a subhalo during writing)
    for ptype in particles:
        hids = particles[ptype]["HaloID"]
        present_on_rank[hids[hids != -1]] = True  # this works because hids are contiguous and 0-indexed

    keep = present_on_rank[subhalo_info.host_field_ids]

    global_to_local_map = np.full(len(subhalo_info.depth), -1, dtype=np.int64)  # depth is proxy for n_subhaloes
    global_to_local_map[np.flatnonzero(keep)] = np.arange(
        keep.sum()
    )  # maps global position to 0-indexed rank-local position

    for ptype in particles:
        subhid = particles[ptype]["SubhaloID"]
        particles[ptype]["SubhaloID"] = np.where(
            (subhid == -1) | ~keep[subhid],
            -1,
            global_to_local_map[subhid],  # the inverted keep mask won't be hit in principle
        )  # set subhid to -1 for non-rank particles

    new_parent_index = np.where(
        subhalo_info.parent_index[keep] < 0,
        -1,
        global_to_local_map[subhalo_info.parent_index[keep]],  # set parent-global index to parent-local
    )

    new_subhalo_info = replace(
        subhalo_info,  # NOTE: n_total_haloes should not be replaced
        host_field_ids=subhalo_info.host_field_ids[keep],
        global_index=subhalo_info.global_index[keep],
        parent_index=new_parent_index,
        depth=subhalo_info.depth[keep],
        n_bound=subhalo_info.n_bound[keep],
        original_sub_ids=subhalo_info.original_sub_ids[keep],
    )

    return new_subhalo_info
