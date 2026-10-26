"""

MPI-parallel version of the periodic 3D friends-of-friends halo finder (fof_halo_algorithm.py), for boxes whose
particles do not fit on one rank. No rank ever holds more than its share of the box.

The box is decomposed into contiguous slabs of x-planes of the FOF cell grid (domains), one per rank:

1. Each rank reads a contiguous chunk of the snapshot (in file order). The domain boundaries are chosen from a global
   histogram of particles per x-plane so each domain holds about the same number of DM particles, and DM particles
   are sent to the rank owning their domain.
2. Each domain receives copies of the STENCIL_REACH x-planes either side of it from its neighbours (ghosts), then
   links owned + ghost particles with the serial kernels on a non-periodic-in-x local grid.
3. Groups crossing a domain boundary appear on both ranks, joined through the ghosts: for each ghost, the pair (the
   owner's group, the receiver's group) is an edge. Edges are gathered on rank 0 and merged with connected
   components; there are only as many as there are groups touching a boundary.
4. Group sizes are summed across ranks, and the haloes (>= min_members) are ordered on rank 0 by descending size,
   with ties broken by the smallest snapshot index of their members. This is the same ordering as the serial
   finder, so the HaloIDs do not depend on the number of ranks.
5. Baryons are sent to the rank owning their domain and attached to their nearest DM particle (owned or ghost)
   within the linking length.
6. All HaloIDs are sent back to the rank which read the particle, aligned with its chunk of the snapshot.

Domains are at least STENCIL_REACH planes thick (2 * STENCIL_REACH for two domains, so that a domain's ghost planes
on either side never overlap), so there are at most n_cells_per_dim // STENCIL_REACH domains; any further ranks hold
no particles during halo finding. With a single domain the grid is the whole periodic box, as in the serial finder.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

# default libraries
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

# workhorses
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..data_management.parallel_reading import RedistributionMap, redistribute_data
from ..log import get_logger
from .fof_halo_algorithm import (
    STENCIL_REACH,
    attach_to_nearest_dm,
    build_periodic_cell_list,
    cells_per_dimension,
    compute_cell_planes,
    link_haloes,
    min_per_root,
    thread_slab_bounds,
)

logger = get_logger()

NO_GROUP = -1  # sentinel for ghost groups with no owned particles, whose edges are redundant
IMBALANCE_WARNING = 2.0  # warn when the busiest domain holds this many times the mean number of DM particles


@dataclass(slots=True, frozen=True)
class Domain:
    """
    A rank's slab of the FOF cell grid:

    - bounds: global x-plane boundaries of every domain (n_domains + 1 entries)
    - index: this rank's domain, or -1 if it has none (more ranks than domains)
    - x_offset: the global x-plane of the local grid's first plane (its first ghost plane)
    - n_planes: number of x-planes in the local grid (owned + ghosts)
    - periodic_x: True when there is a single domain, which then covers the whole periodic box without ghosts
    """

    bounds: np.ndarray
    index: int
    x_offset: int
    n_planes: int
    periodic_x: bool

    @property
    def n_domains(self) -> int:
        return len(self.bounds) - 1

    def owner_of_planes(self, planes: np.ndarray) -> np.ndarray:
        """
        Returns the domain (= rank) owning each global x-plane.
        """
        return np.searchsorted(self.bounds, planes, side="right") - 1


def find_fof_haloes_mpi(
    dm_pos: np.ndarray,
    dm_offset: int,
    boxsize: float,
    comm: Comm,
    b: float = 0.2,
    linking_length: float | None = None,
    baryon_pos: Mapping[str, np.ndarray] | None = None,
    min_members: int = 20,
    n_slabs: int | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Collective (every rank must call it). Identifies haloes with a periodic 3D friends-of-friends on the DM particles
    spread across ranks, then attaches baryons to the halo of their nearest DM particle within the linking length.
    Returns (halo_ids, n_haloes), where halo_ids is a dict keyed by ptype ("dm" plus every key of baryon_pos) of
    HaloID arrays aligned with this rank's input arrays. HaloIDs are as in find_fof_haloes().

    - dm_pos: (N_rank, 3) this rank's DM positions, in the same units as boxsize
    - dm_offset: snapshot index of this rank's first DM particle (its chunks must tile the snapshot in rank order)
    - boxsize: periodic box side length
    - comm: MPI communicator
    - b: linking length in units of the mean DM interparticle separation (ignored if linking_length is given)
    - linking_length: explicit linking length, overriding b
    - baryon_pos: optional mapping of ptype -> (M_rank, 3) positions to attach (read one at a time; every rank must
      have the same keys in the same order)
    - min_members: groups with fewer DM particles than this are discarded
    - n_slabs: number of x-slabs each rank links in parallel; defaults to 8 per thread
    - timings: optional dict which is filled with the wall time (s) of each step
    """
    timings = {} if timings is None else timings
    t0 = perf_counter()

    dm_pos = np.ascontiguousarray(dm_pos, dtype=np.float64)
    n_dm = comm.allreduce(len(dm_pos))

    if linking_length is None:
        linking_length = b * boxsize / np.cbrt(n_dm)
    n_cells_per_dim = cells_per_dimension(boxsize=boxsize, linking_length=linking_length)

    # send DM to the owners of its domain, then swap ghost planes with neighbouring domains
    planes = compute_cell_planes(dm_pos, boxsize, n_cells_per_dim)
    domain = decompose_domains(planes=planes, n_cells_per_dim=n_cells_per_dim, comm=comm)
    to_owner = build_route(destinations=domain.owner_of_planes(planes), comm=comm)
    del planes

    owned_pos = redistribute_data(dm_pos, to_owner, comm)
    owned_index = redistribute_data(np.arange(dm_offset, dm_offset + len(dm_pos), dtype=np.int64), to_owner, comm)
    n_owned = len(owned_pos)

    to_ghost = build_ghost_route(
        planes=compute_cell_planes(owned_pos, boxsize, n_cells_per_dim), domain=domain, comm=comm
    )
    local_pos = np.concatenate([owned_pos, redistribute_data(owned_pos, to_ghost, comm)])
    del owned_pos
    timings["exchanging particles"] = perf_counter() - t0

    # link locally
    t0 = perf_counter()
    n_local = len(local_pos)
    if n_local > 0:
        sort_order, sorted_pos, cell_offsets, unique_cells, row_offsets = build_periodic_cell_list(
            pos=local_pos,
            boxsize=boxsize,
            n_cells_per_dim=n_cells_per_dim,
            x_offset=domain.x_offset,
            n_planes=domain.n_planes,
        )
        parents = link_haloes(
            positions=sorted_pos,
            cell_offsets=cell_offsets,
            unique_cells=unique_cells,
            row_offsets=row_offsets,
            n_cells_per_dim=n_cells_per_dim,
            boxsize=boxsize,
            linking_length=linking_length,
            slab_bounds=thread_slab_bounds(n_planes=domain.n_planes, n_slabs=n_slabs),
            x_offset=domain.x_offset,
            periodic_x=domain.periodic_x,
        )
        # parents are in cell order; bring them back to local order (owned first, then ghosts)
        local_roots = np.empty(n_local, dtype=np.int64)
        local_roots[sort_order] = parents
        del parents
    else:
        sort_order = np.empty(0, dtype=np.int64)
        local_roots = np.empty(0, dtype=np.int64)
    del local_pos
    timings["linking"] = perf_counter() - t0

    # merge groups across domains and label them globally
    t0 = perf_counter()
    owned_labels = label_global_groups(
        local_roots=local_roots,
        n_owned=n_owned,
        owned_index=owned_index,
        to_ghost=to_ghost,
        min_members=min_members,
        comm=comm,
    )
    del owned_index
    n_haloes = max(comm.allgather(int(owned_labels.max(initial=-1)) + 1))

    # ghosts take their owner's label; the receiver's own (partial) view of a ghost's group may be incomplete
    local_labels = np.concatenate([owned_labels, redistribute_data(owned_labels, to_ghost, comm)])
    halo_ids = {"dm": return_to_sender(owned_labels, to_owner, comm, n_original=len(dm_pos))}
    timings["merging groups"] = perf_counter() - t0

    # attach baryons on the domain owning them, then send their HaloIDs back
    t0 = perf_counter()
    if n_local > 0:
        dm_labels_sorted = local_labels[sort_order]
    for ptype, pos in (baryon_pos or {}).items():
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        to_attach = build_route(
            destinations=domain.owner_of_planes(compute_cell_planes(pos, boxsize, n_cells_per_dim)), comm=comm
        )
        attach_pos = redistribute_data(pos, to_attach, comm)

        if n_local > 0:
            attached = attach_to_nearest_dm(
                positions=attach_pos,
                dm_positions=sorted_pos,
                dm_labels=dm_labels_sorted,
                cell_offsets=cell_offsets,
                unique_cells=unique_cells,
                row_offsets=row_offsets,
                n_cells_per_dim=n_cells_per_dim,
                boxsize=boxsize,
                linking_length=linking_length,
                x_offset=domain.x_offset,
                n_planes=domain.n_planes,
                periodic_x=domain.periodic_x,
            )
        else:  # a domain with no DM has nothing to attach to
            attached = np.full(len(attach_pos), -1, dtype=np.int64)

        halo_ids[ptype] = return_to_sender(attached, to_attach, comm, n_original=len(pos))
    timings["attaching baryons"] = perf_counter() - t0

    return halo_ids, n_haloes


def decompose_domains(planes: np.ndarray, n_cells_per_dim: int, comm: Comm) -> Domain:
    """
    Collective. Splits the x-planes of the cell grid into one domain per rank (or fewer, if the grid is too thin),
    balancing the number of particles per domain; planes are this rank's particles' x-planes.
    """
    n_ranks = comm.size
    n_domains = min(n_ranks, n_cells_per_dim // STENCIL_REACH)
    if n_domains == 2 and n_cells_per_dim < 4 * STENCIL_REACH:
        n_domains = 1
    min_thickness = 2 * STENCIL_REACH if n_domains == 2 else STENCIL_REACH

    counts = np.bincount(planes, minlength=n_cells_per_dim).astype(np.int64)
    comm.Allreduce(counts.copy(), counts)  # in place is not portable across MPI implementations
    bounds = balanced_bounds(counts=counts, n_domains=n_domains, min_thickness=min_thickness)
    if comm.rank == 0:
        log_decomposition(counts=counts, bounds=bounds, n_ranks=n_ranks)

    index = comm.rank if comm.rank < n_domains else -1
    if n_domains == 1:
        return Domain(bounds=bounds, index=index, x_offset=0, n_planes=n_cells_per_dim, periodic_x=True)

    thickness = bounds[index + 1] - bounds[index] if index >= 0 else 0
    return Domain(
        bounds=bounds,
        index=index,
        x_offset=int((bounds[max(index, 0)] - STENCIL_REACH) % n_cells_per_dim),
        n_planes=int(thickness + 2 * STENCIL_REACH),
        periodic_x=False,
    )


def log_decomposition(counts: np.ndarray, bounds: np.ndarray, n_ranks: int) -> None:
    """
    Logs how evenly the DM particles are split between domains, warning about idle ranks or a badly unbalanced split.
    """
    n_domains = len(bounds) - 1
    per_domain = np.add.reduceat(counts, bounds[:-1])
    mean = per_domain.mean()
    logger.info(
        f"FOF: box split into {n_domains} slab(s) of {np.diff(bounds).min()}-{np.diff(bounds).max()} cell planes; "
        f"DM particles per slab {per_domain.min():,}-{per_domain.max():,} (mean {mean:,.0f})."
    )

    if n_domains < n_ranks:
        logger.warning(
            f"FOF: the cell grid ({len(counts)} cells per side) only allows {n_domains} slab(s), so "
            f"{n_ranks - n_domains} of {n_ranks} ranks are idle during halo finding."
        )
    if mean > 0 and per_domain.max() > IMBALANCE_WARNING * mean:
        logger.warning(
            f"FOF: the busiest rank holds {per_domain.max() / mean:.1f}x the mean number of DM particles, as a dense "
            f"region cannot be split into slabs thinner than {STENCIL_REACH} cell planes; halo finding will be slower "
            "and that rank needs proportionally more memory."
        )


def balanced_bounds(counts: np.ndarray, n_domains: int, min_thickness: int) -> np.ndarray:
    """
    Returns n_domains + 1 plane boundaries splitting the planes into domains of about equal particle counts, each at
    least min_thickness planes thick.
    """
    n_planes = len(counts)
    total = counts.sum()
    bounds = np.empty(n_domains + 1, dtype=np.int64)
    bounds[0], bounds[-1] = 0, n_planes

    if total > 0:
        targets = total * np.arange(1, n_domains) / n_domains
        bounds[1:-1] = np.searchsorted(np.cumsum(counts), targets, side="left") + 1
    else:
        bounds[1:-1] = np.linspace(0, n_planes, n_domains + 1)[1:-1]

    for k in range(1, n_domains):  # push boundaries up to leave room below...
        bounds[k] = max(bounds[k], bounds[k - 1] + min_thickness)
    for k in range(n_domains - 1, 0, -1):  # ...then down to leave room above
        bounds[k] = min(bounds[k], bounds[k + 1] - min_thickness)

    assert np.all(np.diff(bounds) >= min_thickness), "Domain decomposition failed (grid too thin for the ranks?)"
    return bounds


def build_route(destinations: np.ndarray, comm: Comm) -> RedistributionMap:
    """
    Collective. Returns the RedistributionMap sending each particle to the rank in destinations.
    """
    send_order = np.argsort(destinations, kind="stable")
    send_counts = np.bincount(destinations, minlength=comm.size).astype(np.int64)
    rec_counts = np.empty_like(send_counts)
    comm.Alltoall(sendbuf=send_counts, recvbuf=rec_counts)

    return RedistributionMap(send_order=send_order, send_counts=send_counts, rec_counts=rec_counts)


def build_ghost_route(planes: np.ndarray, domain: Domain, comm: Comm) -> RedistributionMap:
    """
    Collective. Returns the RedistributionMap sending copies of this domain's first STENCIL_REACH planes to the
    previous domain and its last STENCIL_REACH planes to the next (periodically). A particle may go to both.
    """
    n = domain.n_domains
    if n == 1 or domain.index < 0:
        send_order = np.empty(0, dtype=np.int64)
        destinations = np.empty(0, dtype=np.int64)
    else:
        start, stop = domain.bounds[domain.index], domain.bounds[domain.index + 1]
        to_previous = np.flatnonzero(planes < start + STENCIL_REACH)
        to_next = np.flatnonzero(planes >= stop - STENCIL_REACH)
        previous, following = (domain.index - 1) % n, (domain.index + 1) % n
        send_order = np.concatenate([to_previous, to_next])
        destinations = np.concatenate(
            [np.full(len(to_previous), previous, dtype=np.int64), np.full(len(to_next), following, dtype=np.int64)]
        )
        order = np.argsort(destinations, kind="stable")
        send_order, destinations = send_order[order], destinations[order]

    send_counts = np.bincount(destinations, minlength=comm.size).astype(np.int64)
    rec_counts = np.empty_like(send_counts)
    comm.Alltoall(sendbuf=send_counts, recvbuf=rec_counts)

    return RedistributionMap(send_order=send_order, send_counts=send_counts, rec_counts=rec_counts)


def send_back(data: np.ndarray, route: RedistributionMap, comm: Comm) -> np.ndarray:
    """
    Collective. Sends per-particle data on the receiving end of route back along it; returns it aligned with
    route.send_order on the original sender.
    """
    reverse = RedistributionMap(
        send_order=np.arange(len(data), dtype=np.int64),  # received data is already grouped by source rank
        send_counts=route.rec_counts,
        rec_counts=route.send_counts,
    )
    return redistribute_data(data, reverse, comm)


def return_to_sender(data: np.ndarray, route: RedistributionMap, comm: Comm, n_original: int) -> np.ndarray:
    """
    Collective. Like send_back(), but for a route which moved every particle exactly once (build_route()), returning
    the data aligned with the sender's original particle order.
    """
    out = np.empty(n_original, dtype=data.dtype)
    out[route.send_order] = send_back(data, route, comm)
    return out


def label_global_groups(
    local_roots: np.ndarray,
    n_owned: int,
    owned_index: np.ndarray,
    to_ghost: RedistributionMap,
    min_members: int,
    comm: Comm,
) -> np.ndarray:
    """
    Collective. Merges the domains' local groups into global ones and labels them; returns the HaloID of each owned
    particle (-1 if its group has fewer than min_members particles).

    - local_roots: root (local index) of each local particle: owned first, then ghosts in to_ghost receive order
    - owned_index: snapshot index of each owned particle (for the size-ordering tie-break)
    """
    n_local = len(local_roots)

    # local groups are keyed by root, offset by rank so keys are globally unique
    key_offset = comm.exscan(n_local) or 0  # exscan gives None on rank 0
    is_owned = np.zeros(n_local, dtype=np.bool_)
    is_owned[:n_owned] = True
    owned_count = np.bincount(local_roots[:n_owned], minlength=n_local)

    # each ghost tells its owner which group it joined here; groups with no owned particles here add nothing
    ghost_roots = local_roots[n_owned:]
    ghost_keys = np.where(owned_count[ghost_roots] > 0, key_offset + ghost_roots, NO_GROUP)
    receiver_keys = send_back(ghost_keys, to_ghost, comm)  # aligned with to_ghost.send_order
    owner_keys = key_offset + local_roots[to_ghost.send_order]
    linked = receiver_keys != NO_GROUP
    edges = np.unique(np.stack([owner_keys[linked], receiver_keys[linked]], axis=1), axis=0)

    # rank 0 merges the edges into connected components, keyed by their smallest key
    all_edges = comm.gather(edges, root=0)
    if comm.rank == 0:
        merged_keys, merged_roots = merge_edges(np.concatenate(all_edges).reshape(-1, 2))
    else:
        merged_keys = merged_roots = None
    merged_keys, merged_roots = comm.bcast((merged_keys, merged_roots), root=0)

    final_keys = key_offset + np.arange(n_local, dtype=np.int64)
    bridged = np.zeros(n_local, dtype=np.bool_)
    lo, hi = np.searchsorted(merged_keys, [key_offset, key_offset + n_local])
    final_keys[merged_keys[lo:hi] - key_offset] = merged_roots[lo:hi]
    bridged[merged_keys[lo:hi] - key_offset] = True

    # report every group which may be a halo: large enough here, or with members elsewhere
    first_member = min_per_root(parents=local_roots, values=_pad(owned_index, n_local), mask=is_owned)
    reported = np.flatnonzero((owned_count > 0) & (bridged | (owned_count >= min_members)))
    reports = comm.gather((final_keys[reported], owned_count[reported], first_member[reported]), root=0)

    if comm.rank == 0:
        halo_ids_per_rank = rank_haloes(reports=reports, min_members=min_members)
    else:
        halo_ids_per_rank = None
    reported_ids = comm.scatter(halo_ids_per_rank, root=0)

    root_ids = np.full(n_local, -1, dtype=np.int64)
    root_ids[reported] = reported_ids
    return root_ids[local_roots[:n_owned]]


def merge_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (keys, roots): every key appearing in edges (sorted) and the smallest key of its connected component.
    """
    keys, inverse = np.unique(edges, return_inverse=True)
    if len(keys) == 0:
        return keys, keys.copy()
    inverse = inverse.reshape(-1, 2)
    graph = coo_matrix(
        (np.ones(len(inverse), dtype=np.int8), (inverse[:, 0], inverse[:, 1])), shape=(len(keys), len(keys))
    )
    _, components = connected_components(graph, directed=False)

    # keys are sorted, so the first key seen in each component is its smallest
    _, first = np.unique(components, return_index=True)
    return keys, keys[first][components]


def rank_haloes(reports: list[tuple[np.ndarray, np.ndarray, np.ndarray]], min_members: int) -> list[np.ndarray]:
    """
    Sums each reported group's size over ranks and labels those with at least min_members members by descending
    size (ties broken by smallest member index); returns the HaloID (or -1) of each rank's reported groups.
    """
    keys = np.concatenate([r[0] for r in reports])
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sizes = np.bincount(inverse, weights=np.concatenate([r[1] for r in reports])).astype(np.int64)
    first_member = np.full(len(unique_keys), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_member, inverse, np.concatenate([r[2] for r in reports]))

    haloes = np.flatnonzero(sizes >= max(min_members, 1))
    haloes = haloes[np.lexsort((first_member[haloes], -sizes[haloes]))]
    halo_ids = np.full(len(unique_keys), -1, dtype=np.int64)
    halo_ids[haloes] = np.arange(len(haloes), dtype=np.int64)

    splits = np.cumsum([len(r[0]) for r in reports])[:-1]
    return np.split(halo_ids[inverse], splits)


def _pad(values: np.ndarray, length: int) -> np.ndarray:
    """
    Pads values with int64 max up to length (ghosts have no snapshot index here).
    """
    out = np.full(length, np.iinfo(np.int64).max, dtype=np.int64)
    out[: len(values)] = values
    return out
