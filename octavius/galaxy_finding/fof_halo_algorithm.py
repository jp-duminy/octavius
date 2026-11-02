"""

Octavius 3D friends-of-friends halo finder, for snapshots which do not come with HaloIDs.

This reuses the cell linked-list and union-find machinery of the FOF6D galaxy finder, but there are no
pre-identified haloes to parallelise over, so the whole (periodic) box is linked at once. The approach is:

1. Bin DM particles into a periodic grid of cells with diagonal no larger than the linking length (side <= l/sqrt(3))
   and sort by flat cell index (CSR). Every particle in a cell is then within l of every other, so each occupied cell
   is a ready-made group and linking happens between cells rather than particles.
2. Split the grid into slabs of contiguous x-planes. Because the flat index is row-major, each slab is a contiguous
   range of the sorted arrays, so slabs can be linked in parallel (prange) without touching each other's parents.
3. Links which cross a slab boundary (including the periodic wrap in x) are made in a short serial pass afterwards.
4. Groups below min_members are discarded; the remainder are relabelled 0..n-1 in descending size order (ties go to
   the group containing the lowest snapshot index, so the labels are the same however the box is divided).
5. Baryons inherit the HaloID of their nearest DM particle within the linking length (as in SWIFT).

fof_halo_mpi.py runs the same kernels on a slab of the box per MPI rank; the kernels therefore take the grid's first
x-plane (x_offset) and number of x-planes (n_planes), and only wrap in x when the grid is the whole box (periodic_x).

Optimisations relative to a naïve port of dispatch_fof6d:

- Linking is 3D, so no velocity dispersion pass and no phase-space check.
- Since cells are internally connected, a pair of neighbouring cells is skipped outright if they already share a root,
  and otherwise only needs a single pair within l to merge (the search stops at the first). Dense halo cores, where
  a naïve cell list does O(n^2) pair checks per cell, therefore cost close to O(n).
- Neighbour cells are looked up once per cell rather than once per particle, and only the half stencil is visited.
  Lookups go through a dense index of (x, y) rows, so each fetches a whole row of z-neighbours with a short local
  search instead of a binary search over every occupied cell in the box (which is dominated by cache misses).
- Pre-culling isolated particles is deliberately not done: with a cell list, establishing that a particle is
  isolated is the same neighbour search as linking it, so it would cost a second pass for no gain.
- Ranks are int8 (rank <= log2(N)) and the size-ordered relabelling reuses the bincount buffer.

Original FoF: Davis et al. 1985, doi: 10.1086/163168

"""

# default libraries
from collections.abc import Mapping
from time import perf_counter

# workhorses
from numba import get_num_threads, njit, prange
import numpy as np

from .fof6d_algorithm import find_root, union


CELLS_PER_LINKING_LENGTH = np.sqrt(3.0)  # cell diagonal <= linking length, so a cell's particles are all linked
STENCIL_REACH = 2  # ceil(sqrt(3)): cells up to two away can hold particles within the linking length
DENSE_PAIR_THRESHOLD = 1024  # cell pairs with more candidate pairs than this are culled to the facing particles first

_ROW_OFFSETS = [
    (dx, dy) for dx in range(-STENCIL_REACH, STENCIL_REACH + 1) for dy in range(-STENCIL_REACH, STENCIL_REACH + 1)
]

# pairs are symmetric, so linking visits the half stencil (dx, dy, dz) > (0, 0, 0): every dz for rows (dx, dy) > (0, 0)
# and only dz > 0 in the cell's own row. Stored as (dx, dy, lowest dz) per row.
HALF_ROWS = np.array(
    [(dx, dy, -STENCIL_REACH) for dx, dy in _ROW_OFFSETS if (dx, dy) > (0, 0)] + [(0, 0, 1)], dtype=np.int64
)

# for nearest-neighbour searches: rows ordered by their minimum squared separation (in cell units) from the cell, so
# the search can stop once no closer particle is possible
_ROW_GAPS_SQ = [max(abs(dx) - 1, 0) ** 2 + max(abs(dy) - 1, 0) ** 2 for dx, dy in _ROW_OFFSETS]
_NEAREST_FIRST = np.argsort(_ROW_GAPS_SQ, kind="stable")
FULL_ROWS = np.array(_ROW_OFFSETS, dtype=np.int64)[_NEAREST_FIRST]
FULL_ROW_GAPS_SQ = np.array(_ROW_GAPS_SQ, dtype=np.float64)[_NEAREST_FIRST]

def find_fof_haloes(
    dm_pos: np.ndarray,
    boxsize: float,
    b: float = 0.2,
    linking_length: float | None = None,
    baryon_pos: Mapping[str, np.ndarray] | None = None,
    min_members: int = 20,
    n_slabs: int | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Identifies haloes with a periodic 3D friends-of-friends on the DM particles, then attaches baryons to the halo
    of their nearest DM particle within the linking length. Returns a dict keyed by ptype ("dm" plus every key of
    baryon_pos) of HaloID arrays aligned with the input particle order: 0-indexed, ordered by descending DM
    membership, with -1 for particles not in any halo.

    - dm_pos: (N, 3) DM positions, in the same units as boxsize
    - boxsize: periodic box side length
    - b: linking length in units of the mean DM interparticle separation (ignored if linking_length is given)
    - linking_length: explicit linking length, overriding b
    - baryon_pos: optional mapping of ptype -> (M, 3) positions to attach to the DM haloes (read one at a time)
    - min_members: groups with fewer DM particles than this are discarded
    - n_slabs: number of x-slabs to link in parallel; defaults to 8 per thread for load balancing
    - timings: optional dict which is filled with the wall time (s) of each step
    """
    timings = {} if timings is None else timings
    t0 = perf_counter()

    dm_pos = np.ascontiguousarray(dm_pos, dtype=np.float64)
    n_dm = len(dm_pos)

    if linking_length is None:
        linking_length = b * boxsize / np.cbrt(n_dm)
    n_cells_per_dim = cells_per_dimension(boxsize=boxsize, linking_length=linking_length)

    sort_order, sorted_pos, cell_offsets, unique_cells, row_offsets = build_periodic_cell_list(
        pos=dm_pos, boxsize=boxsize, n_cells_per_dim=n_cells_per_dim
    )
    timings["cell grid"] = perf_counter() - t0

    t0 = perf_counter()
    parents = link_haloes(
        positions=sorted_pos,
        cell_offsets=cell_offsets,
        unique_cells=unique_cells,
        row_offsets=row_offsets,
        n_cells_per_dim=n_cells_per_dim,
        boxsize=boxsize,
        linking_length=linking_length,
        slab_bounds=thread_slab_bounds(n_planes=n_cells_per_dim, n_slabs=n_slabs),
        x_offset=0,
        periodic_x=True,
    )
    labels_sorted = label_groups(parents=parents, min_members=min_members, tiebreak=sort_order)
    del parents

    halo_ids = {"dm": np.empty(n_dm, dtype=np.int64)}
    halo_ids["dm"][sort_order] = labels_sorted
    del sort_order
    timings["linking"] = perf_counter() - t0

    t0 = perf_counter()
    for ptype, pos in (baryon_pos or {}).items():
        halo_ids[ptype] = attach_to_nearest_dm(
            positions=gather_wrapped_positions(  # wrap so minimum image is exact
                pos=pos, order=np.arange(len(pos)), boxsize=boxsize
            ),
            dm_positions=sorted_pos,
            dm_labels=labels_sorted,
            cell_offsets=cell_offsets,
            unique_cells=unique_cells,
            row_offsets=row_offsets,
            n_cells_per_dim=n_cells_per_dim,
            boxsize=boxsize,
            linking_length=linking_length,
            x_offset=0,
            n_planes=n_cells_per_dim,
            periodic_x=True,
        )
    timings["attaching baryons"] = perf_counter() - t0

    return halo_ids


def cells_per_dimension(boxsize: float, linking_length: float) -> int:
    """
    Returns the number of cells along each side of the box, such that a cell's diagonal is at most the linking length.
    """
    n_cells_per_dim = int(np.ceil(boxsize * CELLS_PER_LINKING_LENGTH / linking_length))
    if n_cells_per_dim < 2 * STENCIL_REACH + 1:  # otherwise the periodic stencil would visit cells twice
        raise ValueError(
            f"Linking length {linking_length} is too large for the box ({boxsize}); periodic FOF is ill-defined."
        )

    return n_cells_per_dim


def thread_slab_bounds(n_planes: int, n_slabs: int | None = None) -> np.ndarray:
    """
    Returns the x-plane boundaries of the slabs linked in parallel by link_haloes(); defaults to 8 per thread for load
    balancing, and slabs are at least STENCIL_REACH planes thick.
    """
    if n_slabs is None:
        n_slabs = 8 * get_num_threads()
    n_slabs = max(1, min(n_slabs, n_planes // STENCIL_REACH))

    return np.linspace(0, n_planes, n_slabs + 1).astype(np.int64)


def build_periodic_cell_list(
    pos: np.ndarray, boxsize: float, n_cells_per_dim: int, x_offset: int = 0, n_planes: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions particles into a sparse, periodic cell linked-list. By default the grid covers the whole box; for a
    domain of an MPI decomposition it covers only the n_planes x-planes starting at plane x_offset (wrapping), which
    every particle must lie in. Returns a tuple of arrays:

    - sort_order: the indices to sort particles by cell
    - sorted_pos: positions in cell order, wrapped into [0, boxsize)
    - cell_offsets: where each occupied cell begins in the sorted arrays (csr offsets)
    - unique_cells: the flat cell idx of each occupied cell
    - row_offsets: where each (x, y) row of cells begins in unique_cells (dense, n_planes * n_cells_per_dim + 1 entries)

    Cell x coordinates (and so flat indices) are relative to x_offset.
    """
    if n_planes is None:
        n_planes = n_cells_per_dim

    flat_cell_idx = compute_flat_cell_indices(
        pos=pos, boxsize=boxsize, n_cells_per_dim=n_cells_per_dim, x_offset=x_offset
    )
    sort_order = np.argsort(flat_cell_idx)  # numpy's (SIMD) argsort is faster than numba's here
    flat_cell_idx = flat_cell_idx[sort_order]

    changes = np.flatnonzero(flat_cell_idx[1:] != flat_cell_idx[:-1]) + 1
    cell_offsets = np.empty(len(changes) + 2, dtype=np.int64)
    cell_offsets[0] = 0
    cell_offsets[1:-1] = changes
    cell_offsets[-1] = len(pos)
    unique_cells = flat_cell_idx[cell_offsets[:-1]]
    del flat_cell_idx

    sorted_pos = gather_wrapped_positions(pos=pos, order=sort_order, boxsize=boxsize)

    row_starts = np.arange(n_planes * n_cells_per_dim + 1, dtype=np.int64) * n_cells_per_dim
    row_offsets = np.searchsorted(unique_cells, row_starts)

    return sort_order, sorted_pos, cell_offsets, unique_cells, row_offsets


@njit(cache=True, parallel=True)
def gather_wrapped_positions(pos: np.ndarray, order: np.ndarray, boxsize: float) -> np.ndarray:
    """
    Equivalent to np.mod(pos[order], boxsize) but parallel and without the temporary; np.mod is slow on floats.
    """
    out = np.empty((len(order), 3), dtype=np.float64)

    for i in prange(len(order)):
        for d in range(3):
            x = pos[order[i], d]
            if x < 0.0 or x >= boxsize:
                x = x % boxsize
            out[i, d] = x

    return out


@njit(cache=True, parallel=True)
def compute_flat_cell_indices(pos: np.ndarray, boxsize: float, n_cells_per_dim: int, x_offset: int) -> np.ndarray:
    """
    Returns the row-major flat cell index of each particle on a periodic grid of n_cells_per_dim^3 cells, with the x
    cell coordinate counted from plane x_offset (wrapping).
    """
    n_particles = len(pos)
    flat_cell_idx = np.empty(n_particles, dtype=np.int64)

    for i in prange(n_particles):
        cx = _local_plane(pos[i, 0], boxsize, n_cells_per_dim, x_offset)
        cy = _periodic_cell_coordinate(pos[i, 1], boxsize, n_cells_per_dim)
        cz = _periodic_cell_coordinate(pos[i, 2], boxsize, n_cells_per_dim)
        flat_cell_idx[i] = (cx * n_cells_per_dim + cy) * n_cells_per_dim + cz

    return flat_cell_idx


@njit(cache=True, parallel=True)
def compute_cell_planes(pos: np.ndarray, boxsize: float, n_cells_per_dim: int) -> np.ndarray:
    """
    Returns the (global) x cell coordinate of each particle, i.e. which x-plane of cells it lies in.
    """
    planes = np.empty(len(pos), dtype=np.int64)

    for i in prange(len(pos)):
        planes[i] = _periodic_cell_coordinate(pos[i, 0], boxsize, n_cells_per_dim)

    return planes


@njit(cache=True, parallel=True)
def link_haloes(
    positions: np.ndarray,
    cell_offsets: np.ndarray,
    unique_cells: np.ndarray,
    row_offsets: np.ndarray,
    n_cells_per_dim: int,
    boxsize: float,
    linking_length: float,
    slab_bounds: np.ndarray,
    x_offset: int,
    periodic_x: bool,
) -> np.ndarray:
    """
    NOTE: positions must be sorted by cell (from build_periodic_cell_list, with the same x_offset).

    3D friends-of-friends over the cell grid; returns an array of root parent indices (in sorted order). slab_bounds
    span the grid's x-planes. If periodic_x, the grid is the whole box and links wrap in x; otherwise it is an MPI
    domain (plus ghost planes) and nothing is linked beyond its last plane. Links always wrap in y and z.
    """
    n_particles = len(positions)
    n_cells = len(unique_cells)
    parents = np.empty(n_particles, dtype=np.int64)
    rank = np.zeros(n_particles, dtype=np.int8)
    linking_length_sq = linking_length**2
    plane_size = n_cells_per_dim * n_cells_per_dim
    n_slabs = len(slab_bounds) - 1
    n_planes = slab_bounds[-1]

    # cells are internally connected by construction: root each at its first particle
    for k in prange(n_cells):
        start, end = cell_offsets[k], cell_offsets[k + 1]
        for i in range(start, end):
            parents[i] = start
        if end - start > 1:
            rank[start] = 1

    # each slab only links cells within itself, so slabs never write to each other's parents
    for s in prange(n_slabs):
        last_plane = slab_bounds[s + 1] - 1
        k_start = np.searchsorted(unique_cells, slab_bounds[s] * plane_size)
        k_end = np.searchsorted(unique_cells, slab_bounds[s + 1] * plane_size)

        for k in range(k_start, k_end):
            _link_cell(
                positions, parents, rank, cell_offsets, unique_cells, row_offsets, k, n_cells_per_dim, boxsize,
                linking_length_sq, last_plane, False, n_planes, x_offset, periodic_x,
            )

    # serial pass: links from each slab's last STENCIL_REACH planes into the next slab (wrapping periodically)
    for s in range(n_slabs):
        last_plane = slab_bounds[s + 1] - 1
        first_boundary_plane = max(slab_bounds[s], last_plane - STENCIL_REACH + 1)
        k_start = np.searchsorted(unique_cells, first_boundary_plane * plane_size)
        k_end = np.searchsorted(unique_cells, (last_plane + 1) * plane_size)

        for k in range(k_start, k_end):
            _link_cell(
                positions, parents, rank, cell_offsets, unique_cells, row_offsets, k, n_cells_per_dim, boxsize,
                linking_length_sq, last_plane, True, n_planes, x_offset, periodic_x,
            )

    for i in range(n_particles):
        parents[i] = find_root(parents, i)

    return parents


@njit(cache=True)
def _link_cell(
    positions: np.ndarray,
    parents: np.ndarray,
    rank: np.ndarray,
    cell_offsets: np.ndarray,
    unique_cells: np.ndarray,
    row_offsets: np.ndarray,
    k: int,
    n_cells_per_dim: int,
    boxsize: float,
    linking_length_sq: float,
    last_plane: int,
    crossing_slab: bool,
    n_planes: int,
    x_offset: int,
    periodic_x: bool,
) -> None:
    """
    Links occupied cell k to its half-stencil neighbours; mutates parents/rank in place. Only visits neighbours beyond
    last_plane (i.e. in the next slab) if crossing_slab, and only those within the slab otherwise.
    """
    cell_id = unique_cells[k]
    cx = cell_id // (n_cells_per_dim * n_cells_per_dim)
    cy = (cell_id // n_cells_per_dim) % n_cells_per_dim
    cz = cell_id % n_cells_per_dim
    start, end = cell_offsets[k], cell_offsets[k + 1]

    for r in range(len(HALF_ROWS)):
        if (cx + HALF_ROWS[r, 0] > last_plane) != crossing_slab:
            continue
        x = _neighbour_plane(cx, HALF_ROWS[r, 0], n_planes, periodic_x)
        if x < 0:
            continue  # beyond the domain's ghost planes

        row = x * n_cells_per_dim + (cy + HALF_ROWS[r, 1]) % n_cells_per_dim
        z_lo_0, z_hi_0, z_lo_1, z_hi_1 = _wrapped_window(cz + HALF_ROWS[r, 2], cz + STENCIL_REACH, n_cells_per_dim)

        for segment in range(2):
            z_lo, z_hi = (z_lo_0, z_hi_0) if segment == 0 else (z_lo_1, z_hi_1)
            k_lo, k_hi = _cells_in_row(unique_cells, row_offsets, row, z_lo, z_hi, n_cells_per_dim)

            for kn in range(k_lo, k_hi):
                neighbour_start = cell_offsets[kn]
                if find_root(parents, start) == find_root(parents, neighbour_start):
                    continue  # already in the same group, so nothing to check

                if _any_pair_within(
                    positions, start, end, cell_id, neighbour_start, cell_offsets[kn + 1], unique_cells[kn],
                    n_cells_per_dim, boxsize, linking_length_sq, x_offset,
                ):
                    union(parent=parents, rank=rank, idx_i=start, idx_j=neighbour_start)


@njit(cache=True)
def _neighbour_plane(cx: int, dx: int, n_planes: int, periodic_x: bool) -> int:
    """
    Returns the grid x-plane dx away from plane cx, or -1 if it lies outside a non-periodic grid.
    """
    x = cx + dx
    if periodic_x:
        return x % n_planes
    if x < 0 or x >= n_planes:
        return -1
    return x


@njit(cache=True)
def _wrapped_window(z_start: int, z_end: int, n_cells_per_dim: int) -> tuple[int, int, int, int]:
    """
    Splits the periodic z-window [z_start, z_end] into at most two in-range windows (lo_0, hi_0, lo_1, hi_1); an
    absent second window has hi_1 < lo_1.
    """
    if z_start < 0:
        return z_start + n_cells_per_dim, n_cells_per_dim - 1, 0, z_end
    if z_end >= n_cells_per_dim:
        return z_start, n_cells_per_dim - 1, 0, z_end - n_cells_per_dim

    return z_start, z_end, 0, -1


@njit(cache=True)
def _cells_in_row(
    unique_cells: np.ndarray, row_offsets: np.ndarray, row: int, z_lo: int, z_hi: int, n_cells_per_dim: int
) -> tuple[int, int]:
    """
    Returns the range [k_lo, k_hi) of occupied cells in (x, y) row with z in [z_lo, z_hi]; empty if z_hi < z_lo.
    """
    start, end = row_offsets[row], row_offsets[row + 1]
    if z_hi < z_lo or start == end:
        return start, start

    row_cells = unique_cells[start:end]  # short and contiguous, unlike a search over every occupied cell
    base = row * n_cells_per_dim

    return start + np.searchsorted(row_cells, base + z_lo), start + np.searchsorted(row_cells, base + z_hi + 1)


@njit(cache=True)
def _any_pair_within(
    positions: np.ndarray,
    a_start: int,
    a_end: int,
    a_cell: int,
    b_start: int,
    b_end: int,
    b_cell: int,
    n_cells_per_dim: int,
    boxsize: float,
    linking_length_sq: float,
    x_offset: int,
) -> bool:
    """
    Whether any particle of cell a is within the linking length of any particle of cell b; stops at the first. For
    dense cells, particles further than the linking length from the other cell's bounds are culled first, which turns
    the O(n_a * n_b) worst case (unlinked neighbours) into roughly O(n_a + n_b).
    """
    if (a_end - a_start) * (b_end - b_start) <= DENSE_PAIR_THRESHOLD:
        for i in range(a_start, a_end):
            for j in range(b_start, b_end):
                if _periodic_dist_sq_between(positions, i, positions, j, boxsize) <= linking_length_sq:
                    return True
        return False

    candidates_a = _particles_near_cell(
        positions, a_start, a_end, b_cell, n_cells_per_dim, boxsize, linking_length_sq, x_offset
    )
    if len(candidates_a) == 0:
        return False
    candidates_b = _particles_near_cell(
        positions, b_start, b_end, a_cell, n_cells_per_dim, boxsize, linking_length_sq, x_offset
    )

    for i in candidates_a:
        for j in candidates_b:
            if _periodic_dist_sq_between(positions, i, positions, j, boxsize) <= linking_length_sq:
                return True

    return False


@njit(cache=True)
def _particles_near_cell(
    positions: np.ndarray,
    start: int,
    end: int,
    cell_id: int,
    n_cells_per_dim: int,
    boxsize: float,
    linking_length_sq: float,
    x_offset: int,
) -> np.ndarray:
    """
    Returns the indices in [start, end) of particles within the linking length of the bounds of cell cell_id (whose x
    coordinate is counted from plane x_offset).
    """
    cell_size = boxsize / n_cells_per_dim
    half_cell, half_box = 0.5 * cell_size, 0.5 * boxsize
    centre = np.empty(3)
    centre[0] = ((cell_id // (n_cells_per_dim * n_cells_per_dim) + x_offset) % n_cells_per_dim + 0.5) * cell_size
    centre[1] = ((cell_id // n_cells_per_dim) % n_cells_per_dim + 0.5) * cell_size
    centre[2] = (cell_id % n_cells_per_dim + 0.5) * cell_size

    near = np.empty(end - start, dtype=np.int64)
    n_near = 0
    for i in range(start, end):
        gap_sq = 0.0
        for d in range(3):
            delta = abs(positions[i, d] - centre[d])
            if delta > half_box:
                delta = boxsize - delta
            gap = delta - half_cell
            if gap > 0.0:
                gap_sq += gap * gap
        if gap_sq <= linking_length_sq:
            near[n_near] = i
            n_near += 1

    return near[:n_near]


def label_groups(parents: np.ndarray, min_members: int, tiebreak: np.ndarray) -> np.ndarray:
    """
    Converts root parent indices to contiguous HaloIDs ordered by descending group size; groups with fewer than
    min_members particles are given the sentinel -1. Equal-sized groups are ordered by the smallest tiebreak value
    of their members (the original particle index), so the ordering does not depend on how the box was divided.
    """
    lookup = np.bincount(parents, minlength=len(parents))  # group sizes, indexed by root
    roots = np.flatnonzero(lookup >= max(min_members, 1))
    first_member = min_per_root(parents=parents, values=tiebreak, mask=np.ones(len(parents), dtype=np.bool_))
    roots = roots[np.lexsort((first_member[roots], -lookup[roots]))]  # largest first

    lookup[:] = -1  # reuse the buffer as the root -> HaloID map
    lookup[roots] = np.arange(len(roots), dtype=np.int64)

    return lookup[parents]


@njit(cache=True)
def min_per_root(parents: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Returns, indexed by root, the smallest value over each group's masked members (int64 max if it has none).
    """
    out = np.full(len(parents), np.iinfo(np.int64).max, dtype=np.int64)
    for i in range(len(parents)):
        if mask[i] and values[i] < out[parents[i]]:
            out[parents[i]] = values[i]

    return out


@njit(cache=True, parallel=True)
def attach_to_nearest_dm(
    positions: np.ndarray,
    dm_positions: np.ndarray,
    dm_labels: np.ndarray,
    cell_offsets: np.ndarray,
    unique_cells: np.ndarray,
    row_offsets: np.ndarray,
    n_cells_per_dim: int,
    boxsize: float,
    linking_length: float,
    x_offset: int,
    n_planes: int,
    periodic_x: bool,
) -> np.ndarray:
    """
    NOTE: dm_positions/dm_labels must be in the DM cell order (from build_periodic_cell_list, with the same x_offset
    and n_planes). On an MPI domain, positions must lie at least STENCIL_REACH planes inside the grid's x range.

    Returns the HaloID of each particle's nearest DM particle within the linking length, or -1 if there is none.
    """
    n_particles = len(positions)
    halo_ids = np.full(n_particles, -1, dtype=np.int64)
    linking_length_sq = linking_length**2
    cell_size_sq = (boxsize / n_cells_per_dim) ** 2
    for i in prange(n_particles):
        cx = _local_plane(positions[i, 0], boxsize, n_cells_per_dim, x_offset)
        cy = _periodic_cell_coordinate(positions[i, 1], boxsize, n_cells_per_dim)
        cz = _periodic_cell_coordinate(positions[i, 2], boxsize, n_cells_per_dim)

        # fast path: every DM particle in a cell shares a label, so if the particle's own cell holds DM (guaranteed
        # within the linking length) and every cell that could hold a closer one agrees, that is the answer
        label, is_uniform = _uniform_neighbour_label(
            dm_labels, cell_offsets, unique_cells, row_offsets, cx, cy, cz, n_cells_per_dim, linking_length_sq,
            cell_size_sq, n_planes, periodic_x,
        )
        if is_uniform:
            halo_ids[i] = label
            continue

        best_dist_sq = linking_length_sq
        best_label = -1

        for r in range(len(FULL_ROWS)):  # nearest-first, so stop once no closer particle is possible
            if FULL_ROW_GAPS_SQ[r] * cell_size_sq > best_dist_sq:
                break
            x = _neighbour_plane(cx, FULL_ROWS[r, 0], n_planes, periodic_x)
            if x < 0:
                continue

            row = x * n_cells_per_dim + (cy + FULL_ROWS[r, 1]) % n_cells_per_dim
            z_lo_0, z_hi_0, z_lo_1, z_hi_1 = _wrapped_window(cz - STENCIL_REACH, cz + STENCIL_REACH, n_cells_per_dim)

            for segment in range(2):
                z_lo, z_hi = (z_lo_0, z_hi_0) if segment == 0 else (z_lo_1, z_hi_1)
                k_lo, k_hi = _cells_in_row(unique_cells, row_offsets, row, z_lo, z_hi, n_cells_per_dim)

                for kn in range(k_lo, k_hi):
                    z_gap = _periodic_cell_gap(unique_cells[kn] % n_cells_per_dim, cz, n_cells_per_dim)
                    if (FULL_ROW_GAPS_SQ[r] + z_gap * z_gap) * cell_size_sq > best_dist_sq:
                        continue  # no particle in this cell can be closer

                    for j in range(cell_offsets[kn], cell_offsets[kn + 1]):
                        dist_sq = _periodic_dist_sq_between(positions, i, dm_positions, j, boxsize)
                        if dist_sq <= best_dist_sq:
                            best_dist_sq = dist_sq
                            best_label = dm_labels[j]

        halo_ids[i] = best_label

    return halo_ids


@njit(cache=True)
def _uniform_neighbour_label(
    dm_labels: np.ndarray,
    cell_offsets: np.ndarray,
    unique_cells: np.ndarray,
    row_offsets: np.ndarray,
    cx: int,
    cy: int,
    cz: int,
    n_cells_per_dim: int,
    linking_length_sq: float,
    cell_size_sq: float,
    n_planes: int,
    periodic_x: bool,
) -> tuple[int, bool]:
    """
    Returns (label, True) if cell (cx, cy, cz) holds DM and every occupied cell within the linking length of it has the
    same label; otherwise (-1, False), and the exact nearest-neighbour search is needed.
    """
    own_row = cx * n_cells_per_dim + cy
    k_lo, k_hi = _cells_in_row(unique_cells, row_offsets, own_row, cz, cz, n_cells_per_dim)
    if k_lo == k_hi:
        return -1, False  # own cell empty, so there may be no DM within the linking length at all

    label = dm_labels[cell_offsets[k_lo]]

    for r in range(len(FULL_ROWS)):
        if FULL_ROW_GAPS_SQ[r] * cell_size_sq > linking_length_sq:
            break
        x = _neighbour_plane(cx, FULL_ROWS[r, 0], n_planes, periodic_x)
        if x < 0:
            continue

        row = x * n_cells_per_dim + (cy + FULL_ROWS[r, 1]) % n_cells_per_dim
        z_lo_0, z_hi_0, z_lo_1, z_hi_1 = _wrapped_window(cz - STENCIL_REACH, cz + STENCIL_REACH, n_cells_per_dim)

        for segment in range(2):
            z_lo, z_hi = (z_lo_0, z_hi_0) if segment == 0 else (z_lo_1, z_hi_1)
            k_lo, k_hi = _cells_in_row(unique_cells, row_offsets, row, z_lo, z_hi, n_cells_per_dim)

            for kn in range(k_lo, k_hi):
                z_gap = _periodic_cell_gap(unique_cells[kn] % n_cells_per_dim, cz, n_cells_per_dim)
                if (FULL_ROW_GAPS_SQ[r] + z_gap * z_gap) * cell_size_sq > linking_length_sq:
                    continue  # too far to hold a DM particle within the linking length
                if dm_labels[cell_offsets[kn]] != label:
                    return -1, False

    return label, True


@njit(cache=True)
def _periodic_cell_gap(c: int, c_ref: int, n_cells_per_dim: int) -> int:
    """
    The number of whole cells separating cell coordinates c and c_ref along a periodic axis (0 if adjacent).
    """
    dc = abs(c - c_ref)
    dc = min(dc, n_cells_per_dim - dc)
    return max(dc - 1, 0)


@njit(cache=True)
def _periodic_cell_coordinate(x: float, boxsize: float, n_cells_per_dim: int) -> int:
    """
    Wraps a coordinate into the box and returns its cell coordinate along that axis.
    """
    c = int((x % boxsize) * n_cells_per_dim / boxsize)
    return min(c, n_cells_per_dim - 1)  # guards float rounding at x -> boxsize


@njit(cache=True)
def _local_plane(x: float, boxsize: float, n_cells_per_dim: int, x_offset: int) -> int:
    """
    Returns the x cell coordinate of a position on a grid whose first plane is global plane x_offset (wrapping).
    """
    return (_periodic_cell_coordinate(x, boxsize, n_cells_per_dim) - x_offset) % n_cells_per_dim


@njit(cache=True)
def _periodic_dist_sq_between(pos_a: np.ndarray, i: int, pos_b: np.ndarray, j: int, boxsize: float) -> float:
    """
    Minimum-image squared distance between pos_a[i] and pos_b[j].
    """
    half_box = 0.5 * boxsize
    dist_sq = 0.0
    for d in range(3):
        delta = abs(pos_a[i, d] - pos_b[j, d])
        if delta > half_box:
            delta = boxsize - delta
        dist_sq += delta * delta

    return dist_sq
