"""

Octavius 6D friends-of-friends galaxy-finding algorithm which utilises cell linked-lists,
an algorithm originally used in molecular dynamics simulation codes to avoid naïve O(N^2)
checks in groups by partitioning the data into cells with a cutoff radius beyond which particles
are not checked. This is conceptually similar to a k-d tree, however, the difference comes down
to how data is partitioned. k-d trees partition space by data density, which means they are optimal
for adaptive queries. For example, consider wanting to find galaxy local number densities. If this
galaxy is in a cosmic void, its neighbours might be megaparsecs away, whereas in a node it could
be kiloparsecs. The k-d tree adapts to either case as its density-based leaf structure allows
it to traverse voids or zoom in on nodes.

Friends-of-friends is (in our case) based on a fixed linking length. Cell linked-lists are
based on fixed cell sizes; meaning partitioning is spatially-based rather than density-based,
which is therefore more suited to fixed distance queries. We specifically store the linked
list in CSR format.

The previous Octavius FOF6D framework used the scipy.spatial and scipy.sparse framework,
vectorising the galaxy-finding approach with k-d tree -> sparse distance matrix -> sparse adjacency matrix
-> connected components. However, this approach involves O(E) scaling where E is Nk and k
is the average number of nearest neighbours. Therefore, in highly dense regions where k tends
to N, the resulting O(N^2) scaling became untenable in memory. This could also occur for
moderately-dense haloes with enormous number counts. By writing this new algorithm in numba
we use a union find to construct the connected components, thereby avoiding holding all edges
in memory simultaneously.

Original FoF: Davis et al. 1985, doi: 10.1086/163168

"""

# workhorses
from numba import (
    njit,
    prange,
)  # NOTE: do not attach parallel=True to the individual algorithm functions, only the executor!
import numpy as np


@njit(cache=True, parallel=True)
def dispatch_fof6d(
    positions: np.ndarray,
    velocities: np.ndarray,
    parents: np.ndarray,
    ptype_codes: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    linking_length: float,
    velocity_factor: float,
    minstars: int,
    star_ptype_code: np.int8,  # explicit data type for numba
) -> None:
    """
    Parallelises the galaxy finding by dispatching haloes to different cores; modifies the input parents array in place.
    """
    n_haloes = len(starts)

    neighbour_offsets = np.array(  # for symmetric vel criterion
        [(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2) if (dx, dy, dz) > (0, 0, 0)],
        dtype=np.int64,
    )

    for halo_idx in prange(n_haloes):
        s, e = starts[halo_idx], ends[halo_idx]
        n_stars = np.sum(ptype_codes[s:e] == star_ptype_code)

        if n_stars < minstars:
            continue

        halo_pos = positions[s:e]
        halo_vel = velocities[s:e]

        sort_order, cell_ids_sorted, cell_offsets, unique_cells, grid_dims = construct_sparse_cell_linked_list(
            pos=halo_pos, linking_length=linking_length
        )

        sorted_pos = halo_pos[sort_order]
        sorted_vel = halo_vel[sort_order]

        sigmas = compute_local_velocity_dispersions(
            positions=sorted_pos,
            velocities=sorted_vel,
            cell_ids_sorted=cell_ids_sorted,
            cell_offsets=cell_offsets,
            unique_cells=unique_cells,
            grid_dims=grid_dims,
            linking_length=linking_length,
        )

        parents_sorted = link_particles(
            positions=sorted_pos,
            velocities=sorted_vel,
            sigmas=sigmas,
            neighbour_offsets=neighbour_offsets,
            cell_ids_sorted=cell_ids_sorted,
            cell_offsets=cell_offsets,
            unique_cells=unique_cells,
            grid_dims=grid_dims,
            linking_length=linking_length,
            velocity_factor=velocity_factor,
        )

        for i in range(e - s):
            parents[s + sort_order[i]] = sort_order[parents_sorted[i]]


@njit(cache=True)
def construct_sparse_cell_linked_list(
    pos: np.ndarray, linking_length: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions particles in haloes into a (sparse) cell linked-list. Returns a tuple of arrays:

    - sort_order: the indices to sort particles by cell (np.argsort)
    - cell_ids_sorted: sorted per-particle corresponding flat cell idx
    - cell_offsets: where each cell begins in the flat sorted array (classic csr offset)
    - unique_cells: the flat cell idx of each cell
    - grid_dims: the extent of the grid in each direction

    The reason we want this to be sparse is because, while a cell linked-list is more
    naturally suited to the geometry of FOF, there must still be some consideration of
    density; the naïve implementation is sensitive to lone particles at the edges of
    haloes which can affect the entire grid structure.
    """
    n_particles = len(pos)
    pos_min, pos_max = _find_min_max(array=pos)
    inv_linking_length = 1.0 / linking_length  # optimisation: multiplication by inv faster than division

    grid_origin = pos_min - linking_length
    grid_extent = (
        pos_max - pos_min
    ) + 2.0 * linking_length  # this is floor divided; add two linking lengths (top/bottom) to encapsulate all particles
    grid_dims = np.zeros(3, dtype=np.int64)

    for d in range(3):  # NOTE: profiling says np.floor is unideal, so loop over axes
        grid_dims[d] = np.int64(grid_extent[d] * inv_linking_length) + 1  # +1 so if extent is 0, grid_dims is still 1

    flat_cell_idx = np.empty(shape=n_particles, dtype=np.int64)

    # allocate these here so we don't do lookups in the loop
    origin_x, origin_y, origin_z = grid_origin[0], grid_origin[1], grid_origin[2]
    max_x, max_y, max_z = grid_dims[0] - 1, grid_dims[1] - 1, grid_dims[2] - 1

    for i in range(n_particles):
        # this follows numpy/C convention, see https://numpy.org/devdocs/dev/internals.html "Multidimensional array indexing order issues"
        cell_x = max(0, min(int((pos[i, 0] - origin_x) * inv_linking_length), max_x))  # clips to grid dims
        cell_y = max(0, min(int((pos[i, 1] - origin_y) * inv_linking_length), max_y))
        cell_z = max(0, min(int((pos[i, 2] - origin_z) * inv_linking_length), max_z))

        flat_cell_idx[i] = _get_cell_index(cx=cell_x, cy=cell_y, cz=cell_z, grid_dims=grid_dims)

    sort_order = np.argsort(flat_cell_idx, kind="quicksort")  # do not need stable sort (mergesort) for this one

    cell_ids_sorted = flat_cell_idx[sort_order]
    changes = np.flatnonzero(np.diff(cell_ids_sorted)) + 1
    n_cells = len(changes) + 1

    cell_offsets = np.empty(shape=(n_cells + 1), dtype=np.int64)
    cell_offsets[0] = 0
    cell_offsets[1:-1] = changes
    cell_offsets[-1] = n_particles

    unique_cells = cell_ids_sorted[cell_offsets[:-1]]

    return sort_order, cell_ids_sorted, cell_offsets, unique_cells, grid_dims


@njit(cache=True)
def link_particles(
    positions: np.ndarray,
    velocities: np.ndarray,
    sigmas: np.ndarray,
    neighbour_offsets: np.ndarray,
    cell_ids_sorted: np.ndarray,
    cell_offsets: np.ndarray,
    unique_cells: np.ndarray,
    grid_dims: np.ndarray,
    linking_length: float,
    velocity_factor: float,
) -> np.ndarray:
    """
    Links particles via friends-of-friends criteria in 6D phase space; returns an array of parent IDs.
    """
    n_particles = len(positions)
    linking_length_sq = linking_length**2  # we avoid calls to np.sqrt in the loop by doing comparison in squared space
    parents = np.arange(n_particles, dtype=np.int32)
    rank = np.zeros(n_particles, dtype=np.int32)

    scaled_sigmas_sq = (sigmas * velocity_factor) ** 2  # optimisation: precompute these then index

    for i in range(n_particles):  # NOTE: see velocity dispersion function for more comments (duplicated logic)
        cell_id = cell_ids_sorted[i]
        cx = cell_id // (grid_dims[1] * grid_dims[2])  # inverse of the formula in the list construction function
        cy = (cell_id // grid_dims[2]) % grid_dims[1]
        cz = cell_id % grid_dims[2]

        self_k = np.searchsorted(unique_cells, cell_id)  # linking within particle's cell

        for j in range(cell_offsets[self_k], cell_offsets[self_k + 1]):
            if j <= i:
                continue

            _check_link(
                positions=positions,
                velocities=velocities,
                scaled_sigmas_sq=scaled_sigmas_sq,
                parents=parents,
                rank=rank,
                i=i,
                j=j,
                linking_length_sq=linking_length_sq,
            )

        for s in range(len(neighbour_offsets)):  # linking within neighbouring cells
            dx = neighbour_offsets[s, 0]
            dy = neighbour_offsets[s, 1]
            dz = neighbour_offsets[s, 2]

            nx, ny, nz = cx + dx, cy + dy, cz + dz

            if nx < 0 or nx >= grid_dims[0] or ny < 0 or ny >= grid_dims[1] or nz < 0 or nz >= grid_dims[2]:
                continue  # don't need to check neighbouring cells if we are at the edge

            neighbour_flat_idx = _get_cell_index(cx=nx, cy=ny, cz=nz, grid_dims=grid_dims)
            k = np.searchsorted(unique_cells, neighbour_flat_idx)  # only half of the offsets (comparison is symmetric)

            if k < len(unique_cells) and unique_cells[k] == neighbour_flat_idx:
                for j in range(cell_offsets[k], cell_offsets[k + 1]):
                    _check_link(
                        positions=positions,
                        velocities=velocities,
                        scaled_sigmas_sq=scaled_sigmas_sq,
                        parents=parents,
                        rank=rank,
                        i=i,
                        j=j,
                        linking_length_sq=linking_length_sq,
                    )

    for i in range(n_particles):
        parents[i] = find_root(parents, i)

    return parents


@njit(cache=True)
def _check_link(
    positions: np.ndarray,
    velocities: np.ndarray,
    scaled_sigmas_sq: np.ndarray,
    parents: np.ndarray,
    rank: np.ndarray,
    i: int,
    j: int,
    linking_length_sq: float,
) -> None:
    """
    Performs the position and velocity-space checks; mutates parents/rank in place.
    """
    pos_dx = positions[i, 0] - positions[j, 0]
    pos_dy = positions[i, 1] - positions[j, 1]
    pos_dz = positions[i, 2] - positions[j, 2]

    r_sq = pos_dx**2 + pos_dy**2 + pos_dz**2

    if r_sq <= linking_length_sq:  # don't extend to phase space for particles not linked in position space
        vel_dx = velocities[i, 0] - velocities[j, 0]
        vel_dy = velocities[i, 1] - velocities[j, 1]
        vel_dz = velocities[i, 2] - velocities[j, 2]

        dv_sq = vel_dx**2 + vel_dy**2 + vel_dz**2

        # NOTE: max() treats the graph as undirected, whereas min() requires it be strongly-connected
        if dv_sq <= max(scaled_sigmas_sq[i], scaled_sigmas_sq[j]):
            union(parent=parents, rank=rank, idx_i=i, idx_j=j)


@njit(cache=True)
def compute_local_velocity_dispersions(
    positions: np.ndarray,
    velocities: np.ndarray,
    cell_ids_sorted: np.ndarray,
    cell_offsets: np.ndarray,
    unique_cells: np.ndarray,
    grid_dims: np.ndarray,
    linking_length: float,
) -> np.ndarray:
    """
    NOTE: positions/velocities must be ordered with pos/vel[sort_order] (returned from list construction).

    Computes kernel-weighted local velocity dispersions (sigma) for each particle. Returns an array of sigma.
    """
    n_particles = len(positions)
    sigmas = np.zeros(shape=n_particles)
    linking_length_sq = linking_length**2  # we avoid calls to np.sqrt in the loop by doing comparison in squared space

    for i in range(n_particles):
        cell_id = cell_ids_sorted[i]
        cx = cell_id // (grid_dims[1] * grid_dims[2])  # inverse of the formula in the list construction function
        cy = (cell_id // grid_dims[2]) % grid_dims[1]
        cz = cell_id % grid_dims[2]

        weight_sum = 0.0
        weighted_dv_sq_sum = 0.0

        for dx in range(-1, 2):  # by construction, each particle's neighbours are within the nearest 3^3 cells
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    nx, ny, nz = cx + dx, cy + dy, cz + dz

                    if nx < 0 or nx >= grid_dims[0] or ny < 0 or ny >= grid_dims[1] or nz < 0 or nz >= grid_dims[2]:
                        continue  # don't need to check neighbouring cells if we are at the edge

                    neighbour_cell_idx = _get_cell_index(cx=nx, cy=ny, cz=nz, grid_dims=grid_dims)
                    k = np.searchsorted(
                        unique_cells, neighbour_cell_idx
                    )  # insertion index lets us check whether cell is empty

                    if (
                        k < len(unique_cells) and unique_cells[k] == neighbour_cell_idx
                    ):  # ^ as then unique cell id = insertion index
                        for j in range(cell_offsets[k], cell_offsets[k + 1]):
                            if i == j:
                                continue  # particle has 0 distance to itself so would erroneously contribute 1 to its weight

                            pos_dx = (
                                positions[i, 0] - positions[j, 0]
                            )  # NOTE: np.linalg.norm will allocate temporary arrays here, keep it manual
                            pos_dy = positions[i, 1] - positions[j, 1]
                            pos_dz = positions[i, 2] - positions[j, 2]

                            r_sq = pos_dx**2 + pos_dy**2 + pos_dz**2

                            if r_sq > linking_length_sq:
                                continue  # weight is zero beyond the linking length

                            weight = cubic_spline_kernel(r=np.sqrt(r_sq), linking_length=linking_length)

                            vel_dx = velocities[i, 0] - velocities[j, 0]
                            vel_dy = velocities[i, 1] - velocities[j, 1]
                            vel_dz = velocities[i, 2] - velocities[j, 2]

                            dv_sq = vel_dx**2 + vel_dy**2 + vel_dz**2
                            weighted_dv_sq = weight * dv_sq

                            weight_sum += weight
                            weighted_dv_sq_sum += weighted_dv_sq

        if weight_sum > 0.0:  # guard, weights are already zero by construction if weight_sum = 0.0
            sigmas[i] = np.sqrt(weighted_dv_sq_sum / weight_sum)

    return sigmas


@njit(cache=True)
def cubic_spline_kernel(r: float, linking_length: float) -> float:
    """
    Evaluates the one-dimensional SPH cubic spline kernel, returning the
    weight W; necessary for the velocity dispersion.

    (JJ Monaghan 1992, doi: 10.1146/annurev.aa.30.090192.002551)
    """
    q = 2.0 * r / linking_length  # kernel is defined over interval [2, 0]

    if q >= 2.0:
        return 0.0

    elif q >= 1.0:
        return 0.25 * (2.0 - q) ** 3

    else:
        return 1.0 - (1.5 * q**2) + (0.75 * q**3)


@njit(cache=True)
def find_root(parent: np.ndarray, idx: int) -> int:
    """
    One-pass path-compressed root find; returns the parent ID.

    Tarjan, van Leeuwen (1984), doi: 10.1145/62.2160
    """
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]

    return idx


@njit(cache=True)
def union(parent: np.ndarray, rank: np.ndarray, idx_i: int, idx_j: int) -> None:
    """
    Union by rank, implemented from textbook pseudocode; mutates rank array in place.

    Introduction to Algorithms (third ed), MIT; ISBN 978-0-262-03384-8
    """
    root_i = find_root(parent=parent, idx=idx_i)
    root_j = find_root(parent=parent, idx=idx_j)

    if root_i == root_j:
        return

    if rank[root_i] < rank[root_j]:  # root_i must always be the higher rank
        root_i, root_j = root_j, root_i

    parent[root_j] = root_i

    if rank[root_i] == rank[root_j]:
        rank[root_i] += 1  # promote


@njit(cache=True)
def _get_cell_index(cx: int, cy: int, cz: int, grid_dims: np.ndarray) -> int:
    """
    Uses the row-major formula to return the flattened 3D cell index.
    """
    return cx * grid_dims[1] * grid_dims[2] + cy * grid_dims[2] + cz


@njit(cache=True)
def _find_min_max(array: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    The equivalent of np.min/max on an (n, 3) array with arg axis=0, which
    (as of 02/07/26) is not supported in numba. Returns two length-3 arrays:

    - min_vals: the minimum value along x, y, z axes
    - max_vals: the maximum value along x, y, z axes
    """
    min_vals = np.empty(3)  # NOTE: use _vals suffix to avoid conflicting with python's min()/max()
    max_vals = np.empty(3)

    for d in range(3):
        min_vals[d] = array[0, d]
        max_vals[d] = array[0, d]

    for i in range(1, len(array)):
        for d in range(3):
            if array[i, d] < min_vals[d]:
                min_vals[d] = array[i, d]

            if array[i, d] > max_vals[d]:
                max_vals[d] = array[i, d]

    return min_vals, max_vals
