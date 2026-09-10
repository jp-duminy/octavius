"""

Particle-group mapping is represented throughout the codebase in compressed sparse-row (CSR)
format, i.e. you store a list of sorted indices into a ParticleStore, and a corresponding lists
of offsets where offsets[g] is the start of a group, meaning slicing the sorted idx with offsets[g]:offsets[g+1]
will recover the indices of the particles in group g in the ParticleStore.

This is runtime and memory efficient, but also a little tricky to build and get the hang of working with because
you need to keep careful track of your indices.

This file contains the utility functions for working with our membership format.

"""

# other packages
import numpy as np
from numba import njit


def build_group_csr(group_idx: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple of (offsets, sorted_indices) into the group_idx array.

    group_idx must be raw and full-length (i.e. no sentinel value mask applied).
    """
    in_group = group_idx >= 0
    masked_idx = group_idx[in_group]
    counts = np.bincount(masked_idx, minlength=n_groups)  # doesn't accept negative integers
    offsets = np.empty(n_groups + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)

    global_indices = in_group.nonzero()[
        0
    ]  # reindexing needs to happen on the original group_idx order / .nonzero() returns a tuple
    sorted_indices = global_indices[np.argsort(masked_idx, stable=True)]  # .nonzero() returns a tuple (array is first)

    return offsets, sorted_indices


@njit(cache=True)
def propagate_membership_csr(
    offsets: np.ndarray,
    sorted_indices: np.ndarray,
    parent_ids: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses the parent information to propagate membership from a subhalo into its parent, for inclusive properties. Returns a tuple of (inclusive_offsets, inclusive_sorted_indices) which allows a parent subhalo to access its entire membership.
    """
    inclusive_counts = np.empty(n_groups, dtype=np.int64)
    for g in range(n_groups):
        inclusive_counts[g] = offsets[g + 1] - offsets[g]

    # the loops which follow assume parents appear before their children, so ensure this is the case
    depth = np.zeros(n_groups, dtype=np.int64)
    for g in range(n_groups):  # simple loop to recompute depth (easier than passing it)
        current = g
        d = 0
        while parent_ids[current] >= 0:
            current = parent_ids[current]
            d += 1
        depth[g] = d

    reverse_order = np.argsort(depth)[::-1]  # TODO: descending=True in numpy 2.5.0

    for g in reverse_order:  # add child counts to parents
        if parent_ids[g] >= 0:
            inclusive_counts[parent_ids[g]] += inclusive_counts[g]

    inclusive_offsets = np.empty(n_groups + 1, dtype=np.int64)
    inclusive_offsets[0] = 0
    inclusive_offsets[1:] = np.cumsum(inclusive_counts)

    inclusive_sorted_idx = np.empty(
        inclusive_offsets[-1], dtype=sorted_indices.dtype
    )  # contains slots for all particle members
    write_positions = inclusive_offsets[:-1].copy()  # need a running write_positions array for the propagation

    for g in range(
        n_groups
    ):  # loop over groups to place per-group exclusive idx into the total inclusive membership idx array
        start = offsets[g]
        end = offsets[g + 1]
        n_exclusive = end - start
        inclusive_sorted_idx[write_positions[g] : write_positions[g] + n_exclusive] = sorted_indices[start:end]
        write_positions[g] += n_exclusive

    for g in (
        reverse_order
    ):  # now loop again to place the child inclusive idx into the parents' positions in the inclusive array
        parent = parent_ids[g]  # moved this here to reduce verbosity of the fancy indexing
        if parent >= 0:
            start = inclusive_offsets[g]
            end = inclusive_offsets[g + 1]
            n_inclusive = end - start
            inclusive_sorted_idx[write_positions[parent] : write_positions[parent] + n_inclusive] = (
                inclusive_sorted_idx[start:end]
            )
            write_positions[parent] += n_inclusive

    return inclusive_offsets, inclusive_sorted_idx
