"""

Functions to collate per-rank analysis data via MPI and create a final, globally-ordered
catalogue; it is important the global order does not change between serial/parallel runs, to
ensure the output is deterministic.

In practice this becomes really quite finicky, mainly due to the indices dataset, which is
particle-level (so gigabyte-order). Since int64 was only supported in MPI-4 for the operations
which would allow this to be safely done, we have the group-level data get broadcast normally,
then rank 0 calculates where each rank should write its indices output to, then ranks take turns
sequentially writing their indices to the output file.

That said, the functions here get really messy. They work but writing this caused a lot of grief
and if there is a better way to do this, it would be much appreciated.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_management import Internals
    from .pack_catalogue_data import RankPackedData, MembershipArrays
    from mpi4py.MPI import Comm

# defaults
from pathlib import Path
from dataclasses import dataclass

# others
import h5py
import numpy as np

# internal imports
from .conventions import DTYPES
from .parallel_reading import RedistributionMap, redistribute_data
from ..data_management.csr import build_group_csr, propagate_membership_csr
from ..log import get_logger

logger = get_logger()

SORT_COLUMN_BY_KIND: dict[str, str] = {  # this is necessitated by haloes and galaxies having different total mass keys
    "halo": "mass_total",
    "galaxy": "mass_baryon",
}
HALO_POINTER_COLUMNS: frozenset[str] = frozenset(
    {"field_halo_index", "parent_halo_index"}
)  # for hierarchical membership
FILL_VALUES: dict[str, int] = {
    "parent_halo_index": -1,
    "depth": 0,
}  # this is so data doesn't get sent through the pipeline if it doesn't exist (but we still need corresponding sentinel values for the tree pointers)
POST_SORT_COLUMNS: dict[str, frozenset[str]] = {
    "haloes": frozenset({"central_galaxy_index", "field_halo_index"}),
    "galaxies": frozenset({"central_galaxy_index"}),
}


@dataclass(slots=True)
class UnpackedRankData:
    """
    Group-level membership and physics data arrays unpacked from the PackedRankData class.

    - sort_arrays: the array (total mass, but with specific halo/data key for total/baryon) to sort groups by
    - membership_chunks: per-rank membership data
    - group_ids: all group IDs
    - physics_chunks: per-rank physics data

    Key by HDF5 output group name (likewise for columns contained therein).
    """

    sort_arrays: dict[str, list[np.ndarray]]
    membership_chunks: dict[str, dict[str, list[np.ndarray]]]
    group_ids: dict[str, list[np.ndarray]]
    physics_chunks: dict[str, dict[str, list[np.ndarray]]]


def write_catalogue(
    packed_data: RankPackedData,
    catalogue_path: Path,
    internals: Internals,
    comm: Comm | None,
) -> None:
    """
    Writes the Octavius output catalogue.

    - Gathers group-length data to rank 0
    - Rank 0 creates catalogue and writes group data to it
    - Rank 0 computes global ordering and per-rank HDF5 write positions for particle membership data
    - Each rank receives its allocated space in the HDF5 file and writes its share of the global membership data

    MPI-native, and handles serial runs with comm=None too; produces catalogues which have consistent serial/parallel global ordering.
    """
    rank = (
        comm.Get_rank() if comm is not None else 0
    )  # duplicates what's in run_octavius.py for conftest.py to work too
    n_ranks = comm.Get_size() if comm is not None else 1

    lightweight_data = packed_data.without_indices()  # indices array is sizen n_particles (others are n_groups)

    all_lightweight: list[RankPackedData] = (
        comm.gather(lightweight_data, root=0) if comm is not None else [lightweight_data]
    )  # comm.gather is safe on the smaller datasets

    if rank == 0:
        logger.info(f"Synthesising analysis data from {len(all_lightweight)} ranks into output catalogue.")
        unpacked = unpack_data(all_rank_data=all_lightweight, internals=internals)
        sort_orders, resolved = resolve_global_ordering(unpacked=unpacked)

        write_group_data_to_catalogue(
            output_path=catalogue_path,
            sort_orders=sort_orders,
            resolved=resolved,
            physics_chunks=unpacked.physics_chunks,
            internals=internals,
        )

        per_rank_write_positions: dict[tuple[str, str], list[np.ndarray]] = {}

        for group_type in internals.group_types:
            if (
                group_type not in sort_orders
            ):  # guard against absent groups (using sort_orders here works since it is built on what exists)
                continue

            group_params = internals.group_types[group_type]
            hdf5_name = group_params["hdf5_group"]

            for ptype in group_params["ptypes"]:
                key = (hdf5_name, ptype)
                rank_lengths = []

                for rank_data in (
                    all_lightweight
                ):  # there is a fair bit of nesting in the dataclass here but hopefully this reads okay
                    if hdf5_name in rank_data.groups and ptype in rank_data.groups[hdf5_name].particle_lists:
                        rank_lengths.append(
                            np.diff(rank_data.groups[hdf5_name].particle_lists[ptype].offsets)
                        )  # diff(offsets) gives (n-1) lengths array
                    else:
                        rank_lengths.append(
                            np.empty(0, dtype=DTYPES["csr_lengths"])
                        )  # if data isn't on a rank (think high-z snaps) the ordering must be preserved

                flat_lengths = np.concatenate(rank_lengths)
                order = sort_orders[group_type]
                sorted_lengths = flat_lengths[order]
                sorted_offsets = np.empty(len(sorted_lengths) + 1, dtype=DTYPES["csr_offsets"])
                sorted_offsets[0] = 0
                np.cumsum(
                    sorted_lengths, out=sorted_offsets[1:]
                )  # doing np.concat with dtype arg doesn't work for the prepended 0 list

                inverse_order = np.argsort(order, stable=True)
                write_displacements = sorted_offsets[:-1][
                    inverse_order
                ]  # inverse_order[global_index] = sorted_position
                nonzero_mask = sorted_lengths > 0
                nonzero_displacements = sorted_offsets[:-1][nonzero_mask]
                unique_count = len(np.unique(nonzero_displacements))
                assert unique_count == len(nonzero_displacements), (
                    f"{key}: {len(nonzero_displacements) - unique_count} duplicate displacements among nonzero-length groups"
                )

                rank_group_counts = [len(rl) for rl in rank_lengths]
                per_rank_write_positions[key] = np.split(
                    write_displacements, np.cumsum(rank_group_counts[:-1])
                )  # gives per-rank boundaries

                with h5py.File(catalogue_path, "a") as f:  # use "a" not "w"  (group data written above ^)
                    membership_grp = f[hdf5_name]["membership"]

                    # write particle membership lists with data here
                    ds = membership_grp.create_dataset(f"{ptype}_offsets", data=sorted_offsets, compression=1)
                    ds.attrs["description"] = (
                        f"Per-group offsets into {ptype}_indices, where indices[offsets[g]:offsets[g+1]] recovers particles in group g."
                    )

                    ds = membership_grp.create_dataset(f"{ptype}_lengths", data=sorted_lengths, compression=1)
                    ds.attrs["description"] = (
                        f"Per-group lengths in {ptype}_indices; lengths[g] is the number of particles in the group."
                    )

                    ds = membership_grp.create_dataset(
                        f"{ptype}_indices", shape=(sorted_offsets[-1],), dtype=DTYPES["csr_indices"]
                    )  # indices initialised to empty
                    ds.attrs["description"] = (
                        f"Particle-level snapshot indices for {ptype}; for particle p, raw_snapshot_dataset[p], recovers the value of p in the raw snapshot dataset columns."
                    )

    rank_write_positions: dict[tuple[str, str], np.ndarray] = {}
    rank_membership_data: dict[tuple[str, str], MembershipArrays] = {}
    present_groups = set(sort_orders.keys()) if rank == 0 else set()  # for guarding against absent groups
    if comm is not None:
        present_groups = comm.bcast(present_groups, root=0)

    # each rank is given its HDF5 file displacements so it knows where to write the indices array to
    for group_type in internals.group_types:
        if group_type not in present_groups:  # guard
            continue

        group_params = internals.group_types[group_type]
        hdf5_name = group_params["hdf5_group"]

        for ptype in group_params["ptypes"]:
            key = (hdf5_name, ptype)
            send_data = per_rank_write_positions[key] if rank == 0 else None
            rank_write_positions[key] = comm.scatter(send_data, root=0) if comm is not None else send_data[0]

            if hdf5_name in packed_data.groups and ptype in packed_data.groups[hdf5_name].particle_lists:
                rank_membership_data[key] = packed_data.groups[hdf5_name].particle_lists[ptype]

    # writing group-by-group means one HDF5 call per group per ptype, so instead the indices are exchanged so that
    # each rank owns one contiguous block of every indices dataset, which it then writes in a single call
    rank_blocks: dict[tuple[str, str], tuple[int, np.ndarray]] = {}
    for key in rank_write_positions:  # same insertion order on every rank
        csr = rank_membership_data.get(key)
        rank_blocks[key] = gather_contiguous_block(
            displacements=rank_write_positions[key],
            offsets=csr.offsets if csr is not None else np.zeros(1, dtype=DTYPES["csr_offsets"]),
            indices=csr.indices if csr is not None else np.empty(0, dtype=DTYPES["csr_indices"]),
            comm=comm,
        )

    # ranks take turns writing their block of each indices array (globally-ordered) to the output catalogue
    for writer_rank in range(n_ranks):
        if rank == writer_rank:  # == writer_rank so ranks don't write to the file at the same time
            with h5py.File(catalogue_path, "a") as f:  # also use "a" not "w"!
                for (hdf5_name, ptype), (block_start, block) in rank_blocks.items():
                    if len(block) > 0:
                        f[hdf5_name][f"membership/{ptype}_indices"][block_start : block_start + len(block)] = block
        if comm is not None:
            comm.Barrier()  # remember to Barrier() because each rank needs to finish its write first


def gather_contiguous_block(
    displacements: np.ndarray,
    offsets: np.ndarray,
    indices: np.ndarray,
    comm: Comm | None,
) -> tuple[int, np.ndarray]:
    """
    Sends each of a rank's membership indices to the rank owning its position in the output dataset, where the
    dataset is split into n_ranks near-equal contiguous blocks. Returns:

    - block_start: the dataset position of this rank's block
    - block: this rank's block of the indices dataset, in file order
    """
    lengths = np.diff(offsets)
    positions = np.repeat(displacements - offsets[:-1], lengths) + np.arange(len(indices), dtype=np.int64)

    n_ranks = comm.Get_size() if comm is not None else 1
    rank = comm.Get_rank() if comm is not None else 0
    n_total = comm.allreduce(len(indices)) if comm is not None else len(indices)

    bounds = (np.arange(n_ranks + 1, dtype=np.int64) * n_total) // n_ranks
    owner_ranks = np.searchsorted(bounds, positions, side="right") - 1

    send_order = np.argsort(owner_ranks, stable=True)
    send_counts = np.bincount(owner_ranks, minlength=n_ranks).astype(np.int64)
    rec_counts = np.empty_like(send_counts)
    if comm is not None:
        comm.Alltoall(sendbuf=send_counts, recvbuf=rec_counts)
    else:
        rec_counts[:] = send_counts
    redistribution_map = RedistributionMap(send_order=send_order, send_counts=send_counts, rec_counts=rec_counts)

    received_positions = redistribute_data(positions, redistribution_map, comm)
    received_indices = redistribute_data(indices, redistribution_map, comm)

    block_start = int(bounds[rank])
    block = np.empty(bounds[rank + 1] - block_start, dtype=DTYPES["csr_indices"])
    assert len(received_positions) == len(block), "membership indices do not tile the output dataset"
    block[received_positions - block_start] = received_indices

    return block_start, block


def _resolve_pointer_column(chunks: list[np.ndarray], inverse_halo_order: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Concatenates per-rank pointers and reindexes them into final merged catalogue order. Necessary because, after the global
    sort is applied, the local ordering is now invalid.
    """
    merged_pointers = np.concatenate(chunks)
    reindexed = np.where(merged_pointers == -1, -1, inverse_halo_order[merged_pointers])
    return reindexed[order]


def build_galaxy_membership_arrays(
    galaxy_parent_halo_index: np.ndarray, halo_parent: np.ndarray, n_haloes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the galaxy membership CSR arrays and derives the central galaxy index from parent haloes, returning a tuple of:

    - indices: ndarray of galaxy indices
    - offsets: ndarray of galaxy offsets
    - lengths: ndarray of galaxy lengths
    - central_galaxy_index: the index into galaxy_data which corresponds to a halo's most-massive galaxy
    """
    lengths = np.zeros(n_haloes, dtype=DTYPES["csr_lengths"])
    level_halo_rows: list[np.ndarray] = []
    level_galaxy_indices: list[np.ndarray] = []

    current_rows = galaxy_parent_halo_index.astype(np.int64, copy=True)
    galaxy_indices = np.arange(len(galaxy_parent_halo_index), dtype=DTYPES["csr_indices"])

    while True:  # trace the membership hierarchy, depth level by depth level
        valid_mask = current_rows >= 0
        if not valid_mask.any():
            break

        lengths += np.bincount(current_rows[valid_mask], minlength=n_haloes).astype(DTYPES["csr_lengths"])
        level_halo_rows.append(current_rows[valid_mask])
        level_galaxy_indices.append(galaxy_indices[valid_mask])
        current_rows[valid_mask] = halo_parent[current_rows[valid_mask]]  # walk one level up; roots become -1

    offsets = np.empty(len(lengths) + 1, dtype=DTYPES["csr_offsets"])
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:], dtype=DTYPES["csr_offsets"])

    if level_halo_rows:
        flat_halo_rows = np.concatenate(level_halo_rows)
        flat_galaxy_indices = np.concatenate(level_galaxy_indices)
        pair_order = np.lexsort((flat_galaxy_indices, flat_halo_rows))  # by halo row, then galaxy ascending
        indices = flat_galaxy_indices[pair_order]
    else:
        indices = np.empty(0, dtype=DTYPES["csr_indices"])

    central_galaxy_index = np.full(n_haloes, -1, dtype=np.int64)
    non_empty = lengths > 0
    central_galaxy_index[non_empty] = indices[
        offsets[:-1][non_empty]
    ]  # NOTE: this works because the global order uses mass_baryon so the first galaxy automatically has the largest baryonic mass

    return indices, offsets, lengths, central_galaxy_index


def build_subhalo_membership_arrays(
    parent_halo_index: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the subhalo membership CSR arrays, returning a tuple of:

    - indices: indices into halo_data of haloes' subhaloes
    - offsets: where the subhaloes of each halo begins in subhalo_indices
    - lengths: the number of subhaloes in each halo
    - field_halo_index: the field halo index of each halo
    """
    n_haloes = len(parent_halo_index)
    field_halo_index = np.full(shape=n_haloes, fill_value=-1, dtype=np.int64)  # initialise with sentinel value

    # iterate over each depth level to get field haloes (redundant for many subhaloes but this is cheap)
    current_level = parent_halo_index.copy()
    for _ in range(int(depth.max()) if n_haloes > 0 else 0):  # guard on h_haloes > 0 else crashes if no haloes (high-z)
        at_field = (current_level == -1) | (depth[current_level] == 0)  # subhaloes at field-level
        current_level = np.where(
            at_field, current_level, parent_halo_index[current_level]
        )  # increment the depth counter

    field_halo_index[depth > 0] = current_level[depth > 0]  # let field haloes stay at -1

    exclusive_offsets, exclusive_idx = build_group_csr(group_idx=parent_halo_index, n_groups=n_haloes)
    inclusive_offsets, inclusive_idx = propagate_membership_csr(
        offsets=exclusive_offsets,
        sorted_indices=exclusive_idx,
        parent_ids=parent_halo_index,
        n_groups=n_haloes,
    )
    inclusive_lengths = np.diff(inclusive_offsets).astype(DTYPES["csr_lengths"])

    return inclusive_idx, inclusive_offsets, inclusive_lengths, field_halo_index


def unpack_data(
    all_rank_data: list[RankPackedData],
    internals: Internals,
) -> UnpackedRankData:
    """
    Unpacks all the ranks' packed data into an UnpackedRankData object.
    """
    # all these dicts are keyed by group hdf5 name
    group_lengths: dict[str, list[int]] = {group_type: [] for group_type in internals.group_types}
    sort_arrays: dict[str, list[np.ndarray]] = {group_type: [] for group_type in internals.group_types}
    all_group_ids: dict[str, list[np.ndarray]] = {group_type: [] for group_type in internals.group_types}
    physics_chunks: dict[str, dict[str, list[np.ndarray]]] = {group_type: {} for group_type in internals.group_types}

    membership_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        group_type: {
            column_name: []
            for column_name in internals.membership_columns.get(group_type, {})
            if column_name not in POST_SORT_COLUMNS.get(group_type, frozenset())
        }
        for group_type in internals.group_types
    }
    cumulative_haloes = 0

    # NOTE: this is where the functionality begins
    for rank_data in all_rank_data:  # the rank_data dataclass is also (thankfully) keyed by hdf5 names
        for group_type in internals.group_types:
            group_params = internals.group_types[group_type]
            hdf5_name = group_params["hdf5_group"]

            if hdf5_name not in rank_data.groups:
                group_lengths[group_type].append(0)
                continue

            kind = group_params["kind"]
            sort_column = SORT_COLUMN_BY_KIND[
                kind
            ]  # the rank data dict doesn't include hdf5 group prefixes (e.g. properties/core/)

            group = rank_data.groups[hdf5_name]
            n_groups = len(group.group_ids)
            group_lengths[group_type].append(n_groups)
            sort_arrays[group_type].append(group.physics_columns[sort_column])

            if n_groups == 0:
                logger.warning(f"No groups found for {group_type}.")
                continue

            # membership
            for column_name in membership_chunks[group_type]:
                if column_name in group.membership_columns:
                    chunk = group.membership_columns[column_name]
                elif column_name in FILL_VALUES:  # this is for no-subhalo path catalogue consistency (snapshot HaloIDs)
                    column_meta = internals.membership_columns[group_type][column_name]
                    chunk = np.full(n_groups, FILL_VALUES[column_name], dtype=np.dtype(column_meta.dtype))

                if column_name in HALO_POINTER_COLUMNS:
                    chunk = np.where(chunk == -1, -1, chunk + cumulative_haloes)

                membership_chunks[group_type][column_name].append(chunk)

            # physics
            for column_name, column_data in group.physics_columns.items():
                physics_chunks[group_type].setdefault(column_name, []).append(
                    column_data
                )  # need setdefault here (otherwise keyerror)

            # group IDs
            all_group_ids[group_type].append(group.group_ids)

        cumulative_haloes += group_lengths["haloes"][
            -1
        ]  # use the previous rank's contribution so the next knows where to start

    unpacked_data = UnpackedRankData(
        sort_arrays=sort_arrays,
        membership_chunks=membership_chunks,
        group_ids=all_group_ids,
        physics_chunks=physics_chunks,
    )

    return unpacked_data


def resolve_global_ordering(
    unpacked: UnpackedRankData,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """
    Generates the global sorting order, then resolves the per-rank data to correspond to this global ordering; this ensures when data is written to the output file, it appears in an order which does not change between serial, parallel, or varying parallel (mpiexec -n 4 vs 6, etc.) run configurations. Returns:

    - sort_orders: dict keyed by group type which contains the sort mask
    - resolved: membership columns in final catalogue order (necessary because of pointer columns parent/field_halo_index)
    """
    # determine global ordering
    sort_orders: dict[str, np.ndarray] = {}
    for group_type in unpacked.sort_arrays:
        # continue if the group doesn't exist (galaxy finding disabled, or found no galaxies)
        if not unpacked.sort_arrays[group_type]:
            continue

        merged = np.concatenate(
            unpacked.sort_arrays[group_type]
        )  # sort_arrays is needed because galaxies/haloes use baryonic/total mass
        global_ids = np.concatenate(unpacked.group_ids[group_type])
        depth_chunks = unpacked.membership_chunks.get(group_type, {}).get("depth")

        if depth_chunks:
            depth = np.concatenate(depth_chunks)
            sort_orders[group_type] = (
                np.lexsort(  # sorting by the original sort array, with depth as a tiebreaker (subhalo vs halo), then global ID
                    (global_ids, depth, -merged)
                )
            )  # NOTE: lexsort sorts by last key first
        else:
            sort_orders[group_type] = np.lexsort(
                (global_ids, -merged)
            )  # just global IDs for galaxies as a tiebreaker (no hierarchy)

    if "haloes" in sort_orders:
        inverse_halo_order = np.argsort(sort_orders["haloes"], stable=True)

    # resolve the per-rank membership pointer columns into the final catalogue order
    resolved: dict[tuple[str, str], np.ndarray] = {}

    for group_type, columns in unpacked.membership_chunks.items():
        if group_type not in sort_orders:
            continue

        order = sort_orders[group_type]

        for column_name, chunks in columns.items():
            if column_name in HALO_POINTER_COLUMNS:
                resolved[(group_type, column_name)] = _resolve_pointer_column(chunks, inverse_halo_order, order)
            else:
                resolved[(group_type, column_name)] = np.concatenate(chunks)[order]

    return sort_orders, resolved


def write_group_data_to_catalogue(
    output_path: Path,
    sort_orders: dict[str, np.ndarray],
    resolved: dict[tuple[str, str], np.ndarray],
    physics_chunks: dict[str, dict[str, list[np.ndarray]]],
    internals: Internals,
) -> None:
    """
    Writes group-level data to the final output catalogue.
    """
    with h5py.File(output_path, "w") as f_out:  # this is the first write site (future writes must use "a")
        for group_type in internals.group_types:
            group_params = internals.group_types[group_type]
            hdf5_name = group_params["hdf5_group"]

            if group_type not in sort_orders:
                continue

            order = sort_orders[group_type]
            out_grp = f_out.create_group(hdf5_name)
            n_total = len(order)

            # write group-level physics data
            for column_name, chunks in physics_chunks.get(group_type, {}).items():
                merged = np.concatenate(chunks)[order]
                column_meta = internals.output_columns[column_name]
                label = column_meta.label
                label_group = out_grp.require_group(f"properties/{label}")
                ds = label_group.create_dataset(column_name, data=merged, compression=1)
                # write metadata
                ds.attrs["unit"] = column_meta.unit
                ds.attrs["a_exp"] = column_meta.a_exp
                ds.attrs["description"] = column_meta.description

            # ^ see above, final IDs need to be sequential and therefore it is done with np.arange
            out_grp.create_dataset(internals.group_types[group_type]["key"], data=np.arange(n_total), compression=1)

            # write group-level membership data
            membership_grp = out_grp.require_group("membership")
            for column_name, column_meta in internals.membership_columns.get(group_type, {}).items():
                if (group_type, column_name) not in resolved:
                    continue  # central_galaxy_index handled below

                dataset = membership_grp.create_dataset(
                    column_name, data=resolved[(group_type, column_name)], compression=1
                )
                dataset.attrs["description"] = column_meta.description

        # now write the galaxy membership arrays (requires halo info available), not in internals.yaml like particle membership
        if "haloes" in sort_orders:
            n_haloes = len(sort_orders["haloes"])
            halo_membership_grp = f_out[internals.group_types["haloes"]["hdf5_group"]]["membership"]

            subhalo_indices, subhalo_offsets, subhalo_lengths, field_halo_index = build_subhalo_membership_arrays(
                parent_halo_index=resolved[("haloes", "parent_halo_index")],
                depth=resolved[("haloes", "depth")],
            )

            ds = halo_membership_grp.create_dataset("subhalo_indices", data=subhalo_indices, compression=1)
            ds.attrs["description"] = "Catalogue-level indices into halo_data of child subhaloes."

            ds = halo_membership_grp.create_dataset("subhalo_offsets", data=subhalo_offsets, compression=1)
            ds.attrs["description"] = (
                "Per-halo offsets into subhalo_indices, where subhalo_indices[offsets[h]:offsets[h+1]] recovers indexes into halo_data of subhaloes belonging to halo h."
            )

            ds = halo_membership_grp.create_dataset("subhalo_lengths", data=subhalo_lengths, compression=1)
            ds.attrs["description"] = (
                "Per-halo halo_data lengths; subhalo_lengths[h] is the number of subhaloes in halo h."
            )

            field_meta = internals.membership_columns["haloes"]["field_halo_index"]
            ds = halo_membership_grp.create_dataset("field_halo_index", data=field_halo_index, compression=1)
            ds.attrs["description"] = field_meta.description

            if ("galaxies", "parent_halo_index") in resolved:
                galaxy_indices, galaxy_offsets, galaxy_lengths, central_galaxy_index = build_galaxy_membership_arrays(
                    resolved[("galaxies", "parent_halo_index")], resolved[("haloes", "parent_halo_index")], n_haloes
                )

                halo_membership_grp = f_out[internals.group_types["haloes"]["hdf5_group"]]["membership"]

                # write halo-galaxy membership
                ds = halo_membership_grp.create_dataset("galaxy_indices", data=galaxy_indices, compression=1)
                ds.attrs["description"] = "Galaxy-level indices into galaxy_data."

                ds = halo_membership_grp.create_dataset("galaxy_offsets", data=galaxy_offsets, compression=1)
                ds.attrs["description"] = (
                    "Per-halo offsets into galaxy_indices, where galaxy_indices[offsets[h]:offsets[h+1]] recovers indexes into galaxy_data of galaxies belonging to halo h."
                )

                ds = halo_membership_grp.create_dataset("galaxy_lengths", data=galaxy_lengths, compression=1)
                ds.attrs["description"] = (
                    "Per-halo galaxy_data lengths; galaxy_lengths[h] is the number of galaxies in halo h."
                )

                central_meta = internals.membership_columns["haloes"]["central_galaxy_index"]
                central_dataset = halo_membership_grp.create_dataset(
                    "central_galaxy_index", data=central_galaxy_index, compression=1
                )
                central_dataset.attrs["description"] = central_meta.description
