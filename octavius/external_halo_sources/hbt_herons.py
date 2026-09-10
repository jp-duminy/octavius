"""

Machinery for reading and parsing the SWIFT-native subhalo finder HBT-HERONS. This draws
on similar methods to what you will find in their toolbox folder.

HERONS outputs a number split of SubSnap files owing to MPI. In their source code,
toolbox/catalogue_cleanup/SortCatalogues.py can combine these into the format Octavius supports.
The reason we do not support both formats is because variable-length HDF5 reads caused painful
nightmares in early Octavius development and HERONS can post-process this for us.

Website: https://hbt-herons.strw.leidenuniv.nl/
Source code: https://github.com/SWIFTSIM/HBT-HERONS
Source paper: https://academic.oup.com/mnras/article/543/2/1339/8250004
HBT algorithm source paper: https://academic.oup.com/mnras/article/474/1/604/4566529

# NOTE: 'field' in Octavius terminology is the equivalent of the HBT central.
# NOTE: HERONS uses -1 as its sentinel value too.
# NOTE: in a case of "this edge case will never make it to production" making it to production, you
can get subhaloes with no field halo in the HERONS catalogues.
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
import h5py
import numpy as np

# internal imports
from .halo_data_structures import (
    HaloAssignments,
    SubhaloInformation,
    HaloSource,
    distribute_ids,
    apply_lookup,
)
from ..log import get_logger

logger = get_logger()


class HeronsCatalogue(NamedTuple):
    """
    Container for fields from a HBT-HERONS catalogue.
    """

    track_ids: np.ndarray
    host_halo_ids: np.ndarray  # this is the FOFGroupID, which we treat as field haloes
    depth: np.ndarray  # depth ascending where 0 is a field halo
    nested_parent_track_id: np.ndarray
    n_bound: np.ndarray


class HeronsLookups(NamedTuple):
    """
    Container for the halo catalogue and particle lookups.
    """

    field_lookup: np.ndarray
    sub_lookup: np.ndarray
    track_to_field: np.ndarray
    field_mask: np.ndarray
    sub_mask: np.ndarray
    n_field: np.ndarray
    n_sub: np.ndarray


class HeronsHaloSource(HaloSource):
    """
    HBT-HERONS halo ID source; only compatible with SWIFT, and requires SWIFT halo FOF groups to be identified.
    """

    def __init__(self, catalogue_path: Path, reader: SnapshotReader) -> None:

        super().__init__(reader=reader)
        self.catalogue_path = catalogue_path

    @cached_property
    def _catalogue(self) -> HeronsCatalogue:
        """
        All halo-level properties.
        """
        return read_halo_properties(catalogue_path=self.catalogue_path)

    @cached_property
    def _lookups(self) -> HeronsLookups:
        """
        Lookups from particles and haloes to haloes.
        """
        host_halo_ids = self._catalogue.host_halo_ids

        # catalogue indices for field haloes
        field_mask = (self._catalogue.depth == 0) | ((self._catalogue.depth > 0) & (host_halo_ids == -1))
        field_indices = np.flatnonzero(field_mask)
        n_field = len(field_indices)

        # build a lookup from HostHaloIDs (FOFGroupIDs) to contiguous 0-indexed field IDs
        field_lookup = np.full(shape=np.max(host_halo_ids) + 1, fill_value=-1, dtype=np.int64)
        non_orphan = host_halo_ids[field_indices] >= 0  # where haloes have valid FOFGroupIDs

        # filter fields to non-orphans -> filter hosts to fields -> assign non-orphan 0-indexed field IDs
        field_lookup[host_halo_ids[field_indices[non_orphan]]] = np.searchsorted(
            field_indices, field_indices[non_orphan]
        )

        # now do the same thing but for subhaloes (no orphan filter needed)
        sub_mask = (self._catalogue.depth > 0) & (host_halo_ids >= 0)
        sub_indices = np.flatnonzero(sub_mask)
        n_sub = len(sub_indices)
        track_ids = self._catalogue.track_ids
        sub_lookup = np.full(shape=np.max(track_ids) + 1, fill_value=-1, dtype=np.int64)

        sub_lookup[track_ids[sub_indices]] = np.arange(n_sub)

        # also: lone, wandering subhaloes (https://hbt-herons.strw.leidenuniv.nl/algorithm/host_haloes/#hostless-subhaloes)
        promoted_mask = (self._catalogue.depth > 0) & (host_halo_ids == -1)
        promoted_indices = np.flatnonzero(promoted_mask)
        track_to_field = np.full(shape=np.max(track_ids) + 1, fill_value=-1, dtype=np.int64)
        track_to_field[track_ids[promoted_indices]] = np.searchsorted(field_indices, promoted_indices)
        # also applies to field orphans
        field_orphan_indices = field_indices[~non_orphan]
        track_to_field[track_ids[field_orphan_indices]] = np.searchsorted(field_indices, field_orphan_indices)

        lookups = HeronsLookups(
            field_lookup=field_lookup,
            sub_lookup=sub_lookup,
            track_to_field=track_to_field,
            field_mask=field_mask,
            sub_mask=sub_mask,
            n_field=n_field,
            n_sub=n_sub,
        )

        return lookups

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Returns a HaloAssignments dataclass containing the assignments made by HERONS.
        """
        herons_pids, offsets = read_halo_particles(catalogue_path=self.catalogue_path)
        order = np.argsort(
            herons_pids, stable=True
        )  # this is best precomputed otherwise you argsort the pid array 4 times in the merge
        sorted_herons_pids = herons_pids[order]

        track_ids = self._catalogue.track_ids
        host_halo_ids = self._catalogue.host_halo_ids
        depth = self._catalogue.depth

        field_ids, sub_ids = {}, {}

        for (
            ptype
        ) in ptypes:  # to sort the ids we first need all ptypes present to cover all HaloIDs to avoid desync issues
            snapshot_pids = self.reader.read_particle_ids(ptype=ptype)

            ptype_field_ids, ptype_sub_ids = match_herons_particle_ids(
                snapshot_pids=snapshot_pids,
                sorted_herons_pids=sorted_herons_pids,
                order=order,
                particle_offsets=offsets,
                track_ids=track_ids,
                depth=depth,
                host_halo_ids=host_halo_ids,
                track_to_field=self._lookups.track_to_field,
                field_lookup=self._lookups.field_lookup,
            )

            field_ids[ptype] = ptype_field_ids  # sentinels handled by matcher
            sub_ids[ptype] = apply_lookup(ids=ptype_sub_ids, lookup=self._lookups.sub_lookup)

        original_field_ids = self._catalogue.host_halo_ids[self._lookups.field_mask]
        n_total_haloes = self._lookups.n_field

        n_subhaloes = np.sum(self._catalogue.depth > 0)
        logger.info(f"HBT-HERONS: {n_total_haloes:,} field haloes | {n_subhaloes:,} subhaloes.")

        halo_assignments = HaloAssignments(
            field_ids=field_ids,
            original_field_ids=original_field_ids,
            sub_ids=sub_ids,
            n_field_haloes=n_total_haloes,
        )

        for ptype, ids in field_ids.items():
            n_assigned = np.sum(ids != -1)
            logger.info(f"  {ptype}: {n_assigned:,} / {len(ids):,} particles assigned to haloes.")

        return halo_assignments

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

    def read_subhalo_info(self) -> SubhaloInformation:
        """
        Returns a SubhaloInformation dataclass containing HERONS subhalo-level arrays.
        """
        sub_mask = self._lookups.sub_mask
        host_field_ids = apply_lookup(ids=self._catalogue.host_halo_ids[sub_mask], lookup=self._lookups.field_lookup)
        parent_index = apply_lookup(self._catalogue.nested_parent_track_id[sub_mask], lookup=self._lookups.sub_lookup)
        depth = self._catalogue.depth[sub_mask]
        global_index = np.arange(self._lookups.n_sub, dtype=np.int64)
        n_bound = self._catalogue.n_bound[sub_mask]
        original_sub_ids = self._catalogue.track_ids[sub_mask]

        subhalo_info = SubhaloInformation(
            host_field_ids=host_field_ids,
            parent_index=parent_index,
            depth=depth,
            global_index=global_index,
            n_bound=n_bound,
            original_sub_ids=original_sub_ids,
        )

        return subhalo_info

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


def match_herons_particle_ids(
    snapshot_pids: np.ndarray,
    sorted_herons_pids: np.ndarray,
    order: np.ndarray,
    particle_offsets: np.ndarray,
    track_ids: np.ndarray,
    depth: np.ndarray,
    host_halo_ids: np.ndarray,
    track_to_field: np.ndarray,
    field_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matches snapshot particle IDs to their counterparts in the HERONS output. Returns:

    - field_ids: an array of field halo IDs aligned with the snapshot_pids
    - sub_ids: an array of sub IDs aligned with the snapshot_pids
    """
    insertion_idx = np.searchsorted(sorted_herons_pids, snapshot_pids)
    insertion_idx = np.minimum(
        insertion_idx, len(sorted_herons_pids) - 1
    )  # prevent OOB index when max(snapshot_pids) > max(herons_pids)
    matched = sorted_herons_pids[insertion_idx] == snapshot_pids

    field_ids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)
    sub_ids = np.full(shape=len(snapshot_pids), fill_value=-1, dtype=np.int64)

    herons_idx = order[
        insertion_idx[matched]
    ]  # RHS fancy indexing maps matched, which is sorted, to unsorted id arrays
    subhalo_idx = (
        np.searchsorted(particle_offsets, herons_idx, side="right") - 1
    )  # use side=right to match [offsets[i]:offsets[i+1]]; -1 for the subhalo itself

    matched_host = host_halo_ids[subhalo_idx]
    matched_track = track_ids[subhalo_idx]
    matched_depth = depth[subhalo_idx]

    # most particles are particles belonging to subhaloes with valid field haloes
    has_host = matched_host >= 0
    field_ids[matched] = np.where(has_host, field_lookup[matched_host], track_to_field[matched_track])

    # also handle edge case of wandering subhaloes
    sub_ids[matched] = np.where((matched_depth > 0) & has_host, matched_track, -1)

    return field_ids, sub_ids


def read_halo_particles(catalogue_path: Path) -> tuple[np.ndarray, ...]:
    """
    Returns a tuple of ParticleIDs and ParticleOffsets from the merged HERONS catalogue.
    """
    with h5py.File(catalogue_path, "r") as catalogue:
        if "Particles" not in catalogue:
            raise FileNotFoundError(
                f"{catalogue_path} does not contain particle info, please run SortCatalogues.py (included with HBT-HERONS) and enabled the --with-particles flag."
            )

        particle_ids = catalogue["Particles"]["ParticleIDs"][:]
        particle_offsets = catalogue["Subhalos"]["ParticleOffset"][:]

    return particle_ids, particle_offsets


def read_halo_properties(catalogue_path: Path) -> HeronsCatalogue:
    """
    Returns a HeronsCatalogue filled with the relevant parameters from the file.

    NOTE: n_bound == 0 for orphan particles and is not filtered here.
    """
    with h5py.File(catalogue_path, "r") as cat:
        subhaloes = cat["Subhalos"]

        track_ids = subhaloes["TrackId"][:]
        host_halo_ids = subhaloes["HostHaloId"][:]  # this is the FOFGroupID, per their docs
        nested_parent_track_id = subhaloes["NestedParentTrackId"][:]
        depth = subhaloes["Depth"][:]
        n_bound = subhaloes["Nbound"][:]

    herons_catalogue = HeronsCatalogue(
        track_ids=track_ids,
        host_halo_ids=host_halo_ids,
        nested_parent_track_id=nested_parent_track_id,
        depth=depth,
        n_bound=n_bound,
    )

    return herons_catalogue
