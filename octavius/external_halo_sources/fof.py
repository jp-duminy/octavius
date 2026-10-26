"""

Built-in halo source for snapshots without HaloIDs: runs Octavius's periodic 3D friends-of-friends halo finder on
the snapshot itself, so no external catalogue is required.

"""

# type checking (semantic)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_management import SnapshotReader
    from mpi4py.MPI import Comm

# default libraries
from collections.abc import Iterator, Mapping
from time import perf_counter

# other packages
import numpy as np

# internal imports
from .halo_data_structures import HaloSource, HaloAssignments, SubhaloInformation, distribute_ids
from ..galaxy_finding.fof_halo_algorithm import find_fof_haloes
from ..data_management.snapshot_readers_base import HALO_ID_PROVENANCE
from ..galaxy_finding.fof_halo_mpi import find_fof_haloes_mpi
from ..log import get_logger
from ..version import __version__

logger = get_logger()

LARGEST_HALO_WARNING = 0.5  # warn when the largest halo holds more than this fraction of all DM particles


class FOFHaloSource(HaloSource):
    """
    Identifies field haloes with a 3D friends-of-friends on the DM particles; baryons are attached to the halo of
    their nearest DM particle. With MPI, every rank takes part (collective), so no rank needs to hold the whole box;
    serially, rank 0 does it alone. FOF groups carry no subhalo information.
    """

    collective = True

    def __init__(
        self, reader: SnapshotReader, b: float, min_members: int, attach_ptypes: list[str] | None = None
    ) -> None:

        super().__init__(reader=reader)
        self.b = b
        self.min_members = min_members
        self.attach_ptypes = attach_ptypes  # baryonic ptypes to attach to haloes; None attaches all

    def read_halo_ids(self, ptypes: list[str]) -> HaloAssignments:
        """
        Reads all particle positions and runs the halo finder, returning HaloAssignments.
        """
        attached = self._start(ptypes)

        t_start = perf_counter()
        dm_pos = self.reader.read_full_dataset(ptype="dm", dataset="pos")
        t_read_dm = perf_counter() - t_start

        baryon_pos = _LazyPositions(reader=self.reader, ptypes=attached)
        timings: dict[str, float] = {}
        halo_ids = find_fof_haloes(
            dm_pos=dm_pos,
            boxsize=self.reader.simulation_attributes.boxsize,
            b=self.b,
            baryon_pos=baryon_pos,
            min_members=self.min_members,
            timings=timings,
        )
        del dm_pos

        n_haloes = int(halo_ids["dm"].max()) + 1 if len(halo_ids["dm"]) > 0 else 0
        self._finish(halo_ids, ptypes, attached, t_start, t_read_dm, baryon_pos, timings, n_haloes, comm=None)

        return HaloAssignments(
            field_ids=halo_ids,
            n_field_haloes=n_haloes,
            sub_ids=None,
            original_field_ids=None,
        )

    def read_local_halo_ids(
        self, ptypes: list[str], slabs: dict[str, slice], comm: Comm
    ) -> tuple[dict[str, np.ndarray], int]:
        """
        Collective. Each rank reads its slabs of the particle positions and the ranks find haloes together; returns
        the HaloIDs of this rank's slabs and the number of haloes.
        """
        attached = self._start(ptypes)

        t_start = perf_counter()
        dm_pos = self.reader.read_full_dataset(ptype="dm", dataset="pos", slab=slabs["dm"])
        t_read_dm = perf_counter() - t_start

        baryon_pos = _LazyPositions(reader=self.reader, ptypes=attached, slabs=slabs)
        timings: dict[str, float] = {}
        halo_ids, n_haloes = find_fof_haloes_mpi(
            dm_pos=dm_pos,
            dm_offset=slabs["dm"].start,
            boxsize=self.reader.simulation_attributes.boxsize,
            comm=comm,
            b=self.b,
            baryon_pos=baryon_pos,
            min_members=self.min_members,
            timings=timings,
        )
        del dm_pos

        self._finish(halo_ids, ptypes, attached, t_start, t_read_dm, baryon_pos, timings, n_haloes, comm, slabs)

        return halo_ids, n_haloes

    def snapshot_attributes(self) -> dict[str, Any]:
        """
        Provenance stored with the halo IDs if they are written back to the snapshot.
        """
        return {
            HALO_ID_PROVENANCE: "FOF",
            "octavius_version": __version__,
            "halo_b": self.b,
            "min_dm_per_halo": self.min_members,
            "halo_attach_ptypes": ",".join(self.attach_ptypes) if self.attach_ptypes is not None else "gas,star,bh",
        }

    def _start(self, ptypes: list[str]) -> list[str]:
        """
        Checks there are DM particles, logs the start of halo finding, and returns the baryonic ptypes to attach.
        """
        if "dm" not in ptypes:
            raise ValueError("FOF halo finding requires DM particles, but none are available in the snapshot.")

        n_dm = self.reader.particle_counts["dm"]
        logger.info(f"FOF: finding haloes in {n_dm:,} DM particles (b = {self.b}, min. {self.min_members} members).")

        baryonic = [pt for pt in ptypes if pt != "dm"]
        return baryonic if self.attach_ptypes is None else [pt for pt in baryonic if pt in self.attach_ptypes]

    def _finish(
        self,
        halo_ids: dict[str, np.ndarray],
        ptypes: list[str],
        attached: list[str],
        t_start: float,
        t_read_dm: float,
        baryon_pos: _LazyPositions,
        timings: dict[str, float],
        n_haloes: int,
        comm: Comm | None,
        slabs: dict[str, slice] | None = None,
    ) -> None:
        """
        Gives unattached ptypes no halo (in place) and logs the timing breakdown and halo statistics (summed over
        ranks if comm is given).
        """
        for ptype in ptypes:
            if ptype != "dm" and ptype not in attached:
                logger.info(f"FOF: {ptype} not in halo_attach_ptypes, so given no halo.")
                n = self.reader.particle_counts[ptype] if slabs is None else slabs[ptype].stop - slabs[ptype].start
                halo_ids[ptype] = np.full(n, -1, dtype=np.int64)

        # baryon positions are read lazily during attachment, so separate out the read time
        timings["attaching baryons"] -= baryon_pos.read_time
        steps = {"reading positions": t_read_dm + baryon_pos.read_time, **timings}
        breakdown = ", ".join(f"{step} {elapsed:.1f}s" for step, elapsed in steps.items())
        logger.info(f"FOF: halo finding completed in {perf_counter() - t_start:.1f}s ({breakdown}).")

        # haloes are ordered by size, so halo 0 is the largest
        n_largest = int(np.count_nonzero(halo_ids["dm"] == 0))
        if comm is not None:
            n_largest = comm.allreduce(n_largest)
        logger.info(f"FOF: {n_haloes:,} field haloes (largest: {n_largest:,} DM particles) | no subhalo information")

        largest_fraction = n_largest / max(self.reader.particle_counts["dm"], 1)
        if largest_fraction > LARGEST_HALO_WARNING and (comm is None or comm.rank == 0):
            logger.warning(
                f"FOF: the largest halo holds {largest_fraction:.0%} of all DM particles, which suggests the linking "
                f"length is too large (halo_b = {self.b}) or that positions and box size are in different units."
            )

        for ptype in ptypes:  # in the same order on every rank
            n_assigned = int(np.sum(halo_ids[ptype] != -1))
            if comm is not None:
                n_assigned = comm.allreduce(n_assigned)
            logger.info(
                f"  {ptype}: {n_assigned:,} / {self.reader.particle_counts[ptype]:,} particles assigned to haloes."
            )

    def read_subhalo_info(self) -> SubhaloInformation | None:
        """
        No subhaloes on FOFHaloSource.
        """
        return None

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
    ) -> dict[str, np.ndarray] | None:
        """
        No subhaloes on FOFHaloSource.
        """
        return None


class _LazyPositions(Mapping):
    """
    Reads each ptype's positions (or this rank's slab of them) only when the halo finder reaches it, so only one
    baryonic ptype is in memory at once.
    """

    def __init__(self, reader: SnapshotReader, ptypes: list[str], slabs: dict[str, slice] | None = None) -> None:
        self.reader = reader
        self.ptypes = ptypes
        self.slabs = slabs
        self.read_time = 0.0  # cumulative time spent reading, for the timing breakdown

    def __getitem__(self, ptype: str) -> np.ndarray:
        t0 = perf_counter()
        slab = None if self.slabs is None else self.slabs[ptype]
        pos = self.reader.read_full_dataset(ptype=ptype, dataset="pos", slab=slab)
        self.read_time += perf_counter() - t0
        return pos

    def __iter__(self) -> Iterator[str]:
        return iter(self.ptypes)

    def __len__(self) -> int:
        return len(self.ptypes)
