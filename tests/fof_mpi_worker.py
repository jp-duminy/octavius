"""

Run under mpirun by test_fof_haloes_mpi.py: checks that the MPI FOF halo finder gives exactly the serial finder's
HaloIDs, with each rank holding a contiguous chunk of the particles. Exits non-zero on any mismatch.

Usage: mpirun -n <ranks> python -m tests.fof_mpi_worker [--force-messages] <linking length> [<linking length> ...]

--force-messages makes every particle exchange use redistribute_data()'s large-count fallback, in tiny messages.

"""

import sys

import numpy as np
from mpi4py import MPI

from octavius.data_management import parallel_reading
from octavius.galaxy_finding.fof_halo_algorithm import find_fof_haloes
from octavius.galaxy_finding.fof_halo_mpi import find_fof_haloes_mpi
from tests.test_fof_haloes import BOXSIZE, SEED, make_clustered_positions


def chunk(n: int, comm: MPI.Comm) -> slice:
    edges = np.linspace(0, n, comm.size + 1).astype(np.int64)
    return slice(int(edges[comm.rank]), int(edges[comm.rank + 1]))


def main() -> None:
    comm = MPI.COMM_WORLD
    args = sys.argv[1:]
    if "--force-messages" in args:
        args.remove("--force-messages")
        parallel_reading.MPI_MAX_COUNT = 0
        parallel_reading.MAX_MESSAGE_BYTES = 1000  # 41 positions per message
    rng = np.random.default_rng(SEED)
    dm_pos = make_clustered_positions(rng, n_background=20000, n_clusters=40)
    gas_pos = make_clustered_positions(rng, n_background=8000, n_clusters=40) - 0.3  # some fall below 0: wrapped
    dm_chunk, gas_chunk = chunk(len(dm_pos), comm), chunk(len(gas_pos), comm)
    failures = []

    for linking_length in map(float, args):
        for min_members in (1, 20):
            expected = find_fof_haloes(
                dm_pos=dm_pos,
                boxsize=BOXSIZE,
                linking_length=linking_length,
                baryon_pos={"gas": gas_pos},
                min_members=min_members,
            )
            halo_ids, n_haloes = find_fof_haloes_mpi(
                dm_pos=dm_pos[dm_chunk],
                dm_offset=dm_chunk.start,
                boxsize=BOXSIZE,
                comm=comm,
                linking_length=linking_length,
                baryon_pos={"gas": gas_pos[gas_chunk]},
                min_members=min_members,
            )

            for ptype in ("dm", "gas"):
                found = np.concatenate(comm.allgather(halo_ids[ptype]))
                if not np.array_equal(found, expected[ptype]):
                    failures.append(f"{ptype} (l = {linking_length}, min. {min_members})")
            if n_haloes != expected["dm"].max() + 1:
                failures.append(f"n_haloes (l = {linking_length}, min. {min_members})")

    if comm.rank == 0 and failures:
        print(f"{comm.size} ranks: mismatched {', '.join(failures)}", flush=True)
    comm.Barrier()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
