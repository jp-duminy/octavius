"""

Tests the MPI-parallel FOF halo finder (fof_halo_mpi.py): its domain decomposition, its single-rank behaviour (as run
by pytest), and, when mpirun is available, exact agreement with the serial finder across several rank counts.

"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from octavius.galaxy_finding.fof_halo_algorithm import STENCIL_REACH, find_fof_haloes
from octavius.galaxy_finding.fof_halo_mpi import balanced_bounds, find_fof_haloes_mpi, merge_edges
from tests.test_fof_haloes import BOXSIZE, LINKING_LENGTH, SEED, make_clustered_positions

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("n_domains", [1, 2, 5, 16])
def test_balanced_bounds_respect_minimum_thickness(n_domains: int) -> None:
    rng = np.random.default_rng(SEED)
    counts = np.zeros(64, dtype=np.int64)
    counts[10] = 10**6  # one plane holding nearly everything can't be split, so the others must stay valid
    counts += rng.integers(0, 10, size=len(counts))

    bounds = balanced_bounds(counts=counts, n_domains=n_domains, min_thickness=STENCIL_REACH)

    assert bounds[0] == 0 and bounds[-1] == len(counts) and len(bounds) == n_domains + 1
    assert np.all(np.diff(bounds) >= STENCIL_REACH)


def test_balanced_bounds_balance_counts() -> None:
    counts = np.arange(1, 101, dtype=np.int64)  # density rising along x
    bounds = balanced_bounds(counts=counts, n_domains=4, min_thickness=STENCIL_REACH)
    per_domain = np.add.reduceat(counts, bounds[:-1])

    assert per_domain.max() <= 1.1 * per_domain.mean()


def test_merge_edges_labels_components_by_smallest_key() -> None:
    edges = np.array([[7, 3], [3, 12], [40, 41], [12, 5]], dtype=np.int64)
    keys, roots = merge_edges(edges)

    assert np.array_equal(keys, [3, 5, 7, 12, 40, 41])
    assert np.array_equal(roots, [3, 3, 3, 3, 40, 40])


def test_single_rank_matches_serial() -> None:
    rng = np.random.default_rng(SEED)
    dm_pos = make_clustered_positions(rng, n_background=5000, n_clusters=20)
    gas_pos = make_clustered_positions(rng, n_background=3000, n_clusters=20)

    expected = find_fof_haloes(
        dm_pos=dm_pos, boxsize=BOXSIZE, linking_length=LINKING_LENGTH, baryon_pos={"gas": gas_pos}, min_members=20
    )
    halo_ids, n_haloes = find_fof_haloes_mpi(
        dm_pos=dm_pos,
        dm_offset=0,
        boxsize=BOXSIZE,
        comm=MPI.COMM_SELF,
        linking_length=LINKING_LENGTH,
        baryon_pos={"gas": gas_pos},
        min_members=20,
    )

    assert n_haloes == expected["dm"].max() + 1
    for ptype in ("dm", "gas"):
        assert np.array_equal(halo_ids[ptype], expected[ptype])


# linking lengths give grids of 116 cells (many domains), 15 (a halo spanning the box; 3 planes per domain at 5 ranks)
# and 9 (two domains at most, or fewer domains than ranks)
@pytest.mark.parametrize(
    ("n_ranks", "force_messages"), [(2, False), (3, False), (5, False), (3, True)]
)  # force_messages: exchange particles with the large-count fallback
def test_mpi_matches_serial(n_ranks: int, force_messages: bool) -> None:
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("mpirun not available")

    env = {
        # pytest's own process has initialised MPI, whose launcher variables would confuse a nested mpirun
        **{k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "PRTE_"))},
        "OMP_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        # Open MPI (4, 5) refuses more ranks than cores, e.g. on 4-core CI runners; MPICH ignores these
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        "PRTE_MCA_rmaps_default_mapping_policy": ":oversubscribe",
    }
    result = subprocess.run(
        [mpirun, "-n", str(n_ranks), sys.executable, "-m", "tests.fof_mpi_worker"]
        + (["--force-messages"] if force_messages else [])
        + [str(LINKING_LENGTH), "12", "20"],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_decomposition_warns_about_idle_ranks_and_imbalance(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from octavius.galaxy_finding.fof_halo_mpi import log_decomposition

    logger = logging.getLogger("OCTAVIUS")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="OCTAVIUS")
    try:
        counts = np.ones(12, dtype=np.int64)
        counts[0] = 1000  # one dense plane that can't be split
        log_decomposition(counts=counts, bounds=np.array([0, 2, 4, 6, 8, 10, 12]), n_ranks=8)
    finally:
        logger.removeHandler(caplog.handler)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "2 of 8 ranks are idle" in messages
    assert "busiest rank" in messages
