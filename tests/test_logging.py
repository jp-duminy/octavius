"""

Tests the logger's per-rank terminal output and the peak memory reported in the run summary.

"""

import logging
from pathlib import Path

import pytest

from octavius import run_octavius
from octavius.log import SAME_ON_ALL_RANKS, configure_logger


@pytest.mark.parametrize("rank", [0, 1])
def test_messages_common_to_all_ranks_shown_once(rank: int, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Every rank logs the same warning, so only rank 0 shows it; rank-specific warnings show on every rank.
    """
    logger = logging.getLogger("OCTAVIUS")
    saved = logger.handlers[:]
    logger.handlers.clear()  # configure_logger() is a no-op once handlers exist
    try:
        configure_logger(snapshot_path=Path("snapshot.hdf5"), rank=rank)
        logger.warning("common to all ranks", extra=SAME_ON_ALL_RANKS)
        logger.warning("specific to this rank")
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = saved

    terminal = capsys.readouterr().err
    assert ("common to all ranks" in terminal) == (rank == 0)
    assert "specific to this rank" in terminal


def test_peak_memory() -> None:
    """
    Peak memory is a sensible number of GB on Linux and macOS (whose ru_maxrss units differ), and None on Windows.
    """
    peak = run_octavius._peak_memory_gb()
    if run_octavius.resource is None:
        assert peak is None
    else:
        assert 0.01 < peak < 100  # a Python test process uses tens to hundreds of MB


def test_peak_memory_without_resource_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Without the resource module (as on Windows), peak memory is None rather than an error.
    """
    monkeypatch.setattr(run_octavius, "resource", None)
    assert run_octavius._peak_memory_gb() is None
