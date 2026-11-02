"""

Tests the logger's per-rank terminal output.

"""

import logging
from pathlib import Path

import pytest

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
