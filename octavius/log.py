"""

This file contains everything to do with the logger, which prints output to the terminal. It is MPI-aware,
and there is functionality for concatenating per-rank logs at the end of a run.

"""

# default libraries
import logging
from pathlib import Path

# pass as extra= for messages every rank logs identically (e.g. about the config or input files), so the terminal shows
# them once (from rank 0) rather than once per rank; every rank's log file still records them
SAME_ON_ALL_RANKS = {"same_on_all_ranks": True}


class _RankZeroOnly(logging.Filter):
    """
    Drops messages logged with extra=SAME_ON_ALL_RANKS, for the terminal handler of ranks other than 0.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "same_on_all_ranks", False)


BANNER = """
████████████████████████████████████
═                                  ═
═ ░█▀█░█▀▀░▀█▀░█▀█░█░█░▀█▀░█░█░█▀▀ ═
═ ░█░█░█░░░░█░░█▀█░▀▄▀░░█░░█░█░▀▀█ ═
═ ░▀▀▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀▀▀░▀▀▀ ═
═                                  ═
████████████████████████████████████\n
The next-generation simulation analysis toolkit.
"""


def configure_logger(
    snapshot_path: Path, rank: int = 0, output_level: str = "INFO", log_dir: Path | None = None
) -> logging.Logger:
    """
    Creates a logger object for use throughout stages.

    Specify output_log_dir if you'd like a .log file with more detailed information.
    """
    logger = logging.getLogger(name="OCTAVIUS")

    if logger.handlers:
        return logger

    logger.setLevel(level=logging.DEBUG)  # allow all calls through, so handlers can suppress lower levels
    logger.propagate = False

    rank_tag = f"RANK {rank}"
    terminal_format = f"OCTAVIUS [{rank_tag}] | %(levelname)s | %(message)s"
    file_format = f"%(asctime)s | OCTAVIUS [{rank_tag}] | %(levelname)s | %(message)s"

    # if you have todohighlights on this block might look weird (it's the console text)
    console = logging.StreamHandler()
    console.setLevel(
        getattr(logging, output_level.upper()) if rank == 0 else logging.WARNING
    )  # ranks != 0 only flag any errors
    console.setFormatter(logging.Formatter(terminal_format))
    if rank != 0:
        console.addFilter(_RankZeroOnly())  # rank 0 already shows messages common to all ranks
    logger.addHandler(console)  # from the getattr call above, this means output_level defines what you see

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = intermediate_log_path(snapshot_path=snapshot_path, directory=log_dir, rank=rank)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Helper to retrieve the Octavius logger; please call configure_logger() if not already done so, as early in the pipeline as possible.
    """
    logger = logging.getLogger("OCTAVIUS")

    return logger


def intermediate_log_path(snapshot_path: Path, directory: Path, rank: int) -> Path:
    """
    Returns the Path object pointing to the intermediate log filename for a rank.
    """
    return directory / f"octavius_{snapshot_path.stem}_rank_{rank}.log"


def merged_log_path(snapshot_path: Path, output_dir: Path) -> Path:
    """
    Returns the Path object pointing to the merged log filename.
    """
    return output_dir / f"octavius_{snapshot_path.stem}.log"


def clean_logs(
    output_dir: Path,
    snapshot_path: Path,
    n_ranks: int,
    keep_logs: bool,
) -> None:
    """
    Cleans the log directory by either merging per-rank logs into one or deleting them entirely.
    """
    logger = get_logger()
    if keep_logs:  # concatenates the per-rank logs rather than time-based zipper merging
        merged_log = merged_log_path(snapshot_path=snapshot_path, output_dir=output_dir)
        with open(merged_log, "w") as out:
            for i in range(n_ranks):
                rank_log = intermediate_log_path(snapshot_path=snapshot_path, directory=output_dir, rank=i)
                if rank_log.exists():
                    out.write(
                        rank_log.read_text()
                    )  # this writes the logs sequentially so it appears in rank order in the merged log
                    rank_log.unlink()
            logger.info(f"Merged then cleaned up {n_ranks} log files.")
    else:
        for i in range(n_ranks):
            (intermediate_log_path(snapshot_path=snapshot_path, directory=output_dir, rank=i)).unlink(
                missing_ok=True
            )  # just remove intermediate logs
        logger.info(f"Removed {n_ranks} log files.")


def instantiation_message(
    snapshot_path: Path,
    simulation_type: str,
    halo_source: str,
    version: str,
    n_ranks: int,
    cores_per_rank: int,
    stages: list[str],
) -> None:
    """
    Logs the startup information (snapshot path, parallelism info, enabled stages, simulation type,
    halo ID source) in a block.
    """
    logger = get_logger()
    snapshot_size = snapshot_path.stat().st_size / (1024**3)  # in GB

    lines = [
        f"  Snapshot: {snapshot_path.stem} ({snapshot_size:.1f} GB)",
        f"  Type: {simulation_type} | Halo Source: {halo_source}",
        f"  Ranks: {n_ranks} | Per-Rank Threads: {cores_per_rank}",
        f"  Stages: {', '.join(stages)}",  # unpack list
        f"  Version: {version}",
    ]
    width = max(len(line) for line in lines) + 4
    logger.info("=" * max(width, 50))
    for line in lines:
        logger.info(line)
    logger.info("=" * max(width, 50))


def output_summary(
    all_timings: list[dict[str, float]],
    catalogue_path: Path,
    n_ranks: int,
) -> None:
    """
    Logs the analysis summary.
    """
    logger = get_logger()
    catalogue_size = catalogue_path.stat().st_size / (1024**2)

    title = f"  Catalogue: {catalogue_path.name} ({catalogue_size:.1f} MB)"
    width = max(len(title), 50) + 4
    logger.info("=" * width)
    logger.info(title)

    stages = all_timings[0].keys()
    for stage in stages:
        times = [t[stage] for t in all_timings if stage in t]
        if n_ranks == 1:
            logger.info(f"  {stage}: {times[0]:.1f}s")
        else:
            logger.info(f"  {stage}: {max(times):.1f}s (spread of {max(times) - min(times):.1f}s)")

    total = max(sum(t.values()) for t in all_timings)
    logger.info(f"  Total runtime: {total:.1f}s")
    logger.info("=" * width)
