"""

Contains the main() function, and the layout of the end-to-end Octavius analysis pipeline.

"""

# type checking (semantic, do not worry about this)

from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from mpi4py import MPI
    from .data_management import SnapshotReader, RankPackedData, RedistributionMap
    from .external_halo_sources import SubhaloInformation

# default libraries
from pathlib import Path
import argparse
import shutil
from dataclasses import replace
from contextlib import contextmanager
from time import perf_counter

# other packages
import numpy as np
import numba

# internal package imports
from .data_management import (
    GroupStore,
    SimulationData,
    Internals,
    OctaviusConstants,
    OctaviusConfig,
    build_reader,
    build_galaxy_store,
    build_halo_store,
    build_particle_stores,
    build_redistribution_map,
    load_internals,
    resolve_dependencies,
    load_stage_columns,
    release_stage_columns,
    validate_stage_requirements,
    generate_rank_halo_assignments,
    generate_slabs,
    redistribute_data,
    assign_local_subhaloes,
    construct_membership_arrays,
    pack_rank_data,
    output_catalogue_path,
    write_catalogue,
    write_catalogue_headers,
)
from .external_halo_sources import (
    HaloAssignments,
    build_halo_source,
)
from .galaxy_finding import FOF6DResult, find_galaxies
from .aggregate_properties import (
    run_ptype_specific_properties,
    run_core_properties,
    run_local_environment,
    assign_membership,
)
from .photometry import (
    run_photometry,
)
from .log import configure_logger, get_logger, clean_logs, instantiation_message, output_summary, BANNER
from .utils import repack_catalogue
from .version import __version__

# internal filepaths
CONFIG_PATH = Path(__file__).parent / "config.yaml"  # development config
INTERNALS_PATH = Path(__file__).parent / "internals.yaml"


def get_mpi_communicator() -> MPI.Comm | None:
    """
    Checks whether MPI is enabled; if so, returns the comm object.
    """
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except ImportError:
        pass
    return None


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parses the command-line arguments; returns the corresponding Namespace object.
    """
    parser = argparse.ArgumentParser(
        prog="octavius",
        description="The next generation simulation analysis toolkit.",
        epilog="Thank you for using Octavius!",
        suggest_on_error=True,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("analyse", help="Analyses a snapshot.")
    run_parser.add_argument(
        "-s",
        "--snapshot",
        type=Path,
        required=False,
        default=None,
        help="The filepath to the snapshot you would like to analyse.",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help="Directory to which you would like the outputs to be routed.",
    )
    run_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=False,
        help="The filepath to your config; if not provided, the default config.yaml from the repository is parsed, which can be edited directly.",
    )
    run_parser.add_argument(
        "-hc",
        "--halo-catalogue",
        type=Path,
        required=False,
        default=None,
        help="Path to an external halo catalogue (if not using the snapshot haloes).",
    )
    run_parser.add_argument(
        "--quiet", action="store_true", required=False, help="Suppress all terminal output bar warnings."
    )

    init_parser = subparsers.add_parser(
        "init", help="Generate a default config .yaml file to the current working directory."
    )
    init_parser.add_argument(
        "-o", "--output", type=Path, default=Path("."), help="Directory where you would like the config file placed."
    )

    return parser.parse_args(args=args)


def execute_pipeline(
    config: OctaviusConfig,
    internals: Internals,
    constants: OctaviusConstants,
    reader: SnapshotReader,
    halo_assignments: HaloAssignments,
    global_subhalo_info: SubhaloInformation,
    timings: dict[str, float],
) -> RankPackedData:
    """
    Executes each toggled stage of the Octavius pipeline.
    """
    if halo_assignments.n_field_haloes == 0:  # high-z snapshot early return
        logger = get_logger()
        logger.warning("No haloes exist: exiting pipeline.")
        return RankPackedData.empty()

    sim = reader.simulation_attributes
    with timer("Initialise particle data structures", timings=timings):
        particles = build_particle_stores(
            reader=reader, internals=internals, halo_assignments=halo_assignments, process_ptypes=config.process_ptypes
        )
    subhalo_info = assign_local_subhaloes(
        particles=particles, subhalo_info=global_subhalo_info
    )  # this is done locally and is safe, not worth optimising (though an elegant solution is always welcome)

    if config.stages.get("find_galaxies", True):
        with timer("Load FOF6D columns", timings=timings):
            load_stage_columns(particles=particles, reader=reader, stage=internals.stages["find_galaxies"])
        with timer("Find Galaxies", timings=timings):
            fof6d_result = find_galaxies(particles=particles, simulation=sim, config=config, constants=constants)

    else:  # other functions have guards built for no galaxies so we set IDs equal to the sentinel value
        for ptype in particles:
            particles[ptype]["GalID"] = np.full(particles[ptype].n_particles, -1, dtype=np.int64)
        fof6d_result = FOF6DResult.empty()

    with timer("Initialise group data structures", timings=timings):
        groups: dict[str, GroupStore] = {}
        groups["haloes"] = build_halo_store(  # must build halo store first
            particles=particles,
            halo_key=internals.group_types["haloes"]["key"],
            subhalo_key="SubhaloID",
            group_kind=internals.group_types["haloes"]["kind"],
            subhalo_info=subhalo_info,
            original_halo_ids=halo_assignments.original_field_ids,
        )

        if fof6d_result.n_galaxies > 0:
            groups["galaxies"] = build_galaxy_store(
                particles=particles,
                baryonic_ptypes=[pt for pt, store in particles.items() if store.is_baryonic],
                galaxy_key=internals.group_types["galaxies"]["key"],
                group_kind=internals.group_types["galaxies"]["kind"],
            )

    with timer("Prepare pipeline", timings=timings):
        simulation_data = SimulationData(simulation=sim, constants=constants, particles=particles, groups=groups)
        assign_membership(simulation_data=simulation_data, subhalo_info=subhalo_info, config=config)

        requested = [name for name, enabled in config.stages.items() if enabled and name != "find_galaxies"]
        ordered_stages = resolve_dependencies(stages=internals.stages, requested=requested)
        ordered_stages = validate_stage_requirements(
            ordered_stages=ordered_stages, available_ptypes=set(particles.keys())
        )  # overwrite with stages which can actually run

    stage_dispatch = {
        "properties_core": run_core_properties,
        "properties_ptype_specific": run_ptype_specific_properties,
        "properties_local_environment": run_local_environment,
        "photometry": run_photometry,
    }

    for stage_idx, stage in enumerate(ordered_stages):
        with timer(f"Load {stage.name} data", timings=timings):
            load_stage_columns(particles=particles, reader=reader, stage=stage)
        with timer(f"Run {stage.name}", timings=timings):
            stage_dispatch[stage.name](simulation_data=simulation_data, config=config)
        release_stage_columns(particles=particles, current_idx=stage_idx, ordered_stages=ordered_stages)

    if reader.global_indices is not None:
        particle_indices = reader.global_indices
    else:
        particle_indices = {ptype: np.arange(len(particles[ptype]), dtype=np.int64) for ptype in particles}

    with timer("Save data", timings=timings):
        membership_arrays = construct_membership_arrays(
            data=simulation_data, internals=internals, indices=particle_indices
        )
        packed_data = pack_rank_data(
            data=simulation_data, particle_membership_arrays=membership_arrays, internals=internals
        )

    return packed_data


def analyse_snapshot(
    config: OctaviusConfig,
) -> Path:
    """
    Runs the full Octavius analysis pipeline. This function will automatically discern whether it is being
    run in serial or parallel configuration. It is importable and can be run standalone in a Python script;
    for batch processing and larger snapshots, it is recommended users instead use the Octavius command line
    functionality instead.

    Parameters
    ----------
    config: OctaviusConfig
        Configuration object. You can call the from_yaml(yaml_filepath) method on it
        to parse a config.yaml file, or type the parameters manually.

    Returns
    -------
    catalogue_path: pathlib.Path
        Path object pointing towards the analysis catalogue.

    Notes
    -----
    The snapshot filepath in the config can also be specified through command line arguments.
    Please run --help for more information.
    """
    comm = get_mpi_communicator()
    rank = comm.Get_rank() if comm else 0  # top-level rank parallelism
    size = comm.Get_size() if comm else 1
    numba.set_num_threads(n=config.cores_per_rank)  # intra-rank parallelism

    if rank == 0 and not config.quiet:
        print(BANNER, flush=True)

    if comm:
        comm.Barrier()  # so the banner doesn't print after the analysis

    # initialise logger for console output
    configure_logger(
        snapshot_path=config.snapshot_path,
        rank=rank,
        output_level=config.terminal_output_level,
        log_dir=config.output_dir,
    )
    logger = get_logger()
    instantiation_message(
        snapshot_path=config.snapshot_path,
        simulation_type=config.simulation_type,
        halo_source=config.halo_id_source,
        version=__version__,
        n_ranks=size,
        cores_per_rank=config.cores_per_rank,
        stages=[name for name, enabled in config.stages.items() if enabled],
    )
    logger.info("Instantiating analysis data structures.")

    # initialise snapshot/halo readers, constants, internal metadata
    internals = load_internals(internals_filepath=INTERNALS_PATH, config=config)
    oc = OctaviusConstants(mu=config.MU, frad=config.FRAD)
    reader = build_reader(snapshot_path=config.snapshot_path, constants=oc, config=config)
    halo_source = build_halo_source(config=config, reader=reader)

    # parallelism: rank 0 determines which haloes need to go to which rank
    if rank == 0:  # no need for comm.Barrier() here as scatter does it inherently
        all_halo_assignments = halo_source.read_halo_ids(ptypes=reader.available_ptypes)
        subhalo_info = halo_source.read_subhalo_info()
        halo_to_rank = generate_rank_halo_assignments(
            halo_assignments=all_halo_assignments, config=config, n_ranks=size
        )
        original_halo_ids = all_halo_assignments.original_field_ids
        halo_to_rank_length = len(halo_to_rank)
        assert halo_to_rank.dtype == np.int64, (
            "Bcast is receiving wrong dtype (bit corruption)."
        )  # broadcasting is done on assumption it deals with 64-bit data

    else:  # avoid MPI syntax error
        halo_to_rank = None
        subhalo_info = None
        original_halo_ids = None
        halo_to_rank_length = 0

    if comm is not None:
        halo_to_rank_length = comm.bcast(halo_to_rank_length, root=0)  # this is just an int so bcast
        logger.info("Rank halo allocations broadcast successfully.")

        if rank != 0:
            halo_to_rank = np.empty(halo_to_rank_length, dtype=np.int64)  # malloc

        comm.Bcast(halo_to_rank, root=0)  # capital B broadcast for halo_to_rank
        subhalo_info = (
            comm.bcast(subhalo_info, root=0) if comm else subhalo_info
        )  # lowercase b broadcast for the subhalo dataclass (subset of haloes so smaller)
        original_halo_ids = comm.bcast(original_halo_ids, root=0) if comm else original_halo_ids

        # ranks determine which slab of each dataset they will read
        slabs = generate_slabs(rank=rank, n_ranks=size, particle_counts=reader.particle_counts)

        # rank 0 tells other ranks what the (Sub)HaloIDs of the particles on their slabs are
        field_ids = halo_source.distribute_field_ids(
            slabs=slabs, comm=comm, global_ids=all_halo_assignments.field_ids if rank == 0 else None
        )
        sub_ids = halo_source.distribute_sub_ids(
            slabs=slabs, comm=comm, global_subhalo_ids=all_halo_assignments.sub_ids if rank == 0 else None
        )

    else:
        slabs = generate_slabs(rank=0, n_ranks=1, particle_counts=reader.particle_counts)
        field_ids = halo_source.distribute_field_ids(slabs=slabs)
        sub_ids = halo_source.distribute_sub_ids(slabs=slabs)

    # ranks determine the mapping from their slab to other ranks, and the mask for their own allocation of their slab
    masks: dict[str, np.ndarray] = {}
    maps: dict[str, RedistributionMap] = {}
    local_halo_ids: dict[str, np.ndarray] = {}
    local_subhalo_ids: dict[str, np.ndarray] | None = {} if sub_ids is not None else None

    for ptype in sorted(field_ids):  # ranks must iterate in same order
        maps[ptype], masks[ptype] = build_redistribution_map(halo_to_rank, field_ids[ptype], comm)
        local_halo_ids[ptype] = redistribute_data(field_ids[ptype][masks[ptype]], maps[ptype], comm)
        if sub_ids is not None:
            local_subhalo_ids[ptype] = redistribute_data(sub_ids[ptype][masks[ptype]], maps[ptype], comm)

    rank_halo_assignments = HaloAssignments(
        field_ids=local_halo_ids,
        n_field_haloes=len(halo_to_rank),
        sub_ids=local_subhalo_ids,
        original_field_ids=original_halo_ids,
    )

    reader.set_maps(  # store this info on the reader for reading datasets
        slabs=slabs, masks=masks, maps=maps, comm=comm
    )

    # readers now know where all the data goes, so clear rank 0's HaloID allocation
    all_halo_assignments = None
    field_ids = None
    sub_ids = None

    # analysis pipeline
    timings: dict[str, float] = {}
    packed_data = execute_pipeline(
        config=config,
        internals=internals,
        constants=oc,
        reader=reader,
        halo_assignments=rank_halo_assignments,
        global_subhalo_info=subhalo_info,
        timings=timings,
    )

    catalogue_path = output_catalogue_path(snapshot_path=config.snapshot_path, output_dir=config.output_dir)

    write_catalogue(
        packed_data=packed_data,
        catalogue_path=catalogue_path,
        internals=internals,
        comm=comm,
    )

    # diagnostics
    if comm is not None:
        all_timings = comm.gather(timings, root=0)
    else:
        all_timings = [timings]

    if rank == 0:
        write_catalogue_headers(
            catalogue_path=catalogue_path,
            snapshot_path=config.snapshot_path,
            config=config,
            internals=internals,
            sim_attrs=reader.simulation_attributes,
            n_ranks=size,
        )

        if rank == 0:
            if config.compress_catalogue:
                try:
                    repack_catalogue(catalogue_path)
                except FileNotFoundError:
                    logger.warning("h5repack not available; indices in catalogue are uncompressed.")

            output_summary(
                all_timings=all_timings,
                catalogue_path=catalogue_path,
                n_ranks=size,
            )
            clean_logs(
                output_dir=config.output_dir,
                snapshot_path=config.snapshot_path,
                n_ranks=size,
                keep_logs=config.keep_logs,
            )

    return catalogue_path


def generate_config(output_dir: Path = Path(".")) -> None:
    """
    Generates a default config.yaml file in the requested output directory.

    Parameters
    ----------
    output_dir: pathlib.Path
        Path object pointing to the output directory where the config.yaml file should be generated.
        Default: the current working directory.

    Notes
    -----
    The file will be named octavius_config.yaml by default.
    """
    default = CONFIG_PATH
    target = output_dir / "octavius_config.yaml"

    if target.exists():
        raise FileExistsError(f"{target} already exists.")

    shutil.copy(default, target)


@contextmanager
def timer(label: str, timings: dict[str, float]) -> Generator[None, None, None]:
    """
    Logs the time taken for a stage (reduced version of validation suite time_and_memory)
    """
    logger = get_logger()
    t0 = perf_counter()
    yield
    elapsed = perf_counter() - t0
    timings[label] = elapsed
    logger.info(f"{label} completed in {elapsed:.1f}s.")


def main() -> None:
    """
    main() function; currently branches on the init (config generation) and analyse
    (analyse snapshot) paths from the command line.
    """
    args = parse_args()

    if args.command == "init":
        generate_config(output_dir=args.output)

    elif args.command == "analyse":
        config_path = args.config if args.config else CONFIG_PATH
        config = OctaviusConfig.from_yaml(config_path=config_path)

        if args.snapshot is not None:
            config = replace(config, snapshot_path=args.snapshot)
        if args.output is not None:
            config = replace(config, output_dir=args.output)
        if args.halo_catalogue is not None:
            config = replace(config, halo_catalogue_path=args.halo_catalogue)
        if args.quiet:
            config = replace(config, quiet=True, terminal_output_level="ERROR", keep_logs=False)

        if config.snapshot_path is None:
            raise ValueError("Please provide a snapshot path.")
        if config.output_dir is None:
            raise ValueError("Please provide an output directory path.")
        if config.halo_id_source != "SNAPSHOT" and config.halo_catalogue_path is None:
            raise ValueError(
                f"{config.halo_id_source} also requires a catalogue containing ID assignments to be specified in 'halo_catalogue_path'."
            )

        analyse_snapshot(config=config)


if __name__ == "__main__":
    main()
