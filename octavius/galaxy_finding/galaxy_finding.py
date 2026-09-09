"""

The galaxy finding pipeline. This file includes the python-level bindings to the algorithm in
fof6d_algorithm.py, along with the pre-processing and data storage functions.

"""

# type checking (semantic)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import ParticleStore, SimulationAttributes, OctaviusConstants

# defaults
from dataclasses import dataclass

# other libraries
import numpy as np

# internal imports
from .fof6d_algorithm import dispatch_fof6d
from ..data_management import DTYPES, OctaviusConfig
from ..utils import unwrap_positions
from ..log import get_logger

logger = get_logger()
PTYPE_CODES = {"gas": np.int8(0), "star": np.int8(4), "bh": np.int8(5), "dm": np.int8(1)}  # GIZMO number conventions


@dataclass(slots=True, frozen=True)
class FOF6DData:
    """
    Dataclass to concisely store the arrays being passed into FOF6D. Ensure pos is PBC-unwrapped!
    """

    pos: np.ndarray  # these should be unwrapped
    vel: np.ndarray
    ptype_codes: np.ndarray
    write_keys: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


@dataclass(slots=True, frozen=True)
class FOF6DParameters:
    """
    Fixed simulation/runtime parameters which the algorithm needs.
    """

    linking_length: float
    velocity_factor: float
    boxsize: float
    minstars: int


@dataclass(slots=True, frozen=True)
class FOF6DResult:
    """
    Assignments made by FOF6D for writing back into the ParticleStores; n_galaxies is used to
    flag whether galaxies were found, in which case the executor should build the GroupStore for galaxies.
    """

    write_keys: np.ndarray  # indexes back into the ParticleStores
    galaxy_ids: np.ndarray
    ptype_codes: np.ndarray
    n_galaxies: int

    @classmethod  # for early return guards
    def empty(cls) -> FOF6DResult:
        return cls(
            write_keys=np.empty(0, dtype=np.int64),
            galaxy_ids=np.empty(0, dtype=np.int64),
            ptype_codes=np.empty(0, dtype=np.int8),
            n_galaxies=0,
        )


def find_galaxies(
    particles: dict[str, ParticleStore],
    simulation: SimulationAttributes,
    config: OctaviusConfig,
    constants: OctaviusConstants,
) -> FOF6DResult:
    """
    Handles the end-to-end galaxy-finding with FOF6D pipeline; writes back to the ParticleStores and returns
    the FOF6DResult dataclass.
    """
    logger.info("Locating galaxies with the FOF6D algorithm.")

    # early return if there are no stars and therefore no galaxies
    if "star" not in particles or particles["star"].n_particles == 0:
        logger.warning("No stars found: skipping galaxy finding.")

        for ptype in particles:  # this ensures the galaxies GroupStore is not built
            particles[ptype]["GalID"] = np.full(particles[ptype].n_particles, -1, dtype=np.int64)

        return FOF6DResult.empty()

    work_data, params = prepare_fof6d_data(
        particles=particles, simulation=simulation, config=config, constants=constants
    )

    # guard in case the concatenated position arrays are all empty
    if len(work_data.pos) == 0:
        empty_result = FOF6DResult(
            write_keys=np.empty(0, dtype=np.int64),
            galaxy_ids=np.empty(0, dtype=np.int64),
            ptype_codes=np.empty(0, dtype=np.int8),
            n_galaxies=0,
        )
        store_fof6d_results(particles=particles, result=empty_result)
        logger.warning("No particles pass the FOF6D criteria; no galaxies found.")
        return empty_result

    logger.debug(f"Linking length: {params.linking_length:.3f} ckpc")

    parents = np.full(len(work_data.pos), -1, dtype=np.int32)
    dispatch_fof6d(
        positions=work_data.pos,
        velocities=work_data.vel,
        parents=parents,
        ptype_codes=work_data.ptype_codes,
        starts=work_data.starts,
        ends=work_data.ends,
        linking_length=params.linking_length,
        velocity_factor=params.velocity_factor,
        minstars=params.minstars,
        star_ptype_code=PTYPE_CODES["star"],
    )

    result = extract_galaxies_from_parents(work_data=work_data, parents=parents, minstars=params.minstars)

    store_fof6d_results(particles=particles, result=result)

    logger.info(f"Located {result.n_galaxies:,} galaxies.")

    return result


def prepare_fof6d_data(
    particles: dict[str, ParticleStore],
    simulation: SimulationAttributes,
    config: OctaviusConfig,
    constants: OctaviusConstants,
) -> tuple[FOF6DData, FOF6DParameters]:
    """
    Extracts relevant arrays from ParticleStores and FOF6D parameters from the
    SimulationAttributes & user config. Returns a tuple of:

    - data: FOF6DData dataclass
    - params: FOF6DParameters dataclass
    """
    star_halo_ids = particles["star"]["HaloID"]
    max_halo_id = max(
        (
            int(particles[pt]["HaloID"].max())
            for pt in ("star", "gas", "bh")
            if pt in particles and particles[pt].n_particles > 0
        ),
        default=-1,
    )
    n_haloes = max_halo_id + 1  # this is now the number of field haloes (since that's what we operate on)
    star_counts = np.bincount(
        star_halo_ids[star_halo_ids >= 0], minlength=n_haloes
    )  # NOTE: need to mask sentinel value here

    eligible_haloes = np.where(star_counts >= config.min_stars_per_galaxy)[
        0
    ]  # disregard haloes which would have no galaxies
    eligible_set = np.zeros(star_counts.shape[0], dtype=bool)
    eligible_set[eligible_haloes] = True

    gas_mask = apply_gas_mask(gas=particles["gas"], constants=constants, config=config)

    pos_list, vel_list, ptype_list, index_list, hid_list = [], [], [], [], []
    for ptype in ["star", "gas", "bh"]:
        if ptype not in particles:
            continue

        halo_ids = particles[ptype]["HaloID"]
        in_range = (halo_ids >= 0) & (halo_ids < len(eligible_set))
        masked_hids = np.where(in_range, halo_ids, 0)  # have to mask before & operator

        if ptype == "gas":
            mask = gas_mask & in_range & eligible_set[masked_hids]  # dense criterion for gas specifically
        else:
            mask = in_range & eligible_set[masked_hids]

        pos_list.append(particles[ptype]["pos"][mask])
        vel_list.append(particles[ptype]["vel"][mask])
        ptype_list.append(np.full(mask.sum(), PTYPE_CODES[ptype], dtype=np.int8))
        index_list.append(np.arange(particles[ptype].n_particles)[mask])
        hid_list.append(halo_ids[mask])

    linking_length = simulation.mis * config.b

    params = FOF6DParameters(
        linking_length=linking_length,
        velocity_factor=config.velocity_factor,
        boxsize=simulation.boxsize,
        minstars=config.min_stars_per_galaxy,
    )

    # guard if no valid particles pass the mask
    if len(pos_list) == 0:
        logger.warning("No valid particles found: skipping galaxy finding.")
        empty_data = FOF6DData(
            pos=np.empty((0, 3), dtype=DTYPES["pos"]),
            vel=np.empty((0, 3), dtype=DTYPES["vel"]),
            ptype_codes=np.empty(0, dtype=np.int8),
            write_keys=np.empty(0, dtype=np.int64),
            starts=np.empty(0, dtype=np.int64),
            ends=np.empty(0, dtype=np.int64),
        )
        return empty_data, params

    # NOTE: in-halo concatenation, could be expensive
    all_pos, all_vel = np.concatenate(pos_list, dtype=DTYPES["pos"]), np.concatenate(vel_list, dtype=DTYPES["vel"])
    all_ptype, all_halo_ids = np.concatenate(ptype_list), np.concatenate(hid_list)
    all_write_key = np.concatenate(index_list)

    order = np.argsort(all_halo_ids, stable=True)  # keep stable=True on for deterministic results
    all_pos, all_vel = all_pos[order], all_vel[order]
    all_ptype, all_halo_ids = all_ptype[order], all_halo_ids[order]
    all_write_key = all_write_key[order]

    sorted_halo_ids = all_halo_ids  # just for readability
    changes = np.flatnonzero(np.diff(sorted_halo_ids)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(sorted_halo_ids)]))

    sizes = ends - starts
    size_order = np.argsort(sizes)[::-1]  # process largest haloes first

    ordered_starts = starts[size_order]
    ordered_ends = ends[size_order]

    for halo_idx in range(len(ordered_starts)):
        s, e = ordered_starts[halo_idx], ordered_ends[halo_idx]
        unwrap_positions(
            positions=all_pos[s:e], centre=all_pos[s], boxsize=simulation.boxsize
        )  # pass first particle as centre

    data = FOF6DData(
        pos=all_pos,
        vel=all_vel,
        ptype_codes=all_ptype,
        write_keys=all_write_key,
        starts=ordered_starts,
        ends=ordered_ends,
    )

    return data, params


def apply_gas_mask(
    gas: ParticleStore,
    constants: OctaviusConstants,
    config: OctaviusConfig,
) -> np.ndarray:
    """
    Applies the user-configured FOF6D gas mask: dense, star forming, both.
    """
    rho = gas["rho"]
    sfr = gas["sfr"]
    temperature = gas["temperature"]

    # always use dense gas
    nH = (
        rho * (1 - gas["metallicity"] - gas["helium_fraction"]) / constants.PROTON_MASS_G
    )  # metallicity is Z, helium_frac is Y -> gives XH

    dense_mask = nH > config.nH_lim

    if config.gas_criterion == "COLD_OR_STARFORMING":  # sfr usually overrides the other two
        gas_mask = dense_mask & ((temperature < config.T_lim) | (sfr > 0))
    elif config.gas_criterion == "COLD":
        gas_mask = dense_mask & (temperature < config.T_lim)
    elif config.gas_criterion == "STARFORMING":
        gas_mask = dense_mask & (sfr > 0)
    elif config.gas_criterion == "DENSE_ONLY":
        gas_mask = dense_mask

    logger.debug(
        f"Galaxy gas criterion: {np.sum(gas_mask):,} / {len(gas_mask):,} particles passed ({config.gas_criterion})"
    )

    return gas_mask


def extract_galaxies_from_parents(
    work_data: FOF6DData,
    parents: np.ndarray,
    minstars: int,
) -> FOF6DResult:
    """
    Extracts GalIDs from the parents array returned by the FOF6D algorithm.
    """
    n_haloes = len(work_data.starts)
    all_keys, all_gids, all_ptype_codes = [], [], []
    galaxy_id_offset = 0

    for halo_idx in range(n_haloes):
        s, e = work_data.starts[halo_idx], work_data.ends[halo_idx]

        if parents[s] == -1:
            continue

        halo_parents = parents[s:e]
        component_sizes = np.bincount(halo_parents)
        star_mask = work_data.ptype_codes[s:e] == PTYPE_CODES["star"]
        star_counts = np.bincount(halo_parents[star_mask], minlength=len(component_sizes))

        valid_parents = np.where((component_sizes >= minstars) & (star_counts >= minstars))[0]

        if len(valid_parents) == 0:
            continue

        parent_to_gal_id = np.full(len(component_sizes), -1, dtype=np.int64)
        parent_to_gal_id[valid_parents] = np.arange(len(valid_parents), dtype=np.int64) + galaxy_id_offset

        particle_gal_ids = parent_to_gal_id[halo_parents]
        assigned = particle_gal_ids >= 0

        all_keys.append(work_data.write_keys[s:e][assigned])
        all_gids.append(particle_gal_ids[assigned])
        all_ptype_codes.append(work_data.ptype_codes[s:e][assigned])
        galaxy_id_offset += len(valid_parents)

    if galaxy_id_offset == 0:  # guard for an empty group, needs to match type check in signature
        result = FOF6DResult.empty()
    else:
        result = FOF6DResult(
            write_keys=np.concatenate(all_keys),
            galaxy_ids=np.concatenate(all_gids),
            ptype_codes=np.concatenate(all_ptype_codes),
            n_galaxies=galaxy_id_offset,
        )

    return result


def store_fof6d_results(particles: dict[str, ParticleStore], result: FOF6DResult) -> None:
    """
    Appends the galaxy-finding results to the ParticleStores by instantiating the
    GalID column.
    """
    for ptype in particles:
        particles[ptype]["GalID"] = np.full(particles[ptype].n_particles, -1, dtype=np.int64)  # NOTE: sentinel value

        mask = result.ptype_codes == PTYPE_CODES[ptype]

        if not mask.any():
            continue

        particles[ptype]["GalID"][result.write_keys[mask]] = result.galaxy_ids[mask]
