"""

Functionality to assign galaxies their parent haloes. This has two paths: if there is no subhalo information,
it naïvely assigns galaxies a parent_halo_index corresponding to the HaloID of the first particle in the galaxy;
the parent_membership_fraction is then trivially also 1.0. If subhalo information is present, it uses a
plurality vote to determine parent_halo_index and registers parent_membership_fraction accordingly.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import ParticleStore, GroupStore, SimulationData, OctaviusConfig
    from ..external_halo_sources import SubhaloInformation
from ..data_management import DTYPES, build_group_csr, build_galaxy_store
from .aggregate_helpers import first_idx_per_group

# other packages
import numpy as np
from numba import njit

# internal imports
from ..log import get_logger

logger = get_logger()


def assign_membership(
    simulation_data: SimulationData,
    config: OctaviusConfig,
    subhalo_info: SubhaloInformation | None = None,
) -> None:
    """
    Assigns the galaxy groupstore the parent information depending on whether subhalo info exists.
    """
    if "galaxies" not in simulation_data.groups:  # early return in case FOF6D did not find galaxies
        return

    results: dict[str, np.ndarray] = {}
    haloes = simulation_data.groups["haloes"]
    galaxies = simulation_data.groups["galaxies"]
    particles = simulation_data.particles
    available_baryonic = simulation_data.available_baryonic_ptypes
    n_field_haloes = np.sum(haloes["depth"] == 0) if "depth" in haloes else haloes.n_groups

    field_halo_index = assign_galaxy_field_indices(
        particles=particles,
        galaxies=galaxies,
        haloes=haloes,
        available_baryonic_ptypes=available_baryonic,
        n_field_haloes=n_field_haloes,
    )

    if subhalo_info is not None:
        raw_winners, parent_membership_frac = assign_galaxy_halo_indices(
            particles=particles,
            galaxies=galaxies,
            available_baryonic_ptypes=available_baryonic,
            n_subhaloes=len(subhalo_info.depth),
        )
        parent_halo_index = np.where(raw_winners >= 0, raw_winners + n_field_haloes, field_halo_index)
        parent_membership_frac[raw_winners < 0] = 1.0  # interlopers

        # optional FOF6D subhalo finder override: trim and rerun the steps
        if config.subhalo_override:
            logger.info("Overriding FOF6D-assigned subhalo boundaries.")

            _trim_galaxy_interlopers(  # mutates in place
                particles=particles,
                galaxies=galaxies,
                winning_subhalo_idx=raw_winners,
                available_baryonic_ptypes=available_baryonic,
                minstars=config.min_stars_per_galaxy,
                mingas=config.min_gas_per_galaxy,
            )
            galaxies = build_galaxy_store(
                particles=particles,
                baryonic_ptypes=available_baryonic,  # reassign galaxy store in-place
                galaxy_key="GalID",
                group_kind="galaxy",
            )

            # now need to rerun linking steps as the galaxy store's length and particles are different
            field_halo_index = assign_galaxy_field_indices(
                particles=particles,
                galaxies=galaxies,
                haloes=haloes,
                n_field_haloes=n_field_haloes,
                available_baryonic_ptypes=available_baryonic,
            )

            new_winners, parent_membership_frac = assign_galaxy_halo_indices(  # parent frac is now 1.0 by construction
                particles=particles,
                galaxies=galaxies,
                available_baryonic_ptypes=available_baryonic,
                n_subhaloes=len(subhalo_info.depth),
            )
            parent_halo_index = np.where(new_winners >= 0, new_winners + n_field_haloes, field_halo_index)
            parent_membership_frac[new_winners < 0] = 1.0
            simulation_data.groups["galaxies"] = galaxies

    else:
        parent_halo_index = field_halo_index.copy()
        parent_membership_frac = np.ones(galaxies.n_groups, dtype=np.float32)

    results["field_halo_index"] = field_halo_index
    results["parent_halo_index"] = parent_halo_index
    results["parent_membership_fraction"] = parent_membership_frac
    galaxies.write_batch(results=results)


def assign_galaxy_field_indices(
    particles: dict[str, ParticleStore],
    galaxies: GroupStore,
    haloes: GroupStore,
    available_baryonic_ptypes: list[str],
    n_field_haloes: int,
) -> np.ndarray:
    """
    Naïvely assigns galaxies their parent halo indices based on the HaloID of the first particle in the galaxy.
    """
    gids_list, hids_list = [], []

    for ptype in available_baryonic_ptypes:
        store = particles[ptype]
        gids_list.append(store["GalID"])
        hids_list.append(store["HaloID"])

    all_gids, all_hids = (
        np.concatenate(gids_list, dtype=DTYPES["GalID"]),
        np.concatenate(hids_list, dtype=DTYPES["HaloID"]),
    )
    in_galaxy = all_gids >= 0
    all_gids, all_hids = all_gids[in_galaxy], all_hids[in_galaxy]

    galaxy_idx = galaxies.get_indexer(group_id=all_gids)
    offsets, idx_sorted = build_group_csr(group_idx=galaxy_idx, n_groups=galaxies.n_groups)
    first_particle_idx = first_idx_per_group(offsets=offsets, idx_sorted=idx_sorted, n_groups=galaxies.n_groups)
    valid = first_particle_idx >= 0

    invalid_galaxies = np.sum(~valid)
    if invalid_galaxies > 0:
        logger.warning(f"{invalid_galaxies} galaxies have no baryonic particles!")

    galaxy_halo_id = np.full(shape=galaxies.n_groups, fill_value=-1, dtype=DTYPES["HaloID"])
    galaxy_halo_id[valid] = all_hids[first_particle_idx[valid]]

    field_ids = haloes.group_ids[:n_field_haloes]
    positions = np.searchsorted(field_ids, galaxy_halo_id)
    positions = np.clip(positions, 0, len(field_ids) - 1)
    found = field_ids[positions] == galaxy_halo_id
    field_halo_index = np.where(found, positions, -1)

    n_orphan = np.sum(field_halo_index == -1)
    if n_orphan > 0:
        logger.warning(f"{n_orphan} galaxies exist with no valid parent halo.")

    return field_halo_index


def assign_galaxy_halo_indices(
    particles: dict[str, ParticleStore], galaxies: GroupStore, available_baryonic_ptypes: list[str], n_subhaloes: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigns parent halo index by plurality of subhalo membership.
    """
    all_subhids = []
    all_galaxy_idx = []

    for ptype in available_baryonic_ptypes:
        offsets, idx_sorted = galaxies.get_particle_csr(ptype=ptype)
        all_subhids.append(particles[ptype]["SubhaloID"][idx_sorted])
        group_idx = np.searchsorted(offsets, np.arange(len(idx_sorted)), side="right") - 1
        all_galaxy_idx.append(group_idx)

    subhids, galaxy_idx = np.concatenate(all_subhids), np.concatenate(all_galaxy_idx)

    order = np.argsort(galaxy_idx, stable=True)  # make each galaxy's members contiguous across ptypes
    subhids = subhids[order]  # sorted here (so the numba function can assume so)

    lengths = np.bincount(galaxy_idx, minlength=galaxies.n_groups)
    gal_offsets = np.empty(len(lengths) + 1, dtype=DTYPES["csr_offsets"])
    gal_offsets[0] = 0
    np.cumsum(lengths, out=gal_offsets[1:])

    raw_winners, parent_membership_frac = _find_galaxy_parent(
        gal_offsets=gal_offsets, subhids=subhids, n_galaxies=galaxies.n_groups, n_subhaloes=n_subhaloes
    )

    return raw_winners, parent_membership_frac


def _trim_galaxy_interlopers(
    particles: dict[str, ParticleStore],
    galaxies: GroupStore,
    winning_subhalo_idx: np.ndarray,
    available_baryonic_ptypes: list[str],
    minstars: int,
    mingas: int,
) -> None:
    """
    Enforces the subhalo finder as the authority on substructure boundaries by culling particles of
    galaxies where the parent_membership_fraction is not one.
    """
    csr_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # so the reassignment can be vectorised
    for ptype in available_baryonic_ptypes:
        offsets, idx_sorted = galaxies.get_particle_csr(ptype=ptype)
        galaxy_idx = (
            np.searchsorted(offsets, np.arange(len(idx_sorted)), side="right") - 1
        )  # (RHS-1) gives the position in offsets which is the galaxy ID
        sub_ids = particles[ptype]["SubhaloID"][idx_sorted]
        csr_cache[ptype] = (idx_sorted, galaxy_idx)

        # find subhalos where the finder and octavius disagree
        winning_parents = winning_subhalo_idx[galaxy_idx]  # works because ID is really an index
        interloping = (sub_ids != winning_parents) & (winning_parents >= 0)

        particles[ptype]["GalID"][idx_sorted[interloping]] = -1

    if minstars > 0:
        # now we need to trim galaxies which fell below minstars
        star_offsets, star_idx = galaxies.get_particle_csr(ptype="star")
        original_star_counts = np.diff(star_offsets)  # for diagnostic
        star_gal_ids = particles["star"]["GalID"][star_idx]

        # get the star count in each galaxy
        surviving = star_gal_ids >= 0
        star_galaxy_idx = np.searchsorted(star_offsets, np.arange(len(star_idx)), side="right") - 1  # same logic as above
        surviving_star_counts = np.bincount(star_galaxy_idx[surviving], minlength=galaxies.n_groups)
        trim_mask = surviving_star_counts < minstars
        galaxies_to_trim = trim_mask.nonzero()[0]

        for ptype, (idx_sorted, galaxy_idx) in csr_cache.items():
            particles[ptype]["GalID"][idx_sorted[trim_mask[galaxy_idx]]] = -1

        n_stars_lost = original_star_counts.sum() - surviving_star_counts[~trim_mask].sum()
        logger.debug(f"{n_stars_lost} star particles trimmed when deferring to subhalo finder assignments.")
        logger.debug(f"{len(galaxies_to_trim)} galaxies trimmed.")

    if mingas > 0 and "gas" in available_baryonic_ptypes:
        # now we need to trim galaxies which fell below mingas
        gas_offsets, gas_idx = galaxies.get_particle_csr(ptype="gas")
        original_gas_counts = np.diff(gas_offsets)  # for diagnostic
        gas_gal_ids = particles["gas"]["GalID"][gas_idx]

        # get the gas count in each galaxy
        surviving = gas_gal_ids >= 0
        gas_galaxy_idx = np.searchsorted(gas_offsets, np.arange(len(gas_idx)), side="right") - 1  # same logic as above
        surviving_gas_counts = np.bincount(gas_galaxy_idx[surviving], minlength=galaxies.n_groups)
        trim_mask = surviving_gas_counts < mingas
        galaxies_to_trim = trim_mask.nonzero()[0]

        for ptype, (idx_sorted, galaxy_idx) in csr_cache.items():
            particles[ptype]["GalID"][idx_sorted[trim_mask[galaxy_idx]]] = -1

        n_gas_lost = original_gas_counts.sum() - surviving_gas_counts[~trim_mask].sum()
        logger.debug(f"{n_gas_lost} gas particles trimmed when deferring to subhalo finder assignments.")
        logger.debug(f"{len(galaxies_to_trim)} galaxies trimmed.")


@njit(cache=True)
def _find_galaxy_parent(
    gal_offsets: np.ndarray,
    subhids: np.ndarray,
    n_galaxies: int,
    n_subhaloes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each galaxy, determine its parent subhalo index and parent membership fraction from the subhalo info. Returns a tuple of (parent_halo_index, parent_membership_fraction).
    """
    counts = np.zeros(n_subhaloes, dtype=np.int64)
    parent_halo_index = np.full(shape=n_galaxies, fill_value=-1, dtype=np.int64)
    parent_membership_frac = np.full(shape=n_galaxies, fill_value=0, dtype=np.float32)

    for g in range(n_galaxies):
        start = gal_offsets[g]
        end = gal_offsets[g + 1]

        best_id = -1  # 'best' in this case is the subhalo with the plurality vote
        best_count = 0
        total = 0

        for p in range(start, end):  # keep a running track of which subhalo ID is the winner
            subhid = subhids[p]

            if subhid < 0:  # SubhaloIDs contains sentinels too
                continue

            counts[subhid] += 1
            total += 1
            if counts[subhid] > best_count:
                best_count = counts[subhid]
                best_id = subhid

        if total > 0:  # id/total are already initialised to sentinel values (np.full above)
            parent_membership_frac[g] = best_count / total
            parent_halo_index[g] = best_id

        for p in range(
            start, end
        ):  # re-zero the subhids array before the next iteration of the loop (so you don't repeat np.zeros)
            if subhids[p] >= 0:
                counts[subhids[p]] = 0

    return parent_halo_index, parent_membership_frac
