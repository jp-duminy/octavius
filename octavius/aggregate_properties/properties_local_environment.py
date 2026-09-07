"""

Properties related to a structure's local environment (currently only aperture masses).

# TODO: the aperture masses are currently inaccurate for galaxies at the edge of a halo, because particles are split
across ranks by halo and not spatially decomposed.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import ParticleStore, GroupStore, SimulationData, OctaviusConfig

# other packages
import numpy as np
from scipy.spatial import KDTree  # remember, always pass boxsize

# internal imports
from ..log import get_logger
from ..data_management.conventions import DTYPES
from ..data_management import build_group_csr

logger = get_logger()


def run_local_environment(simulation_data: SimulationData, config: OctaviusConfig) -> None:
    """
    Top-level executor for local environment properties.
    """
    if "galaxies" not in simulation_data.groups:
        logger.info("Skipping local environment properties: no galaxies found.")
        return

    galaxies = simulation_data.groups["galaxies"]
    logger.info(f"Running local environment properties: {galaxies.n_groups} members")
    sim = simulation_data.simulation

    for aperture in config.aperture_size:
        aperture_results = compute_galaxy_aperture_masses(
            particles=simulation_data.particles,
            galaxies=galaxies,
            haloes=simulation_data.groups["haloes"],
            boxsize=sim.boxsize,
            aperture_size=aperture,
            cores_per_rank=config.cores_per_rank,
        )
        galaxies.write_batch(results=aperture_results)

    logger.info("Local environment properties computed.")


def compute_galaxy_aperture_masses(  # FIXME: aperture masses of galaxies at edge of halo are less accurate as particles are split across ranks
    particles: dict[str, ParticleStore],
    galaxies: GroupStore,
    haloes: GroupStore,
    boxsize: float,
    aperture_size: float,
    cores_per_rank: int,
) -> dict[str, np.ndarray]:
    """
    Computes mass in an aperture of defined size around galaxies, returning a dict of:

    - mass_{p}_{aperture_size}kpc for p in HI, H2, {ptypes}, total
    """
    results: dict[str, np.ndarray] = {}
    pos_list, mass_list = [], []
    ptype_list, hids_list = [], []
    ptypes = ["star", "gas", "bh", "dm"]
    ptype_to_int = {
        p: i for i, p in enumerate(ptypes)
    }  # integer comparison is faster and cheaper than string comparison

    # originally this got kind of hacked in by inserting these as ptypes, do it explicitly instead
    particle_counts = [len(particles[ptype]) for ptype in ptypes]
    gas_offset = particle_counts[ptypes.index("gas")]  # cumsum up to gas
    gas_start = sum(particle_counts[: ptypes.index("gas")])
    gas_end = gas_start + gas_offset

    total_particles = sum(particle_counts)
    all_mass_HI = np.zeros(total_particles, dtype=DTYPES["mass"])
    all_mass_H2 = np.zeros(total_particles, dtype=DTYPES["mass"])
    all_mass_HI[gas_start:gas_end] = particles["gas"]["mass_HI"]
    all_mass_H2[gas_start:gas_end] = particles["gas"]["mass_H2"]

    for ptype in ptypes:
        data = particles[ptype]
        pos_list.append(data["pos"])
        mass_list.append(data["mass"])
        ptype_list.append(np.full(shape=len(data), fill_value=ptype_to_int[ptype], dtype=DTYPES["ptype"]))
        hids_list.append(data["HaloID"])

    # TODO: find a way around these concats
    all_pos, all_mass = np.concatenate(pos_list, dtype=DTYPES["pos"]), np.concatenate(mass_list, dtype=DTYPES["mass"])
    all_ptypes, all_hids = (
        np.concatenate(ptype_list, dtype=DTYPES["ptype"]),
        np.concatenate(hids_list, dtype=DTYPES["HaloID"]),
    )

    halo_idx = haloes.get_indexer(group_id=all_hids)
    halo_offsets, halo_idx_sorted = build_group_csr(group_idx=halo_idx, n_groups=haloes.n_groups)

    # galaxy membership aligned to their parent haloes
    parent_halo_ids = haloes.group_ids[galaxies["parent_halo_index"]]
    parent_halo_idx = haloes.get_indexer(group_id=parent_halo_ids)
    gal_offsets, gal_idx_sorted = build_group_csr(group_idx=parent_halo_idx, n_groups=haloes.n_groups)

    result = np.zeros(shape=(galaxies.n_groups, len(ptypes)))
    result_HI, result_H2 = np.zeros(shape=galaxies.n_groups), np.zeros(shape=galaxies.n_groups)
    gal_pos = galaxies["com_pos_baryon"]  # avoid repeated lookup

    for h in range(
        haloes.n_groups
    ):  # a galaxy's aperture can and does often extend into multiple haloes, so parallelising this is unsafe
        if gal_offsets[h] == gal_offsets[h + 1]:  # if halo has no galaxies
            continue

        halo_slice: slice = slice(halo_offsets[h], halo_offsets[h + 1])

        # what's in this halo
        halo_pos, halo_mass = all_pos[halo_idx_sorted[halo_slice]], all_mass[halo_idx_sorted[halo_slice]]
        halo_mass_HI, halo_mass_H2 = all_mass_HI[halo_idx_sorted[halo_slice]], all_mass_H2[halo_idx_sorted[halo_slice]]
        halo_ptypes = all_ptypes[halo_idx_sorted[halo_slice]]

        gal_slice: slice = slice(gal_offsets[h], gal_offsets[h + 1])

        gal_indices = gal_idx_sorted[gal_slice]  # indices of galaxies in this halo
        gal_positions = gal_pos[gal_indices]  # positions of galaxies in this halo

        tree = KDTree(data=halo_pos, boxsize=boxsize)
        neighbour_lists = tree.query_ball_point(
            x=gal_positions, r=aperture_size, workers=cores_per_rank
        )  # returns tuple

        for local_idx, neighbours in enumerate(neighbour_lists):
            if len(neighbours) == 0:
                continue

            neighbours = np.array(neighbours)
            result_HI[gal_indices[local_idx]] = halo_mass_HI[neighbours].sum()
            result_H2[gal_indices[local_idx]] = halo_mass_H2[neighbours].sum()
            masses_by_type = np.bincount(halo_ptypes[neighbours], weights=halo_mass[neighbours], minlength=len(ptypes))
            result[gal_indices[local_idx], :] = masses_by_type

    for i, name in enumerate(ptypes):
        results[f"mass_{name}_{int(aperture_size)}kpc"] = result[:, i]

    results[f"mass_HI_{int(aperture_size)}kpc"] = result_HI
    results[f"mass_H2_{int(aperture_size)}kpc"] = result_H2
    results[f"mass_total_{int(aperture_size)}kpc"] = result[:, : len(ptypes)].sum(axis=1)

    return results
