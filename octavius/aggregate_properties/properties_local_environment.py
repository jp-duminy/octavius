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
from numba import njit, prange

# internal imports
from ..log import get_logger
from ..data_management.conventions import DTYPES
from ..data_management import build_group_csr

logger = get_logger()

PTYPE_CODES = {"gas": np.int8(0), "star": np.int8(4), "bh": np.int8(5), "dm": np.int8(1)}  # GIZMO number conventions


def run_local_environment(simulation_data: SimulationData, config: OctaviusConfig) -> None:
    """
    Top-level executor for local environment properties.
    """
    if "galaxies" not in simulation_data.groups:
        logger.info("Skipping local environment properties: no galaxies found.")
        return

    galaxies = simulation_data.groups["galaxies"]
    haloes = simulation_data.groups["haloes"]
    particles = simulation_data.particles
    logger.info(f"Running local environment properties for galaxies: {galaxies.n_groups:,} members")
    sim = simulation_data.simulation

    n_field_haloes = np.sum(haloes["depth"] == 0)  # works by construction of the groupstore
    field_galaxy_offsets, field_galaxy_idx = build_group_csr(
        group_idx=galaxies["field_halo_index"], n_groups=n_field_haloes
    )

    aperture_results = compute_galaxy_aperture_masses(
        particles=particles,
        galaxies=galaxies,
        haloes=haloes,
        field_galaxy_offsets=field_galaxy_offsets,
        field_galaxy_idx=field_galaxy_idx,
        n_field_haloes=n_field_haloes,
        boxsize=sim.boxsize,
        aperture_sizes=config.aperture_size,
        cores_per_rank=config.cores_per_rank,
        available_ptypes=simulation_data.available_ptypes,
    )
    galaxies.write_batch(results=aperture_results)

    if config.distribute_halo_hydrogen and "gas" in particles:
        gas = particles["gas"]
        gas_offsets, gas_idx_sorted = haloes.get_particle_csr(ptype="gas")
        assigned_HI, assigned_H2 = distribute_halo_hydrogen(
            gas_pos=gas["pos"],
            gas_mass_HI=gas["mass_HI"],
            gas_mass_H2=gas["mass_H2"],
            gas_galaxy_ids=gas["GalID"],
            field_gas_offsets=gas_offsets[: n_field_haloes + 1],
            gas_idx_sorted=gas_idx_sorted,
            field_gal_offsets=field_galaxy_offsets,
            field_gal_idx=field_galaxy_idx,
            gal_pos=galaxies["com_pos_baryon"],
            gal_mass=galaxies["mass_baryon"],
            boxsize=sim.boxsize,
            n_galaxies=galaxies.n_groups,
            n_haloes=n_field_haloes,
        )
        galaxies.write_batch(results={"mass_HI_diffuse": assigned_HI, "mass_H2_diffuse": assigned_H2})

    logger.info("Local environment properties computed for galaxies.")


def compute_galaxy_aperture_masses(  # FIXME: aperture masses of galaxies at edge of halo are less accurate as particles are split across ranks
    particles: dict[str, ParticleStore],
    galaxies: GroupStore,
    haloes: GroupStore,
    field_galaxy_offsets: np.ndarray,
    field_galaxy_idx: np.ndarray,
    n_field_haloes: int,
    boxsize: float,
    aperture_sizes: list[float],
    cores_per_rank: int,
    available_ptypes: list[str],
) -> dict[str, np.ndarray]:
    """
    Computes mass in an aperture of defined size around galaxies, returning a dict of:

    - mass_{p}_{aperture_size}kpc for p in HI, H2, {ptypes}, total
    """
    # lots of info we need to start this one and the indexing is complex
    results: dict[str, np.ndarray] = {}
    ptype_to_int = {
        p: i for i, p in enumerate(available_ptypes)
    }  # integer comparison is faster and cheaper than string comparison
    field_csrs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_apertures, n_ptypes = len(aperture_sizes), len(available_ptypes)
    n_gals = galaxies.n_groups
    gas_exists = "gas" in available_ptypes  # guard for gas in the snapshot

    for ptype in available_ptypes:
        offsets, idx_sorted = haloes.get_particle_csr(ptype)
        offsets = offsets[: n_field_haloes + 1]
        field_csrs[ptype] = offsets, idx_sorted

    per_ptype_masses = np.zeros(shape=(n_apertures, n_gals, n_ptypes), dtype=DTYPES["mass"])
    HI_mass = np.zeros(shape=(n_apertures, n_gals), dtype=DTYPES["mass"])
    H2_mass = np.zeros(shape=(n_apertures, n_gals), dtype=DTYPES["mass"])
    gal_pos = galaxies["com_pos_baryon"]

    for halo in range(n_field_haloes):
        if field_galaxy_offsets[halo] == field_galaxy_offsets[halo + 1]:
            continue

        start, end = field_galaxy_offsets[halo], field_galaxy_offsets[halo + 1]

        halo_gals = field_galaxy_idx[start:end]

        member_idx: dict[str, np.ndarray] = {}
        for ptype in available_ptypes:
            offsets, idx_sorted = field_csrs[ptype]
            member_idx[ptype] = idx_sorted[offsets[halo] : offsets[halo + 1]]

        # I know this is quite ugly but the concatenates here are unavoidable
        segment_lengths = [len(member_idx[ptype]) for ptype in available_ptypes]
        halo_pos = np.concatenate(
            [particles[ptype]["pos"][member_idx[ptype]] for ptype in available_ptypes], dtype=DTYPES["pos"]
        )
        halo_mass = np.concatenate(
            [particles[ptype]["mass"][member_idx[ptype]] for ptype in available_ptypes], dtype=DTYPES["mass"]
        )
        halo_ptype_codes = np.repeat(np.arange(n_ptypes, dtype=DTYPES["ptype"]), segment_lengths)

        # hydrogen in gas
        if gas_exists:
            gas_offset = sum(segment_lengths[: ptype_to_int["gas"]])
            n_halo_gas = segment_lengths[ptype_to_int["gas"]]
            halo_gas_HI = particles["gas"]["mass_HI"][member_idx["gas"]]
            halo_gas_H2 = particles["gas"]["mass_H2"][member_idx["gas"]]

        tree = KDTree(data=halo_pos, boxsize=boxsize)

        for aperture_idx, aperture in enumerate(aperture_sizes):
            # query_ball_point returns a list of lists
            neighbour_lists = tree.query_ball_point(x=gal_pos[halo_gals], r=aperture, workers=cores_per_rank)

            for local_idx, neighbour_list in enumerate(neighbour_lists):  # loop over each galaxy in the halo
                galaxy = halo_gals[local_idx]

                if len(neighbour_list) == 0:  # empty aperture guard
                    continue

                neighbours = np.asarray(neighbour_list, dtype=np.int64)
                per_ptype_masses[aperture_idx, galaxy, :] = np.bincount(  # bincount group sum trick
                    halo_ptype_codes[neighbours], weights=halo_mass[neighbours], minlength=n_ptypes
                )

                if gas_exists:  # mask gas allocations to also allocate HI and H2
                    in_gas = (neighbours >= gas_offset) & (neighbours < gas_offset + n_halo_gas)
                    gas_local = neighbours[in_gas] - gas_offset
                    HI_mass[aperture_idx, galaxy] = halo_gas_HI[gas_local].sum()
                    H2_mass[aperture_idx, galaxy] = halo_gas_H2[gas_local].sum()

    # results dict storage
    for aperture_idx, aperture in enumerate(aperture_sizes):
        aperture_label = int(aperture)
        for ptype, ptype_code in ptype_to_int.items():
            results[f"mass_{ptype}_{aperture_label}kpc"] = per_ptype_masses[aperture_idx, :, ptype_code]

        results[f"mass_HI_{aperture_label}kpc"] = HI_mass[aperture_idx]
        results[f"mass_H2_{aperture_label}kpc"] = H2_mass[aperture_idx]
        results[f"mass_total_{aperture_label}kpc"] = np.sum(per_ptype_masses[aperture_idx], axis=1)

    return results


@njit(cache=True, parallel=True)
def distribute_halo_hydrogen(
    gas_pos: np.ndarray,
    gas_mass_HI: np.ndarray,
    gas_mass_H2: np.ndarray,
    gas_galaxy_ids: np.ndarray,
    field_gas_offsets: np.ndarray,
    gas_idx_sorted: np.ndarray,
    field_gal_offsets: np.ndarray,
    field_gal_idx: np.ndarray,
    gal_pos: np.ndarray,
    gal_mass: np.ndarray,
    boxsize: float,
    n_galaxies: int,
    n_haloes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Distributes the HI and H2 in a halo amongst its galaxies, matching Caesar behaviour. This time gated behind a config
    call.

    I appreciate the variable names here are a tad confusing at first.
    """
    assigned_HI = np.zeros(shape=n_galaxies, dtype=np.float64)
    assigned_H2 = np.zeros(shape=n_galaxies, dtype=np.float64)
    half_box = 0.5 * boxsize

    for halo in prange(n_haloes):
        gal_start, gal_end = field_gal_offsets[halo], field_gal_offsets[halo + 1]

        if gal_start == gal_end:  # haloes with no galaxies
            continue

        for gas_idx in range(field_gas_offsets[halo], field_gas_offsets[halo + 1]):
            particle = gas_idx_sorted[gas_idx]

            if gas_mass_HI[particle] == 0.0 and gas_mass_H2[particle] == 0.0:  # particle with no HI/H2
                continue

            if gas_galaxy_ids[particle] != -1:  # particle already belongs to galaxy
                continue

            best_galaxy = -1
            best_weight = -1.0

            for galaxy_idx in range(gal_start, gal_end):
                galaxy = field_gal_idx[galaxy_idx]

                dist_sq = 0.0
                for axis in range(3):  # PBC-aware squared distance
                    delta = gas_pos[particle, axis] - gal_pos[galaxy, axis]
                    if delta > half_box:
                        delta -= boxsize
                    elif delta < -half_box:
                        delta += boxsize
                    dist_sq += delta * delta

                if dist_sq == 0.0:  # shouldn't be hit in principle but avoids a runtime error
                    best_galaxy = galaxy
                    break

                weight = gal_mass[galaxy] / dist_sq  # mass-weighted distance
                if weight > best_weight:
                    best_weight, best_galaxy = weight, galaxy

            if best_galaxy >= 0:
                assigned_HI[best_galaxy] += gas_mass_HI[particle]
                assigned_H2[best_galaxy] += gas_mass_H2[particle]

    return assigned_HI, assigned_H2
