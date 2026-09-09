"""

Core aggregate properties. These include simple computations (per-ptype number counts, masses,
angular momenta, etc.); quantities derived from the basics (baryon/total quantities);
then group-specific derived quantities (e.g. virial quantities for haloes or morphology
indicators for galaxies).

All physical quantities are computed with numba. Please refer to aggregate_computations.py
and aggregate_helpers.py for the backend of this file.

"""

# type checking (semantic)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import (
        ParticleStore,
        GroupStore,
        SimulationAttributes,
        SimulationData,
        OctaviusConstants,
        OctaviusConfig,
    )

# others
import numpy as np

# internal imports
from .aggregate_computations import (
    compute_kinematics,
    compute_rotational_quantities,
    compute_enclosed_mass_radii,
    compute_virial_quantities,
    compute_centre_of_mass,
    compute_vmax_and_rmax,
    compute_radii,
)

from .aggregate_helpers import (
    sum_per_group,
    count_per_group,
    min_idx_per_group,
    first_idx_per_group,
    max_value_per_group,
)

from ..utils import guarded_divide, guarded_arcsin

from ..data_management import (
    build_group_csr,
)

from ..log import get_logger

logger = get_logger()


def run_core_properties(simulation_data: SimulationData, config: OctaviusConfig) -> None:
    """
    Top-level executor for the core aggregate properties.
    """
    constants = simulation_data.constants

    for group_type in simulation_data.groups:  # haloes must run first (halo groupstore is built first)
        logger.info(
            f"Running core properties for {group_type}: {simulation_data.groups[group_type].n_groups:,} members"
        )

        group_store = simulation_data.groups[group_type]
        kind = group_store.kind
        particles = simulation_data.particles
        sim = simulation_data.simulation

        available_ptypes = simulation_data.available_ptypes
        available_baryonic = simulation_data.available_baryonic_ptypes

        # if user wants minpot as the halo centre
        if kind == "halo" and config.halo_centre == "MIN_POT":
            has_potential = all("potential" in ps.columns for ps in particles.values())
            if not has_potential:
                raise ValueError(
                    f"'halo_centre' is set to {config.halo_centre} but potentials are not in the snapshot."
                )

            global_minimum = _prepare_global_minimum_potential(
                particles=particles,
                group_store=group_store,
                ptypes=available_ptypes,
            )
            group_store.write_batch(results=global_minimum)

        run_core_ptype_pass(particles=particles, store=group_store, sim=sim, config=config)
        run_combine(
            store=group_store,
            available_ptypes=available_ptypes,
            available_baryonic_ptypes=available_baryonic,
            boxsize=sim.boxsize,
        )

        run_combined_radial_quantiles(
            particles=particles,
            store=group_store,
            available_ptypes=available_ptypes,
            available_baryonic_ptypes=available_baryonic,
            sim=sim,
            config=config,
        )

        if kind == "halo":
            run_halo_stages(particles=particles, haloes=group_store, sim=sim, constants=constants, config=config)

        elif kind == "galaxy":
            run_galaxy_stages(
                particles=particles,
                galaxies=group_store,
                available_baryonic_ptypes=available_baryonic,
                sim=sim,
            )

        logger.info(f"Core properties computed for {group_type}.")


def run_core_ptype_pass(
    particles: dict[str, ParticleStore],
    store: GroupStore,
    sim: SimulationAttributes,
    config: OctaviusConfig,
) -> None:
    """
    Runs the core common quantities (counts & mass, com, kinematics, radials); writes to GroupStore. Loops over ptypes twice because global minpot/com need to be computed for reference positions.
    """
    ptype_cache: dict[str, tuple[np.ndarray, ...]] = {}  # for things we don't want to recompute on the second loop
    n_groups = store.n_groups

    for ptype in store.ptypes:
        data = particles[ptype]
        offsets, idx_sorted = store.get_particle_csr(ptype=ptype)

        counts_and_mass = _compute_counts_and_mass(
            masses=data["mass"], offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups, ptype=ptype
        )
        store.write_batch(results=counts_and_mass)

        centre_of_mass = _compute_centre_of_mass(
            positions=data["pos"],
            velocities=data["vel"],
            masses=data["mass"],
            offsets=offsets,
            idx_sorted=idx_sorted,
            group_mass=store[f"mass_{ptype}"],
            n_groups=n_groups,
            boxsize=sim.boxsize,
        )
        store.write_batch(results=centre_of_mass, suffix=ptype)

        ptype_cache[ptype] = (offsets, idx_sorted, counts_and_mass)

    if store.kind == "halo":
        if config.halo_centre == "MIN_POT":
            ref_pos = store["minpot_pos"]
            ref_vel = store["minpot_vel"]
        else:
            total_com = _combine_centre_of_mass(
                group_store=store,
                combined_mass=sum(store[f"mass_{pt}"] for pt in particles),
                collective_name="total",
                constituent_ptypes=list(particles.keys()),
                boxsize=sim.boxsize,
            )
            ref_pos = total_com["com_pos_total"]
            ref_vel = total_com["com_vel_total"]
        store.write_batch(results={"_centre_pos": ref_pos, "_centre_vel": ref_vel})

    else:
        available_baryonic = [pt for pt in particles if particles[pt].is_baryonic]
        baryon_com = _combine_centre_of_mass(
            group_store=store,
            combined_mass=sum(store[f"mass_{pt}"] for pt in available_baryonic),
            collective_name="galaxy_ref",
            constituent_ptypes=available_baryonic,
            boxsize=sim.boxsize,
        )
        ref_pos = baryon_com["com_pos_galaxy_ref"]
        ref_vel = baryon_com["com_vel_galaxy_ref"]

    quantile_names = list(config.radial_quantiles)
    quantiles = np.array(list(config.radial_quantiles.values()), dtype=np.float64)

    for ptype in store.ptypes:  # second pass does kinematics which require collective group centres
        data = particles[ptype]
        offsets, idx_sorted, counts_and_mass = ptype_cache[ptype]
        com_vel = store[f"_vel_{ptype}"]

        kinematics = _compute_ptype_kinematics(
            positions=data["pos"],
            velocities=data["vel"],
            masses=data["mass"],
            ref_pos=ref_pos,
            ref_vel=ref_vel,
            com_vel=com_vel,
            offsets=offsets,
            idx_sorted=idx_sorted,
            n_groups=n_groups,
            boxsize=sim.boxsize,
        )
        store.write_batch(results=kinematics, suffix=ptype)

        derived = _derive_kinematics(
            L=store[f"L_{ptype}"],
            dispersion_sum=kinematics["_dispersion_sum"],
            counts=counts_and_mass[f"n_{ptype}"],
            group_mass=store[f"mass_{ptype}"],
        )
        store.write_batch(results=derived, suffix=ptype)

        radii = compute_radii(
            positions=data["pos"],  # these pop out in group order
            ref_pos=ref_pos,
            offsets=offsets,
            idx_sorted=idx_sorted,
            n_groups=n_groups,
            boxsize=sim.boxsize,
        )
        aligned_mass = data["mass"][
            idx_sorted
        ]  # haloes have inclusive membership, so mass/idx must be aligned to group order
        aligned_idx_sorted = np.arange(len(radii), dtype=np.int64)  # one-to-one idx mapping

        radial = _compute_radial_quantities(
            radii=radii,
            masses=aligned_mass,
            offsets=offsets,
            idx_sorted=aligned_idx_sorted,
            n_groups=n_groups,
            quantiles=quantiles,
            quantile_names=quantile_names,
        )
        store.write_batch(results=radial, suffix=ptype)


def run_combine(
    store: GroupStore,
    available_ptypes: list[str],
    available_baryonic_ptypes: list[str],
    boxsize: float,
) -> None:
    """
    Runs functions to combine per-ptype results into baryon and total aggregates (counts, mass, CoM, L, ke_tot, dispersion_sum).
    """
    combined_baryon = _combine_ptype_sums(
        group_store=store,
        collective_name="baryon",
        constituent_ptypes=available_baryonic_ptypes,
        boxsize=boxsize,
    )
    store.write_batch(results=combined_baryon)

    if store.kind == "halo":
        combined_total = _combine_ptype_sums(
            group_store=store,
            collective_name="total",
            constituent_ptypes=available_ptypes,
            boxsize=boxsize,
        )
        store.write_batch(results=combined_total)


def run_halo_stages(
    particles: dict[str, ParticleStore],
    haloes: GroupStore,
    sim: SimulationAttributes,
    constants: OctaviusConstants,
    config: OctaviusConfig,
) -> None:
    """
    Halo-specific virial/mass profile quantities.
    """
    n_groups = haloes.n_groups
    ref_pos = haloes["_centre_pos"]

    all_radii_list, all_masses_list, all_group_idx_list = [], [], []  # ghastly concatenation

    for ptype in haloes.ptypes:
        data = particles[ptype]
        offsets, idx_sorted = haloes.get_particle_csr(ptype=ptype)

        radii = compute_radii(
            positions=data["pos"],
            ref_pos=ref_pos,
            offsets=offsets,
            idx_sorted=idx_sorted,
            n_groups=haloes.n_groups,
            boxsize=sim.boxsize,
        )
        masses = data["mass"][idx_sorted]
        group_idx = np.repeat(np.arange(haloes.n_groups, dtype=np.int64), np.diff(offsets))
        all_radii_list.append(radii)
        all_masses_list.append(masses)
        all_group_idx_list.append(group_idx)

    all_radii = np.concatenate(all_radii_list)
    all_masses = np.concatenate(all_masses_list)
    all_group_idx = np.concatenate(all_group_idx_list)
    all_offsets, all_idx_sorted = build_group_csr(group_idx=all_group_idx, n_groups=n_groups)

    factors = np.array(config.virial_factors)
    mass_profile = _compute_mass_profile_quantities(
        radii=all_radii,
        masses=all_masses,
        offsets=all_offsets,
        idx_sorted=all_idx_sorted,
        n_groups=n_groups,
        factors=factors,
        rhocrit_comoving=sim.rhocrit_comoving,
        scale_factor=sim.scale_factor,
        constants=constants,
    )
    haloes.write_batch(results=mass_profile)

    derived = _derive_halo_quantities(
        group_mass=haloes["mass_total"],
        L_mag=haloes["_L_mag_total"],
        counts=haloes["_n_total"],
        r200_factor=sim.r200_factor,
        scale_factor=sim.scale_factor,
        constants=constants,
    )
    haloes.write_batch(results=derived)


def run_galaxy_stages(
    particles: dict[str, ParticleStore],
    galaxies: GroupStore,
    available_baryonic_ptypes: list[str],
    sim: SimulationAttributes,
) -> None:
    """
    Galaxy-specific morphological quantities (requires another pass for alignment of particle L axes).
    """
    n_groups = galaxies.n_groups
    combined_L = galaxies["L_baryon"]
    ref_pos = galaxies["com_pos_baryon"]
    ref_vel = galaxies["com_vel_baryon"]

    combined_ke_rot = np.zeros(n_groups)
    combined_counter_rotating_mass = np.zeros(n_groups)

    for ptype in available_baryonic_ptypes:
        data = particles[ptype]
        offsets, idx_sorted = galaxies.get_particle_csr(ptype=ptype)

        counter_rot, ke_rot = compute_rotational_quantities(
            positions=data["pos"],
            velocities=data["vel"],
            masses=data["mass"],
            ref_pos=ref_pos,
            ref_vel=ref_vel,
            L_group=combined_L,
            offsets=offsets,
            idx_sorted=idx_sorted,
            n_groups=n_groups,
            boxsize=sim.boxsize,
        )

        combined_ke_rot += ke_rot
        combined_counter_rotating_mass += counter_rot

        per_ptype_derived = _derive_galaxy_quantities(
            ke_tot=galaxies[f"_ke_tot_{ptype}"],
            ke_rot=ke_rot,
            counter_rotating_mass=counter_rot,
            group_mass=galaxies[f"mass_{ptype}"],
            counts=galaxies[f"n_{ptype}"],
        )

        galaxies.write_batch(results=per_ptype_derived, suffix=ptype)

    derived = _derive_galaxy_quantities(
        ke_tot=galaxies["_ke_tot_baryon"],
        ke_rot=combined_ke_rot,
        counter_rotating_mass=combined_counter_rotating_mass,
        group_mass=galaxies["mass_baryon"],
        counts=galaxies["_n_baryon"],
    )
    galaxies.write_batch(results=derived, suffix="baryon")


def _combined_quantiles(
    particles: dict[str, ParticleStore],
    store: GroupStore,
    ptypes: list[str],
    ref_pos: np.ndarray,
    sim: SimulationAttributes,
    quantiles: np.ndarray,
    quantile_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Concatenates, then computes combined quantiles, returning a dict of the same quantities computed by +compute_radial_quantities.
    """
    radii_list, masses_list, group_idx_list = [], [], []

    for ptype in ptypes:
        data = particles[ptype]
        offsets, idx_sorted = store.get_particle_csr(ptype)

        radii = compute_radii(
            positions=data["pos"],
            ref_pos=ref_pos,
            offsets=offsets,
            idx_sorted=idx_sorted,
            n_groups=store.n_groups,
            boxsize=sim.boxsize,
        )
        masses = data["mass"][idx_sorted]  # align to group-order
        group_idx = np.repeat(
            np.arange(store.n_groups, dtype=np.int64), np.diff(offsets)
        )  # reconstruct flat group idx array
        radii_list.append(radii)
        masses_list.append(masses)
        group_idx_list.append(group_idx)

    all_group_idx = np.concatenate(group_idx_list)
    all_offsets, all_idx_sorted = build_group_csr(group_idx=all_group_idx, n_groups=store.n_groups)

    radial = _compute_radial_quantities(
        radii=np.concatenate(radii_list),
        masses=np.concatenate(masses_list),
        offsets=all_offsets,
        idx_sorted=all_idx_sorted,
        n_groups=store.n_groups,
        quantiles=quantiles,
        quantile_names=quantile_names,
    )

    return radial


def run_combined_radial_quantiles(
    particles: dict[str, ParticleStore],
    store: GroupStore,
    available_ptypes: list[str],
    available_baryonic_ptypes: list[str],
    sim: SimulationAttributes,
    config: OctaviusConfig,
) -> None:
    """
    Computes radial quantiles for combined ptype sets (baryon, and total for haloes).
    """
    quantile_names = list(config.radial_quantiles)
    quantiles = np.array(list(config.radial_quantiles.values()), dtype=np.float64)

    if store.kind == "halo":
        baryon_ref = store["_centre_pos"]
    else:
        baryon_ref = store["com_pos_baryon"]

    baryon_quantiles = _combined_quantiles(
        particles=particles,
        store=store,
        ptypes=available_baryonic_ptypes,
        ref_pos=baryon_ref,
        sim=sim,
        quantiles=quantiles,
        quantile_names=quantile_names,
    )

    store.write_batch(results=baryon_quantiles, suffix="baryon")

    if store.kind == "halo":
        total_quantiles = _combined_quantiles(
            particles=particles,
            store=store,
            ptypes=available_ptypes,
            ref_pos=store["_centre_pos"],
            sim=sim,
            quantiles=quantiles,
            quantile_names=quantile_names,
        )

        store.write_batch(results=total_quantiles, suffix="total")


def _prepare_global_minimum_potential(
    particles: dict[str, ParticleStore], group_store: GroupStore, ptypes: list[str]
) -> dict[str, np.ndarray]:
    """
    Finds the global minimum potential across multiple particle types.

    Neccesary for baryon/total aggregate properties. Returns a dict of:

    - minpot_pos
    - minpot_vel
    """
    results: dict[str, np.ndarray] = {}
    n_groups = group_store.n_groups
    best_pot = np.full(shape=n_groups, fill_value=np.inf)
    best_pos = np.full(shape=(n_groups, 3), fill_value=np.nan)
    best_vel = np.full(shape=(n_groups, 3), fill_value=np.nan)

    for ptype in ptypes:
        data = particles[ptype]
        offsets, idx_sorted = group_store.get_particle_csr(ptype=ptype)

        potentials = data["potential"]
        positions = data["pos"]
        velocities = data["vel"]

        min_idx = min_idx_per_group(values=potentials, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
        has_min = min_idx >= 0
        ptype_min_pot = np.full(n_groups, np.inf)
        ptype_min_pot[has_min] = potentials[min_idx[has_min]]

        better = ptype_min_pot < best_pot
        best_pot[better] = ptype_min_pot[better]
        best_pos[better] = positions[min_idx[better]]  # you get the idea
        best_vel[better] = velocities[min_idx[better]]

    results["minpot_pos"] = best_pos
    results["minpot_vel"] = best_vel

    return results


def _compute_counts_and_mass(
    masses: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    ptype: str,
) -> dict[str, np.ndarray]:
    """
    Computes number counts and total masses for the input ptype, returning a dict of:

    - n_{ptype}
    - mass_{ptype}
    """
    results: dict[str, np.ndarray] = {}

    results[f"n_{ptype}"] = count_per_group(offsets=offsets, n_groups=n_groups)
    results[f"mass_{ptype}"] = sum_per_group(values=masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)

    return results


def _compute_centre_of_mass(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    group_mass: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    boxsize: float,
) -> dict[str, np.ndarray]:
    """
    Computes centre-of-mass positions and velocities (with PBC handling), returning a dict of:

    - _pos
    - _vel
    """
    results: dict[str, np.ndarray] = {}

    anchor_idx = first_idx_per_group(
        offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )  # HACK: anchor can be any particle, so use first member of each group
    anchor_positions = np.full((n_groups, 3), np.nan)
    valid_anchors = anchor_idx >= 0
    anchor_positions[valid_anchors] = positions[anchor_idx[valid_anchors]]

    com_positions, com_velocities = compute_centre_of_mass(
        positions=positions,
        velocities=velocities,
        masses=masses,
        anchor_pos=anchor_positions,
        group_mass=group_mass,
        offsets=offsets,
        idx_sorted=idx_sorted,
        n_groups=n_groups,
        boxsize=boxsize,
    )

    results["_pos"] = com_positions
    results["_vel"] = com_velocities

    return results


def _compute_ptype_kinematics(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    com_vel: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    boxsize: float,
) -> dict[str, np.ndarray]:
    """
    Computes kinematic quantities, where ref_pos/vel is wrt the group centre (com for galaxies, minpot for haloes), returning a dict of:

    - L
    - _dispersion_sum
    - _ke_tot
    """
    L, ke_tot, dispersion_sum, inertia_tensor = compute_kinematics(
        positions=positions,
        velocities=velocities,
        masses=masses,
        ref_pos=ref_pos,
        ref_vel=ref_vel,
        com_vel=com_vel,
        offsets=offsets,
        idx_sorted=idx_sorted,
        n_groups=n_groups,
        boxsize=boxsize,
    )

    # inertia tensors isn't derived but follows the same masking laws as combine_ptype_sums does, so do the masking here
    counts = np.diff(offsets)  # offsets is an (n_groups+1) lengths array so np.diff shifts it to mimic counts
    empty = counts == 0
    small = (counts > 0) & (counts < 3)
    inertia_tensor[empty] = np.nan
    inertia_tensor[small] = 0

    results: dict[str, np.ndarray] = {}

    results["L"] = L
    results["_ke_tot"] = ke_tot
    results["_dispersion_sum"] = dispersion_sum
    results["inertia_tensor"] = inertia_tensor

    return results


def _compute_radial_quantities(
    radii: np.ndarray,
    masses: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    quantiles: np.ndarray,
    quantile_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Computes radial quantities (r20, half_mass, etc.) defined from config, returning a dict of:

    - r20
    - half mass
    - r80

    Or others specified in config.yaml
    """
    results: dict[str, np.ndarray] = {}

    radial_results = compute_enclosed_mass_radii(
        radii=radii, masses=masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups, quantiles=quantiles
    )
    max_radii = max_value_per_group(values=radii, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    max_radii[max_radii == -np.inf] = 0.0  # the engine room function instantiates with -np.inf

    for q, col_name in enumerate(quantile_names):
        results[f"radius_{col_name}"] = radial_results[:, q]
    results["radius_max"] = max_radii

    return results


def _compute_mass_profile_quantities(
    radii: np.ndarray,
    masses: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    factors: np.ndarray,
    rhocrit_comoving: float,
    scale_factor: float,
    constants: OctaviusConstants,
) -> dict[str, np.ndarray]:
    """
    Computes mass profile quantities (virial and vmax/rmax), (which as of 19/06/26 are only required for haloes). Returns a dict of:

    - radius_{f}c for f in {factors}
    - mass_{f}c
    - vmax
    - rmax
    """
    results: dict[str, np.ndarray] = {}

    virial_radius, virial_mass = compute_virial_quantities(
        radii=radii,
        masses=masses,
        offsets=offsets,
        idx_sorted=idx_sorted,
        n_groups=n_groups,
        rhocrit=rhocrit_comoving,
        factors=factors,
    )

    for f, factor in enumerate(factors.astype(int)):  # cast array to ints for f-string name
        results[f"radius_{factor}c"] = virial_radius[:, f]
        results[f"mass_{factor}c"] = virial_mass[:, f]

    vmax, rmax = compute_vmax_and_rmax(
        radii=radii,
        masses=masses,
        offsets=offsets,
        idx_sorted=idx_sorted,
        G=constants.G_VCIRC,
        scale_factor=scale_factor,
        n_groups=n_groups,
    )

    results["vmax"], results["rmax"] = vmax, rmax

    return results


def _derive_kinematics(
    L: np.ndarray,
    dispersion_sum: np.ndarray,
    counts: np.ndarray,
    group_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute derived kinematic quantities (simple arithmetic on the outputs of _compute_kinematics), returning a dict of:

    - velocity_dispersion
    - L_azimuth: azimuthal angular momentum angle (phi)
    - L_elevation: elevation angular momentum angle (theta in polar coords)
    """
    results: dict[str, np.ndarray] = {}

    L_mag = np.linalg.norm(L, axis=1)
    L_azimuth = np.arctan2(L[:, 1], L[:, 0])
    L_elevation = guarded_arcsin(guarded_divide(L[:, 2], L_mag, fill_value=0.0))
    velocity_dispersions = np.where(counts > 0, np.sqrt(guarded_divide(dispersion_sum, group_mass)), np.nan)

    small = (counts > 0) & (
        counts < 3
    )  # groups with fewer than 3 counts have ill-defined rotational quantities (mask away)
    empty = counts == 0

    if small.sum() >= 0.5 * len(velocity_dispersions):
        logger.debug(f"{small.sum()}/{len(velocity_dispersions)} groups hit the small flag in _derive_kinematics.")

    for quantity in [velocity_dispersions, L, L_mag, L_azimuth, L_elevation]:
        quantity[empty] = np.nan
        quantity[small] = 0.0

    results["_L_mag"] = L_mag
    results["L_azimuth"], results["L_elevation"] = L_azimuth, L_elevation
    results["velocity_dispersion"] = velocity_dispersions

    return results


def _derive_halo_quantities(
    group_mass: np.ndarray,
    L_mag: np.ndarray,
    counts: np.ndarray,
    r200_factor: float,
    scale_factor: float,
    constants: OctaviusConstants,
) -> dict[str, np.ndarray]:
    """
    Derived halo quantities, returning a dict of:

    - r200m: radius at which matter overdensity is 200x the mean matter density
    - velocity_circular
    - temperature_virial
    - spin_param: (Bullock) spin parameter
    """
    results: dict[str, np.ndarray] = {}
    r200m = r200_factor * group_mass ** (1.0 / 3.0)  # NOTE: comoving
    v_circ = np.sqrt(
        guarded_divide(numerator=(constants.G_VCIRC * group_mass), denominator=(r200m * scale_factor))
    )  # v_circ needs physical r200m

    temperature_virial = constants.VIRIAL_TEMP_FACTOR * v_circ**2
    spin_param = guarded_divide(numerator=L_mag, denominator=(np.sqrt(2) * group_mass * v_circ * r200m))

    empty = counts == 0
    logger.debug(f"{empty.sum()} haloes are empty (NaN for their halo quantities).")

    for arr in [r200m, v_circ, temperature_virial, spin_param]:
        arr[empty] = np.nan

    results["r200m"] = r200m
    results["velocity_circular"] = v_circ
    results["temperature_virial"] = temperature_virial
    results["spin_param"] = spin_param

    return results


def _derive_galaxy_quantities(
    ke_tot: np.ndarray,
    ke_rot: np.ndarray,
    counter_rotating_mass: np.ndarray,
    group_mass: np.ndarray,
    counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Derived galaxy quantities, returning a dict of:

    - BoverT: bulge-to-total kinematic ratio
    - kappa_rot: fraction of kinetic energy in rotation (Sales et al. (2012))
    """
    results: dict[str, np.ndarray] = {}

    BoverT = guarded_divide(numerator=(2 * counter_rotating_mass), denominator=group_mass)
    kappa_rot = guarded_divide(numerator=ke_rot, denominator=ke_tot)

    small = (counts > 0) & (
        counts < 3
    )  # groups with fewer than 3 counts have ill-defined rotational quantities (mask away)
    empty = counts == 0

    for quantity in [BoverT, kappa_rot]:
        quantity[empty] = np.nan
        quantity[small] = 0.0

    results["BoverT"], results["kappa_rot"] = BoverT, kappa_rot

    return results


def _combine_counts_and_mass(
    group_store: GroupStore, collective_name: str, constituent_ptypes: list[str]
) -> dict[str, np.ndarray]:
    """
    Combines counts and centre-of-mass results from constituent_ptypes (additive), returning a dict of:

    - _n_{collective_name}
    - mass_{collective_name}
    """
    results: dict[str, np.ndarray] = {}

    counts = sum(group_store[f"n_{pt}"] for pt in constituent_ptypes)
    mass = sum(group_store[f"mass_{pt}"] for pt in constituent_ptypes)

    results[f"_n_{collective_name}"] = counts
    results[f"mass_{collective_name}"] = mass

    return results


def _combine_centre_of_mass(
    group_store: GroupStore,
    combined_mass: np.ndarray,
    collective_name: str,
    constituent_ptypes: list[str],
    boxsize: float,
) -> dict[str, np.ndarray]:
    """
    Combines centre of masses from constituent_ptypes. PBC-aware, returns a dict of

    - com_pos_{collective_name}
    - com_vel_{collective_name}
    """
    results: dict[str, np.ndarray] = {}
    n_groups = group_store.n_groups
    anchor_ptype = constituent_ptypes[0]  # since the anchor can be anywhere, use first ptype
    anchor_com = group_store[f"_pos_{anchor_ptype}"]

    weighted_shift = np.zeros(shape=(n_groups, 3))
    weighted_vel = np.zeros(shape=(n_groups, 3))

    for pt in constituent_ptypes:
        pt_com = group_store[f"_pos_{pt}"]
        pt_vel = group_store[f"_vel_{pt}"]
        pt_mass = group_store[f"mass_{pt}"][:, np.newaxis]  # upgrade dimensionality so this works with (n, 3) arrays

        shift = pt_com - anchor_com
        shift -= boxsize * np.round(shift / boxsize)
        weighted_shift += pt_mass * shift
        weighted_vel += pt_mass * pt_vel

    combined_com = guarded_divide(numerator=(anchor_com + weighted_shift), denominator=(combined_mass[:, np.newaxis]))
    combined_com %= boxsize
    combined_vel = guarded_divide(numerator=weighted_vel, denominator=combined_mass[:, np.newaxis])

    results[f"com_pos_{collective_name}"] = combined_com
    results[f"com_vel_{collective_name}"] = combined_vel

    return results


def _combine_ptype_sums(
    group_store: GroupStore, collective_name: str, constituent_ptypes: list[str], boxsize: float
) -> dict[str, np.ndarray]:
    """
    Combines results from individual particle type arrays (wraps above functions), returning a dict of:

    - quantities in combine_counts_and_mass
    - quantities in combine_centre_of_mass
    - L_{collective_name}
    """
    results: dict[str, np.ndarray] = {}
    n_groups = group_store.n_groups

    counts_and_mass = _combine_counts_and_mass(
        group_store=group_store, collective_name=collective_name, constituent_ptypes=constituent_ptypes
    )
    centre_of_mass = _combine_centre_of_mass(
        group_store=group_store,
        combined_mass=counts_and_mass[f"mass_{collective_name}"],
        collective_name=collective_name,
        constituent_ptypes=constituent_ptypes,
        boxsize=boxsize,
    )

    combined_L = np.zeros(shape=(n_groups, 3))
    combined_ke = np.zeros(shape=n_groups)
    combined_dispersion_sum = np.zeros(shape=n_groups)
    combined_com_vel = centre_of_mass[f"com_vel_{collective_name}"]
    combined_inertia_tensor = np.zeros(shape=(n_groups, 3, 3))

    for pt in constituent_ptypes:
        ptype_com_vel = group_store[f"_vel_{pt}"]
        ptype_mass = group_store[f"mass_{pt}"]

        delta_vel = ptype_com_vel - combined_com_vel
        delta_vel_sq = np.sum(delta_vel**2, axis=1)

        # nan_to_num zeros NaNs before summing so they don't corrupt collective results
        ptype_dispersion_sum = np.nan_to_num(group_store[f"_dispersion_sum_{pt}"], nan=0.0) + ptype_mass * delta_vel_sq
        ptype_ke = np.nan_to_num(group_store[f"_ke_tot_{pt}"], nan=0.0) + 0.5 * ptype_mass * delta_vel_sq
        ptype_L = np.nan_to_num(group_store[f"L_{pt}"], nan=0.0)
        ptype_I = np.nan_to_num(group_store[f"inertia_tensor_{pt}"], nan=0.0)

        combined_L += ptype_L
        combined_ke += ptype_ke
        combined_dispersion_sum += ptype_dispersion_sum
        combined_inertia_tensor += ptype_I

    combined_counts = counts_and_mass[f"_n_{collective_name}"]

    combined_L_mag = np.linalg.norm(combined_L, axis=1)
    combined_velocity_dispersion = np.where(
        combined_counts > 0,
        np.sqrt(
            guarded_divide(numerator=combined_dispersion_sum, denominator=counts_and_mass[f"mass_{collective_name}"])
        ),
        np.nan,
    )
    combined_L_azimuth = np.arctan2(combined_L[:, 1], combined_L[:, 0])
    combined_L_elevation = guarded_arcsin(guarded_divide(numerator=combined_L[:, 2], denominator=combined_L_mag))

    small = (combined_counts > 0) & (combined_counts < 3)
    empty = combined_counts == 0

    for quantity in [combined_L_mag, combined_L, combined_L_azimuth, combined_L_elevation, combined_inertia_tensor]:
        quantity[empty] = np.nan
        quantity[small] = 0.0

    results[f"L_{collective_name}"] = combined_L
    results[f"L_azimuth_{collective_name}"] = combined_L_azimuth
    results[f"L_elevation_{collective_name}"] = combined_L_elevation
    results[f"_L_mag_{collective_name}"] = combined_L_mag
    results[f"_ke_tot_{collective_name}"] = combined_ke
    results[f"_dispersion_sum_{collective_name}"] = combined_dispersion_sum
    results[f"velocity_dispersion_{collective_name}"] = combined_velocity_dispersion
    results[f"inertia_tensor_{collective_name}"] = combined_inertia_tensor

    return results | counts_and_mass | centre_of_mass
