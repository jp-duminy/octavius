"""

Tests whether the aggregate property internal functions correctly recover properties from the numpy implementation with known arrays.

For the heavier compute functions which have more intricate logic, it is not possible to loop over groups with numpy to verify the results independently; you must replicate the same logic as the engine room functions, which means you will always get the same answer. As such, I decided it was better to do some simple asserts (is this result physical?) more akin to the testing_suite.py functions.

"""

import numpy as np

from octavius.aggregate_properties.aggregate_computations import (
    compute_kinematics,
    compute_rotational_quantities,
    compute_enclosed_mass_radii,
    compute_virial_quantities,
    compute_centre_of_mass,
    compute_vmax_and_rmax,
    compute_radii,
)

from octavius.aggregate_properties.aggregate_helpers import (
    sum_per_group,
    count_per_group,
    max_value_per_group,
    min_value_per_group,
    min_idx_per_group,
    max_idx_per_group,
    first_idx_per_group,
)

from octavius.aggregate_properties.properties_core import _combine_centre_of_mass
from octavius.aggregate_properties.properties_local_environment import distribute_halo_hydrogen


from octavius.data_management import (
    GroupStore,
    build_group_csr,
)

from octavius.data_management import OctaviusConstants

oc = OctaviusConstants()

SEED = 2317434

N_GROUPS = 3
BOXSIZE = 100.0

GROUP_IDX = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int64)

rng = np.random.default_rng(seed=SEED)
pos = rng.uniform(0, BOXSIZE, size=(len(GROUP_IDX), 3))
vel = rng.normal(0, 70.0, size=(len(GROUP_IDX), 3))

pos[0] = [99.0, 50.0, 50.0]  # make these groups straddle the boundary for PBC checks
pos[1] = [1.0, 50.0, 50.0]

masses = rng.uniform(1e5, 1e7, size=len(GROUP_IDX))
ref_pos = np.array([pos[GROUP_IDX == g].mean(axis=0) for g in range(N_GROUPS)])
ref_vel = np.array([vel[GROUP_IDX == g].mean(axis=0) for g in range(N_GROUPS)])

OFFSETS, IDX_SORTED = build_group_csr(group_idx=GROUP_IDX, n_groups=N_GROUPS)


def test_count_per_group() -> None:
    """
    Tests aggregate helper count_per_group (bincount wrapper).
    """
    expected = [np.count_nonzero(GROUP_IDX == g) for g in range(N_GROUPS)]
    result = count_per_group(offsets=OFFSETS, n_groups=N_GROUPS)

    np.testing.assert_array_equal(expected, result, err_msg="count_per_group failed.")


def test_sum_per_group() -> None:
    """
    Tests aggregate helper sum_per_group (bincount wrapper) with the masses array.
    """
    expected = [np.sum(masses[GROUP_IDX == g]) for g in range(N_GROUPS)]
    result = sum_per_group(values=masses, offsets=OFFSETS, idx_sorted=IDX_SORTED, n_groups=N_GROUPS)

    np.testing.assert_allclose(
        expected, result, rtol=1e-14, err_msg="sum_per_group failed."
    )  # floating point differences can occur here


def test_value_per_group() -> None:
    """
    Tests aggregate helpers max/min_value_per_group with the masses array.
    """
    max_expected = [np.max(masses[GROUP_IDX == g]) for g in range(N_GROUPS)]
    max_result = max_value_per_group(values=masses, offsets=OFFSETS, idx_sorted=IDX_SORTED, n_groups=N_GROUPS)

    np.testing.assert_allclose(max_expected, max_result, rtol=1e-14, err_msg="max_value_per_group failed.")

    min_expected = [np.min(masses[GROUP_IDX == g]) for g in range(N_GROUPS)]
    min_result = min_value_per_group(values=masses, offsets=OFFSETS, idx_sorted=IDX_SORTED, n_groups=N_GROUPS)

    np.testing.assert_allclose(min_expected, min_result, rtol=1e-14, err_msg="min_value_per_group failed.")


def test_idx_per_group() -> None:
    """
    Tests aggregate helpers max/min_idx_per_group with the masses array.
    """
    max_expected = [np.flatnonzero(GROUP_IDX == g)[np.argmax(masses[GROUP_IDX == g])] for g in range(N_GROUPS)]
    max_result = max_idx_per_group(values=masses, offsets=OFFSETS, idx_sorted=IDX_SORTED, n_groups=N_GROUPS)

    np.testing.assert_array_equal(max_expected, max_result, err_msg="max_idx_per_group failed.")

    min_expected = [np.flatnonzero(GROUP_IDX == g)[np.argmin(masses[GROUP_IDX == g])] for g in range(N_GROUPS)]
    min_result = min_idx_per_group(values=masses, offsets=OFFSETS, idx_sorted=IDX_SORTED, n_groups=N_GROUPS)

    np.testing.assert_array_equal(min_expected, min_result, err_msg="min_idx_per_group failed.")


def test_first_idx_per_group() -> None:
    """
    Tests aggregate helper first_idx_per_group.
    """
    first_expected = [np.flatnonzero(GROUP_IDX == g)[0] for g in range(N_GROUPS)]
    first_result = first_idx_per_group(offsets=OFFSETS, idx_sorted=IDX_SORTED, n_groups=N_GROUPS)

    np.testing.assert_array_equal(first_expected, first_result, err_msg="first_idx_per_group failed.")


def test_radii() -> None:
    """
    Tests engine room function compute_radii.
    """
    delta = pos - ref_pos[GROUP_IDX]
    delta -= BOXSIZE * np.round(delta / BOXSIZE)
    expected = np.linalg.norm(delta, axis=1)

    result = compute_radii(
        positions=pos,
        ref_pos=ref_pos,
        offsets=OFFSETS,
        idx_sorted=IDX_SORTED,
        n_groups=N_GROUPS,
        boxsize=BOXSIZE,
    )

    np.testing.assert_allclose(
        result, expected[IDX_SORTED], rtol=1e-14
    )  # result is csr-aligned, expected is particle-aligned


def test_com() -> None:
    """
    Tests engine room function compute_centre_of_mass.
    """
    group_mass = np.array([masses[GROUP_IDX == g].sum() for g in range(N_GROUPS)])
    anchor_pos = np.array(
        [pos[np.flatnonzero(GROUP_IDX == g)[0]] for g in range(N_GROUPS)]
    )  # mirror convention of anchoring to first particle

    expected_pos = np.zeros((N_GROUPS, 3))
    expected_vel = np.zeros((N_GROUPS, 3))

    for g in range(N_GROUPS):
        mask = GROUP_IDX == g
        delta = pos[mask] - anchor_pos[g]
        delta -= BOXSIZE * np.round(delta / BOXSIZE)
        expected_pos[g] = (anchor_pos[g] + np.average(delta, weights=masses[mask], axis=0)) % BOXSIZE
        expected_vel[g] = np.average(vel[mask], weights=masses[mask], axis=0)

    result_pos, result_vel = compute_centre_of_mass(
        positions=pos,
        velocities=vel,
        masses=masses,
        anchor_pos=anchor_pos,
        group_mass=group_mass,
        offsets=OFFSETS,
        idx_sorted=IDX_SORTED,
        n_groups=N_GROUPS,
        boxsize=BOXSIZE,
    )

    np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-12)
    np.testing.assert_allclose(result_vel, expected_vel, rtol=1e-12)


def test_kinematics() -> None:
    """
    Tests engine room function compute_kinematics.
    """
    com_vel = np.array(
        [np.average(vel[GROUP_IDX == g], weights=masses[GROUP_IDX == g], axis=0) for g in range(N_GROUPS)]
    )

    expected_L = np.zeros((N_GROUPS, 3))
    expected_ke = np.zeros(N_GROUPS)
    expected_disp = np.zeros(N_GROUPS)
    expected_I_tensor = np.zeros((N_GROUPS, 3, 3))

    for g in range(N_GROUPS):
        mask = GROUP_IDX == g
        delta_pos = pos[mask] - ref_pos[g]
        delta_pos -= BOXSIZE * np.round(delta_pos / BOXSIZE)
        delta_vel_ref = vel[mask] - ref_vel[g]
        delta_vel_com = vel[mask] - com_vel[g]
        m = masses[mask]
        r_sq = np.sum(delta_pos**2, axis=1)

        expected_L[g] = np.sum(m[:, None] * np.cross(delta_pos, delta_vel_ref), axis=0)
        expected_ke[g] = 0.5 * np.sum(m * np.sum(delta_vel_ref**2, axis=1))
        expected_disp[g] = np.sum(m * np.sum(delta_vel_com**2, axis=1))
        expected_I_tensor[g] = np.sum(m * r_sq) * np.eye(3) - np.einsum("i, ij, ik->jk", m, delta_pos, delta_pos)

    delta = pos - ref_pos[GROUP_IDX]
    delta -= BOXSIZE * np.round(delta / BOXSIZE)

    result_L, result_ke, result_disp, result_I = compute_kinematics(
        positions=pos,
        velocities=vel,
        masses=masses,
        ref_pos=ref_pos,
        ref_vel=ref_vel,
        com_vel=com_vel,
        offsets=OFFSETS,
        idx_sorted=IDX_SORTED,
        n_groups=N_GROUPS,
        boxsize=BOXSIZE,
    )

    np.testing.assert_allclose(result_L, expected_L, rtol=1e-12)
    np.testing.assert_allclose(result_ke, expected_ke, rtol=1e-12)
    np.testing.assert_allclose(result_disp, expected_disp, rtol=1e-12)
    np.testing.assert_allclose(result_I, expected_I_tensor, rtol=1e-12)


def test_rotational_quantities() -> None:
    """
    Tests engine room function compute_rotational_quantities.
    """
    L_group = np.zeros((N_GROUPS, 3))

    for g in range(N_GROUPS):
        mask = GROUP_IDX == g
        delta_pos = pos[mask] - ref_pos[g]
        delta_pos -= BOXSIZE * np.round(delta_pos / BOXSIZE)
        delta_vel = vel[mask] - ref_vel[g]
        L_group[g] = np.sum(masses[mask, None] * np.cross(delta_pos, delta_vel), axis=0)

    expected_counter = np.zeros(N_GROUPS)
    expected_krot = np.zeros(N_GROUPS)

    for g in range(N_GROUPS):
        mask = GROUP_IDX == g
        delta_pos = pos[mask] - ref_pos[g]
        delta_pos -= BOXSIZE * np.round(delta_pos / BOXSIZE)
        delta_vel = vel[mask] - ref_vel[g]
        m = masses[mask]

        momentum = m[:, None] * delta_vel
        L_particles = np.cross(delta_pos, momentum)
        L_dot = np.sum(L_particles * L_group[g], axis=1)

        expected_counter[g] = np.sum(m[L_dot < 0])

        R_cyl = np.linalg.norm(np.cross(delta_pos, L_group[g]), axis=1)

        for i in range(len(m)):
            if R_cyl[i] > 0.0:
                v_circ = L_dot[i] / (R_cyl[i] * m[i])
                expected_krot[g] += 0.5 * m[i] * v_circ**2

    result_counter, result_krot = compute_rotational_quantities(
        positions=pos,
        velocities=vel,
        masses=masses,
        ref_pos=ref_pos,
        ref_vel=ref_vel,
        L_group=L_group,
        offsets=OFFSETS,
        idx_sorted=IDX_SORTED,
        n_groups=N_GROUPS,
        boxsize=BOXSIZE,
    )

    np.testing.assert_allclose(result_counter, expected_counter, rtol=1e-12)
    np.testing.assert_allclose(result_krot, expected_krot, rtol=1e-12)


def test_enclosed_mass_radii() -> None:
    """
    Verifies engine room function enclosed_mass_radii output is sensible.
    """
    delta = pos - ref_pos[GROUP_IDX]
    delta -= BOXSIZE * np.round(delta / BOXSIZE)
    radii = np.linalg.norm(delta, axis=1)

    quantiles = np.array([0.2, 0.5, 0.8])

    result = compute_enclosed_mass_radii(
        radii=radii, masses=masses, offsets=OFFSETS, idx_sorted=IDX_SORTED, n_groups=N_GROUPS, quantiles=quantiles
    )

    for g in range(N_GROUPS):
        row = result[g]
        valid = ~np.isnan(row)
        assert np.all(np.diff(row[valid]) >= 0), "enclosed_mass_radii_failed: not monotonically increasing."
        assert np.all(row[valid] <= np.max(radii[GROUP_IDX == g])), (
            "enclosed_mass_radii_failed: enclosed radius exceeds max particle radius."
        )


def test_virial_quantities() -> None:
    """
    Verifies engine room function compute_virial_quantities output is sensible.
    """
    delta = pos - ref_pos[GROUP_IDX]
    delta -= BOXSIZE * np.round(delta / BOXSIZE)
    radii = np.linalg.norm(delta, axis=1)

    rhocrit = 1e-29
    factors = np.array([200.0, 500.0])

    result_r, result_m = compute_virial_quantities(
        radii=radii,
        masses=masses,
        offsets=OFFSETS,
        idx_sorted=IDX_SORTED,
        n_groups=N_GROUPS,
        rhocrit=rhocrit,
        factors=factors,
    )

    for g in range(N_GROUPS):
        if not np.isnan(result_r[g, 0]):
            assert result_r[g, 1] <= result_r[g, 0], "compute_virial_quantities failed: r500c > r200c."
            assert result_m[g, 1] <= result_m[g, 0], "compute_virial_quantities failed: m500c > m200c."
            assert result_r[g, 0] <= np.max(radii[GROUP_IDX == g]), (
                "compute_virial_quantities failed: virial radius exceeds max particle radius."
            )


def test_vmax_rmax() -> None:
    """
    Verifies engine room function compute_vmax_and_rmax output is sensible.
    """
    delta = pos - ref_pos[GROUP_IDX]
    delta -= BOXSIZE * np.round(delta / BOXSIZE)
    radii = np.linalg.norm(delta, axis=1)
    scale_factor = 0.5

    result_vmax, result_rmax = compute_vmax_and_rmax(
        radii=radii,
        masses=masses,
        offsets=OFFSETS,
        idx_sorted=IDX_SORTED,
        G=oc.G_VCIRC,
        scale_factor=scale_factor,
        n_groups=N_GROUPS,
    )

    for g in range(N_GROUPS):
        if not np.isnan(result_vmax[g]):
            assert result_vmax[g] > 0, "compute_vmax_and_rmax failed: negative vmax."
            assert result_rmax[g] > 0, "compute_vmax_and_rmax failed: negative rmax."
            assert result_rmax[g] <= np.max(radii[GROUP_IDX == g]), (
                "compute_vmax_and_rmax failed: rmax exceeds max particle radius."
            )
            total_mass = np.sum(masses[GROUP_IDX == g])
            v_at_outer = np.sqrt(oc.G_VCIRC * total_mass / np.max(radii[GROUP_IDX == g]))
            assert result_vmax[g] >= v_at_outer * 0.99, "compute_vmax_and_rmax failed: vmax below circular velocity."


def test_combined_com() -> None:
    """
    Tests whether the combined centre of mass is correct, in response to the v0.9.3 issue.
    """
    store = GroupStore(group_ids=np.arange(3), group_key="GalID", kind="galaxy")
    constituent_ptypes = ["gas", "star", "bh"]

    # one galaxy with all ptypes; two galaxies, each missing ptypes
    store["mass_gas"] = np.array([1.0, 1.0, 0.0])  # one galaxy with no gas mass
    store["mass_star"] = np.array([3.0, 3.0, 1.0])
    store["mass_bh"] = np.array([1.0, 0.0, 3.0])  # one with no bh mass
    combined_mass = sum(store[f"mass_{pt}"] for pt in constituent_ptypes)

    # positions assuming a box of 100 length in arbitrary units, x = y = z for simplicity
    boxsize = 100.0
    # put the ptype pos vals across boundaries
    store["_pos_gas"] = boxsize * np.repeat(np.array([[0.45], [0.02], [0.0]]), repeats=3, axis=1)
    store["_pos_star"] = boxsize * np.repeat(np.array([[0.55], [0.98], [0.49]]), repeats=3, axis=1)
    store["_pos_bh"] = boxsize * np.repeat(np.array([[0.0], [0.0], [0.51]]), repeats=3, axis=1)

    # this is very rote so I'm going to make velocity ascending
    store["_vel_gas"] = np.full(shape=(3, 3), fill_value=1.0)
    store["_vel_star"] = np.full(shape=(3, 3), fill_value=2.0)
    store["_vel_bh"] = np.full(shape=(3, 3), fill_value=3.0)

    # group 0: star wins (m = 3), and is at 0.55
    # gas shift is -0.1 which stays as such, bh shift is -0.55 which wraps to 0.45
    # weighted shift is then 0.07 and the com with offset is 0.62
    # doing this for the others gives:

    expected_pos = boxsize * np.repeat(np.array([[0.62], [0.99], [0.505]]), repeats=3, axis=1)
    expected_vel = np.repeat(np.array([[2.0], [1.75], [2.75]]), repeats=3, axis=1)
    # repeat for others and you get [0.525, 0.99, 0.505] * boxsize
    # same for vel gives [1.75, 1.75, 2.75]

    results = _combine_centre_of_mass(
        group_store=store,
        combined_mass=combined_mass,
        collective_name="baryon",
        constituent_ptypes=constituent_ptypes,
        boxsize=boxsize,
    )

    np.testing.assert_allclose(
        results["com_pos_baryon"],
        expected_pos,
        err_msg="test_combined_com failed: positions do not match",
    )

    np.testing.assert_allclose(
        results["com_vel_baryon"],
        expected_vel,
        err_msg="test_combined_com failed: velocities do not match",
    )


def test_halo_gas_redistribution() -> None:
    """
    Tests whether the distribute_halo_hydrogen function will give all of a halo's HI and H2 to
    a lone galaxy if the config flag is enabled.
    """
    # galaxy with no gas, but its halo has 3 gas particles with some HI and H2
    # only 3 particles so the index stuff is easy to hand write
    gas_pos = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    gas_mass_HI = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    gas_mass_H2 = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    gas_galaxy_ids = np.full(3, -1, dtype=np.int64)

    assigned_HI, assigned_H2 = distribute_halo_hydrogen(
        gas_pos=gas_pos,
        gas_mass_HI=gas_mass_HI,
        gas_mass_H2=gas_mass_H2,
        gas_galaxy_ids=gas_galaxy_ids,
        field_gas_offsets=np.array([0, 3], dtype=np.int64),
        gas_idx_sorted=np.arange(3, dtype=np.int64),
        field_gal_offsets=np.array([0, 1], dtype=np.int64),
        field_gal_idx=np.array([0], dtype=np.int64),
        gal_pos=np.zeros((1, 3), dtype=np.float64),
        gal_mass=np.array([1.0e10], dtype=np.float64),
        boxsize=100.0,
        n_galaxies=1,
        n_haloes=1,
    )

    np.testing.assert_allclose(
        assigned_HI[0], np.sum(gas_mass_HI), err_msg="test_halo_gas_redistribution failed: not all HI went to galaxy."
    )

    np.testing.assert_allclose(
        assigned_H2[0], np.sum(gas_mass_H2), err_msg="test_halo_gas_redistribution failed: not all H2 went to galaxy."
    )
