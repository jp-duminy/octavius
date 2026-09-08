"""

The photometry engine room. Photometry has a less well-defined boundary than aggregate_properties
between simple helpers and heavyweight physics computations, but for the most part, these functions
are the ones which dominate runtime and are called by the run_photometry function.

All physics computations are written in numba for JIT compilation. Numba provides an inherent raw speed
advantage (the compilation is cached on the machine, per cache=True), but more importantly, avoids materialising
intermediate arrays where numpy otherwise would. This is especially important in photometry, where the computations
are rather demanding and so intermediates can cause nasty memory spikes.

"""

# other packages
from numba import njit, prange
import numpy as np

# internal imports
from .photometry_helpers import (
    interpolate_ssp,
    get_interpolation_idx,
    StarData,
    GasData,
    SSPData,
    FilterData,
    DustData,
    PhotometryConstants,
)
from ..utils import unwrap_positions


@njit(cache=True, parallel=True)
def compute_photometric_properties(
    # namedtuple data containers
    star_data: StarData,
    gas_data: GasData,
    ssp_data: SSPData,
    filter_data: FilterData,
    dust_data: DustData,
    phot_constants: PhotometryConstants,
    gal_com_pos: np.ndarray,
    n_galaxies: int,
    # uv slope/fir luminosity quantities
    delta_nu: np.ndarray,
    wavelengths: np.ndarray,
    uv_start_idx: int,
    uv_end_idx: int,
    # misc
    madau_transmission: np.ndarray,
    kernel_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs all photometric property computations for galaxies in parallel. Returns:

    - mag_abs: (n_galaxies) absolute AB magnitudes
    - mag_abs_nodust: (n_galaxies) absolute AB magnitudes (no dust correction)
    - mag_app: (n_galaxies) apparent AB magnitudes
    - mag_app_nodust: (n_galaxies) apparent AB magnitudes (no dust correction)
    - luminosity_fir: (n_galaxies) dust-reprocessed luminosities
    - beta: uv slope
    - beta_nodust: uv slope (no dust correction)
    - out_spectra_dust: the spectrum with dust attenuation (only if spectra flag is enabled)
    - out_spectra_nodust: the spectrum with no dust attenuation (only if spectra flag is enabled)
    """
    n_bands = len(filter_data.transmission_norm_abs)
    n_lambdas = len(wavelengths)
    log_wavelengths = np.log10(wavelengths)

    # make allocations for outputs
    mag_abs = np.full(shape=(n_galaxies, n_bands), fill_value=np.nan)
    mag_abs_nodust = np.full(shape=(n_galaxies, n_bands), fill_value=np.nan)
    mag_app = np.full(shape=(n_galaxies, n_bands), fill_value=np.nan)
    mag_app_nodust = np.full(shape=(n_galaxies, n_bands), fill_value=np.nan)
    luminosity_fir = np.full(shape=n_galaxies, fill_value=np.nan)
    beta = np.full(shape=n_galaxies, fill_value=np.nan)
    beta_nodust = np.full(shape=n_galaxies, fill_value=np.nan)

    # HACK: enable standalone photometry spectra by allocating (0, 0) empty array and disabling in pipeline
    if phot_constants.keep_spectra:
        out_spectra_dust = np.empty(shape=(n_galaxies, n_lambdas), dtype=np.float64)
        out_spectra_nodust = np.empty(shape=(n_galaxies, n_lambdas), dtype=np.float64)
    else:
        out_spectra_dust = np.empty(shape=(0, 0), dtype=np.float64)
        out_spectra_nodust = np.empty(shape=(0, 0), dtype=np.float64)

    # misc
    neighbour_offsets = np.array(  # for dust cells
        [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)],
        dtype=np.int64,
    )
    dtm_slope = -0.104 * phot_constants.redshift + 0.97  # for nodust Li 2019 branch
    dtm_intercept = -0.059 * phot_constants.redshift + 0.005

    for gal_idx in prange(n_galaxies):
        # per galaxy allocations
        spectrum_dust = np.zeros(shape=n_lambdas)
        spectrum_nodust = np.zeros(shape=n_lambdas)
        extinction_curve = np.empty(shape=n_lambdas)

        # dust extinction/attenutation
        apply_extinction_law(
            out_curve=extinction_curve,
            log_ssfr=dust_data.gal_ssfr[gal_idx],
            log_Z_solar=dust_data.gal_Z[gal_idx],
            dust_curves=dust_data.dust_curves,
            dust_law=dust_data.dust_law_idx,
        )

        # index mapping
        gas_start = gas_data.offsets[gal_idx]
        gas_end = gas_data.offsets[gal_idx + 1]
        gas_slice = gas_data.idx_sorted[gas_start:gas_end]

        star_start = star_data.offsets[gal_idx]
        star_end = star_data.offsets[gal_idx + 1]
        star_slice = star_data.idx_sorted[star_start:star_end]

        # optional rotation matrices
        if phot_constants.use_rotation:
            rot = phot_constants.rotation_matrices[gal_idx]
            star_pos = star_data.pos[star_slice] @ rot.T
            star_vel_los = (star_data.vel[star_slice] @ rot.T)[:, phot_constants.los_axis]
            com_pos = gal_com_pos[gal_idx] @ rot.T
            gas_pos = gas_data.pos[gas_slice] @ rot.T
        else:
            star_pos = star_data.pos[star_slice]
            star_vel_los = star_data.vel[star_slice, phot_constants.los_axis]
            com_pos = gal_com_pos[gal_idx]
            gas_pos = gas_data.pos[gas_slice]

        # compute metal column density

        if gas_start == gas_end:
            Z_col = np.zeros(len(star_slice), dtype=np.float64)
        else:
            Z_col = compute_metal_column_densities(
                star_pos=star_pos,
                gas_pos=gas_pos,
                gas_mass=gas_data.dust_mass[gas_slice],
                gas_metallicity=gas_data.metallicity[gas_slice],
                smoothing_lengths=gas_data.smoothing_lengths[gas_slice],
                gal_centre=com_pos,
                neighbour_offsets=neighbour_offsets,
                kernel_table=kernel_table,
                los_axis=phot_constants.los_axis,
                boxsize=phot_constants.boxsize,
            )

        # convert metal column density -> A_v
        star_A_v = np.empty(shape=len(Z_col), dtype=np.float64)
        star_A_v[:] = Z_col * phot_constants.Z_col_to_A_v

        # Li et al. 2019 correction (https://arxiv.org/pdf/1906.09277)
        # this is ported from Caesar cyloser, I am unsure where their coefficients come from, but I have assumed they are correct
        if not phot_constants.use_dust:
            for s in range(len(star_A_v)):
                if Z_col[s] > 0.0:
                    z_solar = Z_col[s] / phot_constants.Z_sun  # normalise
                    dtm = 10.0 ** (dtm_slope * np.log10(z_solar) + dtm_intercept)

                    if dtm < phot_constants.mw_dust_to_metal:
                        star_A_v[s] *= dtm / phot_constants.mw_dust_to_metal

        # the SSP table is spread across ~5.5k wavelengths, but most stars are clustered in a few bins
        # if you therefore sort the stars, you get a performance boost because the spectrum computations
        # will hit the cached values in the bin rather than jumping about
        n_stars_gal = star_end - star_start
        n_age_grid = len(ssp_data.ages)
        ssp_keys = np.empty(n_stars_gal, dtype=np.int64)
        local_ages = star_data.age[star_slice]
        local_metallicities = star_data.metallicity[star_slice]

        for s in range(n_stars_gal):
            age_idx, _ = get_interpolation_idx(grid=ssp_data.ages, value=np.log10(local_ages[s]) + 9.0)
            Z_idx, _ = get_interpolation_idx(grid=ssp_data.metallicities, value=np.log10(local_metallicities[s]))
            ssp_keys[s] = Z_idx * n_age_grid + age_idx  # row-major ordering

        ssp_order = np.argsort(ssp_keys)

        compute_spectrum(
            star_masses=star_data.mass[star_slice][ssp_order],
            star_ages=star_data.age[star_slice][ssp_order],
            star_metallicities=star_data.metallicity[star_slice][ssp_order],
            star_los_velocities=star_vel_los[ssp_order],
            star_A_v=star_A_v[ssp_order],
            attenuation_curve=extinction_curve,
            ssp_spectra=ssp_data.spectra,
            ssp_ages=ssp_data.ages,
            ssp_metallicities=ssp_data.metallicities,
            ssp_mass_remaining=ssp_data.mass_remaining,
            wavelengths=wavelengths,
            c_kms=phot_constants.c_kms,
            split_age=phot_constants.split_age,
            out_spectrum_dust=spectrum_dust,
            out_spectrum_nodust=spectrum_nodust,
        )

        # absolute magnitudes
        for band in range(n_bands):
            mag_abs[gal_idx, band] = compute_ab_magnitude(
                spectrum=spectrum_dust,
                transmission_weighted=filter_data.transmission_weighted_abs[band],
                transmission_norm=filter_data.transmission_norm_abs[band],
                flux_factor=filter_data.flux_factor_abs,
            )
            mag_abs_nodust[gal_idx, band] = compute_ab_magnitude(
                spectrum=spectrum_nodust,
                transmission_weighted=filter_data.transmission_weighted_abs[band],
                transmission_norm=filter_data.transmission_norm_abs[band],
                flux_factor=filter_data.flux_factor_abs,
            )

        # apply cosmic extinction if desired (Madau 1995)
        if phot_constants.use_cosmic_ext:
            spectrum_dust_app = spectrum_dust * madau_transmission
            spectrum_nodust_app = spectrum_nodust * madau_transmission
        else:
            spectrum_dust_app = spectrum_dust
            spectrum_nodust_app = spectrum_nodust

        # apparent magnitudes
        for band in range(n_bands):
            mag_app[gal_idx, band] = compute_ab_magnitude(
                spectrum=spectrum_dust_app,
                transmission_weighted=filter_data.transmission_weighted_app[band],
                transmission_norm=filter_data.transmission_norm_app[band],
                flux_factor=filter_data.flux_factor_app,
            )
            mag_app_nodust[gal_idx, band] = compute_ab_magnitude(
                spectrum=spectrum_nodust_app,
                transmission_weighted=filter_data.transmission_weighted_app[band],
                transmission_norm=filter_data.transmission_norm_app[band],
                flux_factor=filter_data.flux_factor_app,
            )

        # dust-reprocessed luminosity
        luminosity_fir[gal_idx] = compute_fir_luminosity(
            spectrum_dust=spectrum_dust,
            spectrum_nodust=spectrum_nodust,
            bin_weights=delta_nu,
        )

        # uv slope
        beta[gal_idx] = compute_uv_slope(
            spectrum=spectrum_dust,
            log_wavelengths=log_wavelengths,
            uv_start_idx=uv_start_idx,
            uv_end_idx=uv_end_idx,
        )
        beta_nodust[gal_idx] = compute_uv_slope(
            spectrum=spectrum_nodust,
            log_wavelengths=log_wavelengths,
            uv_start_idx=uv_start_idx,
            uv_end_idx=uv_end_idx,
        )

        if phot_constants.keep_spectra:
            out_spectra_dust[gal_idx] = spectrum_dust
            out_spectra_nodust[gal_idx] = spectrum_nodust

    return (
        mag_abs,
        mag_abs_nodust,
        mag_app,
        mag_app_nodust,
        luminosity_fir,
        beta,
        beta_nodust,
        out_spectra_dust,
        out_spectra_nodust,
    )


@njit(cache=True)
def compute_spectrum(
    star_masses: np.ndarray,
    star_ages: np.ndarray,
    star_metallicities: np.ndarray,
    star_los_velocities: np.ndarray,
    star_A_v: np.ndarray,
    attenuation_curve: np.ndarray,
    ssp_spectra: np.ndarray,
    ssp_ages: np.ndarray,
    ssp_metallicities: np.ndarray,
    ssp_mass_remaining: np.ndarray,
    wavelengths: np.ndarray,
    c_kms: float,
    split_age: float,
    out_spectrum_dust: np.ndarray,
    out_spectrum_nodust: np.ndarray,
) -> None:
    """
    Computes the spectrum for a galaxy from its individual stars; writes in-place to the two output arrays.
    """
    n_wave = len(wavelengths)
    n_stars = len(star_masses)
    star_spectrum = np.empty(shape=n_wave, dtype=np.float64)
    out_spectrum_dust[:] = 0.0  # we reuse the output arrays, so set them equal to zero at instantiation
    out_spectrum_nodust[:] = 0.0

    for i in range(n_stars):
        star_age = star_ages[i]
        los_velocity = star_los_velocities[i]
        shift_factor = 1.0 + los_velocity / c_kms  # for doppler shift
        log_Z = np.log10(star_metallicities[i])

        # if stars are below split_age, we split its age into time bins
        if star_age < split_age and star_age > 0.0:
            n_bins = min(int(split_age / star_age), 10)  # upper limit of ten bins for stars > split_age
            age_step = star_age / (n_bins + 1)

            # this is called just for mass_remaining
            mass_remaining = interpolate_ssp(
                log_age=np.log10(star_age) + 9.0,  # convert to log10(yr)
                log_Z=log_Z,
                age_grid=ssp_ages,
                Z_grid=ssp_metallicities,
                spectra=ssp_spectra,
                mass_remaining=ssp_mass_remaining,
                out_spectrum=star_spectrum,
            )
            formation_mass = star_masses[i] / mass_remaining
            split_mass = formation_mass / (2 * n_bins + 1)

            for j in range(-n_bins, n_bins + 1):
                sub_age = star_age + j * age_step
                sub_age_log_yr = np.log10(sub_age) + 9

                interpolate_ssp(  # discard the mass_remaining from here
                    log_age=sub_age_log_yr,
                    log_Z=log_Z,
                    age_grid=ssp_ages,
                    Z_grid=ssp_metallicities,
                    spectra=ssp_spectra,
                    mass_remaining=ssp_mass_remaining,
                    out_spectrum=star_spectrum,  # writes to this pre-allocated array
                )

                # doppler shift
                _compute_doppler_shift(
                    star_spectrum=star_spectrum,
                    wavelengths=wavelengths,
                    shift_factor=shift_factor,
                    A_v=star_A_v[i],
                    attenuation_curve=attenuation_curve,
                    mass=split_mass,
                    out_spectrum_dust=out_spectrum_dust,
                    out_spectrum_nodust=out_spectrum_nodust,
                )

        elif star_age <= 0.0:  # prevents log(age) going to -inf
            continue

        else:  # otherwise don't bother
            mass_remaining = interpolate_ssp(
                log_age=np.log10(star_age) + 9.0,  # convert to log10(yr)
                log_Z=log_Z,
                age_grid=ssp_ages,
                Z_grid=ssp_metallicities,
                spectra=ssp_spectra,
                mass_remaining=ssp_mass_remaining,
                out_spectrum=star_spectrum,
            )

            formation_mass = star_masses[i] / mass_remaining

            _compute_doppler_shift(
                star_spectrum=star_spectrum,
                wavelengths=wavelengths,
                shift_factor=shift_factor,
                A_v=star_A_v[i],
                attenuation_curve=attenuation_curve,
                mass=formation_mass,
                out_spectrum_dust=out_spectrum_dust,
                out_spectrum_nodust=out_spectrum_nodust,
            )


@njit(cache=True)
def compute_fir_luminosity(
    spectrum_dust: np.ndarray,
    spectrum_nodust: np.ndarray,
    bin_weights: np.ndarray,
) -> float:
    """
    Returns:

    - luminosity_fir: the dust-reprocessed luminosity.
    """
    luminosity_fir = 0.0

    for i in range(len(spectrum_dust)):
        luminosity_fir += (spectrum_nodust[i] - spectrum_dust[i]) * bin_weights[i]

    return luminosity_fir


@njit(cache=True)
def compute_uv_slope(
    spectrum: np.ndarray,
    log_wavelengths: np.ndarray,
    uv_start_idx: int,
    uv_end_idx: int,
) -> float:
    """
    Returns:

    - beta: the UV slope.
    """
    # NOTE: numba is yet to support numpy.polynomial classes so polyfit is not available
    sum_xy = 0.0
    sum_x = 0.0  # x = log10(lambda)
    sum_x_sq = 0.0
    sum_y = 0.0  # y = log10(L)
    n = 0

    for i in range(uv_start_idx, uv_end_idx):
        if spectrum[i] > 0.0:
            log_L = np.log10(spectrum[i])
            log_lambda = log_wavelengths[i]

            sum_xy += log_L * log_lambda
            sum_x += log_lambda
            sum_x_sq += log_lambda**2
            sum_y += log_L
            n += 1

    if n < 2:  # undetermined for n=1 too
        return np.nan

    slope = (n * sum_xy - (sum_x * sum_y)) / ((n * sum_x_sq) - sum_x**2)
    beta = slope - 2.0  # c/lambda^2 conversion factor gives -2 in logspace

    return beta


@njit(cache=True)
def _compute_doppler_shift(
    star_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    shift_factor: float,
    A_v: float,
    attenuation_curve: np.ndarray,
    mass: float,
    out_spectrum_dust: np.ndarray,
    out_spectrum_nodust: np.ndarray,
) -> None:
    """
    Computes the Doppler shift for a star and writes in-place to the out spectra.
    """
    n_wave = len(wavelengths)
    j = 0  # idx into the spectra
    for i in range(n_wave):
        # initial guess as to where we expect to shift to
        target_wavelength = wavelengths[i] * shift_factor

        while j < n_wave - 1 and wavelengths[j + 1] <= target_wavelength:  # go to initial guess
            j += 1

        # check whether j or (j+1) is closest
        if j < n_wave - 1 and abs(wavelengths[j + 1] - target_wavelength) < abs(wavelengths[j] - target_wavelength):
            j += 1

        dust_correction = np.exp(-A_v * attenuation_curve[i])

        spectrum = star_spectrum[i]
        out_spectrum_dust[j] += mass * dust_correction * spectrum
        out_spectrum_nodust[j] += mass * spectrum


@njit(cache=True)
def apply_extinction_law(
    out_curve: np.ndarray,
    log_ssfr: float,
    log_Z_solar: float,
    dust_curves: np.ndarray,
    dust_law: int,
) -> None:
    """
    Applies the user-selected extinction law (uses the DUST_CURVE_IDX mapping). Writes in-place to out_curve.
    """
    n_lambdas = len(out_curve)

    if dust_law <= 5:
        out_curve[:] = dust_curves[dust_law]

    else:  # calzetti for log_ssfr > 0, MW for log_ssfr < -1; linear mix in between
        ssfr_weight = min(
            max(log_ssfr + 1.0, 0.0), 1.0
        )  # shift to [0, 1] for weighting (np.clip only works on scalars in numba)

        for i in range(n_lambdas):
            out_curve[i] = (dust_curves[1][i] * ssfr_weight) + (dust_curves[3][i] * (1 - ssfr_weight))

        if dust_law == 7:  # mix in SMC curve at low metallicities, ramping up between log(Z) = -1 to -2
            Z_weight = min(max(log_Z_solar + 2, 0.0), 1.0)  # shift to [0, 1] again

            for i in range(n_lambdas):
                out_curve[i] = (out_curve[i] * Z_weight) + (dust_curves[4][i] * (1 - Z_weight))


@njit(cache=True)
def compute_ab_magnitude(
    spectrum: np.ndarray,
    transmission_weighted: np.ndarray,
    transmission_norm: float,
    flux_factor: float,
) -> float:
    """
    Computes the AB magnitude; expects flux_factor to absorb distance so this function can do both absolute & apparent magnitudes. Returns:

    - magnitude: the AB magnitude.
    """
    band_luminosity = 0.0

    for i in range(len(spectrum)):
        band_luminosity += spectrum[i] * transmission_weighted[i]

    band_luminosity /= transmission_norm
    band_flux = band_luminosity * flux_factor  # flux_factor should have distance baked in

    if band_flux <= 0.0:
        return np.nan  # sentinel value for no/negative flux

    magnitude = -2.5 * np.log10(band_flux) - 48.6  # cgs form

    return magnitude


@njit(cache=True)
def compute_metal_column_densities(
    star_pos: np.ndarray,
    gas_pos: np.ndarray,
    gas_mass: np.ndarray,
    gas_metallicity: np.ndarray,
    smoothing_lengths: np.ndarray,
    neighbour_offsets: np.ndarray,
    gal_centre: np.ndarray,  # galaxy com
    kernel_table: np.ndarray,
    los_axis: int,
    boxsize: float,
) -> np.ndarray:
    """
    Computes the total metal column density from gas along the LOS for the stars in a galaxy along the LOS. star_pos should be the stars in the galaxy; gas quantities should be halo-level. Returns:

    - Z_col: the total metal column density from gas along the LOS
    """
    if len(star_pos) == 0 or len(gas_pos) == 0:  # avoid wasted computations if no gas (or stars)
        return np.zeros(shape=len(star_pos), dtype=np.float64)

    # NOTE: this internal unwrapper mutates in place, which is ok, because star/gas pos are copied slices
    unwrap_positions(positions=star_pos, centre=gal_centre, boxsize=boxsize)
    unwrap_positions(positions=gas_pos, centre=gal_centre, boxsize=boxsize)

    # orthogonal axes
    ax0 = (los_axis + 1) % 3
    ax1 = (los_axis + 2) % 3

    sort_order, cell_offsets, n_cells_x, n_cells_y, origin_x, origin_y, cell_width = build_dust_cell_list(
        gas_pos=gas_pos,
        smoothing_lengths=smoothing_lengths,
        ax0=ax0,
        ax1=ax1,
    )

    n_bins = len(kernel_table) - 1
    n_stars = len(star_pos)
    Z_col = np.zeros(shape=n_stars, dtype=np.float64)
    dx = np.empty(3, dtype=np.float64)  # allocate this here and overwrite in the loop

    for i in range(n_stars):  # outer loop over stars
        # the cell in which the star lives
        cx = int((star_pos[i, ax0] - origin_x) / cell_width)
        cy = int((star_pos[i, ax1] - origin_y) / cell_width)
        cx = min(
            max(cx, 0), (n_cells_x - 1)
        )  # clip because cells were built on gas so stars can be outside the covered region
        cy = min(max(cy, 0), (n_cells_y - 1))

        for j in range(len(neighbour_offsets)):  # loop over cells (cells are 2D so we only check 9 adjacent cells)
            nx = cx + neighbour_offsets[j, 0]
            ny = cy + neighbour_offsets[j, 1]

            if nx < 0 or nx >= n_cells_x or ny < 0 or ny >= n_cells_y:
                continue  # don't need to check neighbouring cells if we are at the edge

            cell_id = nx * n_cells_y + ny  # row major ordering again
            start = cell_offsets[cell_id]  # slice into the sort_order array to get the gas particles in the cell
            end = cell_offsets[cell_id + 1]

            for idx in range(start, end):  # inner loop over gas in each cell
                g = sort_order[idx]

                for d in range(3):  # inherited convention: observer lives at -infinity
                    dx[d] = gas_pos[g, d] - star_pos[i, d]

                if dx[los_axis] > 0:
                    continue  # if gas is behind star it contributes 0 to attenuation

                b_sq = dx[ax0] ** 2 + dx[ax1] ** 2
                h = smoothing_lengths[g]
                h_sq = h**2

                if b_sq >= h_sq:
                    continue  # gas particles beyond 1 smoothing length away have 0 weight

                b_over_h = np.sqrt(b_sq) / h  # kernel table is keyed by b/h
                table_idx = int(n_bins * b_over_h)
                kernel_weight = kernel_table[table_idx]

                # metal mass with kernel weight normalised to surface element
                Z_col[i] += gas_mass[g] * gas_metallicity[g] * kernel_weight / h_sq

    return Z_col


@njit(cache=True)
def build_dust_cell_list(  # TODO: this per-halo structure is currently constructed for each galaxy, precompute and key by field_halo_idx
    gas_pos: np.ndarray,
    smoothing_lengths: np.ndarray,
    ax0: int,
    ax1: int,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float, float]:
    """
    Creates a cell linked list for dust attenutation. Returns:

    - sort_order: the indices to sort particles by cell (np.argsort)
    - cell_offsets: where each cell begins in the flat sorted array (classic csr offset)
    - n_cells_{x/y}: the number of cells in the orthogonal directions
    - origin_{x/y}: the origin of the cell list in the orthogonal directions
    - cell_width: the width of each cell (max smoothing length)
    """
    if len(gas_pos) == 0:  # you can't hit this in principle, but I'm leaving it here in case the code changes in future
        empty = np.empty(0, dtype=np.int64)
        offsets = np.zeros(1, dtype=np.int64)
        return empty, offsets, 0, 0, 0.0, 0.0, 0.0  # match type check

    # set the maximum cell width to h_max so a star will always access all gas which contributes to its attenutation
    cell_width = np.max(smoothing_lengths)

    # np.min/max with axis arg is not supported in numba (yet)
    origin_x = np.min(gas_pos[:, ax0])
    origin_y = np.min(gas_pos[:, ax1])
    max_x = np.max(gas_pos[:, ax0])
    max_y = np.max(gas_pos[:, ax1])
    n_cells_x = int((max_x - origin_x) / cell_width) + 1  # padding so particles at the edge are always included
    n_cells_y = int((max_y - origin_y) / cell_width) + 1

    n_particles = len(gas_pos)
    cell_ids = np.empty(
        shape=n_particles, dtype=np.int32
    )  # NOTE: int32 should suffice here since this is performance-critical

    for i in range(n_particles):
        ix = int((gas_pos[i, ax0] - origin_x) / cell_width)
        iy = int((gas_pos[i, ax1] - origin_y) / cell_width)
        cell_idx = ix * n_cells_y + iy  # row-major ordering
        cell_ids[i] = cell_idx

    sort_order = np.argsort(cell_ids, kind="quicksort")  # NOTE: dense, not sparse, due to the geometry of the cells
    n_total_cells = n_cells_x * n_cells_y
    cell_offsets = np.zeros(n_total_cells + 1, dtype=np.int64)

    # add number of particles in each cell (offsets becomes the number of particles in cell i-1)
    for i in range(n_particles):
        cell_offsets[cell_ids[i] + 1] += 1

    # then add per-cell counts to the next offset so offsets becomes actually particle-indexable
    for i in range(n_total_cells):
        cell_offsets[i + 1] += cell_offsets[i]

    return sort_order, cell_offsets, n_cells_x, n_cells_y, origin_x, origin_y, cell_width
