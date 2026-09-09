"""

Functions to handle executing the photometry pipeline (physics in photometry_computations.py). This is from where the pipeline imports.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import SimulationData, OctaviusConfig, GroupStore
    from .photometry_tables import PhotometryTable

# other packages
import numpy as np
from astropy import units as u

# internal imports
from .dust_curves import (
    atten_calzetti,
    atten_conroy,
    atten_power_law,
    extinct_cardelli,
    extinct_lmc,
    extinct_smc,
)
from .photometry_helpers import (
    extinct_madau,
    build_interpolation_table,
    StarData,
    GasData,
    SSPData,
    DustData,
    FilterData,
    PhotometryConstants,
    DUST_CURVE_IDX,
    LOS_AXIS_MAP,
    KERNEL_INT_MAP,
)
from .photometry_tables import read_photometry_table
from .photometry_computations import compute_photometric_properties
from ..data_management import CODE_UNITS
from ..utils import guarded_divide
from ..log import get_logger

logger = get_logger()


def run_photometry(simulation_data: SimulationData, config: OctaviusConfig) -> None:
    """
    Top-level executor for photometry.
    """
    if "galaxies" not in simulation_data.groups:
        logger.info("Skipping photometry: no galaxies found.")
        return
    logger.info("Preparing photometry data.")

    photometry_table = read_photometry_table(table_path=config.photometry_table_path)
    constants = simulation_data.constants
    sim = simulation_data.simulation
    kernel_table = build_interpolation_table(
        n_bins=config.interpolation_bins, kernel_type=KERNEL_INT_MAP[config.kernel_type]
    )
    los_axis = LOS_AXIS_MAP[config.viewing_axis]

    # construct dust curves
    wavelengths = photometry_table.wavelengths
    n_wavelengths = len(wavelengths)
    dust_curves = _build_dust_curves(wavelengths=wavelengths, n_wavelengths=n_wavelengths, alpha=config.power_law_alpha)

    # construct delta_nu (c/lambda**2 * bin width)
    wavelengths_cm = wavelengths * u.angstrom.to(u.cm)  # convert to CGS
    delta_lambda = np.gradient(wavelengths_cm)
    delta_nu = (constants.C_CGS / wavelengths_cm**2) * delta_lambda

    # metal column density -> stellar A_v conversion factor
    Z_col_to_A_v = (
        constants.M_SUN_G
        * constants.X_H
        * (1 + sim.redshift) ** 2
        / (constants.KPC_CM**2 * constants.PROTON_MASS_G * constants.AV_TO_NH * constants.Z_SUN_WATSON)
    )

    # precompute cosmic extinction (Madau 1995): ssp table is in rest frame so shift
    madau_transmission = extinct_madau(wavelengths=wavelengths * (1.0 + sim.redshift), redshift=sim.redshift)

    # prefactor for absolute magnitude flux (10pc)
    flux_factor_abs = constants.L_SUN_CGS / (4.0 * np.pi * (10.0 * constants.PC_CM) ** 2)
    # prefactor for apparent magnitude flux (luminosity distance)
    if sim.redshift < 1e-10:
        flux_factor_app = flux_factor_abs
        logger.debug(
            "Snapshot is at z = 0; apparent magnitudes are equal to absolute magnitudes as luminosity distance is zero."
        )
    else:
        flux_factor_app = (
            constants.L_SUN_CGS
            / (4.0 * np.pi * sim.cosmology.luminosity_distance(sim.redshift).to("cm").value ** 2)
            * (1.0 + sim.redshift)
        )  # apparent magnitude is computed by blueshifting band; correct for corresponding (1+z) flux reduction

    # UV bounds
    uv_start_idx = np.searchsorted(wavelengths, 1500.0)
    uv_end_idx = np.searchsorted(wavelengths, 3000.0)

    # galaxy data
    galaxies = simulation_data.groups["galaxies"]
    gal_com_pos = galaxies["com_pos_baryon"]
    gal_ssfr, gal_Z = _prepare_galaxy_quantities(
        galaxies=galaxies, Z_sun=constants.Z_SUN_ASPLUND
    )  # REVIEW: asplund metallicity is inherited convention
    star_offsets, star_idx = galaxies.get_particle_csr(ptype="star")
    gas_offsets, gas_idx = galaxies.get_particle_csr(ptype="gas")

    # filter data transmission coefficients [0, 1] + normalisations for the apparent and absolute magnitude integrands
    transmission_weighted_abs, transmission_norm_abs, transmission_weighted_app, transmission_norm_app = (
        _build_filter_transmissions(
            wavelengths=wavelengths,
            n_wavelengths=n_wavelengths,
            delta_nu=delta_nu,
            phot_table=photometry_table,
            bands=config.bands,
            redshift=sim.redshift,
        )
    )

    # build namedtuples
    stars = simulation_data.particles["star"]
    star_data = StarData(
        pos=stars["pos"],
        vel=stars["vel"],
        mass=stars["mass"],
        age=stars["age"],
        metallicity=stars["metallicity"],
        offsets=star_offsets,
        idx_sorted=star_idx,
    )

    # ascertain whether to include dust
    gas = simulation_data.particles["gas"]
    use_dust = config.use_dust and ("dust_mass" in gas.columns)  # both flags have to pass

    if use_dust:
        dust_mass = gas["dust_mass"]
        dust_metallicity = np.ones(shape=len(gas), dtype=np.float64)
        logger.debug("Using dust from snapshot.")
    else:
        dust_mass = gas["mass"]
        dust_metallicity = gas["metallicity"]
        logger.debug("Computing dust from Li (2019) relation.")

    gas_data = GasData(
        pos=gas["pos"],
        smoothing_lengths=gas["smoothing_length"],
        dust_mass=dust_mass,
        metallicity=dust_metallicity,
        offsets=gas_offsets,
        idx_sorted=gas_idx,
    )

    ssp_data = SSPData(
        spectra=photometry_table.spectra,
        mass_remaining=photometry_table.mass_remaining,
        ages=photometry_table.ages,
        metallicities=photometry_table.metallicities,
    )

    if "_rotation_matrices" in galaxies.columns:
        use_rotation = True
        rotation_matrices = galaxies["_rotation_matrices"]
    else:
        use_rotation = False
        rotation_matrices = np.zeros((0, 3, 3))

    phot_constants = PhotometryConstants(
        split_age=config.split_age,
        los_axis=los_axis,
        c_kms=constants.C_KMS,
        boxsize=sim.boxsize,
        redshift=sim.redshift,
        mw_dust_to_metal=constants.MW_DUST_TO_METAL,
        Z_sun=constants.Z_SUN_ASPLUND,  # 0.0134
        Z_col_to_A_v=Z_col_to_A_v,
        use_cosmic_ext=config.use_cosmic_extinction,
        use_dust=use_dust,
        use_rotation=use_rotation,
        keep_spectra=config._keep_spectra,  # use for standalone photometry only
        rotation_matrices=rotation_matrices,
    )

    dust_data = DustData(
        dust_curves=dust_curves,
        dust_law_idx=DUST_CURVE_IDX[config.extinction_law],
        gal_ssfr=gal_ssfr,
        gal_Z=gal_Z,
    )

    filter_data = FilterData(
        transmission_weighted_abs=transmission_weighted_abs,
        transmission_norm_abs=transmission_norm_abs,
        transmission_weighted_app=transmission_weighted_app,
        transmission_norm_app=transmission_norm_app,
        flux_factor_abs=flux_factor_abs,
        flux_factor_app=flux_factor_app,
    )

    logger.info(f"Computing photometric properties for galaxies: {galaxies.n_groups:,} members")
    (
        mag_abs,
        mag_abs_nodust,
        mag_app,
        mag_app_nodust,
        luminosity_fir,
        beta,
        beta_nodust,
        spectra_dust,
        spectra_nodust,
    ) = compute_photometric_properties(
        star_data=star_data,
        gas_data=gas_data,
        ssp_data=ssp_data,
        filter_data=filter_data,
        dust_data=dust_data,
        phot_constants=phot_constants,
        gal_com_pos=gal_com_pos,
        n_galaxies=galaxies.n_groups,
        delta_nu=delta_nu,
        wavelengths=wavelengths,
        uv_start_idx=uv_start_idx,
        uv_end_idx=uv_end_idx,
        madau_transmission=madau_transmission,
        kernel_table=kernel_table,
    )

    results: dict[str, np.ndarray] = {}

    for band_idx, band_name in enumerate(config.bands):
        results[f"mag_abs_{band_name}"] = mag_abs[:, band_idx]
        results[f"mag_abs_nodust_{band_name}"] = mag_abs_nodust[:, band_idx]
        results[f"mag_app_{band_name}"] = mag_app[:, band_idx]
        results[f"mag_app_nodust_{band_name}"] = mag_app_nodust[:, band_idx]

    results["luminosity_fir"] = luminosity_fir
    results["beta"] = beta
    results["beta_nodust"] = beta_nodust

    if config._keep_spectra:
        results["spectra"] = spectra_dust
        results["spectra_nodust"] = spectra_nodust

    galaxies.write_batch(results=results)  # use write_batch for shape checking

    logger.info(f"Successfully computed photometric properties for {galaxies.n_groups:,} galaxies.")


def _prepare_galaxy_quantities(galaxies: GroupStore, Z_sun: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares galaxy quantities needed for photometry with astropy-backed unit conversions. Returns:

    - gal_ssfr: specific ssfr of galaxies in log10(Gyr**-1)
    - gal_Z: sfr-weighted galaxy metallicity in log10(dimensionless)
    """
    ssfr_to_gyr_inv = (CODE_UNITS["sfr"].unit / CODE_UNITS["mass"].unit).to(u.Gyr**-1)

    ssfr_gyr = guarded_divide(
        numerator=(galaxies["sfr"] * ssfr_to_gyr_inv),
        denominator=(galaxies["mass_star"]),
        fill_value=0.0,
    )
    ssfr_gyr[ssfr_gyr <= 0.0] = 1e-30  # prevent NaN/inf
    gal_ssfr = np.log10(ssfr_gyr)
    Z_sfr_weighted = np.nan_to_num(
        galaxies["metallicity_gas_sfr_weighted"], nan=0.0
    )  # masked to NaN for undefined groups; set to zero as we want magnitudes for all galaxies
    Z_ratio = guarded_divide(
        numerator=Z_sfr_weighted,
        denominator=Z_sun,
        fill_value=0.0,
    )
    Z_ratio[Z_ratio <= 0.0] = 1e-30  # prevent NaN/inf
    gal_Z = np.log10(Z_ratio)

    return gal_ssfr, gal_Z


def _build_dust_curves(wavelengths: np.ndarray, n_wavelengths: int, alpha: float) -> np.ndarray:
    """
    Constructs the 2D dust curves array, shaped automatically according to the dust curve indices dict and the number of available wavelengths n_wavelengths. Returns:

    - dust_curves: (n_dust_curves, n_wavelengths) array.
    """
    calzetti = atten_calzetti(wavelengths=wavelengths)
    conroy = atten_conroy(wavelengths=wavelengths)
    power_law = atten_power_law(wavelengths=wavelengths, alpha=alpha)
    cardelli = extinct_cardelli(wavelengths=wavelengths)
    smc = extinct_smc(wavelengths=wavelengths)
    lmc = extinct_lmc(wavelengths=wavelengths)

    # pack dust curves into the expected 2D array form
    dust_curves = np.empty(shape=(len(DUST_CURVE_IDX), n_wavelengths), dtype=np.float64)

    dust_curves[DUST_CURVE_IDX["CALZETTI"]] = calzetti
    dust_curves[DUST_CURVE_IDX["CONROY"]] = conroy
    dust_curves[DUST_CURVE_IDX["CARDELLI"]] = cardelli
    dust_curves[DUST_CURVE_IDX["SMC"]] = smc
    dust_curves[DUST_CURVE_IDX["LMC"]] = lmc
    dust_curves[DUST_CURVE_IDX["POWER_LAW"]] = power_law

    return dust_curves


def _build_filter_transmissions(
    wavelengths: np.ndarray,
    n_wavelengths: int,
    delta_nu: np.ndarray,
    phot_table: PhotometryTable,
    bands: list[str],
    redshift: float,
) -> tuple[np.ndarray, ...]:
    """
    Constructs the weights and normalisation for the filter integrands in apparent and absolute magnitudes.
    """
    # allocate output arrays
    n_bands = len(bands)
    transmission_weighted_abs = np.zeros((n_bands, n_wavelengths), dtype=np.float64)
    transmission_norm_abs = np.empty(n_bands, dtype=np.float64)
    transmission_weighted_app = np.zeros((n_bands, n_wavelengths), dtype=np.float64)
    transmission_norm_app = np.empty(n_bands, dtype=np.float64)

    for band_idx, band_name in enumerate(bands):
        curve = phot_table.filters[band_name]

        # absolute magnitude: rest frame (what ssp table outputs)
        T_abs = np.interp(
            wavelengths, curve.wavelength, curve.transmission, left=0.0, right=0.0
        )  # set to 0 outside valid range
        transmission_weighted_abs[band_idx] = T_abs * delta_nu
        transmission_norm_abs[band_idx] = np.sum(T_abs * delta_nu)

        # apparent magnitude: observer frame, shift ssp output
        T_app = np.interp(
            wavelengths, curve.wavelength / (1.0 + redshift), curve.transmission, left=0.0, right=0.0
        )  # set to 0 outside valid range
        transmission_weighted_app[band_idx] = T_app * delta_nu
        transmission_norm_app[band_idx] = np.sum(T_app * delta_nu)

    return transmission_weighted_abs, transmission_norm_abs, transmission_weighted_app, transmission_norm_app
