"""

Helper functions for the photometry pipeline.


"""

# default packages
from collections import namedtuple

# other packages
import numpy as np
from numba import njit

# internal imports
from ..log import get_logger

logger = get_logger()


def extinct_madau(
    wavelengths: np.ndarray,
    redshift: float,
) -> np.ndarray:
    """
    Applies IGM extinction for redshifted wavelengths according to the formula described by Madau (1995).
    The form used here is the approximation described in the footnote of p21, which is claimed to be
    accurate to within 5% and is used as the standard formula in codes such as synphot. Returns:

    - transmission: exp(-tau) lying between [0, 1]

    Madau (1995) ApJ 441 18 (doi: 10.1086/175332).
    """
    if redshift < 1e-10:  # in principle this won't be asked for from the config but this guards in case it is
        logger.debug(
            f"Requested Madau (1995) extinction but snapshot is at redshift {redshift}; Madau terms are all ones."
        )
        return np.ones_like(wavelengths, dtype=np.float64)

    logger.debug("Applying Madau (1995) IGM extinction.")
    lyman_limit = 912.0
    lyman_rest_wavelengths = np.array([1216, 1026, 973, 950])  # p21, after eq. 15
    a_vals = np.array([3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4])  # p21 after eq. 15

    # integral bounds
    x_e = 1 + redshift
    in_bounds = wavelengths < lyman_limit * x_e
    x_c = wavelengths[in_bounds] / lyman_limit

    tau = np.zeros_like(wavelengths, dtype=np.float64)

    for rest_wavelength, coeff in zip(lyman_rest_wavelengths, a_vals):
        mask = wavelengths <= rest_wavelength * x_e
        tau[mask] += coeff * (wavelengths[mask] / rest_wavelength) ** 3.46

    # footnote equation
    tau[in_bounds] += (
        0.25 * (x_c**3.0) * (x_e**0.46 - x_c**0.46)
        + 9.4 * (x_c**1.5) * (x_e**0.18 - x_c**0.18)
        - 0.7 * (x_c**3.0) * (x_c**-1.32 - x_e**-1.32)
        - 0.023 * (x_e**1.68 - x_c**1.68)
    )

    transmission = np.clip(np.exp(-tau), 0.0, 1.0)  # this clip is more a guard and shouldn't be hit in practice

    return transmission


@njit(cache=True)
def build_interpolation_table(
    n_bins: int,
    kernel_type: int,
) -> np.ndarray:
    """
    Precomputes the kernel weight integral via numerical integration over the range of
    impact parameter values with n_bins controlling the granularity. Returns:

    - table: a table of the kernel weights for impact parameters. Key with the impact parameter
    normalised by smoothing length (b/h).
    """
    bin_width = 1.0 / n_bins
    table = np.zeros(shape=(n_bins + 1), dtype=np.float64)

    for i in range(n_bins + 1):  # outer loop over impact params (b)
        b_sq = (i * bin_width) ** 2
        z_max = np.sqrt(1.0 - b_sq)
        integral = 0.0

        for j in range(n_bins):  # inner loop over radial coord along LOS (z)
            z_lo = j * bin_width
            z_hi = z_lo + bin_width

            if z_lo >= z_max:
                break

            q_lo = np.sqrt(z_lo**2 + b_sq)
            q_hi = np.sqrt(z_hi**2 + b_sq)

            if kernel_type == 0:  # 0 = cubic
                w_lo = _cubic_kernel(q=q_lo)
                w_hi = _cubic_kernel(q=q_hi)
            else:  # 1 = quintic
                w_lo = _quintic_kernel(q=q_lo)
                w_hi = _quintic_kernel(q=q_hi)

            integral += 0.5 * bin_width * (w_lo + w_hi)  # trapezoid rule

        table[i] = 2.0 * integral  # factor of 2 comes from symmetry in z

    return table


@njit(cache=True)
def _cubic_kernel(
    q: float,
) -> float:
    """
    Evaluates the 3D cubic kernel and returns the corresponding weight.

    (JJ Monaghan 1992, doi: 10.1146/annurev.aa.30.090192.002551)

    NOTE: different to the FOF6D cubic_spline_kernel() function.
    """
    # defined over interval [0, 1] coming from setting h' = 2h.
    if q >= 1.0:
        return 0.0

    normalisation = 2.0 / np.pi
    u = 2.0 * q  # so the formulae follow the form in the paper (but rescaled)

    if q >= 0.5:
        return normalisation * (2.0 - u) ** 3

    else:
        return normalisation * ((2.0 - u) ** 3 - 4.0 * (1.0 - u) ** 3)


@njit(cache=True)
def _quintic_kernel(q: float) -> float:
    """
    Evaluates the 3D quintic kernel and returns the corresponding weight.

    (Liu and Liu 2010, doi: https://doi.org/10.1007/s11831-010-9040-7); originally described by Morris (1996).
    """
    # defined over interval [0, 1] coming from setting h' = 3h.

    if q >= 1.0:
        return 0.0

    normalisation = 81.0 / (359.0 * np.pi)
    u = 3.0 * q  # so the formulae follow the form in the paper (but rescaled)

    if q > (2.0 / 3.0):
        return normalisation * (3.0 - u) ** 5

    elif q > (1.0 / 3.0):
        return normalisation * ((3.0 - u) ** 5 - 6.0 * (2.0 - u) ** 5)

    else:
        return normalisation * ((3.0 - u) ** 5 - 6.0 * (2.0 - u) ** 5 + 15.0 * (1.0 - u) ** 5)


@njit(cache=True)
def interpolate_ssp(
    log_age: float,
    log_Z: float,
    age_grid: np.ndarray,
    Z_grid: np.ndarray,
    spectra: np.ndarray,
    mass_remaining: np.ndarray,
    out_spectrum: np.ndarray,
) -> float:
    """
    Interpolates the log(age) and log(Z) values for a star into the SSP table to
    recover the spectra and remaining mass. Returns:

    - mass_remaining: the fraction of original mass the star has now
    - (overwrites out_spectrum in place)
    """
    # NOTE: this function is hefty and runs per-star, and therefore requires careful optimisation to avoid memory spikes; hence it overwrites out_spectrum in place and avoids materialising intermediates through fancy indexing

    age_idx, age_frac = _get_interpolation_idx(grid=age_grid, value=log_age)
    Z_idx, Z_frac = _get_interpolation_idx(grid=Z_grid, value=log_Z)

    # formula from wikipedia: bilinear interpolation, "on the unit square", where x is age and y is metallicity
    w11 = (1 - age_frac) * (1 - Z_frac)
    w12 = (1 - age_frac) * Z_frac
    w21 = (1 - Z_frac) * age_frac
    w22 = age_frac * Z_frac

    for wave_idx in range(len(out_spectrum)):  # spectra is (Z, age, wavelength)
        out_spectrum[wave_idx] = (
            w11 * spectra[Z_idx, age_idx, wave_idx]
            + w12 * spectra[Z_idx + 1, age_idx, wave_idx]  # NOTE: columns are in (y, x)
            + w21 * spectra[Z_idx, age_idx + 1, wave_idx]
            + w22 * spectra[Z_idx + 1, age_idx + 1, wave_idx]
        )

    mass_frac = (  # mass_remaining is (Z, age)
        w11 * mass_remaining[Z_idx, age_idx]
        + w12 * mass_remaining[Z_idx + 1, age_idx]
        + w21 * mass_remaining[Z_idx, age_idx + 1]
        + w22 * mass_remaining[Z_idx + 1, age_idx + 1]
    )

    return mass_frac


@njit(cache=True)
def _get_interpolation_idx(grid: np.ndarray, value: float) -> tuple[int, float]:
    """
    Finds the idx into 'grid' where 'value' can be interpolated from and the corresponding coefficient. Returns:

    - idx: index into 'grid' from which the value should be interpolated
    - coefficient: the fraction coefficient for linear polynomial interpolation
    """
    n_grid = len(grid)

    # find position: side="right" returns last idx where value can be inserted to maintain order
    idx = np.searchsorted(grid, value, side="right") - 1  # -1 floors idx to the value in grid
    idx = min(max(idx, 0), (n_grid - 2))  # prevents negative/OOB idx

    fraction = (value - grid[idx]) / (grid[idx + 1] - grid[idx])
    fraction = min(max(fraction, 0.0), 1.0)  # clip to physical range

    return idx, fraction


# for code tidiness (otherwise you get a horrific 35-argument function)
StarData = namedtuple(
    "StarData",
    [
        "pos",
        "vel",
        "mass",
        "age",
        "metallicity",
        "offsets",
        "idx_sorted",
    ],
)

GasData = namedtuple(
    "GasData",
    [
        "pos",
        "smoothing_lengths",
        "dust_mass",
        "metallicity",
        "offsets",
        "idx_sorted",
    ],
)

SSPData = namedtuple(
    "SSPData",
    [
        "spectra",
        "mass_remaining",
        "ages",
        "metallicities",
    ],
)

FilterData = namedtuple(
    "FilterData",
    [
        "transmission_weighted_abs",
        "transmission_norm_abs",
        "transmission_weighted_app",
        "transmission_norm_app",
        "flux_factor_abs",
        "flux_factor_app",
    ],
)

DustData = namedtuple(
    "DustData",
    [
        "dust_curves",
        "dust_law_idx",
        "gal_ssfr",
        "gal_Z",
    ],
)

PhotometryConstants = namedtuple(
    "PhotometryConstants",
    [
        "los_axis",
        "split_age",
        "c_kms",
        "boxsize",
        "redshift",
        "mw_dust_to_metal",
        "Z_sun",
        "Z_col_to_A_v",
        "use_cosmic_ext",
        "use_dust",
        "use_rotation",
        "keep_spectra",
        "rotation_matrices",
    ],
)


DUST_CURVE_IDX = {
    "POWER_LAW": 0,
    "CALZETTI": 1,
    "CONROY": 2,
    "CARDELLI": 3,
    "SMC": 4,
    "LMC": 5,
    "MIX_CALZ_MW": 6,
    "COMPOSITE": 7,
}

LOS_AXIS_MAP = {
    "X": 0,
    "Y": 1,
    "Z": 2,
}

KERNEL_INT_MAP = {
    "CUBIC": 0,
    "QUINTIC": 1,
}
