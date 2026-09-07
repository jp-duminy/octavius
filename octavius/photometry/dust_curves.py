"""

Standard attenuation curve formulae which compute the optical depth tau as a function of wavelength.

Top of file: attenuation laws
Bottom of file: extinction laws.

The convention I have gone with in the dust law functions is their bounds key off the values
and units of what the paper defines, e.g. for Calzetti we use angstrom whereas Cardelli we use x = um^-1.

All functions expect to receive a wavelengths array in angstrom. They convert internally to match the coefficients and
conventions of their respective papers for readability and correctedness.

"""

# other packages
import numpy as np
from numba import njit

NORMALISATION = 5500.0  # angstrom


@njit(cache=True)
def atten_power_law(
    wavelengths: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Attenuate based on a simple power law in -(alpha), normalised by default to 5,500 angstrom. Returns:

    - taus: the optical depth at each wavelength.
    """
    taus = (wavelengths / NORMALISATION) ** (-alpha)

    return taus


@njit(cache=True)
def _calzetti_ir(
    wavelength: float,
    R_v: float = 4.05,
) -> float:
    """
    Returns k(lambda) from the Calzetti IR attenuation law (defined between 6.3um and 22um)
    evaluated at the input wavelength (in angstrom).
    """
    wavelength_um = 1e4 / wavelength
    k_lambda = 2.659 * (-1.857 + (1.040 * wavelength_um)) + R_v

    return k_lambda


@njit(cache=True)
def _calzetti_uv(
    wavelength: float,
    R_v: float = 4.05,
) -> float:
    """
    Returns k(lambda) from the Calzetti UV attenuation law (defined between 0.12um and 6.3um)
    evaluated at the input wavelength (in angstrom).
    """
    wavelength_um = 1e4 / wavelength
    k_lambda = (
        2.659 * (-2.156 + (1.509 * wavelength_um) - (0.198 * wavelength_um**2) + (0.011 * wavelength_um**3)) + R_v
    )

    return k_lambda


@njit(cache=True)
def atten_calzetti(wavelengths: np.ndarray) -> np.ndarray:
    """
    Attenuate based on the starburst curve described by Calzetti et al. (2000). The law is
    defined between 0.12um and 22um, so extrapolations to the far-UV and near-IR are made
    using the slope at the edge of either domain. Returns:

    - taus: the optical depth at each wavelength.

    Daniela Calzetti et al. 2000 ApJ 533 682 (doi: 10.1086/308692) Eq. 4
    """
    calzetti_rv = 4.05  # +/- 0.8

    # UV extrapolation: evaluate the UV power law at 1,100 and 1,200 angstrom, then use its slope.
    k_1100 = _calzetti_uv(wavelength=1100, R_v=calzetti_rv)
    k_1200 = _calzetti_uv(wavelength=1200, R_v=calzetti_rv)
    uv_slope = (k_1200 - k_1100) / 100

    # IR extrapolation: evaluate the IR power law at 21,000 and 22,000 angstrom
    k_21900 = _calzetti_ir(wavelength=21900, R_v=calzetti_rv)
    k_22000 = _calzetti_ir(wavelength=22000, R_v=calzetti_rv)
    ir_slope = (k_22000 - k_21900) / 100

    # normalisation
    normalisation = _calzetti_uv(wavelength=NORMALISATION, R_v=calzetti_rv) / calzetti_rv

    n_wave = wavelengths.shape[0]
    taus = np.empty(shape=n_wave, dtype=np.float64)

    for i in range(n_wave):
        wavelength = wavelengths[i]

        if wavelength < 1200.0:
            k = k_1100 + (wavelength - 1100.0) * uv_slope

        elif wavelength < 6300.0:
            k = _calzetti_uv(wavelength=wavelength, R_v=calzetti_rv)

        elif wavelength <= 22000.0:
            k = _calzetti_ir(wavelength=wavelength, R_v=calzetti_rv)

        else:
            k = k_22000 + (wavelength - 22000.0) * ir_slope

        if k < 0.0:
            k = 0.0

        taus[i] = k / calzetti_rv / normalisation

    return taus


@njit(cache=True)
def _conroy_ir(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy IR attenuation law (defined between 0.3um^-1 and 1.1um^-1)
    evaluated at x.
    """
    a = 0.574 * (x**1.61)
    b = -0.527 * (x**1.61)
    result = a + b / R_v

    return result


@njit(cache=True)
def _conroy_optical(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy optical/near-IR attenuation law
    (defined between 1.1um^-1 and 3.3um^-1) evaluated at x.
    """
    y = x - 1.82  # NOTE: in the paper they define y = x - 1.82 where x is in um^-1

    a = (
        1
        + (0.177 * y)
        - (0.504 * y**2)
        - (0.0243 * y**3)
        + (0.721 * y**4)
        + (0.0198 * y**5)
        - (0.775 * y**6)
        + (0.330 * y**7)
    )
    b = (
        (1.413 * y)
        + (2.283 * y**2)
        + (1.072 * y**3)
        - (5.384 * y**4)
        - (0.622 * y**5)
        + (5.303 * y**6)
        - (2.090 * y**7)
    )

    result = a + b / R_v

    return result


@njit(cache=True)
def _conroy_mid_uv(
    x: float,
    f_bump: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy near/mid UV attenuation law
    (defined between 3.3um^-1 and 5.9um^-1) evaluated at x.
    """
    fa = (3.3 / x) ** 6 * (
        -0.0370 + (0.0469 * f_bump) - (0.601 * f_bump / R_v) + (0.542 / R_v)
    )  # this is the paper variable name

    a = 1.752 - (0.316 * x) - ((0.104 * f_bump) / ((x - 4.67) ** 2 + 0.341)) + fa
    b = -3.09 + (1.825 * x) + ((1.206 * f_bump) / ((x - 4.62) ** 2 + 0.263))

    result = a + b / R_v

    return result


@njit(cache=True)
def _conroy_far_uv(
    x: float,
    f_bump: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Conroy far-UV attenuation law
    (defined between 5.9um^-1 and 8.0um^-1) evaluated at x.
    """
    fa = -0.0447 * (x - 5.9) ** 2 - 0.00978 * (x - 5.9) ** 3  # this is the paper variable name
    fb = 0.213 * (x - 5.9) ** 2 + 0.121 * (x - 5.9) ** 3  # this is the paper variable name

    a = 1.752 - (0.316 * x) - ((0.104 * f_bump) / ((x - 4.67) ** 2 + 0.341)) + fa
    b = -3.09 + (1.825 * x) + ((1.206 * f_bump) / ((x - 4.62) ** 2 + 0.263)) + fb

    result = a + b / R_v

    return result


@njit(cache=True)
def atten_conroy(wavelengths: np.ndarray, f_bump: float = 0.6) -> np.ndarray:  # pg192 of Conroy recommends f_bump = 0.6
    """
    Attenuate based on the Milky Way extinction curve with arbitrary UV bump
    (parametrised by f_bump) described by Conroy et al. (2010). This is defined from 0.3um^-1 to
    8.0um^-1. At f_bump = 0, there is no bump; 1.0 returns the standard Milky Way UV bump. Returns:

    - taus: the optical depth at each wavelength.

    Charlie Conroy et al 2010 ApJ 718 184 (doi: 10.1088/0004-637X/718/1/184)
    """
    conroy_rv = 3.1

    wavelengths_inverse_um = 1e4 / wavelengths  # angstrom -> um^-1
    n_wave = wavelengths_inverse_um.shape[0]
    taus = np.empty(shape=n_wave, dtype=np.float64)

    upper_value = 8.0  # where Conroy ends
    end_of_domain = _conroy_far_uv(x=upper_value, f_bump=f_bump, R_v=conroy_rv)  # tau at the value where Conroy ends

    for i in range(n_wave):
        x = wavelengths_inverse_um[
            i
        ]  # wavelength in inverse microns (what the paper calls it, and why the loop looks slightly weird))

        if x > upper_value:  # X-UV extrapolation
            tau = (upper_value / x) ** -1.3 * end_of_domain  # power law taken from pyloser in Caesar (undocumented)
        elif x > 5.9:
            tau = _conroy_far_uv(x=x, f_bump=f_bump, R_v=conroy_rv)
        elif x > 3.3:
            tau = _conroy_mid_uv(x=x, f_bump=f_bump, R_v=conroy_rv)
        elif x > 1.1:
            tau = _conroy_optical(x=x, R_v=conroy_rv)
        else:
            tau = _conroy_ir(x=x, R_v=conroy_rv)

        taus[i] = tau

    return taus


@njit(cache=True)
def _cardelli_ir(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Cardelli IR extinction law
    (defined between 0.3um^-1 and 1.1um^-1) evaluated at x.
    """
    a = 0.574 * (x**1.61)
    b = -0.527 * (x**1.61)
    result = a + b / R_v

    return result


@njit(cache=True)
def _cardelli_optical(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Cardelli optical extinction law
    (defined between 1.1um^-1 and 3.3um^-1) evaluated at x.
    """
    y = x - 1.82  # NOTE: this is the paper convention

    a = (
        1
        + (0.17699 * y)
        - (0.50447 * y**2)
        - (0.02427 * y**3)
        + (0.72085 * y**4)
        + (0.01979 * y**5)
        - (0.77530 * y**6)
        + (0.32999 * y**7)
    )
    b = (
        (1.41338 * y)
        + (2.28305 * y**2)
        + (1.07233 * y**3)
        - (5.38434 * y**4)
        - (0.62251 * y**5)
        + (5.30260 * y**6)
        - (2.09002 * y**7)
    )

    result = a + b / R_v

    return result


@njit(cache=True)
def _cardelli_mid_uv(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Cardelli mid-UV extinction law
    (defined between 3.3um^-1 and 5.9um^-1) evaluated at x.
    """
    a = 1.752 - (0.316 * x) - (0.104 / ((x - 4.67) ** 2 + 0.341))
    b = -3.09 + (1.825 * x) + (1.206 / ((x - 4.62) ** 2 + 0.263))

    result = a + b / R_v

    return result


@njit(cache=True)
def _cardelli_far_uv(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Cardelli far-UV extinction law
    (defined between 5.9um^-1 and 8.0um^-1) evaluated at x.
    """
    fa = -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3
    fb = 0.2130 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3

    a = 1.752 - (0.316 * x) - (0.104 / ((x - 4.67) ** 2 + 0.341)) + fa
    b = -3.09 + (1.825 * x) + (1.206 / ((x - 4.62) ** 2 + 0.263)) + fb

    result = a + b / R_v

    return result


@njit(cache=True)
def _cardelli_xuv(
    x: float,
    R_v: float = 3.1,
) -> float:
    """
    Returns a(lambda) + b(lambda)/R_v from the Cardelli XUV extinction law
    (defined between 8.0um^-1 and 10.0um^-1) evaluated at x.
    """
    y = x - 8.0

    a = -1.073 - (0.628 * y) + (0.137 * y**2) - (0.070 * y**3)
    b = 13.670 + (4.257 * y) - (0.420 * y**2) + (0.374 * y**3)

    result = a + b / R_v

    return result


@njit(cache=True)
def extinct_cardelli(wavelengths: np.ndarray) -> np.ndarray:
    """
    Extinct based on the Milky Way extinction curve described by Cardelli,
    Clayton and Mathis (1989), which is defined from 0.3um^-1 to 10um^-1.

    Cardelli, Clayton and Mathis 1989 ApJ 345 245 (doi: 10.1086/167900)
    """
    cardelli_rv = 3.1

    wavelengths_inverse_um = 1e4 / wavelengths  # angstrom to um^-1
    n_wave = wavelengths_inverse_um.shape[0]
    taus = np.empty(shape=n_wave, dtype=np.float64)

    for i in range(n_wave):
        x = wavelengths_inverse_um[i]

        if x < 1.1:
            tau = _cardelli_ir(x=x, R_v=cardelli_rv)
        elif x < 3.3:
            tau = _cardelli_optical(x=x, R_v=cardelli_rv)
        elif x < 5.9:
            tau = _cardelli_mid_uv(x=x, R_v=cardelli_rv)
        elif x < 8.0:
            tau = _cardelli_far_uv(x=x, R_v=cardelli_rv)
        else:
            tau = _cardelli_xuv(x=x, R_v=cardelli_rv)

        taus[i] = tau

    return taus


@njit(cache=True)
def _pei_extinction(
    wavelength_um: float,
    a_vals: np.ndarray,
    lambda_vals: np.ndarray,
    b_vals: np.ndarray,
    n_vals: np.ndarray,
) -> float:
    """
    Returns tau from the general Pei (1992) Drude profile extinction formula.

    Terms in order represent "background" (BKG); far-ultraviolet (FUV);
    far-infrared (FIR) extinctions; 2175 angstrom; 9.7 um; and 18 um extinction features.
    """
    lambda_v = 0.55  # pg 131 (and obsastro)
    xi = 0.0  # (greek letter xi)
    v_conversion = (
        0.0  # the paper formula gives A(lambda)/A(B); evaluate at the v-band wavelength to also get A(V)/A(B)
    )

    for i in range(lambda_vals.shape[0]):
        xi += a_vals[i] / (
            (wavelength_um / lambda_vals[i]) ** n_vals[i] + (lambda_vals[i] / wavelength_um) ** n_vals[i] + b_vals[i]
        )
        v_conversion += a_vals[i] / (
            (lambda_v / lambda_vals[i]) ** n_vals[i] + (lambda_vals[i] / lambda_v) ** n_vals[i] + b_vals[i]
        )

    result = xi / v_conversion  # divide the paper formula to get A(lambda)/A(V)

    return result


@njit(cache=True)
def extinct_smc(wavelengths: np.ndarray) -> np.ndarray:
    """
    Extinct based on the Small Magellanic Cloud extinction formula described by Pei (1992).

    Yichuan Pei 1992 ApJ 395 130 (doi: 10.1086/171637)
    """
    a_vals = np.array([185.0, 27.0, 0.005, 0.010, 0.012, 0.030])
    lambda_vals = np.array([0.042, 0.08, 0.22, 9.7, 18.0, 25.0])
    b_vals = np.array([90.0, 5.50, -1.95, -1.95, -1.80, 0.00])
    n_vals = np.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0])

    wavelengths_um = wavelengths / 1e4
    n_wave = wavelengths_um.shape[0]
    taus = np.empty(shape=n_wave, dtype=np.float64)

    for i in range(n_wave):
        taus[i] = _pei_extinction(
            wavelength_um=wavelengths_um[i],
            a_vals=a_vals,
            lambda_vals=lambda_vals,
            b_vals=b_vals,
            n_vals=n_vals,
        )

    return taus


@njit(cache=True)
def extinct_lmc(wavelengths: np.ndarray) -> np.ndarray:
    """
    Extinct based on the Large Magellanic Cloud extinction formula described by Pei (1992).

    Yichuan Pei 1992 ApJ 395 130 (doi: 10.1086/171637)
    """
    a_vals = np.array([175.0, 19.0, 0.023, 0.005, 0.006, 0.020])
    lambda_vals = np.array([0.046, 0.08, 0.22, 9.7, 18.0, 25.0])  # only BKG term changes
    b_vals = np.array([90.0, 5.50, -1.95, -1.95, -1.80, 0.00])  # same as SMC
    n_vals = np.array([2.0, 4.5, 2.0, 2.0, 2.0, 2.0])  # only FUV term changes

    wavelengths_um = wavelengths / 1e4
    n_wave = wavelengths_um.shape[0]
    taus = np.empty(shape=n_wave, dtype=np.float64)

    for i in range(n_wave):
        taus[i] = _pei_extinction(
            wavelength_um=wavelengths_um[i],
            a_vals=a_vals,
            lambda_vals=lambda_vals,
            b_vals=b_vals,
            n_vals=n_vals,
        )

    return taus
