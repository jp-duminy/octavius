"""

Physics calculations which SnapshotReaders need to use for parsing format-specific
data into the agnostic-expected format.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.cosmology import FLRW
    from .conventions import OctaviusConstants

# default packages
from dataclasses import dataclass

# other packages
import numpy as np
import astropy.units as u

# internal imports
from .data_structures import SimulationAttributes
from ..log import get_logger

logger = get_logger()


def derive_simulation_attributes(
    cosmology: FLRW,
    h: float,
    a: float,
    redshift: float,
    omega_matter: float,
    omega_lambda: float,
    w_0: float,
    w_a: float,
    boxsize: float,
    n_star: int,
    n_gas: int,
    constants: OctaviusConstants,
) -> SimulationAttributes:
    """
    Derives cosmology and calculates mean interparticle separation.

    Returns a SimulationAttributes dataclass.
    """
    time_gyr = cosmology.age(redshift).value
    Hz = cosmology.H(redshift).to(1 / u.s).value
    E_z = cosmology.efunc(redshift)
    rhocrit = cosmology.critical_density(redshift).to(u.M_sun / u.kpc**3).value
    omega_matter_z = cosmology.Om(redshift)

    r200_factor = (200 * 4.0 / 3.0 * np.pi * omega_matter_z * rhocrit * a**3) ** (-1.0 / 3.0)
    mis = calculate_mean_interparticle_separation(n_star=n_star, n_gas=n_gas, boxsize=boxsize)

    return SimulationAttributes(
        h=h,
        boxsize=boxsize,
        scale_factor=a,
        redshift=redshift,
        w_0=w_0,
        w_a=w_a,
        omega_matter=omega_matter,
        omega_lambda=omega_lambda,
        mis=mis,
        cosmology=cosmology,
        time_gyr=time_gyr,
        time=time_gyr * constants.GYR_S,
        Hz=Hz,
        rhocrit=rhocrit,
        rhocrit_comoving=rhocrit * a**3,
        E_z=E_z,
        omega_matter_z=omega_matter_z,
        r200_factor=r200_factor,
    )


def derive_stellar_age(formation_time: np.ndarray, time_gyr: float, cosmology: FLRW) -> np.ndarray:
    """
    Converts stellar formation time (stored as a scale factor by gizmo/swift) into stellar age in GYr.
    """
    redshifts = 1.0 / formation_time - 1.0
    return time_gyr - cosmology.age(redshifts).to_value(u.Gyr)  # see astropy for integration details


def calculate_temperature(
    internal_energy: np.ndarray,
    electron_abundance: np.ndarray,
    helium_fraction: np.ndarray,
    constants: OctaviusConstants,
    gamma: float = 5 / 3,
) -> np.ndarray:
    """
    Calculates temperature from internal energy and electron abundance.
    """
    y_helium = helium_fraction / (4 * (1 - helium_fraction))
    mu = (1 + 4 * y_helium) / (1 + y_helium + electron_abundance)

    mean_molecular_weight = mu * constants.PROTON_MASS_G

    temperature = mean_molecular_weight * (gamma - 1) * internal_energy / constants.BOLTZMANN_CGS

    return temperature


def calculate_mean_interparticle_separation(
    n_star: int,
    n_gas: int,
    boxsize: float,
) -> float:
    """
    Computes the baryonic mean interparticle separation from the number of star and gas particles; black holes make up a small-enough subset of the box to be safely disregarded.

    Returns the mean interparticle separation.
    """
    mis = boxsize / (n_star + n_gas) ** (1.0 / 3.0)
    logger.debug(f"Mean baryonic interparticle separation: {mis:.2f}")

    return mis


@dataclass(frozen=True, slots=True)
class TNGConstants:
    """
    TNG-specific derivation constants, mostly from Stevens et al. (2019).
    """

    BLITZ_ALPHA: float = 0.92  # table 2 in Blitz-Rosolowsky (2006)
    BLITZ_P0: float = 4.3e4  # table 2 in Blitz-Rosolowsky (2006)
    A_0: float = 573.0
    A_EXP: float = -0.8
    T_SN: float = 5.73e7
    T_COLD: float = 1.0e3
    NH_THRESH: float = 0.1065  # https://www.tng-project.org/data/forum/topic/970/star-formation-threshold-density/


def calculate_tng_x_neutral(
    internal_energy: np.ndarray,
    sfr: np.ndarray,
    neutral_fraction: np.ndarray,
    rho: np.ndarray,
    hydrogen_fraction: np.ndarray,
    constants: OctaviusConstants,
    tng_constants: TNGConstants,
) -> np.ndarray:
    """
    Calculates gas fractions for TNG simulations according to Stevens et al. (2019) prescriptions
    in the appendix.
    """
    star_forming = sfr > 0

    mu_cold = 4.0 / (1.0 + 3.0 * hydrogen_fraction[star_forming])
    u_cold = constants.BOLTZMANN_CGS * tng_constants.T_COLD / ((5 / 3 - 1) * mu_cold * constants.PROTON_MASS_G)
    u_SN = constants.BOLTZMANN_CGS * tng_constants.T_SN / ((5 / 3 - 1) * mu_cold * constants.PROTON_MASS_G)

    nH_sf = rho[star_forming] * hydrogen_fraction[star_forming] / constants.PROTON_MASS_G
    A_ = tng_constants.A_0 * (nH_sf / tng_constants.NH_THRESH) ** (tng_constants.A_EXP)
    u_hot = u_cold + u_SN / (1.0 + A_)

    x_neutral = np.empty_like(neutral_fraction)

    cold_fraction = (u_hot - internal_energy[star_forming]) / (u_hot - u_cold)
    x_neutral[star_forming] = np.clip(cold_fraction, 0.0, 1.0)
    x_neutral[~star_forming] = neutral_fraction[~star_forming]

    return x_neutral
