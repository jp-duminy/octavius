"""

Functions for generating and parsing a bespoke .hdf5 file containing all the data needed
for Octavius's photometry pipeline. This localises the FSPS dependency (which is significantly
heavyweight) to a pre-processing step which only needs to run once.

Also contains the method for hacking the config.

"""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import fsps

# default libraries
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

# other packages
import numpy as np
import h5py

# internal imports
from ..version import __version__
from ..log import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class PhotometryTable:
    """
    Dataclass containing pre-computed FSPS photometry data. Where n_{param} is the number of grid points for a param:

    - spectra: (n_Z, n_age, n_wavelength) array of spectra
    - mass_remaining: (n_Z, n_age) array of the remaining mass
    - ages: (n_age) array of ages (log10)
    - metallicities: (n_Z) array of metallicities (log10, absolute)
    - wavelengths: (n_wavelength) array of wavelengths in angstrom
    - filters: dict of filter curves (with typed access to wavelength and transmission arrays)
    """

    spectra: np.ndarray  # (n_Z, n_age, n_wavelength) float64
    mass_remaining: np.ndarray  # (n_Z, n_age) float64
    ages: np.ndarray  # (n_age,) float64, log10(yr)
    metallicities: np.ndarray  # (n_Z,) float64, log10(Z) absolute
    wavelengths: np.ndarray  # (n_wavelength,) float64, Angstrom
    filters: dict[str, FilterCurve]


class FilterCurve(NamedTuple):
    """
    NamedTuple helper for convenience in accessing photometry table attributes.
    """

    wavelength: np.ndarray
    transmission: np.ndarray
    lambda_eff: float


def read_photometry_table(table_path: Path) -> PhotometryTable:
    """
    Parses the photometry .hdf5 file generated from generate_photometry_table(_from_sp). Returns:

    - PhotometryTable dataclass with all fields populated. Generator ensures all fields are float64.
    """
    logger.info(f"Parsing photometry table at {table_path}")

    with h5py.File(table_path, "r") as tab:
        # debugging
        oversample = tab.attrs["oversample"]
        spectral_library = tab.attrs["spectral_library"]
        logger.debug(f"Oversampling: {oversample}")
        logger.debug(f"Spectral Library: {spectral_library}")

        table_version = tab.attrs["octavius_version"]
        if table_version != __version__:
            logger.warning(
                f"Photometry table was generated with version {table_version}; you are running version {__version__}"
            )

        # SSP datasets
        ssp = tab["ssp"]  # top-level SSP group
        age_dataset = ssp["ages"]
        metal_dataset = ssp["metallicities"]
        wavelength_dataset = ssp["wavelengths"]
        spectra_dataset = ssp["spectra"]
        mass_dataset = ssp["mass_remaining"]

        # ensure the multidimensional datasets have the proper array ordering
        assert spectra_dataset.shape == (metal_dataset.shape[0], age_dataset.shape[0], wavelength_dataset.shape[0]), (
            "spectra dataset does not match expected (Z, age, wavelength) shape."
        )
        assert mass_dataset.shape == (metal_dataset.shape[0], age_dataset.shape[0]), (
            "mass_remaining dataset does not match expected (Z, age) shape."
        )

        # filter datasets
        filters: dict[str, FilterCurve] = {}
        filters_group = tab["filters"]

        for (
            filter_name
        ) in filters_group:  # loop over groupd without .items() otherwise VS code can't infer types (annoying)
            filter_group = filters_group[filter_name]
            wave = filter_group["wavelength"][:]
            trans = filter_group["transmission"][:]
            lambda_eff = filter_group.attrs["lambda_eff"]
            curve = FilterCurve(wavelength=wave, transmission=trans, lambda_eff=lambda_eff)
            filters[filter_name] = curve

        # build PhotometryTable
        table = PhotometryTable(
            spectra=spectra_dataset[:],
            mass_remaining=mass_dataset[:],
            ages=age_dataset[:],
            metallicities=metal_dataset[:],
            wavelengths=wavelength_dataset[:],
            filters=filters,
        )

    logger.info("Successfully parsed photometry table.")

    return table


def generate_photometry_table(
    output_path: Path,
    imf: str = "chabrier",
    nebular_emission: bool = True,
    oversample: tuple[int, int] = (2, 2),
) -> None:
    """
    Generates a bespoke .hdf5 file containing all SSP and filter curve data needed for the Octavius photometry
    pipeline. The config field photometry_table should be pointed at this file to be parsed at runtime.

    Parameters
    ----------
    output_path: pathlib.Path
        Path object pointing to where you would like to save the photometry table.
    imf: str
        The choice of IMF to use with FSPS. All FSPS IMFs are supported (option 4, piecewise, is named "piecewise"); default: ``"chabrier"``.
    nebular_emission: bool
        Whether or not to include nebular emission with FSPS. Default: ``True``.
    oversample: tuple[int, int]
        The factors in [age, metallicity] by which to oversample the native FSPS grid ranges from for more accurate interpolation; will increase filesize.
    """
    import fsps  # avoid fsps being a default dependency

    imf_to_int = {  # conversion from the IMF names to their fsps counterparts
        "salpeter": 0,
        "chabrier": 1,
        "kroupa": 2,
        "dave": 3,
        "piecewise": 4,
    }

    if imf not in imf_to_int:
        raise ValueError(f"{imf} is not included in the available IMFs: {imf_to_int.keys()}")
    imf_int = imf_to_int[imf]

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=imf_int, add_neb_emission=nebular_emission)

    generate_photometry_table_from_sp(output_path=output_path, sp=sp, oversample=oversample)


def generate_photometry_table_from_sp(
    output_path: Path,
    sp: fsps.StellarPopulation,
    oversample: tuple[int, int] = (2, 2),
) -> None:
    """
    Uses an existing python FSPS StellarPopulation (sp) object to generate a bespoke .hdf5
    file containing all SSP and filter curve data needed for the Octavius photometry pipeline.
    The config field photometry_table should be pointed at this file to be parsed at runtime.

    Parameters
    ----------
    output_path: pathlib.Path
        Path object pointing to where you would like to save the photometry table.
    sp: fsps.StellarPopulation
        The StellarPopulation object you would like to use with photometry.
    oversample: tuple[int, int]
        The factors in [age, metallicity] by which to oversample the native FSPS grid ranges from for more accurate interpolation; will increase filesize.
    """
    import fsps

    if sp.params["sfh"] != 0:
        raise ValueError(f"Value of parameters sfh (current: {sp.params['sfh']}) is incompatible (required: 0).")
    if sp._zcontinuous != 1:
        raise ValueError(
            f"Values of parameters zcontinuous (current: {sp.params['zcontinuous']}) is incompatible (required: 1)."
        )

    # values from the sp
    Z_sun = (
        sp.solar_metallicity
    )  # use this directly: FSPS internally uses a recent, 2025 reference for its solar metallicity
    raw_ages = sp.ssp_ages  # log_10(years)
    raw_metallicities = sp.zlegend  # absolute metallicity (linear: I checked the fortran code to confirm)
    raw_wavelengths = sp.wavelengths  # angstroms

    # add oversampling
    oversampled_ages = _oversample_grid(raw_values=raw_ages, factor=oversample[0])
    oversampled_metallicities = _oversample_grid(raw_values=raw_metallicities, factor=oversample[1])
    log_metallicities = np.log10(oversampled_metallicities)

    # allocate output arrays
    n_Z = len(oversampled_metallicities)
    n_age = len(oversampled_ages)
    n_wave = len(raw_wavelengths)
    spectra = np.empty((n_Z, n_age, n_wave), dtype=np.float64)
    mass_remaining = np.empty((n_Z, n_age), dtype=np.float64)

    # loop over populations in metallicity then ages for that metallicity to get spectra and remaining mass from sp
    for i_Z, log_Z in enumerate(log_metallicities):  # enumerate to index into the preallocated arrays
        sp.params["logzsol"] = log_Z - np.log10(Z_sun)  # FSPS expects solar-scaled units

        for i_age, log_age in enumerate(oversampled_ages):
            tage_gyr = 10.0 ** (log_age - 9.0)
            spectrum = sp.get_spectrum(tage=tage_gyr)[1]
            spectra[i_Z, i_age, :] = spectrum
            mass_remaining[i_Z, i_age] = sp.stellar_mass

    # get the filters
    filters = {}
    for filter_name in fsps.list_filters():
        band = fsps.get_filter(filter_name)
        wave, trans = band.transmission  # this isn't typed but it is a tuple of 2 arrays
        filters[filter_name] = (wave, trans)

    # write the hdf5 file
    with h5py.File(output_path, "w") as f:
        ssp_group = f.create_group("ssp")
        ssp_group.create_dataset("spectra", data=spectra)
        ssp_group.create_dataset("mass_remaining", data=mass_remaining)
        ssp_group.create_dataset("ages", data=oversampled_ages)
        ssp_group.create_dataset("metallicities", data=log_metallicities)
        ssp_group.create_dataset("wavelengths", data=raw_wavelengths)

        filter_group = f.create_group("filters")
        for filter_name, (wave, trans) in filters.items():
            band_group = filter_group.create_group(filter_name)
            band_group.create_dataset("wavelength", data=wave)
            band_group.create_dataset("transmission", data=trans)

            # add effective wavelength (transmission-weighted mean wavelength) too
            lambda_eff = np.sum((wave * trans)) / np.sum(trans)
            band_group.attrs["lambda_eff"] = lambda_eff

        # add some metadata
        f.attrs["fsps_version"] = fsps.__version__
        f.attrs["octavius_version"] = __version__
        f.attrs["spectral_library"] = sp.spec_library
        f.attrs["imf"] = sp.params["imf_type"]
        f.attrs["nebular_emission"] = bool(sp.params["add_neb_emission"])
        f.attrs["solar_metallicity"] = Z_sun
        f.attrs["oversample"] = oversample
        f.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()


def read_filter_names(table_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Opens the photometry table and reads the filter names and their effective wavelengths.
    This is so load_internals() can handle the output column expansion when users input a
    shorthand (e.g. sdss expands to sdss_{bands}, which then all get parsed into outputs in internals.yaml)
    """
    with h5py.File(table_path, "r") as f:
        names = list(f["filters"].keys())
        lambda_effs = np.array([f["filters"][n].attrs["lambda_eff"] for n in names])

    return names, lambda_effs


def resolve_band_names(
    requested: list[str],
    available_filters: list[str],
    effective_wavelengths: np.ndarray,
) -> list[str]:
    """
    Matches and expands the user-requested bands in the photometry section of the config.
    This means users can do 'sdss' and get all sdss bands, for example; engineered against
    fsps.list_filters(), all of which are present in our photometry table. Returns:

    - filters_to_process: resolved list of filters to pull out of PhotometryTable, automatically deduplicated
    """
    filters_to_process: list[str] = []

    for filter_name in requested:
        if filter_name == "all":
            return available_filters

        elif filter_name == "uvoir":  # all filters bluewards of 5 microns
            filters_to_process += [
                f for f, lambda_eff in zip(available_filters, effective_wavelengths) if lambda_eff < 50000.0
            ]

        elif filter_name in available_filters:
            filters_to_process.append(filter_name)

        else:  # filter_name is the instrument name usually followed by {_band} so we can grab all the bands associated with a filter
            matches = [f for f in available_filters if f.startswith(filter_name + "_")]

            if not matches:
                raise KeyError(
                    f"{filter_name} is not in the list of available filters: check typo? (run fsps.list_filters() for all filters)"
                )

            filters_to_process += matches

    filters_to_process = list(
        dict.fromkeys(filters_to_process)
    )  # HACK: deduplicate any filters by converting to a dict then back to a list

    if "v" not in filters_to_process:  # v must always be processed to get A_v
        filters_to_process.insert(0, "v")  # and put it first

    return filters_to_process


def _oversample_grid(raw_values: np.ndarray, factor: int) -> np.ndarray:
    """
    Helper to apply the user-requested oversampling to a grid.
    """
    if factor <= 1:  # if no oversampling
        return raw_values.copy()

    n_raw = len(raw_values)
    n_final = (n_raw - 1) * factor + 1  # e.g. [0, x, 1, x, 2, x, 3, x, 4, x, 5] for 2x oversampling

    result = np.empty(shape=n_final, dtype=raw_values.dtype)

    for i in range(n_raw - 1):
        start = i * factor
        result[start : start + factor] = np.linspace(
            raw_values[i], raw_values[i + 1], factor, endpoint=False
        )  # endpoint arg to avoid keying outside range

    result[-1] = raw_values[
        -1
    ]  # the final value will be an interpolated one (from range(n_raw - 1)); set it to the original final value

    return result
