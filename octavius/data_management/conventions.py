"""

This file can be thought of as a backend warehouse. It contains code units, dtypes,
snapshot unit conversions and physical constants. Also defines the snapshot reader.

"""

# default packages
from dataclasses import dataclass, field, fields
from typing import Any
from yaml import safe_load
from pathlib import Path

# other packages
from astropy.constants import codata2022 as codata
from astropy.constants import iau2015 as iau
import astropy.units as u
from astropy.cosmology import FLRW
import numpy as np

# internal imports
from ..log import get_logger

logger = get_logger()

# for config parsing
CONFIG_FIELDS = frozenset({"thresholds", "physics", "fof6d", "properties", "photometry", "parallelism", "logging"})
FILEPATHS = frozenset({"snapshot_path", "output_dir", "halo_catalogue_path", "photometry_table_path"})
VALID_SIM_TYPES = frozenset({"SIMBA", "SWIFT-KIARA", "SWIFT-EAGLE", "SWIFT-COLIBRE", "TNG"})
VALID_HALO_CATS = frozenset({"SNAPSHOT", "AHF", "HBT-HERONS", "SUBFIND"})
VALID_HALO_CENTRES = frozenset({"MIN_POT", "COM"})
VALID_EXT_LAWS = frozenset({"COMPOSITE", "POWER_LAW", "CARDELLI", "CONROY", "CALZETTI", "MIX_CALZ_MW", "SMC", "LMC"})
VALID_KERNELS = frozenset(["CUBIC", "QUINTIC"])
VALID_VIEW_AXES = frozenset({"X", "Y", "Z"})
VALID_GAS_CRITERIA = frozenset({"COLD", "STARFORMING", "COLD_OR_STARFORMING", "DENSE_ONLY"})
ALWAYS_POSITIVE = frozenset(
    {"b", "velocity_factor", "n_io_chunks", "interpolation_bins", "aperture_size", "virial_factors"}
)

VALID_ENTRIES: dict[str, frozenset[str]] = {
    "simulation_type": VALID_SIM_TYPES,
    "halo_id_source": VALID_HALO_CATS,
    "extinction_law": VALID_EXT_LAWS,
    "viewing_axis": VALID_VIEW_AXES,
    "halo_centre": VALID_HALO_CENTRES,
    "kernel_type": VALID_KERNELS,
}

VALID_COMBOS: dict[str, frozenset[str]] = {
    "SWIFT": frozenset({"AHF", "SNAPSHOT", "HBT-HERONS"}),
    "SIMBA": frozenset({"AHF", "SNAPSHOT"}),
    "TNG": frozenset({"SUBFIND"}),
}


@dataclass(frozen=True, slots=True)
class OctaviusConfig:
    """
    Octavius config object containing user-configured physics/runtime parameters. This can either be directly parsed from a YAML file with the class method (recommended), or created directly.

    Methods
    -------
    from_yaml()
        Parses a YAML file into the config (recommended usage).

    Notes
    -----

    - 'b' controls what fraction of the mean interparticle separation the linking length is. This is usually set to 0.02.
    - 'velocity_factor' is in units of the local velocity dispersion, and controls how many standard deviations from the local velocity dispersion a particle considers its neighbours to be linked in phase space.
    - FRAD is the radiative efficiency (in the accretion formula, usually 0.1).
    - MU is the mean molecular weight (in the virial scaling formulae, usually 0.6)
    - n_io_chunks does not represent chunking through the full pipeline, rather, chunking on reading datasets. This is to alleviate stress on filesystems from multiple ranks simultaneously requesting multi-GB reads, but in practice leaving this at 10 is fine.
    """

    # NOTE: since config is frozen (immutable) the dicts (default mutable) must be created with field(default_factory)
    # default_factory expects a callable so wrap it in lambda:

    snapshot_path: Path
    output_dir: Path
    simulation_type: str
    halo_id_source: str
    cores_per_rank: int

    n_io_chunks: int = 10
    stages: dict[str, bool] = field(
        default_factory=lambda: {
            "find_galaxies": True,
            "properties_core": True,
            "properties_ptype_specific": True,
            "properties_local_environment": True,
            "photometry": True,
        }
    )

    process_ptypes: dict[str, bool] = field(
        default_factory=lambda: {
            "gas": True,
            "star": True,
            "dm": True,
            "bh": True,
        }
    )

    min_stars_per_galaxy: int = 16
    min_dm_per_halo: int = 24

    nH_lim: float = 0.13
    T_lim: float = 1.0e5
    FRAD: float = 0.1
    MU: float = 0.6

    radial_quantiles: dict[str, float] = field(
        default_factory=lambda: {
            "r20": 0.2,
            "half_mass": 0.5,
            "r80": 0.8,
        }
    )
    aperture_size: list[int] = field(default_factory=lambda: [30])
    virial_factors: list[int] = field(default_factory=lambda: [200, 500, 2500])
    halo_centre: str = "MIN_POT"

    b: float = 0.02
    velocity_factor: float = 1.0
    subhalo_override: bool = False
    gas_criterion: str = "COLD_OR_STARFORMING"

    bands: list[str] = field(default_factory=lambda: ["all"])
    extinction_law: str = "COMPOSITE"
    viewing_axis: str = "Z"
    use_dust: bool = True
    use_cosmic_extinction: bool = True
    interpolation_bins: int = 5000
    kernel_type: str = "CUBIC"
    power_law_alpha: float = 1.0
    split_age: float = 0.01
    _keep_spectra: bool = False  # used for standalone photometry, not in YAML file

    terminal_output_level: str = "INFO"
    keep_logs: bool = False
    quiet: bool = False

    compress_catalogue: bool = True
    halo_catalogue_path: Path | None = None
    photometry_table_path: Path | None = None

    def __post_init__(self) -> None:
        """
        Post-initialisation validation of both sensible and valid config parameters to prevent errors at runtime.
        """
        # uppercase all strings which are expected to be uppercase
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, str) and f.name not in FILEPATHS:
                object.__setattr__(self, f.name, value.upper())

        # str fields which only have certain allowed inputs
        for field_name, valid_entries in VALID_ENTRIES.items():
            user_entered = getattr(self, field_name)
            if user_entered not in valid_entries:
                raise ValueError(
                    f"'{user_entered}' is not a valid choice; please select from {', '.join(valid_entries)}."
                )

        # fields which cannot be negative
        for field_name in ALWAYS_POSITIVE:
            user_entered = getattr(self, field_name)

            if isinstance(user_entered, list):
                if any(v <= 0 for v in user_entered):
                    raise ValueError(f"'{field_name}': all entries must be positive.")

            else:
                if user_entered <= 0:
                    raise ValueError(f"'{field_name}' must be positive.")

        # valid permutations of simulation & halo catalogue
        sim_prefix = self.simulation_type.split("-")[0]
        valid_sources = VALID_COMBOS.get(sim_prefix)
        if valid_sources is not None and self.halo_id_source not in valid_sources:
            raise ValueError(
                f"'{self.halo_id_source}' is not currently supported for '{self.simulation_type}'; "
                f"please select from {', '.join(sorted(valid_sources))}."
            )

        # HACK: auto-expand photometry bands for user convenience
        if self.stages.get("photometry", False) and self.photometry_table_path is not None:
            from ..photometry.photometry_tables import resolve_band_names, read_filter_names  # avoid circular import

            names, lambda_effs = read_filter_names(self.photometry_table_path)
            object.__setattr__(self, "bands", resolve_band_names(self.bands, names, lambda_effs))

    @classmethod
    def from_yaml(cls, config_path: Path, **overrides: Any) -> OctaviusConfig:
        """
        Parses an octavius_config.yaml parameter file into the dataclass used internally.
        Names of entries themselves and the file layout must not be changed.

        Parameters
        ----------
        config_path: pathlib.Path
            Path object pointing to the config file.
        **overrides: Any
            Any overrides to config fields specified in the YAML.

        Returns
        -------
        config: OctaviusConfig
            The config dataclass.
        """
        with open(config_path, "r") as f:
            raw = safe_load(f)

        flat = _flatten_config(raw)

        for key in FILEPATHS:
            if key in flat and flat[key] is not None:
                flat[key] = Path(flat[key]).expanduser()

        flat.update(overrides)

        return cls(**flat)  # keyword unpacking saves us from a 35 argument instantiation


def _flatten_config(raw: dict) -> dict:
    """
    Flattens the raw config (in dict form, parsed with yaml) from its nested structure into flat fields (obeying old behaviour so you can just key off config).
    """
    flat: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict) and key in CONFIG_FIELDS:
            for inner_key in value:
                if inner_key in flat:
                    raise ValueError(f"{inner_key} is duplicated in the config.")
            flat.update(value)
        else:
            flat[key] = value

    return flat


@dataclass(frozen=True, slots=True)
class OctaviusConstants:
    """
    The internal constants and scaling relations used in the Octavius pipeline.

    Notes
    -----

    - This is backed by the astropy API and uses the CODATA 2022 & IAU 2015 datasets.
    - Contains some scaling factors.

    To get the astropy datasets please run:

    - from astropy.constants import iau2015 as iau
    - from astropy.constants import codata2022 as codata
    - import astropy.units as u
    """

    # config-dependent parameters
    mu: float = 0.6  # (assuming X=0.7, Y=0.28, Z=0.02)
    frad: float = 0.1

    # fundamental values (CODATA/IAU)
    G_CGS: float = codata.G.cgs.value
    C_CGS: float = codata.c.cgs.value
    C_KMS: float = codata.c.to(u.km / u.s).value
    PROTON_MASS_G: float = codata.m_p.cgs.value
    BOLTZMANN_CGS: float = codata.k_B.cgs.value
    SIGMA_T_CGS: float = codata.sigma_T.cgs.value
    M_SUN_G: float = iau.M_sun.cgs.value
    L_SUN_CGS: float = iau.L_sun.cgs.value
    KPC_CM: float = iau.kpc.cgs.value
    PC_CM: float = iau.pc.cgs.value
    KPC_M: float = iau.kpc.si.value
    GYR_S: float = (1 * u.Gyr).to(u.s).value

    X_H: float = 0.76  # primoridal hydrogen fraction
    MW_DUST_TO_METAL: float = 0.4 / 0.6  # dwek 1998, watson 2011
    Z_SUN_WATSON: float = 0.0189  # watson 2011
    Z_SUN_ASPLUND: float = 0.0134  # asplund 2009
    AV_TO_NH: float = 2.2e21  # watson 2011

    # derived unit conversions
    G_VCIRC: float = codata.G.to(u.km**2 * u.kpc / (u.M_sun * u.s**2)).value

    # derived factors
    VIRIAL_TEMP_FACTOR: float = field(init=False)
    EDD_FACTOR: float = field(init=False)
    Z_COL_TO_AV: float = field(init=False)

    def __post_init__(self) -> None:

        object.__setattr__(  # expect ~3.6e5
            self,
            "VIRIAL_TEMP_FACTOR",
            self.mu * codata.m_p.cgs.value * u.km.to(u.cm) ** 2 / (2 * codata.k_B.cgs.value),
        )

        object.__setattr__(  # expect ~2.2e-8
            self,
            "EDD_FACTOR",
            (4 * np.pi * codata.G.cgs * codata.m_p.cgs / (self.frad * codata.c.cgs * codata.sigma_T.cgs))
            .to(1 / u.yr)
            .value,
        )


@dataclass(slots=True, frozen=True)
class SimulationAttributes:
    """
    Simulation attributes (e.g. hubble parameter), read/derived from the header information.
    """

    # header attributes
    boxsize: float
    h: float
    scale_factor: float
    w_0: float
    w_a: float
    redshift: float
    omega_matter: float  # I renamed this because O0 isn't very readable
    omega_lambda: float
    mis: float

    # derived
    cosmology: FLRW
    time_gyr: float
    time: float
    Hz: float
    rhocrit: float
    rhocrit_comoving: float
    E_z: float
    omega_matter_z: float
    r200_factor: float


@dataclass(frozen=True, slots=True)
class DatasetUnits:
    """
    Astropy unit with the a_exponent baked in; SWIFT cares not for h factors and we divide them out in GIZMO, so we omit it.
    """

    unit: u.UnitBase
    a_exponent: int = 0


CODE_UNITS = {
    "pos": DatasetUnits(unit=u.kpc, a_exponent=1),  # kpc * a
    "vel": DatasetUnits(unit=(u.km / u.s)),  # peculiar
    "potential": DatasetUnits(unit=u.km**2 / u.s**2),  # (km/s)**2
    "mass": DatasetUnits(unit=u.M_sun),  # solar masses
    "mass_HI": DatasetUnits(unit=u.M_sun),  # solar masses
    "mass_H2": DatasetUnits(unit=u.M_sun),  # solar masses
    "dust_mass": DatasetUnits(unit=(u.M_sun)),  # solar masses
    "dust_mass_fractions": DatasetUnits(unit=(u.dimensionless_unscaled)),
    "species_HI": DatasetUnits(unit=u.dimensionless_unscaled),
    "species_H2": DatasetUnits(unit=u.dimensionless_unscaled),
    "HI_abundance": DatasetUnits(unit=u.dimensionless_unscaled),
    "H2_fraction": DatasetUnits(unit=u.dimensionless_unscaled),
    "internal_energy": DatasetUnits(unit=(u.cm**2 / u.s**2)),  # CGS
    "temperature": DatasetUnits(unit=u.K),  # kelvin
    "metallicity": DatasetUnits(unit=u.dimensionless_unscaled),  # dimensionless
    "age": DatasetUnits(unit=u.Gyr),
    "rho": DatasetUnits(unit=(u.g / u.cm**3)),  # CGS
    "rhocrit": DatasetUnits(unit=(u.M_sun / u.kpc**3)),
    "helium_fraction": DatasetUnits(unit=u.dimensionless_unscaled),
    "electron_abundance": DatasetUnits(unit=u.dimensionless_unscaled),
    "sfr": DatasetUnits(unit=(u.M_sun / u.yr)),  # solar masses/yr
    "bhmass": DatasetUnits(unit=u.M_sun),  # solar masses
    "bhmdot": DatasetUnits(unit=(u.M_sun / u.yr)),  # solar masses/yr
    "smoothing_length": DatasetUnits(unit=u.kpc, a_exponent=1),  # ckpc
}

DTYPES = {
    "particle_id": np.int64,
    "ptype": np.int8,  # this allows up to 256 ptypes
    # NOTE: quantities requiring 64-bit precision
    "pos": np.float64,
    "vel": np.float64,
    "mass": np.float64,
    "dust_mass": np.float64,
    "rho": np.float64,
    "internal_energy": np.float64,
    "sfr": np.float64,
    "metallicity": np.float64,
    "mass_HI": np.float64,
    "mass_H2": np.float64,
    "potential": np.float64,
    "age": np.float64,
    "bhmass": np.float64,
    "bhmdot": np.float64,
    # NOTE: external halo readers (AHF) sometimes store absurdly large integers for HIDs
    "HaloID": np.int64,
    "GalID": np.int64,
    "parent_halo": np.int64,
    "ngas": np.int32,  # largest halo in simba had 19mil particles; this should suffice
    "nstar": np.int32,
    "ndm": np.int32,
    "nbh": np.int32,
    "csr_offsets": np.int64,
    "csr_lengths": np.int32,  # same justification as for nparticles
    "csr_indices": np.int64,
}


def gadget_unit_conversion_factor(dataset: str, h: float, a: float) -> float:
    """
    Factors for converting standard GADGET units to internal code units.
    Please see http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps-units
    Astropy for automatic dimensional analysis.
    """
    conversions = {
        "pos": (u.kpc * a / h),
        "vel": (u.km / u.s * np.sqrt(a)),
        "potential": (u.km**2 / u.s**2),
        "mass": (1e10 * u.M_sun / h),  # 1e10 factor is just a Gizmo scaling convention
        "rho": (1e10 * u.M_sun * h**2 / (u.kpc**3 * a**3)),
        "internal_energy": (u.km**2 / u.s**2),
        "sfr": (u.M_sun / u.yr),
        "metallicity": (u.dimensionless_unscaled),
        "fHI": (u.dimensionless_unscaled),
        "fH2": (u.dimensionless_unscaled),
        "age": (u.Gyr),
        "bhmass": (1e10 * u.M_sun / h),
        "bhmdot": (1e10 * u.M_sun / (h * u.kpc / (u.km / u.s))),
        "helium_fraction": (u.dimensionless_unscaled),
        "electron_abundance": (u.dimensionless_unscaled),
        "dust_mass": (1e10 * u.M_sun / h),
        "smoothing_length": (u.kpc * a / h),
    }

    snap_unit = conversions[dataset]
    target = CODE_UNITS[dataset]
    target_quantity = a**target.a_exponent * target.unit

    factor = (
        (1 * snap_unit) / target_quantity
    ).decompose()  # 1 * snap_unit is needed to convert to an astropy Quantity
    assert factor.unit == u.dimensionless_unscaled, f"Unit mismatch for {dataset}"

    return factor.value


def output_catalogue_path(snapshot_path: Path, output_dir: Path) -> Path:
    """
    Returns the Path object pointing to the final production output analysis catalogue filename.
    """
    return output_dir / f"octavius_{snapshot_path.stem}.hdf5"
