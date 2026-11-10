"""

Particle type-specific aggregate properties. For example:

- Hydrogen mass fractions (gas)
- Star formation histories (stars)
- Eddington fractions (black holes)
- 1.4 GHz radio luminosities from star formation (gas) and jets (black holes)

In practice this file exists more for modularity purposes; it is by far the lightest stage
of the pipeline at present.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import SimulationData, OctaviusConfig, OctaviusConstants

# other packages
import numpy as np

# internal imports
from ..utils import guarded_divide
from ..aggregate_properties.aggregate_helpers import (
    sum_per_group,
    max_idx_per_group,
)
from ..log import get_logger

logger = get_logger()


def run_ptype_specific_properties(simulation_data: SimulationData, config: OctaviusConfig) -> None:
    """
    Top-level executor for the ptype-specific aggregate properties.
    """
    particles = simulation_data.particles
    constants = simulation_data.constants

    for group_type in simulation_data.groups:
        logger.info(f"Running particle-specific properties for {group_type}.")

        group_store = simulation_data.groups[group_type]
        n_groups = group_store.n_groups

        # gas
        if "gas" in particles:
            gas = particles["gas"]
            gas_offsets, gas_idx = group_store.get_particle_csr(ptype="gas")

            # this lets us decouple core properties as a dependency for ptype properties
            if "mass_gas" in group_store:
                gas_mass = group_store["mass_gas"]
            else:
                gas_mass = sum_per_group(
                    values=gas["mass"], offsets=gas_offsets, idx_sorted=gas_idx, n_groups=group_store.n_groups
                )

            gas_results = compute_gas_properties(
                masses=gas["mass"],
                masses_HI=gas["mass_HI"],
                masses_H2=gas["mass_H2"],
                metallicities=gas["metallicity"],
                temperatures=gas["temperature"],
                sfrs=gas["sfr"],
                gas_mass=gas_mass,
                offsets=gas_offsets,
                idx_sorted=gas_idx,
                n_groups=n_groups,
                Tlim=config.T_lim,
            )
            group_store.write_batch(results=gas_results)
            group_store.write_batch(
                results=compute_radio_sf(
                    sfr=gas_results["sfr"],
                    massive_star_fraction=config.sfr_massive_star_fraction,
                    constants=constants,
                )
            )

            if group_store.kind == "halo":
                cgm_results = compute_cgm_properties(
                    masses=gas["mass"],
                    metallicities=gas["metallicity"],
                    temperatures=gas["temperature"],
                    helium_frac=gas["helium_fraction"],
                    rho=gas["rho"],
                    offsets=gas_offsets,
                    idx_sorted=gas_idx,
                    n_groups=n_groups,
                    nH_lim=config.nH_lim,
                    proton_mass_cgs=constants.PROTON_MASS_G,
                )
                group_store.write_batch(results=cgm_results)

        # stars
        if "star" in particles:
            star = particles["star"]
            star_offsets, star_idx = group_store.get_particle_csr(ptype="star")

            # this lets us decouple core properties as a dependency for ptype properties
            if "mass_star" in group_store:
                star_mass = group_store["mass_star"]
            else:
                star_mass = sum_per_group(
                    values=star["mass"], offsets=star_offsets, idx_sorted=star_idx, n_groups=group_store.n_groups
                )

            star_results = compute_star_properties(
                masses=star["mass"],
                metallicities=star["metallicity"],
                ages=star["age"],
                star_mass=star_mass,
                offsets=star_offsets,
                idx_sorted=star_idx,
                n_groups=n_groups,
            )
            group_store.write_batch(results=star_results)

        # black holes
        if "bh" in particles:
            bh = particles["bh"]
            bh_offsets, bh_idx = group_store.get_particle_csr(ptype="bh")
            bh_results = compute_bh_properties(
                masses=bh["bhmass"],
                bhmdots=bh["bhmdot"],
                offsets=bh_offsets,
                idx_sorted=bh_idx,
                n_groups=n_groups,
                edd_factor=constants.EDD_FACTOR,
            )
            group_store.write_batch(results=bh_results)
            radio_agn_results = compute_radio_agn(
                masses=bh["bhmass"],
                bhmdots=bh["bhmdot"],
                offsets=bh_offsets,
                idx_sorted=bh_idx,
                n_groups=n_groups,
                jet_bh_mass_min=config.jet_bh_mass_min,
                jet_fedd_max=config.jet_fedd_max,
                constants=constants,
            )
            group_store.write_batch(results=radio_agn_results)

        logger.info(f"Particle-specific properties computed for {group_type}.")


def compute_gas_properties(
    masses: np.ndarray,
    masses_HI: np.ndarray,
    masses_H2: np.ndarray,
    metallicities: np.ndarray,
    temperatures: np.ndarray,
    sfrs: np.ndarray,
    gas_mass: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    Tlim: float,
) -> dict[str, np.ndarray]:
    """
    Computes gas-specific properties, returning a dict of:

    - mass_HI, mass_H2
    - sfr
    - metallicity_{mass/sfr}_weighted
    - temperature_gas_mass_weighted
    - mass_gas_hot: gas mass above Tlim
    - mass_gas_cold: gas mass below Tlim

    And writes HI/H2 masses back to ParticleStore for local environment properties.
    """
    results: dict[str, np.ndarray] = {}

    mass_HI = sum_per_group(values=masses_HI, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    mass_H2 = sum_per_group(values=masses_H2, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    sfr = sum_per_group(values=sfrs, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    metal_mass = sum_per_group(
        values=(metallicities * masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    metal_sfr = sum_per_group(values=(metallicities * sfrs), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    temp_mass = sum_per_group(values=(temperatures * masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)

    cold_masses = np.where(temperatures < Tlim, masses, 0.0)
    mass_gas_cold = sum_per_group(values=cold_masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    hot_masses = np.where(temperatures >= Tlim, masses, 0.0)
    mass_gas_hot = sum_per_group(values=hot_masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)

    results["mass_HI"] = mass_HI
    results["mass_H2"] = mass_H2
    results["mass_gas_cold"] = mass_gas_cold
    results["mass_gas_hot"] = mass_gas_hot
    results["sfr"] = sfr
    results["metallicity_gas_mass_weighted"] = guarded_divide(numerator=metal_mass, denominator=gas_mass)
    results["metallicity_gas_sfr_weighted"] = guarded_divide(numerator=metal_sfr, denominator=sfr)
    results["temperature_gas_mass_weighted"] = guarded_divide(numerator=temp_mass, denominator=gas_mass)

    return results


def compute_cgm_properties(
    masses: np.ndarray,
    temperatures: np.ndarray,
    metallicities: np.ndarray,
    helium_frac: np.ndarray,
    rho: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    nH_lim: float,
    proton_mass_cgs: float,
) -> dict[str, np.ndarray]:
    """
    Computes gas CGM-specific quantities (which are only well-defined for haloes), returning a dict of:

    - mass_cgm
    - temp_{mass/metal}_weighted_cgm
    - metallicity_{mass/temp}_weighted_cgm
    """
    results: dict[str, np.ndarray] = {}
    nH = rho * (1 - metallicities - helium_frac) / proton_mass_cgs

    cgm_criterion = nH < nH_lim
    cgm_masses = np.where(cgm_criterion, masses, 0.0)  # cannot mask directly on inclusive csr: must use np.where
    cgm_temperatures = np.where(cgm_criterion, temperatures, 0.0)
    cgm_metallicities = np.where(cgm_criterion, metallicities, 0.0)

    cgm_mass = sum_per_group(values=cgm_masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    cgm_temp_mass = sum_per_group(
        values=(cgm_temperatures * cgm_masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    cgm_temp_metal = sum_per_group(
        values=(cgm_temperatures * cgm_masses * cgm_metallicities),
        offsets=offsets,
        idx_sorted=idx_sorted,
        n_groups=n_groups,
    )
    cgm_metal_mass = sum_per_group(
        values=(cgm_masses * cgm_metallicities), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )

    results["mass_cgm"] = cgm_mass
    results["temperature_gas_mass_weighted_cgm"] = guarded_divide(numerator=cgm_temp_mass, denominator=cgm_mass)
    results["temperature_gas_metal_weighted_cgm"] = guarded_divide(numerator=cgm_temp_metal, denominator=cgm_metal_mass)
    results["metallicity_gas_mass_weighted_cgm"] = guarded_divide(numerator=cgm_metal_mass, denominator=cgm_mass)
    results["metallicity_gas_temperature_weighted_cgm"] = guarded_divide(
        numerator=cgm_temp_metal, denominator=cgm_temp_mass
    )

    return results


def compute_star_properties(
    masses: np.ndarray,
    metallicities: np.ndarray,
    ages: np.ndarray,
    star_mass: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
) -> dict[str, np.ndarray]:
    """
    Computes star-specific properties, returning a dict of:

    - metallicity_star_mass_weighted
    - age_{mass/metal}_weighted
    """
    results: dict[str, np.ndarray] = {}

    metal_mass = sum_per_group(
        values=(masses * metallicities), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )
    age_mass = sum_per_group(values=(ages * masses), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    age_metal = sum_per_group(
        values=(ages * masses * metallicities), offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )

    results["metallicity_star_mass_weighted"] = guarded_divide(numerator=metal_mass, denominator=star_mass)
    results["age_star_mass_weighted"] = guarded_divide(numerator=age_mass, denominator=star_mass)
    results["age_star_metal_weighted"] = guarded_divide(numerator=age_metal, denominator=metal_mass)

    return results


def compute_bh_properties(
    masses: np.ndarray,
    bhmdots: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    edd_factor: float,
) -> dict[str, np.ndarray]:
    """
    Computes (supermassive) black-hole specific properties, returning a dict of:

    - smbh_mdot: mass accretion rate of SMBH
    - smbh_fedd: Eddington fraction of SMBH
    - smbh_mass: mass of SMBH
    """
    results: dict[str, np.ndarray] = {}

    smbh_idx = max_idx_per_group(
        values=masses, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups
    )  # also assigns -1 as sentinel
    with_bh = smbh_idx >= 0

    smbh_mass = np.full(shape=n_groups, fill_value=np.nan)
    smbh_mdot = np.full(shape=n_groups, fill_value=np.nan)

    smbh_mass[with_bh] = masses[smbh_idx[with_bh]]
    smbh_mdot[with_bh] = bhmdots[smbh_idx[with_bh]]
    smbh_fedd = guarded_divide(numerator=smbh_mdot, denominator=(edd_factor * smbh_mass))

    results["smbh_mdot"] = smbh_mdot
    results["smbh_fedd"] = smbh_fedd
    results["smbh_mass"] = smbh_mass

    return results


def compute_radio_sf(
    sfr: np.ndarray,
    massive_star_fraction: float,
    constants: OctaviusConstants,
) -> dict[str, np.ndarray]:
    """
    Computes the 1.4 GHz radio luminosity from star formation (W/Hz), following Thomas et al. (2021): the
    non-thermal and thermal emission of Condon (1992), proportional to the rate of forming stars above 5 Msun, which
    is massive_star_fraction times the total SFR (0.290 for a Chabrier 2003 IMF).

    AI-generated: claude-opus-5-5
    """
    nu = constants.RADIO_FREQUENCY_GHZ
    luminosity_per_sfr = constants.CONDON_NONTHERMAL_W_HZ * nu**-0.8 + constants.CONDON_THERMAL_W_HZ * nu**-0.1

    return {"L_radio_sf": luminosity_per_sfr * massive_star_fraction * sfr}


def compute_radio_agn(
    masses: np.ndarray,
    bhmdots: np.ndarray,
    offsets: np.ndarray,
    idx_sorted: np.ndarray,
    n_groups: int,
    jet_bh_mass_min: float,
    jet_fedd_max: float,
    constants: OctaviusConstants,
) -> dict[str, np.ndarray]:
    """
    Computes the 1.4 GHz core radio luminosity of each group's black holes with jets (W/Hz), following Thomas et al.
    (2021): a black hole has jets if its mass exceeds jet_bh_mass_min and 0 < f_Edd < jet_fedd_max, and then emits
    nu * P_nu = 1e30 erg/s (Mdot / 4e17 g/s)^(17/12) (Kording et al. 2008). The luminosities of all of a group's black
    holes with jets are summed. Mdot is the snapshot's (instantaneous) accretion rate, where Thomas et al. averaged
    it over the past 50 Myr.

    AI-generated: claude-opus-5-5
    """
    fedd = guarded_divide(numerator=bhmdots, denominator=(constants.EDD_FACTOR * masses))
    has_jets = (masses > jet_bh_mass_min) & (fedd > 0.0) & (fedd < jet_fedd_max)

    mdot_g_per_s = np.where(has_jets, bhmdots, 0.0) * constants.MSUN_PER_YR_TO_G_PER_S
    core_power_erg_per_s = constants.KORDING_POWER_ERG_S * (mdot_g_per_s / constants.KORDING_MDOT_G_S) ** (
        constants.KORDING_INDEX
    )
    frequency_hz = constants.RADIO_FREQUENCY_GHZ * 1.0e9
    luminosity_w_per_hz = np.where(has_jets, core_power_erg_per_s / frequency_hz * 1.0e-7, 0.0)  # erg/s -> W

    L_radio_agn = sum_per_group(values=luminosity_w_per_hz, offsets=offsets, idx_sorted=idx_sorted, n_groups=n_groups)
    return {"L_radio_agn": L_radio_agn}
