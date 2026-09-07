# Aggregate Properties

Octavius computes a wide range of aggregate properties for haloes and galaxies. These come from the pipeline stages `properties_core`, `properties_ptype_specific`, and `properties_local_environment`. Properties are always inclusive, meaning a field halo will have its properties computed from all its constituent particles which can also contribute to the properties of its subhaloes.

:::{note}
Properties are computed for each particle type individually, as well as baryonic (and, for haloes, total). The particle type or aggregation which a quantity is computed for is in its suffix: `{ptype}` refers to `gas`, `star`, `bh`, `baryon`, in addition to `dm` and `total` for haloes.
:::

:::{tip}
When a stage is enabled, all of its outputs will appear in the output catalogue; it is not possible to control the outputs of a stage. Furthermore, the dependency resolution at runtime will re-enable stages which produce outputs required by the user-requested stages if disabled.
:::

:::{tip}
The datasets which are present in a catalogue for either group can be listed through the `describe()` method on an `OctaviusCatalogue` on `haloes` or `galaxies`.
:::

(core-properties)=
## Core Properties

This stage is the most heavy of the three. The 'core' properties are an assortment of familiar astronomical properties such as kinematic, rotational, radial, virial, and morphological quantities.

### Configurable Parameters

- `radial_quantiles`: the enclosed-mass radius quantiles and their names.

- `virial_factors`: the factors of the critical density for which halo virial quantities are computed.

### Per-group outputs

- `n_{ptype}`: the number of particles.

- `mass_{ptype}`: the mass.

- `com_pos_{ptype}`: the centre-of-mass position vector.

- `com_vel_{ptype}`: the centre-of-mass velocity vector.

- `L_{ptype}`: the angular momentum vector $\vec{L}$.

- `L_azimuth_{ptype}`: the azimuthal angle $\phi$ of $\vec{L}$.

- `L_elevation_{ptype}`: the elevation angle $\theta$ of $\vec{L}$.

- `velocity_dispersion_{ptype}`: the mass-weighted velocity dispersion vector $\sigma$.

- `inertia_tensor_{ptype}`: the inertia tensor $\boldsymbol{I}$.

- `radius_{quantile}_{ptype}`: the enclosed-mass radii, where the quantiles are defined in the configuration file.

- `radius_max_{ptype}`: the maximum member particle radius.

### Halo outputs

- `r200m`: the radius which encloses 200x the mean matter density $r_{200m}$.

- `spin_param`: the Bullock spin parameter $\lambda$ (uses $r_{200m}$).

- `temperature_virial`: the virial temperature $T$ (computed using configuration file `MU`) (uses $r_{200m}$).

- `velocity_circular`: the circular velocity at $r_{200m}$.

- `vmax`: the maximum circular velocity $v_{max}$.

- `rmax`: the radius at $v_{max}$.

- `r{factor}c`: the radii enclosing the factor of critical density (e.g. $r_{200c}$), where the factors are defined in the configuration file.

- `m{factor}c`: the mass enclosed within `r{factor}c` (e.g. $m_{200c}$), where the factors are defined in the configuration file.

- `minpot_pos_{ptype}`: the position vector of the particle at the minimum potential (only appears if `halo_centre` is `MIN_POT`).

- `minpot_vel_{ptype}`: the velocity vector of the particle at the minimum potential (only appears if `halo_centre` is `MIN_POT`).

:::{tip}
The minimum potential is often more useful than the centre-of-mass for haloes, owing to the irregular shapes of FOF haloes.
:::

### Galaxy outputs

- `BoverT_{ptype}`: the bulge-to-total kinematic ratio, defined in terms of the counter rotating mass.

- `kappa_rot_{ptype}`: the ratio of rotational-to-total kinetic energy.

(particle-specific-properties)=
## Particle-Specific Properties

This stage computes properties which only pertain to a specific particle type; they are not keyed by their individual ptype suffixes.

### Configurable Parameters

- `nH_lim`: the density, in $n_{H} \ cm^{-3}$, below which gas is considered part of the CGM.

- `T_lim`: the temperature, in $K$, below which gas is considered cold.

### Per-group outputs

Gas:

- `mass_HI`: the neutral hydrogen mass.

- `mass_H2`: the molecular hydrogen mass.

- `sfr`: the total star forming rate.

- `metallicity_gas_mass_weighted`: the mass-weighted metallicity.

- `metallicity_gas_sfr_weighted`: the sfr-weighted metallicity.

- `temperature_gas_mass_weighted`: the mass-weighted temperature.

- `mass_gas_cold`: the mass below `T_lim`, as specified in the configuration file.

- `mass_gas_hot`: the mass above `T_lim`, as specified in the configuration file.

Stars:

- `metallicity_star_mass_weighted` the mass-weighted metallicity.

- `age_star_mass_weighted`: the mean mass-weighted age.

- `age_star_metal_weighted`: the mean metallicity-weighted age.

Black holes:

- `smbh_mass`: the mass of the most massive black hole.

- `smbh_fedd`: the Eddington fraction of the most massive black hole.

- `smbh_mdot`: the accretion rate of the most massive black hole.

### Halo outputs

Gas properties are also computed for the halo CGM, which is defined as being gas with density below `nH_lim` in the configuration file:

- `mass_cgm`: the CGM mass.

- `metallicity_gas_mass_weighted_cgm`: the CGM mass-weighted metallicity.

- `metallicity_gas_temperature_weighted_cgm`: the CGM temperature-weighted metallicity. 

- `temperature_gas_mass_weighted_cgm`: the CGM mass-weighted temperature.

- `temperature_gas_metal_weighted_cgm`: the metallicity-weighted temperature.

(local-environment-properties)=
## Local Environment Properties

This stage runs for galaxies and computes the properties of their local environment: currently, this is only the mass contained within a specified aperture radius from their centres-of-mass. Please see the `OctaviusAnalyser` {ref}`local densities documentation <usage-local-densities>` for information on how to compute local mass and number densities for groups.

:::{warning}
Aperture masses are slightly less accurate for galaxies at the edge of haloes, as they are computed using only in-halo particles.
:::

### Configurable Parameters

- `aperture_size`: the radii in $\mathrm{kpc}$ for which aperture masses are computed.

### Galaxy Outputs

- `mass_{ptype}_{aperture}kpc`: the mass contained within the specified {aperture} in $\mathrm{kpc}$, where the aperture sizes are defined in the configuration file. Do note the the `dm` mass in the aperture is also provided.

- `mass_HI_{aperture}kpc`: the neutral hydrogen mass contained within the specified $\mathrm{kpc}$ aperture.

- `mass_H2_{aperture}kpc`: the molecular hydrogen mass contained within the specified $\mathrm{kpc}$ aperture.

