# Photometry

Octavius includes a full photometry pipeline which can compute absolute and apparent magnitudes in all [FSPS](https://python-fsps.readthedocs.io/en/latest/)-supported bands with dust extinction. The routine is as follows:

- Partition the parent halo into cells sized by the maximum smoothing length
- Compute the dust extinction to each star along the line of sight
- Attenuate the spectra with the user-desired extinction law specified in the configuration file
- Sum the spectra of the stars in each galaxy
- Apply the desired bandpasses

:::{note}
Octavius accounts for dust but does not do full radiative transfer, which should be delegated to a tool like `powderday`.
:::

The [standalone analyser](../guide/standalone_analysis.md) enhances the functionality of photometry, allowing you to rotate galaxies and directly access their SEDs. Please refer to the {ref}`photometry section <usage-photometry>` for more information.

## FSPS Data File

The photometry pipeline requires a bespoke HDF5 data file known as the photometry table which is generated via `FSPS`. The provided method `generate_photometry_table()` includes basic access to the choice of IMF (`default: Chabrier`) and whether to use nebular emission (`default: True`); for full control over the parameters, an `fsps.StellarPopulation` can be passed directly to the provided `generate_photometry_table_from_sp()` method. Any invalid parameter choices will raise an error. 

The `oversample` argument appears on both methods and exists to oversample in `[age, metallicity]` by the specified factors from the native `FSPS` ranges. Star particles will be interpolated onto the table, so the oversampling improves the accuracy of the interpolation through increased granularity. However, the caveat is this will increase the filesize of the data file by the product of the specified factors. As this table is generated once independently of the pipeline, the oversampling is set to `[2, 2]` by default, which produces a file of around `81 MB`.

:::{tip}
The photometry table contains metadata in its top-level attributes, which can be accessed via `h5py`; warnings will be raised if the version of Octavius running the analysis is mismatched with the version which generated the table.
:::

## Configuring Photometry

Photometry includes the most user-configured parameters of the other stages in the configuration file. Functionality is included to:

- Specify a threshold age for stars below which to create sub-bins to improve the accuracy of their spectra
- Control the extinction law
- Apply cosmic extinction
- Increase the granularity of the interpolated approximation of the kernel LOS integral
- Control the type of kernel used

Standalone photometry allows you to:

- Rotate galaxies
- Return SEDs

Please see the {ref}`configuration documentation <photometry-parameters>` for more information.

## Note on Performance

Photometry is the most computationally-expensive stage of the pipeline, taking longer than all other stages combined at scale. This time is dominated by the most massive galaxies: on large snapshots, single galaxies can take minutes by themselves. This is to be expected as the routine is inherently intricate. Optimisations such as partitioning the parent halo into cells for the LOS dust extinction have been implemented, but please allow more time for photometry than the rest of the pipeline. Future optimisation efforts will focus here. 

:::{tip}
The most demanding stage of the routine is computing the extinction for stars and their spectra, so it is recommended to liberally select bands in the configuration file. 
:::

## Galaxy Outputs

Magnitudes:

- `mag_app_{band}`: apparent magnitudes in the specified band with dust extinction
- `mag_app_nodust_{band}`: apparent magnitudes in the specified band (no dust extinction)
- `mag_abs_{band}`: absolute magnitudes in the specified band with dust extinction
- `mag_abs_nodust_{band}`: absolute magnitudes in the specified band (no dust extinction)

Where apparent magnitudes are computed with the luminosity distance. At redshift zero snapshots, the apparent magnitudes are equal to absolute.

Others:

- `luminosity_fir`: the far-infrared luminosity
- `beta`: the UV spectral slope $\beta$ with dust extinction
- `beta_nodust`: the UV spectral slope $\beta$ (no dust extinction)

If running standalone photometry, three new datasets will exist if you enable the `keep_spectra` flag:

- `spectra`: SEDs in $L_\odot \, Hz^{-1}$ with dust extinction
- `spectra_nodust`: SEDs in $L_\odot \, Hz^{-1}$ (no dust extinction)
- `wavelengths`: the corresponding wavelengths in $\AA$
