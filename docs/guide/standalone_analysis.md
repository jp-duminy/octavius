# Standalone Analysis

Octavius includes functionality for rerunning analysis routines from the pipeline on groups in its catalogues.

:::{warning}
This method is designed for targeted analysis of subsets of the catalogue; when in the territory of thousands of groups, you should simply rerun the pipeline. The `OctaviusAnalyser` takes full advantage of threaded parallelism, but not MPI.
:::

:::{note}
The standalone analyser will return datasets in the default catalogue format. This format can be accessed with the `get_units()` and `is_comoving()` methods on `galaxies` or `haloes`.
:::

## The OctaviusAnalyser

The `OctaviusAnalyser` is able to call pipeline stages by reconstructing the necessary data structures from the catalogue. It can be created with the `build_analyser()` method, and requires you to provide both `OctaviusConfig` and `OctaviusCatalogue` objects. You may also optionally override the snapshot path stored in the config.

```python
from pathlib import Path
import octavius as oc

catalogue_path = Path("/path/to/catalogue.hdf5")
config_path = Path("/path/to/config.yaml")

config = oc.OctaviusConfig.from_yaml(config_path)  # or type manually
catalogue = oc.load_catalogue(catalogue_path)
analyser = oc.build_analyser(catalogue=catalogue, config=config)
```

:::{warning}
You should ensure the `OctaviusCatalogue` and the config snapshot (under `snapshot_path`) are consistent. `build_analyser()` accepts an optional `snapshot_path` argument for safety.
:::

:::{tip}
When being run in standalone mode, stages which depend on other stages' outputs will need to load those outputs from the catalogue; if they are not present, an error will be raised. It is therefore best to always create catalogues with as much information as possible.
:::

## Usage

The analyser currently supports three standalone stages:

- `properties_core`
- `properties_ptype_specific`
- `photometry`

And one standalone computation:

- `local_densities`

You will need to specify the catalogue indices of the groups of interest via `group_indices`. The groups must all be of the same group type, which is specified by `group_type`. The analyser will return a `StageResult` dataclass, from which the dataset columns can be accessed.

```python
analyser = oc.build_analyser(snapshot_path=snapshot_path, catalogue=catalogue, config=config)
galaxies_of_interest = [0, 1, 3, 22, 47]  # or array
haloes_of_interest = [1, 3]

galaxy_core_properties = analyser.compute_core_properties(group_indices=galaxies_of_interest, group_type="galaxies")
halo_ptype_properties = analyser.compute_ptype_specific_properties(group_indices=haloes_of_interest, group_type="haloes")

galaxy_kappa_rot = galaxy_core_properties["kappa_rot_baryon"]
hot_halo_gas_mass = halo_ptype_properties["mass_gas_hot"]
```

(usage-photometry)=
### Usage: Photometry

Standalone photometry provides enhanced functionality over the pipeline version:

- Galaxy SEDs can be obtained directly
- Galaxies can be rotated

Rotations can be applied using the `orientation` parameter. `face-on` or `side-on` shorthands can be used; alternatively, you can pass a bespoke rotation matrix. The requested rotation is applied to all galaxies. To return spectra, you can enable the `keep_spectra` flag (increases memory footprint). This will cause `spectra`, `spectra_nodust`, and `wavelengths` to appear in the result dataclass: these have units of $L_\odot \, Hz^{-1}$ and $\AA$ respectively.

```python
analyser = oc.build_analyser(catalogue=catalogue, config=config)
galaxies_of_interest = [0, 1, 3, 22, 47]  # or array

side_on_result = analyser.compute_photometry(
    group_indices=galaxies_of_interest, orientation="side-on", keep_spectra=True
)
side_on_spectra = side_on_spectra.columns["spectra"]

# rotate 90 degrees about the x-axis
from scipy.spatial.transform import Rotation
rotation_matrix = Rotation.from_rotvec(np.pi / 2 * np.array([1, 0, 0])).as_matrix()

rotated_result = analyser.compute_photometry(
    group_indices=[0, 1, 2], orientation=rotation_matrix, keep_spectra=True
)
rotated_spectra = rotated_spectra.columns["spectra"]
```

:::{note}
Line-of-sight extinction is computed along the axis specified by `viewing_axis` in the config. If using `face-on` or `side-on` shorthands, the axis will be adjusted automatically. 
:::

:::{tip}
Photometry will need to use gas from the parent field haloes for dust extinction. This is handled internally, but it means `halo_data` must be present in the HDF5 catalogue for standalone photometry to work. 
:::

(usage-local-densities)=
### Usage: Local Densities

The `OctaviusAnalyser` provides a method called `compute_local_densities()` for spatial queries of the local mass and number densities of groups at the specified `radii`, which should be in comoving $\mathrm{kpc}$. This will construct a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) over the centre positions of groups belonging to the specified `group_type`, and return:

- `local_mass_density_{radius}kpc`: the local mass density in $M_\odot \, \mathrm{kpc}^{-3}$ for each $\mathrm{kpc}$ `radius` in `radii`.

- `local_number_density_{radius}kpc`: the local number density in $\mathrm{kpc}^{-3}$ for each $\mathrm{kpc}$ `radius` in `radii`.

```python
analyser = oc.build_analyser(catalogue=catalogue, config=config)
galaxies_of_interest = [0, 1, 3, 22, 47]  # or array

local_result = analyser.compute_local_densities(
    group_indices=galaxies_of_interest, 
    radii=[300, 1000, 3000], 
    group_type="galaxies"
)

local_mass_densities = local_result["local_mass_density_1000kpc"]
```

## Updating Config Parameters

A convenience method on the analyser, `update_config()`, is provided to update the stored OctaviusConfig between runs. This enables you to rerun the analysis with different config parameters.

```python
analyser = oc.build_analyser(catalogue=catalogue, config=config)
galaxies_of_interest = [0, 1, 3, 22, 47]  # or array

analyser.update_config(extinction_law="COMPOSITE")
composite_properties = analyser.compute_photometry(
    group_indices=galaxies_of_interest,
)

analyser.update_config(extinction_law="CARDELLI")
cardelli_properties = analyser.compute_photometry(
    group_indices=galaxies_of_interest,
)
```
