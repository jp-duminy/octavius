# Quickstart

Once installed, the only prerequisite to analysing a snapshot is a configuration YAML file, which contains all of the parameters and settings necessary to run the pipeline. A configuration file can be automatically generated in your current working directory from the terminal:

```bash
octavius init
```

Photometry requires a bespoke data file containing filters and SSP information from [FSPS](https://github.com/dfm/python-fsps). This can be generated from a Python script using the provided method:

```python
from pathlib import Path  # pathlib is in the standard library
from octavius import generate_photometry_table, generate_photometry_table_from_sp

photometry_table_filepath = Path("/path/to/table.hdf5")

generate_photometry_table(photometry_table_filepath)  # default method with basic options 

import fsps
sp = fsps.StellarPopulation(...)  # full control over FSPS options
generate_photometry_table_from_sp(photometry_table_filepath, sp)
```

Snapshots can be analysed from either the command line or within a Python script: both methods are straightforward. For large snapshots and more complex workflows, it is recommended to run from the command line for greater flexibility:

```bash
octavius analyse --help  # for a list of command-line arguments
mpiexec -n 2 octavius analyse -c /path/to/config.yaml
```

The analyse_snapshot function is directly importable; it is natively MPI-aware.

```python
from pathlib import Path
from octavius import analyse_snapshot, OctaviusConfig

config_filepath = Path("/path/to/config.yaml")
config = OctaviusConfig.from_yaml(config_filepath)  # can optionally add overrides

# you can also type the parameters manually
config = OctaviusConfig(...)

catalogue_path = analyse_snapshot(config)
```

An object-oriented API is provided for loading and conveniently interfacing with catalogues. This includes methods of mapping the galaxies and haloes in the catalogue to their constituent particles in the raw snapshot as well as group hierarchies (including subhaloes).

```python
from pathlib import Path
from octavius import load_catalogue

catalogue_filepath = Path("/path/to/catalogue.hdf5")

catalogue = load_catalogue(catalogue_filepath)  # returns catalogue object

print(catalogue)  # shows basic cataloguealogue information
catalogue.galaxies.describe()  # lists datasets available in galaxies

# array of galaxy stellar masses in grams
star_mass_grams = catalogue.galaxies.get_dataset("mass_star", to_units="g")

# get the stellar masses of their field haloes
field_halo_idx = catalogue.galaxies.get_membership(name="field_halo_index")  # indices into halo data
halo_star_mass_grams = catalogue.haloes.get_dataset("mass_star", to_units="g", mask=field_halo_idx)

# get the indices into the original snapshot of stars in galaxy 0
star_indices = catalogue.galaxies.get_particle_indices(ptype="star", group_index=0)

# get simulation info
boxsize = catalogue.sim_info("boxsize", to_units="Mpc", physical=True)    
```

Catalogue objects provide unit/physical conversions, masks, and metadata information for group properties, as well as simulation metadata and cosmological information.