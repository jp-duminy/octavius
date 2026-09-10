<div align="center">
<pre>
████████████████████████████████████
▄                                  ▄
▄ ░█▀█░█▀▀░▀█▀░█▀█░█░█░▀█▀░█░█░█▀▀ ▄
▄ ░█░█░█░░░░█░░█▀█░▀▄▀░░█░░█░█░▀▀█ ▄
▄ ░▀▀▀░▀▀▀░░▀░░▀░▀░░▀░░▀▀▀░▀▀▀░▀▀▀ ▄
▄                                  ▄
████████████████████████████████████
</pre>

[![PyPI](https://img.shields.io/pypi/v/octavius)](https://pypi.org/project/octavius/)
[![DOI](https://zenodo.org/badge/1136349333.svg)](https://doi.org/10.5281/zenodo.22166418)
[![Python](https://img.shields.io/pypi/pyversions/octavius)](https://pypi.org/project/octavius/)
[![CI](https://github.com/jp-duminy/octavius/actions/workflows/ci.yml/badge.svg)](https://github.com/jp-duminy/octavius/actions)
[![codecov](https://codecov.io/gh/jp-duminy/octavius/graph/badge.svg?token=71Y900J2OY)](https://codecov.io/gh/jp-duminy/octavius)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENCE)
[![Docs](https://readthedocs.org/projects/octavius/badge/?version=latest)](https://octavius.readthedocs.io/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Dependencies](https://img.shields.io/librariesio/release/pypi/octavius)](https://libraries.io/pypi/octavius)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

# Octavius: The Next-Generation Simulation Analysis Toolkit

Octavius is a high-performance, fully-parallel code designed to produce analysis catalogues of hydrodynamical simulation snapshots. Written entirely in Python with minimal dependencies and all features ready out-of-the-box, it is designed to cleanly slot into analysis workflows.

Catalogues are stored as HDF5 files containing physical properties and membership mapping for both haloes and galaxies, along with cosmological information from the snapshot; the package includes a user-friendly API for working with its catalogues.

Please refer to the [documentation](https://octavius.readthedocs.io/) for more information.

## Install

Octavius is packaged on pip, with a planned conda-forge deployment in the near-future. It can be quickly installed from the terminal by running:

```terminal
    pip install octavius
```

## Features

- Supports SWIFT (EAGLE, KIARA, COLIBRE), TNG, and SIMBA snapshots
- Supports AHF, HBT-HERONS and SUBFIND halo catalogues
- Produces snapshot-agnostic catalogues
- Built-in galaxy finding with a 6D friends-of-friends algorithm
- Computes over fifty properties for haloes and galaxies (including subhaloes)
- Photometry in all FSPS-compatible bands with dust attenuation (no radiative transfer)
- User-friendly API for working with analysis catalogues
- Comprehensive membership mapping, including hierarchies
- Standalone analysis tools (on-the-fly pipeline stages)
- Comprehensive unit and regression tests

## Quickstart Guide

To analyse a snapshot, the only prerequisite is to configure Octavius: this can be done through a [YAML file](octavius/config.yaml). Once installed, a YAML file for you to fill in can be generated in your current directory by running:

```terminal
    octavius init
```

The analysis can then be called either from the command line, or a Python script.

```terminal
    mpiexec -n 2 octavius analyse -c /path/to/config.yaml
```

```python
    from pathlib import Path
    import octavius as oc

    # generate table for photometry (optional)
    photometry_table_path = Path("/path/to/photometry_table.hdf5")
    oc.generate_photometry_table(photometry_table_path)

    # from YAML
    config_filepath = Path("/path/to/config.yaml")
    config = oc.OctaviusConfig.from_yaml(config_filepath)

    # can also type parameters manually
    config = oc.OctaviusConfig(...)

    # analyse a snapshot
    catalogue_path = oc.analyse_snapshot(config)
```

It is recommended to use the command-line method for larger snapshots and more complex workflows, as it is more flexible. For more information, please run:

```terminal
    octavius analyse --help
```

An object-oriented API is provided for loading and conveniently interfacing with catalogues. This includes methods of mapping the galaxies and haloes in the catalogue to their constituent particles in the raw snapshot as well as group hierarchies (including subhaloes).

```python
    from pathlib import Path
    import octavius as oc

    catalogue_path = Path("/path/to/catalogue.hdf5")

    catalogue = oc.load_catalogue(catalogue_path)
    # get array of galaxy stellar masses in grams
    star_mass_grams = catalogue.galaxies.get_dataset("mass_star", to_units="g")
```

Furthermore, a standalone analyser is provided to run analysis routines on subsets of haloes and galaxies from the catalogue. This reduces complex workflows to a few lines in a Python script:

```python
    from pathlib import Path
    import octavius as oc

    snapshot_path = Path("/path/to/snapshot.hdf5")
    config_path = Path("/path/to/config.yaml")

    config = oc.OctaviusConfig.from_yaml(config_path)

    catalogue_path = oc.analyse_snapshot(config)
    catalogue = oc.load_catalogue(catalogue_path)
    analyser = oc.build_analyser(catalogue=catalogue, config=config)

    # compute face-on photometry for galaxies 4, 22, 37
    photometry_data = analyser.compute_photometry(group_indices=[4, 22, 37], orientation="face-on")  
```

Octavius is mostly written in [numba](https://numba.readthedocs.io/en/stable/), a just-in-time compiler for Python code; this can add a minute or two of overhead when first run, but thereafter, the compiled functions are cached.

## Dependencies

- [numba](https://numba.readthedocs.io/en/stable/)
- [numpy](https://numpy.org/doc/stable/)
- [astropy](https://docs.astropy.org/en/stable/index.html)
- [h5py](https://docs.h5py.org/en/stable/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

Optional:

- [fsps](https://python-fsps.readthedocs.io/en/latest/) (for photometry data tables)

## Citing

Octavius can be cited with [Zenodo](https://doi.org/10.5281/zenodo.22166418) `doi: 10.5281/zenodo.22166634`.

## Beta

The package is currently in beta pre-release; please keep this in mind if using it for analysis. You are encouraged to please play around with the code and [report any bugs or inaccuracies](https://github.com/jp-duminy/octavius/issues) which may arise.

## Caesar

Octavius is the spiritual successor to [Caesar](https://caesar.readthedocs.io/) ([source code](https://github.com/dnarayanan/caesar)). There is a short guide for `caesar` users in the [documentation](https://octavius.readthedocs.io/en/latest/guide/caesar_users_guide.html).

## Licence

Octavius is licenced under the BSD 3-Clause licence.

## Contributing

Contributions and bug reports are warmly encouraged. The package is currently pinned to relatively recent versions of its dependencies, and any investigation as to whether these can be relaxed would be most welcome; otherwise, please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed information.

## Contributors

- JP Duminy, University of Edinburgh
- Jakub Szpila, Nicolaus Copernicus Astronomical Center

<small>Last updated by JP Duminy, 10/09/2026.</small>