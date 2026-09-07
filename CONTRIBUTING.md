# Contributing to Octavius

Thank you for taking the time to contribute! Octavius warmly welcomes all contributions.

[I want a quick example](#quick-example)

[Full developer manual](https://octavius.readthedocs.io/en/latest/developers/index.html)

## Where to Begin?

The most important place to begin is the [`internals.yaml`](octavius/internals.yaml) file. This file is authoritative, containing crucial information such as pipeline stages, the data they require, the outputs they produce, and metadata for the catalogue. This file is automatically parsed by [`pipeline_management.py`](octavius/data_management/pipeline_management.py) to resolve stage dependencies, verify the code will produce the intended outputs, ensure data arrives in the right place, and write the analysis to the output catalogue with correct metadata. It is also important to consult [`conventions.py`](octavius/data_management/conventions.py), where units, datatypes, constants, and config parameters are defined.

There are four data structures most fundamental to the analysis, which exist in [`data_structures.py`](octavius/data_management/data_structures.py):

- `SnapshotReader`, which interfaces between Octavius and a raw, external snapshot. 
- `HaloSource`, which maps particles to their haloes.
- `ParticleStore`, a container of particles of the same particle type resembling a dictionary.
- `GroupStore`, a container similar to the ParticleStore, but containing groups of the same type (galaxies, haloes).

Octavius operates on four particle types (known throughout the codebase as `ptypes`): `star`, `gas`, `bh` (black hole), and `dm` (dark matter).

## The Stores

How do we map from a `ParticleStore` to a `GroupStore`? 

The `ParticleStore` contains ndarrays of data for its particles. These include `HaloID`, `SubhaloID`, and `GalID`, which are group ID mappings used to construct a `GroupStore` for a certain group type. It is important to understand how the mapping works because all the analysis is parallelised and computed at group-level. 

The codebase-wide membership representation convention is compressed sparse-row (CSR) format. In this representation there is a flat array of indices (henceforth sorted_idx), where keying sorted_idx by p will produce the index into the ParticleStore for particle p, and a flat array of offsets, which dictates the offset from the start of sorted_idx where the particles of each group start. So, if we want to access the star particles in galaxies, we would call get_csr_membership(ptype="star") on the galaxy GroupStore, and then to iterate over a quantity of particles in galaxy g, we would do:

```python
    for idx in range(offsets[g], offsets[g+1]):
        particle = sorted_idx[idx]
        value = quantity_array[particle]
```

Or when doing fancy indexing, simply do `[offsets[g]:offsets[g+1]]`. This comes with a (sometimes useful) property that `np.diff(offsets)` will give you the number of particles in each group.

It is important to remember Octavius supports one-to-many group mapping to compute inclusive properties in the case of particles belonging to multiple haloes; so you should always iterate over groups and index their particles as opposed to trying go the other way. 

Every `ParticleStore` & `GroupStore` is packaged into respective dictionaries named `particles` and `groups`, which are in turn stored in the `SimulationData` dataclass which is mutable and contains all data for the pipeline. 

## Numba

The analysis functions of Octavius are entirely written in numba, which provides machine-level speed through JIT compilation. The convention is to use nopython mode (njit) for maximum speed. Please refer to numba documentation for [supported numpy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html) and [supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html). On any functions written in numba, the cache=True argument should be passed so the compilation cost is only paid on the first run. Numba also provides parallelisation automatically through parallel=True and prange. 

## Sentinel Values

The [sentinel value](https://en.wikipedia.org/wiki/Sentinel_value), used to denote something does not exist or is unassigned, is -1. For example: a particle with GalID -1 does not belong to any galaxy.

## Tests and Code Quality

Octavius contains an ever-growing assortment of unit and regression tests to ensure the code remains high-quality and performant. These can be found in the [`tests`](tests/) directory. All commits and pull requests must pass the CI workflow set up on github, which runs pytest. It is therefore important to ensure modifications to the code pass the tests, and additions to the code please come with associated tests.

For tests against real data, the validation suite ([`validation_suite.py`](tests/validation/validation_suite.py)) is designed to provide rigorous regression tests as well as detailed memory and runtime profiling. While unit tests call the user-facing pipeline, this file contains a heavier-yet-identical version for the aforementioned purposes. A hand-maintained list of output columns in [`validation_columns.py`](tests/validation/validation_columns.py), though at-times tedious to keep track of, is essential for catching bugs which do not break the code but meaningfully affect physics results.

Ruff is used for linting and formatting; installing the dev configuration will come with pre-commit hooks, which will automatically lint and reformat your code to a standardised style before submitting.

This infrastructure exists to ensure long-term codebase health.

## Code Comments

On functions or methods which form part of the user-facing API, the numpy docstring style is used. Otherwise, docstrings including a brief description of what a function does and what it returns is the general convention. Code comments are encouraged where possible, please.

Where possible, comment annotations (e.g. FIXME: , TODO: , etc.) are also encouraged.

## Ethos

Octavius is intended to be a lean, minimal-dependency catalogue builder for simulations with impressive speed for a Python-first package. The idea is users can quickly install the package and get cracking with analyses. To this end, there are some fundamental principles:

- The internal pipeline should be agnostic to its inputs (SnapshotReaders translate raw snapshots into the internal format)
- MPI happens at the read-in and writing layers, not in pipeline functions
- Where possible, code is accelerated with numba
- Users should be able to have full confidence in their outputs through tests
- The pipeline should scale appropriately to tackle enormous simulations

## Quick Example

Say I would like to add a new property to Octavius. For the sake of a straightforward example, we will go with inertia tensors.

Firstly, we must declare its existence in internals.yaml. From examining the existing stages, we can see the requisite data for the inertia tensor will exist in the properties_core stage:

```yaml
  properties_core:
    label: core
    applies_to: [haloes, galaxies]
    requires: []
    needs_particle_columns:
      all: [pos, vel, mass, potential]
```

We would like to compute it over all ptypes, and total/baryon too. Therefore, we will add it to both:

```yaml
    columns: [..., "inertia_tensor_{ptype}", ...]
        over: {ptype: [gas, star, dm, bh]}
    columns: [..., "inertia_tensor_{combined}", ...]
        over: {combined: [total, baryon]}
```

Now the pipeline expects the dataset to come out of the core properties stage. We must also declare it will be in the output catalogue, and its metadata (the 'over:' field is a shorthand for fields to iterate over):

```yaml
    inertia_tensor_{ptype}:
        over:         {ptype: [gas, star, dm, bh, total, baryon]}
        dtype:        float64
        unit:         "Msun*kpc**2"
        a_exp:        2
        description:  "Moment of inertia tensor."
```

And now we must actually compute it. In [`aggregate_computations.py`](octavius/aggregate_properties/aggregate_computations.py), we can see a function in the engine room called compute_kinematics() is already looping over groups to compute kinematic quantities. Firstly, we allocate the output array:

```python
I_tensor = np.zeros(shape=(n_groups, 3, 3))
```

Then we can simply work it into the loop and return statement, like so:

```python
            # inertia tensor diagonals
            I_tensor[g, 0, 0] += mass * (ry**2 + rz**2)
            I_tensor[g, 1, 1] += mass * (rx**2 + rz**2)
            I_tensor[g, 2, 2] += mass * (rx**2 + ry**2)

            # inertia tensor off-diagonals
            I_tensor[g, 0, 1] -= mass * rx * ry
            I_tensor[g, 1, 0] -= mass * rx * ry
            I_tensor[g, 0, 2] -= mass * rx * rz
            I_tensor[g, 2, 0] -= mass * rx * rz
            I_tensor[g, 1, 2] -= mass * ry * rz
            I_tensor[g, 2, 1] -= mass * ry * rz

    return L, ke_tot, dispersion_sum, I_tensor
```

Then, we can trace it into [`properties_core.py`](octavius/aggregate_properties/properties_core.py). It can be computed for each ptype, then simply added (as compute_kinematics() will compute it for each ptype about the same axis) for the total and baryonic values. 

```python
    L, ke_tot, dispersion_sum, inertia_tensor = compute_kinematics(...)
    # we will set empty groups equal to NaN and zero the degenerate cases
    counts = np.diff(offsets) 
    empty = counts == 0
    small = (counts > 0) & (counts < 3)
    inertia_tensor[empty] = np.nan
    inertia_tensor[small] = 0
    results["inertia_tensor"] = inertia_tensor  # results will be absorbed into GroupStore later in the code
```

The standard method of getting data into a `GroupStore` is to use the `write_batch()` method with a dictionary; alternatively, the column can be written directly by keying the `GroupStore` like a dictionary. It is recommended to store result arrays for quantities in a results dictionary and then use .write_batch() so the `GroupStore` can verify the shape of the data matches, so in this case we write the inertia tensor to the existing results dictionary which is later absorbed.

Then, where combined kinematic quantities are being computed, we can simply add the combined quantity to the loop:

```python
    combined_inertia_tensor = np.zeros(shape=(n_groups, 3, 3))
    for pt in constituent_ptypes:
        ptype_I = np.nan_to_num(group_store[f"inertia_tensor_{pt}"], nan=0.0)  # so NaNs from empty ptypes in the group do not corrupt the total
        combined_inertia_tensor += ptype_I
```

The inertia tensor will now appear in the output catalogue.

New stages should follow the existing templates in `internals.yaml`. Each stage should have an associated `run_stage_name` function which takes `SimulationData` and the `OctaviusConfig` as its only arguments, which can be called from the pipeline: then, define what the stage needs and outputs in `internals.yaml`; the data will automatically be loaded and released for the stage when slotted into the pipeline. 