# Lightcones

Octavius can build lightcones of galaxies or haloes from a set of catalogues of one simulation at different redshifts. Only the catalogues are needed; the raw snapshots are not read.

## Building a lightcone

```python
from pathlib import Path
import octavius as oc

catalogue_paths = sorted(Path("/path/to/catalogues").glob("octavius_*.hdf5"))

(lightcone_path,) = oc.build_lightcone(
    catalogue_paths=catalogue_paths,
    output_path=Path("lightcone.hdf5"),
    area=2.0,                         # deg^2
    z_min=0.0,
    z_max=3.0,
    group_type="galaxies",            # or "haloes"
    properties=["mass_star", "sfr"],  # any catalogue datasets
    redshift_space=True,              # also compute z_obs
    seed=42,
)
```

`build_lightcone()` returns a list of the files written. With `n_lightcones > 1`, independent lightcones are written to `lightcone_000.hdf5`, `lightcone_001.hdf5`, and so on.

The same is available from the command line (`octavius lightcone -h` lists all options):

```terminal
octavius lightcone /path/to/catalogues/octavius_*.hdf5 --area 2 --z-min 0 --z-max 3 \
    -g galaxies -p mass_star sfr --redshift-space --seed 42 -o lightcone.hdf5
```

### Field shape

The field is centred on the line of sight at (RA, Dec) = (0, 0). It is `square` by default (|RA|, |Dec| ≤ half-width), or `circle` (all objects within a radius of the centre), which may suit radio or X-ray surveys. In both cases the field covers exactly `area` deg² on the sphere. The half-width (or radius) is stored in the header as `half_angle`.

### How the lightcone is filled

The observer sits at a random position, and the lightcone is split into comoving-distance shells, one per catalogue: each distance is filled from the snapshot nearest to it, so shell edges sit midway between snapshot distances. Shells nearer than the lowest-redshift snapshot (or further than the highest) are filled from that snapshot, with a warning when `z_max` is beyond the last one. Distances and redshifts use the simulation's cosmology from the catalogue headers, which must agree between catalogues.

The box is periodically replicated to cover each shell, so the field may be wider than the box. Each replica is a randomly rotated, reflected and shifted copy of the box; `randomise` sets how often a new transform is drawn:

- `snapshot` (default): once per shell. Structures are continuous within a shell, but repeat across the field when it is wider than the box.
- `replica`: for every box replica. Fewer repeated structures, but discontinuities at replica faces.
- `none`: one transform for the whole lightcone.

:::{note}
The same object can appear more than once in a lightcone: in different replicas of its box, or (at different epochs) from different snapshots. The `replica` column gives the box replica each entry came from.
:::

### Redshift space

With `redshift_space=True`, the lightcone also stores the observed redshift `z_obs`, including the line-of-sight peculiar velocity: 1 + z_obs = (1 + z_cos)(1 + v_los / c). Shell membership and the `z_min`/`z_max` cut always use the cosmological redshift `z_cos`.

### Positions

Galaxies are placed at `com_pos_baryon` with velocity `com_vel_baryon`, and haloes at `minpot_pos` with velocity `com_vel_total`. Use `position_column` and `velocity_column` to choose others.

## Reading a lightcone

```python
with oc.load_lightcone(lightcone_path) as lc:
    print(lc)
    ra, dec, z = lc.get_dataset("ra"), lc.get_dataset("dec"), lc.get_dataset("z_obs")
    mstar = lc.get_dataset("mass_star")
```

Each lightcone file contains:

- `header`: construction parameters (area, shape, redshift range, seed, ...) and cosmology, available as `lc.header`.
- `shells`: per source catalogue, its path, redshift and the shell (`d_min`, `d_max`, comoving Mpc) it fills.
- `objects`: per-object datasets, sorted by comoving distance: `ra`, `dec` (deg), `comoving_distance` (Mpc), `z_cos`, `z_obs` (if requested), `snapshot_index`, `group_index`, `replica`, and the requested properties with their catalogue units.

`snapshot_index` indexes `lc.catalogue_paths`, and `group_index` is the object's row in that catalogue, so any other catalogue information can be joined back:

```python
snapshot_index, group_index = lc.get_dataset("snapshot_index"), lc.get_dataset("group_index")
with oc.load_catalogue(lc.catalogue_paths[0]) as cat:
    rows = group_index[snapshot_index == 0]
    sfr = cat.galaxies.get_dataset("sfr")[rows]
```

`get_dataset(name, to_physical=True)` converts comoving properties to physical using the scale factor of each object's source snapshot.

:::{note}
This module is based on the [Simba lightcone code](https://github.com/christopherlovell/simba_lightcone_generation) of Christopher Lovell.
:::
