# Membership

:::{seealso}
Please see the [catalogues](../guide/catalogues.md) documentation for information.
:::

Membership arrays are stored in the catalogue, and provide mappings between related groups and their constituent particles in the raw snapshot. These datasets are accessible with the methods `get_membership()`, `get_particle_indices()`, and, for haloes, `get_galaxies()` and `get_subhaloes()`.

:::{note}
Group-level index arrays are indices into catalogue-level data, unless specified otherwise.
:::

## Halo Membership

- `field_halo_index`: the index into `halo_data` of the parent field halo (-1 for field haloes).

- `parent_halo_index`: the index into `halo_data` of the immediate parent (sub)halo, equal to `field_halo_index` if the parent is the field halo (-1 for field haloes).

- `depth`: the depth in the hierarchy in ascending order (0 for field haloes).

- `central_galaxy_index`: the index into `galaxy_data` of the most massive galaxy belonging to this halo (-1 if the halo has no galaxies).

- `original_ids`: the original halo catalogue IDs (for progenitor matching).

## Galaxy Membership

- `field_halo_index`: the index into `halo_data` of the parent field halo.

- `parent_halo_index`: the index into `halo_data` of the immediate parent (sub)halo, equal to `field_halo_index` if the parent is the field halo.

- `parent_membership_fraction`: the fraction of subhalo-assigned particles in this galaxy which belong to the halo at `parent_halo_index`. In practice, this is usually 1.0 as expected.

## Particle Membership

The particle membership arrays are stored in CSR format for both haloes and galaxies.

- `{ptype}_indices`: the indices into particle-level data from the original snapshot, aligned to group order.

- `{ptype}_offsets`: the offsets into `{ptype}_indices` for each group, such that the constituent particles of group $g$ live at `indices[offsets[g]:offsets[g+1]]`.

- `{ptype}_lengths`: the number of particles in each group, equal to `np.diff(offsets)`.