# Format-Specific Information

The internals of the pipeline are agnostic to the simulation (and halo catalogue) type which it receives: snapshot and membership data are parsed into internal data structures which follow defined conventions.

## Halo Catalogues

(subhalo-info)=
### General

- The original finder IDs of haloes entering the pipeline are stored under `original_ids` in the catalogue `halo_data`.

When run on a halo catalogue which includes subhaloes, Octavius will map out the hierarchical substructure according to the following prescription:

- Field haloes are identified as haloes with no immediate parent.

- Subhaloes are identified as haloes with an immediate parent.

- For inclusive properties, the particles of subhaloes will be propagated into the membership arrays of their parents.

- Galaxy finding operates at the field halo level: in practice this aligns with the results of the subhalo assignments, but the config flag `subhalo_override` can be enabled to strictly respect the subhalo boundaries as laid out by the finder.

Subhaloes can be identified with the `depth` column in `halo_data`: the column is stored ascending, where depth 0 indicates a field halo.

### AHF

- When specifying the halo catalogue path, you should specify the stem of the catalogue (everything before .AHF_halos), e.g. `AHF_C200_m12.5n128.0000.z3.003.`

- Field halo assignments of particles are made according to the parent field halo of the deepest subhalo to which they belong.

### HBT-HERONS

- The pipeline has currently only been tested on SWIFT-KIARA snapshots with associated HBT-HERONS catalogues.

### FOF

- Under MPI, halo finding is spread across every rank: the box is split into slabs along x with roughly equal numbers of DM particles, and each rank only holds its own slab (plus a thin layer of neighbouring particles), so the box size is limited by the total memory of all ranks rather than one. Particle exchanges too large for a single MPI call are automatically split into smaller messages, so any number of ranks works. Each rank uses `cores_per_rank` threads. The resulting HaloIDs do not depend on the number of ranks. Slabs must be at least two FOF cells (about one linking length) thick, so ranks beyond that limit sit idle during halo finding; this only matters for tiny boxes.

- Without MPI (no `mpi4py`), halo finding runs in one process, which must hold every particle's position (plus roughly 65 bytes per DM particle of working memory).

- HaloIDs are ordered by descending DM membership, so halo 0 is the largest. A warning is logged if the largest halo holds more than half of all DM particles, which usually means the linking length is too large or the positions and box size are in different units.

- Under MPI, the log reports how evenly the DM particles are split between ranks, and warns if some ranks are idle or one rank holds more than twice its share (a dense region cannot be split into slabs thinner than two cells).

- With `write_halo_ids`, ranks write their share of the halo IDs to the snapshot one after another; this is a one-off cost of writing 8 bytes per particle. For multi-file SWIFT snapshots the halo IDs are stored as ordinary datasets in the virtual-dataset file, not in the individual files.

### SUBFIND

- The pipeline will assume the particles in the original snapshot have been re-ordered according to the SUBFIND catalogue assignments.

## Simulation-Specific

### SWIFT

- When running on multi-file SWIFT snapshots, you should specify the virtual dataset HDF5 file as the snapshot path.

- Support for COLIBRE and EAGLE simulations is still nascent.

### TNG

- Hydrogen gas fractions for TNG snapshots are computed according to the derivation laid out in [Stevens et al. (2019)](https://doi.org/10.1093/mnras/sty3451) (Appendix A1).

- Default GADGET units are assumed.

- Multi-file snapshots are not currently supported.

### SIMBA

- Default GADGET units are assumed.