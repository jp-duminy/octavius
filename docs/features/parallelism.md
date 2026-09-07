# Parallelism

The Octavius analysis pipeline is fully hybrid-parallelised by default, controlled through both [message passing interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface) and [built-in threaded parallelism provided by numba](https://numba.readthedocs.io/en/stable/user/threading-layer.html).

## MPI

The top-level parallelism using MPI via `mpi4py` is done by considering haloes as a unit of work. As Octavius consumes an external halo finder catalogue, the workload can be [embarrassingly](https://en.wikipedia.org/wiki/Embarrassingly_parallel) parallelised by dividing haloes amongst MPI ranks. 

To divide haloes equally amongst ranks, it is not sufficient to simply bin them to ranks by particle count. The stages of the pipeline are sensitive to different parameters: the FOF6D algorithm, for example, only uses baryonic particles, and is sensitive to the number of stars within each halo. Aggregate properties scales with the total number of particles, including dark matter; photometry only requires stars and gas. To this end, we need a method of quantifying computational weight. 

An empirically-adjusted relation is therefore used to bin haloes across ranks. The weights are tuned as follows:

$W_{FOF6D} = N_{stars}^{1.2} + N_{gas}$

$W_{AP} = N_{stars} + N_{gas} + N_{dm}$

$W_{P} = (N_{stars} + N_{gas})^{1.1}$. 

These computational weights are a proxy for processing time. A [simple binning algorithm](https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling) is then used to bin haloes to ranks. It initialises all ranks at zero weights; then, it iterates through the haloes list by weight descending, sequentially assigning haloes to the rank with the lowest weight at each timestep. The end result produces ranks weighted equally according to the empirical law.

This algorithm is shown to produce well-balanced runtimes across ranks. 

:::{warning}
Computing photometry for enormous galaxies in large snapshots can cause weight imbalancing; this is an artefact of the computational cost of photometry, rather than the parallelism itself.
:::

## Threaded Layers

Intra-rank parallelism is provided by `numba`. This is controlled by the `cores_per_rank` parameter in the configuration file. In principle, the pipeline can be executed without MPI by settings `cores_per_rank` equal to -1. This allows MPI to be used for cross-node communication, parallel IO, redistribution of data. Within a rank, the pipeline itself is embarrassingly parallel, allowing Octavius to take advantage of the simple, yet highly optimised `prange` and `parallel=True` instructions. This keeps MPI communication out of the pipeline beyond IO read-in and read-out.

## File IO

The strategy for reading datasets off disc in parallel with is as follows:

- Rank 0 knows which haloes belong to each rank from the binning algorithm
- Rank 0 broadcasts the allocation to the ranks
- Each dataset is divided into $r$ contiguous slabs, where $r$ is the number of ranks
- Each rank reads its slab of the dataset in chunks
- Now ranks own a slab of the dataset and, from rank 0, know where the particles belong
- Once the dataset is read in, a `comm.Alltoallv` call is used to redistribute the data from raw slabs to their destination ranks based on the halo assignments

This is somewhat abstract owing to the halo assignments, but it allows the ranks to cooperate in loading data in a manner which bounds the peak memory to the size of the dataset.

In line with the minimal-dependencies ethos, Octavius does not directly use parallel HDF5 for the writes. The strategy is as follows:

- Group-level data is small relative to particle-level data, so this is broadcast to rank 0
- The tricky exception here are the particle indices, which are large and so cannot be broadcast to rank 0
- Rank 0 uses the global group-level data to compute the global sort order
- Rank 0 writes group-level data into the catalogue according to the global order (mass descending with tiebreaks using `np.lexsort`)
- From the group-level membership arrays (specifically the offsets), rank 0 determines what slice of the HDF5 dataset each rank should write its indices to such that the indices are written in globally-sorted order
- Rank 0 broadcasts to the other ranks where they must write their indices arrays
- All ranks take turns writing their particle indices to the output catalogue

As the ranks do sequential writes, the indices array is not compressed at write time. After the ranks have finished, h5repack is (optionally) used to compress the indices array.