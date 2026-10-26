# AGENTS Instructions

## Codebase Context

`octavius` is a BSD-licensed, high-performance code for analysing the outputs of cosmological simulations of galaxy formation. It is written in pure Python and is the spiritual successor to [`caesar`](https://github.com/dnarayanan/caesar). `octavius` is designed for the next generation of simulations and has an _ab initio_ focus on extensibility and scalability.

## Interactions on Github

- Never perform issue or contribution-related actions to the main repository (jp-duminy/octavius) on behalf of the user. This includes, but is not limited to, interactions such as pushing code, opening issues or pull requests, and replying to other users.

- The user is required to disclose your agent information. This must include the model and optionally the harness and reasoning level. 

## Coding Standards

- **AI-generated code must be marked:** a function/class written with no human input should include the model number at the end of its docstring as `AI-generated: <model>`.
- **Before editing, inspect the code:** its implementation, documentation, associated callers, and related examples. Do not assume APIs, datasets, or methods exist before checking. Most of the existing codebase structure has arisen from iteration, so understanding the history of abstractions is useful.
- **Performance-intensive code should be written in `numba`.** The `cache=True` flag should be toggled on `njit` functions.
- The code is parallelised through MPI and the multithreading included with `numba`.
- **Separation of responsibilities is strictly enforced**: analysis routines should utilise a top-level Python executor named run_{routine} (see the aggregate_properties folder for examples) which accept the `SimulationData` and `OctaviusConfig` data structures, and declare what they need in `internals.yaml`. No analysis routine should ever directly open a HDF5 file and must always be strictly agnostic to simulation formats. Refer to `docs/developers/contributions.md` for examples.
- **Use the declarative architecture:** `internals.yaml` and the associated `Internals` dataclass exist to make adding new code easy. Package configurables into `OctaviusConfig` and fixed constants into `OctaviusConstants`.
- **Currently the codebase uses minimal dependencies:** adding dependencies should therefore be justified, and kept to optional where possible.
- **Always check and format files with Ruff after writing them**
- **Function signatures should always be type annotated:** so should more complex variables
- **Imports at top of file:** with exceptions to avoid circular imports. Use `TYPE_CHECKING` for type-only imports. `from __future__ import annotations` is not needed as we are on Python 3.14.
- **Tests should be human-readable and simple to follow:** strictly avoid overengineering when adding new tests. Refer to file tests/test_computations.py for examples. Tests are to aid debugging, and should avoid extras like explicit error handling: this is not enterprise software. The tests mainly use `np.testing` for assertions and `pytest` for orchestration. Follow the existing codebase test pattern.
- **Avoid computer-science domain language:** this is a codebase for physics research, so avoid using CS terminology which would be unfamiliar to a physicist.
- **Avoid writing documentation:** I wrote all the documentation by hand and thus would prefer user-facing API docstrings (numpy-style) and documentation preferably be written by a human. If you are writing documentation, include it as a separate commit so a human can read it and rewrite it as they please.
- **CSR:** inclusive properties means the membership CSR arrays duplicate particles. Always access particles from the membership arrays of groups, rather than the other way around.
- **Beware PBCs and h/a factors:** PBCs should be handled as-and-when, h factors should be dealt with in the read-in layer, and a factors should only be used when necessary
- **Refactoring:** avoid overly complex abstractions and overengineering. Try to extend existing code rather than rewriting it.
- **Triple check kernels:** SPH kernels have been a perennial source of errors, so ensure their analytic form and normalisation is consistent.

## Conventions

- **Sentinel values:** unassigned means -1, e.g. a particle with no host galaxy has GalID -1. 
- **Halo naming:** a field halo is a halo with no immediate parent. HaloID refers to a particle's field halo ID; SubhaloID refers to its lowest-level subhalo ID. The in-betweens are matched using the `SubhaloInformation` dataclass. Galaxies have both a parent halo, which can be a subhalo, and a field halo; these are equal if its immediate parent is a field halo. Galaxies __always__ belong to a halo.
- **Keyword arguments:** it is a general convention to pass keyword arguments, as `numba` does not accept the dataclasses directly. This is encouraged but positional arguments are acceptable, especially for smaller functions.
- **Domain decomposition:** MPI parallelisation for the pipeline is over haloes as a unit of work rather than spatial, so a field halo, its subhalos and galaxies are always on the same rank. PBC unwrapping is then done via minimum image.
- **Helper functions:** prefix helper functions with an underscore (_) and check utils/helpers.py for safe division and PBC handling.
- **Variable names:** try to use more descriptive variable names and avoid one or two-letter variables
- **Module docstrings:** for each file, include a module-level docstring describing what it contains and does.
- **Task tags:** add task tags (usually TODO, FIXME, and NOTE) where warranted for users with TODO highlights on.

## Commands

- Test: `pytest` and `mpiexec -n 1 pytest` from repo root
- Docs: `make html` in /docs, CI will check for issues in building documentation
- Pre-commit: `pre-commit run --all-files`
- Dev install: `pip install -e .[dev]`

## Boundaries
- Ask first
    - Large cross-codebase refactors.
    - New dependencies with broad impact.
    - Changing default, established behaviour.
    - Remove or modify an existing task tag
- Never
    - Commit secrets, credentials, or tokens.
    - Use destructive git operations unless explicitly requested.
    - `--no-verify` tag on commits.