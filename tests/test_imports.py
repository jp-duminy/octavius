"""

Tests external and internal imports.

"""


def test_external_dependencies() -> None:
    """
    Checks whether all external dependencies are importable.
    """
    # the big five
    import numpy as np
    import numba
    import h5py
    import astropy.units as u
    from mpi4py import MPI

    # random operations so ruff does not delete the imports
    assert np.array([1, 2, 3]).sum() == 6
    assert numba.njit(lambda x: x + 1)(1) == 2
    assert u.kpc.to(u.cm) > 0
    assert h5py.string_dtype() is not None
    assert MPI.COMM_WORLD.Get_size() >= 1


def test_package_imports() -> None:
    """
    Tests whether internal imports work.
    """
    import octavius

    assert hasattr(octavius, "load_catalogue")  # this will fail if the load function gets renamed


def test_quiet_flag_parsed():
    """
    Tests the command line arguments.
    """
    from octavius.run_octavius import parse_args

    args = parse_args(["analyse", "--config", "dummy.yaml", "--quiet"])
    assert args.quiet is True
