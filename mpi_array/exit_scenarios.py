"""
==========================================
The :mod:`mpi_array.exit_scenarios` Module
==========================================

Programs which end while arrays are still allocated, normally or because of
an unhandled exception. The test driver :samp:`utils/run_mpi_tests.py` runs each
scenario as its own MPI job and checks the exit status of the job (and that it
doesn't hang), e.g.::

   mpirun -n 4 python -m mpi4py -c "import mpi_array.exit_scenarios as s; s.normal_exit()"
   mpirun -n 4 python -c "import mpi_array.exit_scenarios as s; s.normal_exit()"

The arrays are kept in this module, so that they are allocated at exit (they are freed in
creation order by an :mod:`atexit` function, see :obj:`mpi_array.comms.DeferredFrees`).

Functions
=========

.. autosummary::
   :toctree: generated/

   allocate_arrays - Allocates arrays which stay allocated until exit.
   normal_exit - Arrays are allocated at a normal exit.
   raise_on_all_ranks - Every process raises an unhandled exception.
   raise_on_one_rank - One process raises an unhandled exception.
   raise_on_one_rank_in_with - One process raises an unhandled exception in a :samp:`with` block.
   raise_on_one_rank_with_temporary - One process raises while a temporary array is on the stack.
   normal_exit_with_deferred_frees - Arrays garbage collected on some processes only at exit.

"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi

from .license import license as _license, copyright as _copyright, version as _version
from . import globale_creation as _globale_creation

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

#: The arrays which stay allocated until the interpreter shuts down.
_arrays = []


class ExitScenarioError(RuntimeError):

    """
    The exception raised by the scenarios.
    """
    pass


def allocate_arrays():
    """
    Allocates arrays which stay allocated until exit. The processes have different
    sets of MPI windows: only the inter-locale processes have an inter-locale window,
    and an array is freed after creating its windows.
    """
    shape = (10 * _mpi.COMM_WORLD.size,)
    gary = _globale_creation.zeros(shape)
    gary.initialise_windows()
    _arrays.append(_globale_creation.zeros(shape))
    gary.free()
    _arrays.extend(_globale_creation.zeros(shape) for i in range(3))


def normal_exit():
    """
    The program ends normally while arrays are allocated, the job must exit with status 0.
    """
    allocate_arrays()


def raise_on_all_ranks():
    """
    Every process raises an unhandled exception while arrays are allocated,
    the job must exit with a non-zero status.
    """
    allocate_arrays()
    raise ExitScenarioError("Raised on all processes.")


def raise_on_one_rank():
    """
    Rank 0 raises an unhandled exception while arrays are allocated, and the other
    processes wait in a barrier. With :samp:`python -m mpi4py`, the job must abort.
    """
    allocate_arrays()
    if _mpi.COMM_WORLD.rank == 0:
        raise ExitScenarioError("Raised on rank 0 only.")
    _mpi.COMM_WORLD.barrier()


def raise_on_one_rank_in_with():
    """
    Rank 0 raises an unhandled exception in a :samp:`with` block, and the other
    processes wait in a barrier. With :samp:`python -m mpi4py`, the job must abort.
    """
    with _globale_creation.zeros((10 * _mpi.COMM_WORLD.size,)) as gary:
        gary.initialise_windows()
        if _mpi.COMM_WORLD.rank == 0:
            raise ExitScenarioError("Raised on rank 0 only.")
        _mpi.COMM_WORLD.barrier()


def raise_on_one_rank_with_temporary():
    """
    Rank 0 raises an unhandled exception while a temporary array (the result
    of :samp:`a + 1`) is on the stack, and the other processes wait in a barrier.
    Unwinding the stack garbage collects the temporary on rank 0 only (its free is
    deferred, no MPI calls). With :samp:`python -m mpi4py`, the job must abort.
    """
    def f():
        if _mpi.COMM_WORLD.rank == 0:
            raise ExitScenarioError("Raised on rank 0 only.")
        return 2

    with _globale_creation.zeros((10 * _mpi.COMM_WORLD.size,)) as gary:
        gary.initialise_windows()
        (gary + 1) * f()
        _mpi.COMM_WORLD.barrier()


def normal_exit_with_deferred_frees():
    """
    The program ends normally while arrays (and fetched regions of arrays) are allocated,
    some of them garbage collected on some processes only (their frees are deferred),
    the job must exit with status 0: the windows which are still allocated are freed in
    creation order at exit.
    """
    allocate_arrays()
    world = _mpi.COMM_WORLD
    shape = (10 * world.size,)
    # Locales of two processes, so that the fetched region is shared by two processes.
    intra_locale_comm = world.Split(world.rank // 2, world.rank)
    garbage = \
        [
            _globale_creation.zeros(shape, peer_comm=world, intra_locale_comm=intra_locale_comm)
            for i in range(3)
        ]
    garbage[1].initialise_windows()
    garbage[1].inter_locale_barrier()
    fetched = garbage[1].locale_get(start=(0,), stop=shape)
    garbage[1].inter_locale_barrier()
    garbage[1].intra_locale_barrier()
    if world.rank % 2 == 0:
        del garbage[:], fetched
    else:
        _arrays.extend(garbage)
        _arrays.append(fetched)
    _arrays.append(_globale_creation.ones(shape))


__all__ = [s for s in dir() if not s.startswith('_')]
