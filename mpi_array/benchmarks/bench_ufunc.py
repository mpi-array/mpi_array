"""
Benchmarks for ufuncs.
"""

from .utils import try_import_for_setup as _try_import_for_setup
from .core import Bench


class UfuncBench(Bench):
    """
    Base class for array ufunc benchmarks.
    """

    #: Number of repetitions to run each benchmark
    repeat = 10

    #: The set of array-shape parameters.
    params = [[(100, 100, 100,), ((1000, 100, 100,)), ((1024, 1024, 1024,))], ]

    #: The name of the array-shape parameters.
    param_names = ["shape"]

    # Goal time (seconds) for a single repeat.
    goal_time = 2.0

    # Execute benchmark for this time (seconds) as a *warm-up* prior to real timing.
    warmup_time = 1.0

    def __init__(self):
        """
        Initialise, set :attr:`module` to :samp:`None`.
        """
        self._module = None

    @property
    def module(self):
        """
        The :obj:`module` used to create array instances.
        """
        return self._module

    @module.setter
    def module(self, module):
        self._module = module

    def setup(self, shape):
        """
        Should be over-ridden in sub-classes.
        """
        pass

    def get_globale_shape(self, locale_shape):
        """
        Returns a *globale* array shape for the given shape of the *locale* array.

        :type locale_shape: sequence of :samp:`int`
        :param locale_shape: The shape of the array to be allocated on each *locale*.
        """
        if self.cart_comm_dims is None:
            from ..comms import CartLocaleComms as _CartLocaleComms
            import numpy as _np
            import mpi4py.MPI as _mpi
            comms = _CartLocaleComms(ndims=len(locale_shape), peer_comm=_mpi.COMM_WORLD)
            self.cart_comm_dims = _np.asarray(comms.dims)
        return tuple(self.cart_comm_dims * locale_shape)

    def free_mpi_array_obj(self, a):
        """
        Free MPI communicators and MPI windows for the given object.

        :type a: free-able
        :param a: A :obj:`mpi_array.globale.gndarray` instance
           or a :obj:`mpi_array.comms.LocaleComms` instance.
        """
        if hasattr(a, "locale_comms"):
            a.locale_comms.free()
        if hasattr(a, "free"):
            a.free()

    def free(self, a):
        """
        Clean up array resources, over-ridden in sub-classes.
        """
        pass


class NumpyUfuncBench(UfuncBench):
    """
    Comparison benchmarks for :func:`numpy.ufunc` instances.
    """

    def setup(self, shape):
        """
        Import :mod:`numpy` module and assign to :attr:`module`.
        """
        self.array = None
        self.module = _try_import_for_setup("numpy")
        mpi = _try_import_for_setup("mpi4py.MPI")
        if mpi.COMM_WORLD.size > 1:
            raise NotImplementedError("only runs for single process")


class MpiArrayUfuncBench(UfuncBench):
    """
    Benchmarks for :mod:`mpi_array` ufuncs..
    """

    def setup(self, shape):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.module = _try_import_for_setup("mpi_array")

    def free(self, a):
        """
        See :meth:`free_mpi_array_obj`.
        """
        self.free_mpi_array_obj(a)


__all__ = [s for s in dir() if not s.startswith('_')]
