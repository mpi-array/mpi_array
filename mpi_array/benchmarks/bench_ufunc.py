"""
Benchmarks for ufuncs.
"""

from .utils import try_import_for_setup as _try_import_for_setup
from .core import Bench


class UfuncBench(Bench):
    """
    Base class for array ufunc benchmarks.
    """

    def __init__(self):
        """
        Initialise.
        """
        self._a_ary = None
        self._b_ary = None
        self._c_ary = None
        self._b_scalar = None

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

    @property
    def a_ary(self):
        """
        """
        return self._a_ary

    @a_ary.setter
    def a_ary(self, value):
        """
        """
        self._a_ary = value

    @property
    def b_ary(self):
        """
        """
        return self._b_ary

    @b_ary.setter
    def b_ary(self, value):
        """
        """
        self._b_ary = value

    @property
    def c_ary(self):
        """
        """
        return self._c_ary

    @c_ary.setter
    def c_ary(self, value):
        """
        """
        self._c_ary = value

    @property
    def b_scalar(self):
        """
        """
        return self._b_scalar

    @b_scalar.setter
    def b_scalar(self, value):
        """
        """
        self._b_scalar = value

    def initialise_arrays(self, shape):
        """
        Initialise arrays/scalars passed to ufunc instances.
        """
        self.a_ary = self.module.empty(self.get_globale_shape(shape), dtype=self.dtype)
        self.b_ary = self.module.empty(self.get_globale_shape(shape), dtype=self.dtype)
        self.c_ary = self.module.empty(self.get_globale_shape(shape), dtype=self.dtype)
        self.b_scalar = self.dtype.type()

    def setup(self, shape):
        """
        Should be over-ridden in sub-classes.
        """
        self.module = None

    def teardown(self, shape):
        """
        Free arrays allocated during :meth:`setup`.
        """
        self.free(self.a_ary)
        self.free(self.b_ary)
        self.free(self.c_ary)

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
