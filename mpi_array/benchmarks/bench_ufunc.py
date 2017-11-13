"""
Benchmarks for ufuncs.
"""

import sys as _sys
from .utils import try_import_for_setup as _try_import_for_setup
from .core import Bench


class UfuncBench(Bench):
    """
    Base class for array ufunc benchmarks.
    Run benchmarks for calls of the form::

       numpy.exp(self.a_ary, out=self.c_ary)
       numpy.add(self.a_ary, self.b_ary, out=self.c_ary)

    where the :attr:`a_ary`, :attr:`b_ary` and :attr:`c_ary` arrays
    are initialised (with uniform random scalars) during :meth:`setup`.
    """

    def __init__(self, ufunc_name=None):
        """
        Initialise.

        :type ufunc_name: :obj:`str`
        :param ufunc_name: Name of the ufunc, should be the name of an attribute of :attr:`module`.
        """
        Bench.__init__(self)
        self._ufunc_name = ufunc_name
        self._ufunc = None
        self._a_ary = None
        self._b_ary = None
        self._c_ary = None
        self._b_scalar = None

    #: Number of repetitions to run each benchmark
    repeat = 8

    #: The set of array-shape parameters.
    params = [[((1024**2) // 8,), ((1000, 100, 100,)), ((1024 // 8, 1024, 1024,))], ]

    #: The name of the array-shape parameters.
    param_names = ["shape"]

    # Goal time (seconds) for a single repeat.
    goal_time = 1.0

    # Execute benchmark for this time (seconds) as a *warm-up* prior to real timing.
    warmup_time = 1.0

    @property
    def ufunc(self):
        """
        The :samp:`numpy.ufunc` for this benchmark.
        """
        return self._ufunc

    @ufunc.setter
    def ufunc(self, value):
        self._ufunc = value

    @property
    def a_ary(self):
        """
        First input array.
        """
        return self._a_ary

    @a_ary.setter
    def a_ary(self, value):
        self._a_ary = value

    @property
    def a_ary_range(self):
        """
        A :samp:`(low, high)` tuple indicating the uniform random range of scalars for
        the :attr:`a_ary` array elements.
        """
        return (0.66, 1.66)

    @property
    def b_ary(self):
        """
        Second input array.
        """
        return self._b_ary

    @b_ary.setter
    def b_ary(self, value):
        self._b_ary = value

    @property
    def b_ary_range(self):
        """
        A :samp:`(low, high)` tuple inficating the uniform random range of scalars for
        the :attr:`b_ary` array elements.
        """
        return (0.7, 1.3)

    @property
    def c_ary(self):
        """
        Output array.
        """
        return self._c_ary

    @c_ary.setter
    def c_ary(self, value):
        self._c_ary = value

    @property
    def b_scalar(self):
        """
        Second scalar input.
        """
        return self._b_scalar

    @b_scalar.setter
    def b_scalar(self, value):
        self._b_scalar = value

    @property
    def random_state(self):
        """
        A :obj:`numpy.random.RandomState` instance, used to generate random
        values in arrays.
        """
        _np = self.try_import_for_setup("numpy")
        _mpi = self.try_import_for_setup("mpi4py.MPI")
        seed_str = str(2 ** 31)[1:]
        rank_str = str(_mpi.COMM_WORLD.rank + 1)
        seed_str = rank_str + seed_str[len(rank_str):]
        seed_str = seed_str[0:-len(rank_str)] + rank_str[::-1]
        random_state = _np.random.RandomState(seed=int(seed_str))

        return random_state

    def initialise_arrays(self, shape):
        """
        Initialise arrays/scalars passed to ufunc instances.

        :type shape: sequence of :obj:`int`
        :param shape: Shape of the arrays to be passed to ufuncs.
        """
        shape = self.get_globale_shape(shape)
        random_state = self.random_state
        self.a_ary = \
            random_state.uniform(
                low=self.a_ary_range[0],
                high=self.a_ary_range[1],
                size=shape
            ).astype(self.dtype)
        self.b_ary = \
            random_state.uniform(
                low=self.b_ary_range[0],
                high=self.b_ary_range[1],
                size=shape
            ).astype(self.dtype)
        self.c_ary = self.module.empty(shape, dtype=self.dtype)
        self.b_scalar = \
            self.dtype.type(random_state.uniform(low=self.b_ary_range[0], high=self.b_ary_range[1]))

    def initialise(self, shape, dtype, module_name):
        """
        Sets the :attr:`module`, :attr:`dtype` and :attr:`ufunc` attributes
        and initialises the :attr:`a_ary`, :attr:`b_ary`, :attr:`b_scalar` and :attr:`c_ary`
        arrays.
        """
        self.module = _try_import_for_setup(module_name)
        self.dtype = self.module.dtype(dtype)
        self.ufunc = getattr(self.module, self._ufunc_name)
        self.initialise_arrays(shape)

    def setup(self, shape, dtype="float64"):
        """
        Should be over-ridden in sub-classes.
        """
        pass

    def teardown(self, shape):
        """
        Free arrays allocated during :meth:`setup`.
        """
        self.free(self.a_ary)
        self.a_ary = None
        self.free(self.b_ary)
        self.b_ary = None
        self.free(self.c_ary)
        self.c_ary = None
        self.b_scalar = None

    def free(self, a):
        """
        Clean up array resources, over-ridden in sub-classes.
        """
        pass


class NumpyUfuncBench(UfuncBench):
    """
    Comparison benchmarks for :obj:`numpy.ufunc` instances.
    """

    def initialise_arrays(self, shape):
        """
        Initialise arrays/scalars passed to ufunc instances.

        :type shape: sequence of :obj:`int`
        :param shape: Shape of the arrays to be passed to ufuncs.
        """
        shape = self.get_globale_shape(shape)
        random_state = self.random_state
        self.a_ary = \
            random_state.uniform(
                low=self.a_ary_range[0],
                high=self.a_ary_range[1],
                size=shape
            ).astype(self.dtype)
        self.b_ary = \
            random_state.uniform(
                low=self.b_ary_range[0],
                high=self.b_ary_range[1],
                size=shape
            ).astype(self.dtype)
        self.c_ary = self.module.empty(shape, dtype=self.dtype)
        self.b_scalar = \
            self.dtype.type(random_state.uniform(low=self.b_ary_range[0], high=self.b_ary_range[1]))

    def setup(self, shape, dtype="float64"):
        """
        Import :mod:`numpy` module and assign to :attr:`module`.
        """

        mpi = _try_import_for_setup("mpi4py.MPI")
        if mpi.COMM_WORLD.size > 1:
            raise NotImplementedError("only runs for single process")

        self.initialise(shape=shape, dtype=dtype, module_name="numpy")


class MpiArrayUfuncBench(UfuncBench):
    """
    Benchmarks for :mod:`mpi_array` ufuncs..
    """

    def initialise_arrays(self, shape):
        """
        Initialise arrays/scalars passed to ufunc instances.

        :type shape: sequence of :obj:`int`
        :param shape: Shape of the arrays to be passed to ufuncs.
        """
        shape = self.get_globale_shape(shape)
        random_state = self.random_state
        self.a_ary = self.module.empty(shape, dtype=self.dtype)
        self.b_ary = self.module.empty(shape, dtype=self.dtype)
        self.c_ary = self.module.empty(shape, dtype=self.dtype)

        self.a_ary.rank_view_n[...] = \
            random_state.uniform(
                low=self.a_ary_range[0], high=self.a_ary_range[1], size=self.a_ary.rank_view_n.shape
        ).astype(self.dtype)
        self.b_ary.rank_view_n[...] = \
            random_state.uniform(
                low=self.b_ary_range[0], high=self.b_ary_range[1], size=self.b_ary.rank_view_n.shape
        ).astype(self.dtype)

        self.b_scalar = \
            self.dtype.type(random_state.uniform(low=self.b_ary_range[0], high=self.b_ary_range[1]))
        self.a_ary.locale_comms.intra_locale_comm.barrier()

    def setup(self, shape, dtype="float64"):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.initialise(shape=shape, dtype=dtype, module_name="mpi_array")

    def free(self, a):
        """
        See :meth:`free_mpi_array_obj`.
        """
        self.free_mpi_array_obj(a)


class NumpyUfuncBench_invsqrt(NumpyUfuncBench):
    """
    Benchmarks for :samp:`numpy.power(ary, -0.5)` ufunc for :obj:`numpy.ndarray` input.
    """

    def __init__(self):
        MpiArrayUfuncBench.__init__(self, ufunc_name="power")

    def time_array_op(self, shape, dtype="float64"):
        """
        Timing for single argument ufuncs.
        """
        self._ufunc(self._a_ary, -0.5, out=self._c_ary)


class MpiArrayUfuncBench_invsqrt(MpiArrayUfuncBench):
    """
    Benchmarks for :samp:`mpi_array.power(ary, -0.5)` ufunc for :obj:`mpi_array.globale.gndarray`
    input.
    """

    def __init__(self):
        MpiArrayUfuncBench.__init__(self, ufunc_name="power")

    def time_array_op(self, shape, dtype="float64"):
        """
        Timing for single argument ufuncs.
        """
        self._ufunc(self._a_ary, -0.5, out=self._c_ary)


_module_name_to_sphinx_doc_array_type = {
    "mpi_array": "mpi_array.globale.gndarray",
    "numpy": "numpy.ndarray"
}
_module_name_to_bench_type = {"mpi_array": MpiArrayUfuncBench, "numpy": NumpyUfuncBench}


def create_ufunc_bench(module_name, ufunc_name, method_dict):
    """
    Creates a new benchmark type for the ufunc :samp:`getattr({module_name}, {ufunc_name})`.
    """

    try:
        bench_type = _module_name_to_bench_type[module_name]
        ufunc_bench_type = None

        def __init__(self):
            bench_type.__init__(self, ufunc_name=ufunc_name)

        modyule = _try_import_for_setup(module_name)
        ufunc = getattr(modyule, ufunc_name)
        ufunc_bench_method_dict = {"__init__": __init__}
        if ufunc.nin == 1:
            ufunc_bench_method_dict["time_array_op"] = \
                method_dict["time_array_op"]
        if ufunc.nin == 2:
            ufunc_bench_method_dict["time_array_array_op"] = \
                method_dict["time_array_array_op"]
            ufunc_bench_method_dict["time_array_scalar_op"] = \
                method_dict["time_array_scalar_op"]

        ufunc_bench_type = \
            type(
                bench_type.__name__ + "_" + ufunc_name,
                (bench_type,),
                ufunc_bench_method_dict
            )
        setattr(
            ufunc_bench_type,
            "__doc__",
            ("Benchmark for :obj:`numpy." + ufunc_name + "` with :obj:`%s` array inputs.")
            %
            (_module_name_to_sphinx_doc_array_type[module_name],)
        )
    except (AttributeError, ImportError, NotImplementedError):
        raise
    return ufunc_bench_type


def create_ufunc_benchmarks(ufunc_names, module_names, module=None):
    """
    Creates a new benchmark type for each ufunc name in :samp:`ufunc_names`
    and for each module name in :samp:`{module_names}`. Sets created types
    as attributes of the module :samp:`{module}`.
    """

    def time_array_op(self, shape, dtype="float64"):
        """
        Timing for single input ufuncs (single array input).
        """
        self._ufunc(self._a_ary, out=self._c_ary)

    def time_array_array_op(self, shape, dtype="float64"):
        """
        Timing for two input ufuncs (two array inputs).
        """
        self._ufunc(self._a_ary, self._b_ary, out=self._c_ary)

    def time_array_scalar_op(self, shape, dtype="float64"):
        """
        Timing for two input  ufuncs (one array input, one scalar input).
        """
        self._ufunc(self._a_ary, self._b_scalar, out=self._c_ary)

    method_dict = \
        {
            "time_array_op": time_array_op,
            "time_array_array_op": time_array_array_op,
            "time_array_scalar_op": time_array_scalar_op
        }

    for module_name in module_names:
        for ufunc_name in ufunc_names:
            bench_type = create_ufunc_bench(module_name, ufunc_name, method_dict)
            if (module is not None) and (bench_type is not None):
                setattr(module, bench_type.__name__, bench_type)


def find_scipy_ufuncs():
    """
    Imports the :func:`scipy.special.erf` ufunc and sets it as an attribute
    of the :mod:`numpy` and :mod:`mpi_array` modules.
    """
    try:
        import scipy.special as special
        import numpy as np
        import mpi_array as mpia
        for m in (np, mpia):
            setattr(m, "erf", special.erf)
    except Exception:
        pass


find_scipy_ufuncs()
UFUNC_NAMES = ["add", "subtract", "multiply", "true_divide", "log", "log10", "erf"]

create_ufunc_benchmarks(
    module_names=("mpi_array", "numpy"),
    ufunc_names=UFUNC_NAMES,
    module=_sys.modules[__name__]
)

__all__ = [s for s in dir() if not s.startswith('_')]
