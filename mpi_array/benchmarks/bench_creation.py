"""
Benchmarks for array creation.
"""

from .utils import try_import_for_setup as _try_import_for_setup


class CreateBench(object):
    """
    Base class for array creation benchmarks.
    """

    #: Number of repetitions to run each benchmark
    repeat = 10

    #: The set of array-shape parameters.
    params = [[(100, 100, 100,), ((1000, 100, 100,)), ((1024, 1024, 1024,))], ]

    #: The name of the array-shape parameters.
    param_names = ["shape"]

    # Goal time (seconds) for a single repeat.
    goal_time = 5.0

    # Execute benchmark for this time (seconds) as a *warm-up* prior to real timing.
    warmup_time = 2.0

    #: Inter-locale cartesian communicator dims
    cart_comm_dims = None

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


class NumpyCreateBench(CreateBench):
    """
    Comparison benchmarks for :func:`numpy.empty` and :func:`numpy.zeros`.
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

    def time_empty(self, shape):
        """
        Time uninitialised array creation.
        """
        self.free(self.module.empty(self.get_globale_shape(shape), dtype="int32"))

    def time_zeros(self, shape):
        """
        Time zero-initialised array creation.
        """
        self.free(self.module.zeros(self.get_globale_shape(shape), dtype="int32"))


class MpiArrayCreateBench(NumpyCreateBench):
    """
    Benchmarks for :func:`mpi_array.empty` and :func:`mpi_array.zeros`.
    """

    def setup(self, shape):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.module = _try_import_for_setup("mpi_array")

    def free(self, a):
        """
        .. seealso:: :meth:`free_mpi_array_obj`
        """
        self.free_mpi_array_obj(a)


class CommsCreateBench(CreateBench):
    """
    Benchmarks for :obj:`mpi_array.comms.LocaleComms`
    and :obj:`mpi_array.comms.CartLocaleComms` construction.
    """

    #: No param, is :samp:`None`.
    params = None

    def setup(self):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.module = _try_import_for_setup("mpi_array")

    def free(self, a):
        """
        .. seealso:: :meth:`free_mpi_array_obj`
        """
        self.free_mpi_array_obj(a)

    def time_locale_comms(self):
        """
        Time construction of :obj:`mpi_array.comms.LocaleComms`.
        """
        self.free(self.module.comms.LocaleComms())

    def time_cart_locale_comms(self):
        """
        Time construction of :obj:`mpi_array.comms.CartLocaleComms`.
        """
        self.free(self.module.comms.CartLocaleComms(ndims=3))


class CommsAllocBench(CreateBench):
    """
    Benchmarks for :meth:`mpi_array.comms.LocaleComms.alloc_locale_buffer`.
    """

    #: No param, is :samp:`None`.
    params = None

    @property
    def locale_comms(self):
        """
        A :obj:`mpi_array.comms.CartLocaleComms` instance used to allocate memory.
        """
        return self._locale_comms

    def setup(self):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        Also initialise :attr:`locale_comms` with a :obj:`mpi_array.comms.CartLocaleComms`
        instance.
        """
        self.module = _try_import_for_setup("mpi_array.comms")
        self._locale_comms = self.module.CartLocaleComms(ndims=3)

    def teardown(self):
        """
        Free :attr:`locale_comms` communicators.
        """
        self.free(self.locale_comms)

    def free(self, a):
        """
        .. seealso:: :meth:`free_mpi_array_obj`
        """
        self.free_mpi_array_obj(a)

    def time_alloc_locale_buffer(self):
        """
        Time call of :meth:`mpi_array.comms.LocaleComms.alloc_locale_buffer`.
        """
        self.free(self.locale_comms.alloc_locale_buffer(shape=(128, 1024, 1024), dtype="int32"))


class MangoCreateBench(NumpyCreateBench):
    """
    Benchmarks for :func:`mango.empty` and :func:`mango.zeros`
    (`mango tomography software <https://physics.anu.edu.au/appmaths/capabilities/mango.php>`_).
    """

    def setup(self, shape):
        """
        Import :mod:`mango` module and assign to :samp:`self.module`.
        """
        self.module = _try_import_for_setup("mango")


# class Ga4pyGainCreateBench(NumpyCreateBench):
#    """
#    Benchmarks for `ga4py.gain <https://github.com/GlobalArrays/ga4py>`_ :func:`ga4py.gain.empty`
#    and :func:`ga4py.gain.zeros`.
#    """
#
#    def setup(self, shape):
#        """
#        Import :mod:`mango` module and assign to :samp:`self.module`.
#        """
#        self._ga = _try_import_for_setup("ga4py.ga")
#        self.module = _try_import_for_setup("ga4py.gain")
#        # Turn off the global-arrays caching of arrays.
#        _try_import_for_setup("ga4py.gain.core").ga_cache.level = 0
#
#    def free(self, a):
#        self._ga.sync()
#        self._ga.destroy(a.handle)
#        a._base = 0

__all__ = [s for s in dir() if not s.startswith('_')]
