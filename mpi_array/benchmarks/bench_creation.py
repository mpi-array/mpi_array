"""
Benchmarks for array creation.
"""

from .utils import try_import_for_setup as _try_import_for_setup


class CreateBench(object):
    """
    Base class for array creation benchmarks.
    """

    #: Number of repetitions to run each benchmark
    repeat = 16

    #: The set of array-shape parameters.
    params = [[(100, 100, 100,), ((1000, 100, 100,)), ((1024 // 4, 1024, 1024,))], ]

    #: The name of the array-shape parameters.
    param_names = ["shape"]

    goal_time = 1.0
    warmup_time = 1.0

    #: Inter-locale cartesian communicator dims
    cart_comm_dims = None

    def __init__(self):
        self.array = None
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
        """
        pass

    def get_globale_shape(self, locale_shape):
        if self.cart_comm_dims is None:
            from ..comms import CartLocaleComms as _CartLocaleComms
            import numpy as _np
            import mpi4py.MPI as _mpi
            comms = _CartLocaleComms(ndims=len(locale_shape), peer_comm=_mpi.COMM_WORLD)
            self.cart_comm_dims = _np.asarray(comms.dims)
        return tuple(self.cart_comm_dims * locale_shape)

    def free(self, a):
        """
        Clean up array resources.
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
        if hasattr(a, "locale_comms"):
            a.locale_comms.free()
        if hasattr(a, "free"):
            a.free()


class CommsCreateBench(CreateBench):
    """
    Benchmarks for :obj:`mpi_array.comms.LocaleComms`
    and :obj:`mpi_array.comms.CartLocaleComms` construction.
    """

    params = None

    def setup(self):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.module = _try_import_for_setup("mpi_array")

    def free(self, a):
        if hasattr(a, "locale_comms"):
            a.locale_comms.free()
        if hasattr(a, "free"):
            a.free()

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
