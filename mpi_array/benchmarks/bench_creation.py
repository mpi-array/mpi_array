"""
Benchmarks for array creation.
"""
from __future__ import absolute_import
from ..license import license as _license, copyright as _copyright, version as _version

from .utils.misc import try_import_for_setup as _try_import_for_setup
from .core import Bench as _Bench

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class CreateBench(_Bench):
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

    def free(self, a):
        """
        Clean up array resources, over-ridden in sub-classes.
        """
        pass


class NumpyCreateBench(CreateBench):
    """
    Comparison benchmarks for :func:`numpy.empty`, :func:`numpy.zeros` etc.
    """

    def setup(self, shape):
        """
        Import :mod:`numpy` module and assign to :attr:`module`.
        """
        self.array = None
        self.module = self.try_import_for_setup("numpy")
        mpi = self.try_import_for_setup("mpi4py.MPI")
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

    def time_ones(self, shape):
        """
        Time one-initialised array creation.
        """
        self.free(self.module.ones(self.get_globale_shape(shape), dtype="int32"))

    def time_full(self, shape):
        """
        Time value-initialised array creation.
        """
        self.free(
            self.module.full(self.get_globale_shape(shape), fill_value=(2**31) - 1, dtype="int32")
        )


class MpiArrayCreateBench(NumpyCreateBench):
    """
    Benchmarks for :func:`mpi_array.empty`, :func:`mpi_array.zeros` etc.
    """

    def setup(self, shape):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.module = self.try_import_for_setup("mpi_array")

    def free(self, a):
        """
        See :meth:`free_mpi_array_obj`.
        """
        self.free_mpi_array_obj(a)


class BlockPartitionCreateBench(object):
    """
    Benchmarks for :obj:`mpi_array.distribution.BlockPartition`
    construction.
    """

    #: Number of locales.
    params = [64, 128, 256]
    param_names = ["num_locales"]

    goal_time = 0.25
    warmup_time = 0.15

    repeat = 8

    def setup(self, num_locales):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.module = _try_import_for_setup("mpi_array.distribution")
        np = _try_import_for_setup("numpy")
        split = _try_import_for_setup("array_split.split")
        self.globale_extent = self.module.GlobaleExtent(start=(0, 0, 0), stop=(8192, 8192, 8192))
        self.dims = split.shape_factors(num_locales, self.globale_extent.ndim)
        self.cart_coord_to_cart_rank = \
            {tuple(np.unravel_index(rank, self.dims)): rank for rank in range(0, num_locales)}

    def time_block_partition(self, num_locales):
        """
        Time construction of :obj:`mpi_array.distribution.BlockPartition`.
        """
        self.module.BlockPartition(
            globale_extent=self.globale_extent,
            dims=self.dims,
            cart_coord_to_cart_rank=self.cart_coord_to_cart_rank
        )


class MpiArrayCreateLikeBench(CreateBench):
    """
    Benchmarks for :func:`mpi_array.empty_like`, :func:`mpi_array.zeros_like` etc.
    """

    def setup(self, shape):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        self.module = self.try_import_for_setup("mpi_array")
        self._like_array = self.module.empty(self.get_globale_shape(shape), dtype="int32")

    def teardown(self, shape):
        """
        Free the *like* array.
        """
        self.free(self._like_array)

    def free(self, a):
        """
        See :meth:`free_mpi_array_obj`.
        """
        self.free_mpi_array_obj(a)

    def time_empty_like(self, shape):
        """
        Test creation of *uninitialised* array using another array as template.
        """
        self.free(self.module.empty_like(self._like_array))

    def time_ones_like(self, shape):
        """
        Test creation of *one initialised* array using another array as template.
        """
        self.free(self.module.ones_like(self._like_array))


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
        self.module = self.try_import_for_setup("mpi_array")

    def free(self, a):
        """
        See :meth:`free_mpi_array_obj`.
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

    #: The set of array-shape parameters.
    params = [[(4, 1024, 1024,), (64, 1024, 1024,), (1024, 1024, 1024,)], ]

    @property
    def locale_comms(self):
        """
        A :obj:`mpi_array.comms.CartLocaleComms` instance used to allocate memory.
        """
        return self._locale_comms

    def setup(self, shape):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        Also initialise :attr:`locale_comms` with a :obj:`mpi_array.comms.CartLocaleComms`
        instance.
        """
        self.module = self.try_import_for_setup("mpi_array.comms")
        self._locale_comms = self.module.CartLocaleComms(ndims=3)

    def teardown(self, shape):
        """
        Free :attr:`locale_comms` communicators.
        """
        self.free(self.locale_comms)

    def free(self, a):
        """
        See :meth:`free_mpi_array_obj`.
        """
        self.free_mpi_array_obj(a)

    def time_alloc_locale_buffer(self, shape):
        """
        Time call of :meth:`mpi_array.comms.LocaleComms.alloc_locale_buffer`.
        """
        self.free(self.locale_comms.alloc_locale_buffer(shape=shape, dtype="int32"))


class MangoCreateBench(NumpyCreateBench):
    """
    Benchmarks for :func:`mango.empty` and :func:`mango.zeros`
    (`mango tomography software <https://physics.anu.edu.au/appmaths/capabilities/mango.php>`_).
    """

    def setup(self, shape):
        """
        Import :mod:`mango` module and assign to :samp:`self.module`.
        """
        self.module = self.try_import_for_setup("mango")

    def time_full(self, shape):
        """
        No :samp:`full` function in :samp:`mango`.
        """
        pass

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
#        self._ga = self.try_import_for_setup("ga4py.ga")
#        self.module = self.try_import_for_setup("ga4py.gain")
#        # Turn off the global-arrays caching of arrays.
#        self.try_import_for_setup("ga4py.gain.core").ga_cache.level = 0
#
#    def free(self, a):
#        self._ga.sync()
#        self._ga.destroy(a.handle)
#        a._base = 0


__all__ = [s for s in dir() if not s.startswith('_')]
