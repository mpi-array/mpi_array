"""
======================================
The :mod:`mpi_array.comms_test` Module
======================================

Module defining :mod:`mpi_array.comms` unit-tests.
Execute as::

   python -m mpi_array.comms_test

or::

   mpirun -n 4 python -m mpi_array.comms_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   LocaleCommsTest - Tests for :obj:`mpi_array.comms.LocaleComms`.
   CartLocaleCommsTest - Tests for :obj:`mpi_array.comms.CartLocaleComms`.
   CreateDistributionTest - Tests for :func:`mpi_array.comms.create_distribution`.

"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401

from .comms import CartLocaleComms, LocaleComms, CommsCache
from .comms import create_single_locale_distribution, create_locale_comms, create_distribution
from .comms import check_distrib_type, DT_BLOCK, DT_SLAB, DT_CLONED, DT_SINGLE_LOCALE
from .comms import check_locale_type, LT_NODE, LT_PROCESS
from .comms import reshape_comms_distribution, shape_squeeze
from .comms import agree_on_error, get_cannonical_dims
from .utils import get_shared_mem_usage_percent_string, LruCache
from .distribution import SingleLocaleDistribution as _SingleLocaleDistribution

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class LocaleCommsTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.comms.LocaleComms`.
    """

    def test_get_shared_mem_usage_percent_string(self):
        """
        Coverage for :func:`mpi_array.comms.get_shared_mem_usage_percent_string`.
        """
        p = \
            get_shared_mem_usage_percent_string(
                shm_file_name="/probably/does/not_exist/on_file/system"
            )
        self.assertEqual("unknown", p)

    def test_construct(self):
        """
        Test :meth:`mpi_array.comms.LocaleComms.__init__`
        """
        i = LocaleComms(peer_comm=_mpi.COMM_WORLD)

        self.assertTrue(i.intra_locale_comm is not None)
        self.assertTrue(i.intra_locale_comm.size >= 1)
        self.assertTrue(i.peer_comm is not None)
        self.assertTrue(i.peer_comm.size >= 1)
        self.assertEqual(i.num_locales, len(i.peer_ranks_per_locale))
        self.assertEqual(
            i.peer_comm.size,
            sum(len(i.peer_ranks_per_locale[r]) for r in range(i.num_locales))
        )
        prpl_ranks = \
            sorted(sum((tuple(i.peer_ranks_per_locale[r]) for r in range(i.num_locales)), ()))
        self.assertSequenceEqual(
            list(range(0, i.peer_comm.size)),
            prpl_ranks
        )

        i = LocaleComms()

        self.assertTrue(i.intra_locale_comm is not None)
        self.assertTrue(i.intra_locale_comm.size >= 1)
        self.assertTrue(i.peer_comm is not None)
        self.assertTrue(i.peer_comm.size >= 1)
        i.inter_locale_comm = _mpi.COMM_NULL
        self.assertEqual(_mpi.COMM_NULL, i.inter_locale_comm)
        i.inter_locale_comm = None
        self.assertEqual(None, i.inter_locale_comm)

    def test_construct_invalid_comms(self):
        """
        Test :meth:`mpi_array.comms.LocaleComms.__init__`
        """
        if _mpi.COMM_WORLD.size != _mpi.COMM_SELF.size:
            self.assertRaises(
                ValueError,
                LocaleComms,
                _mpi.COMM_SELF,  # peer
                _mpi.COMM_SELF,  # intra
                _mpi.COMM_WORLD  # inter
            )
        lc = LocaleComms()
        if lc.intra_locale_comm.size > 1:
            self.assertRaises(
                ValueError,
                LocaleComms,
                lc.peer_comm,  # peer
                lc.intra_locale_comm,  # intra
                lc.peer_comm  # inter
            )

    def test_construct_invalid_comms_message(self):
        """
        The :obj:`ValueError` for inter-locale processes which are not rank 0 of their
        intra-locale communicator has a heading line, followed by one line for each
        of them. The message was built with an implicit string concatenation, which
        joined the lines with the heading.
        """
        lc = LocaleComms()
        if lc.intra_locale_comm.size > 1:
            with self.assertRaises(ValueError) as context:
                LocaleComms(lc.peer_comm, lc.intra_locale_comm, lc.peer_comm)
            lines = str(context.exception).split("\n")
            self.assertEqual(
                "Got invalid inter_locale_comm rank to intra_locale_comm rank translation:",
                lines[0]
            )
            self.assertEqual(lc.peer_comm.size - lc.num_locales, len(lines) - 1)
            for line in lines[1:]:
                self.assertRegex(
                    line,
                    r"^peer rank=\d+, inter_locale_comm rank=\d+ translated to "
                    + r"intra_locale_comm rank=[1-9]\d*, should be intra_locale_comm rank=0$"
                )

    def test_construct_no_shared(self):
        lc = LocaleComms(intra_locale_comm=_mpi.COMM_SELF)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))
        self.assertEqual(1, lc.intra_locale_comm.size)
        self.assertNotEqual(_mpi.COMM_WORLD, _mpi.COMM_NULL)

    def test_comms_cache(self):
        """
        Tests for :obj:`mpi_array.comms.CommsCache`: communicators and :samp:`None` match
        by identity, other key elements by value. A hit is a dictionary look-up (the keys of
        all entries used to be compared), and the debug message isn't made when debug
        messages aren't logged.
        """
        cache = CommsCache()
        comm = _mpi.COMM_WORLD
        logger = LocaleComms().rank_logger

        class Value(object):
            rank_logger = logger

        v0 = Value()
        cache.add((comm, None, _np.array([2, 0])), v0)
        self.assertTrue(cache.find((comm, None, [2, 0])) is v0)
        self.assertTrue(cache.find((comm, None, (2, 0))) is v0)
        self.assertTrue(cache.find((comm, comm, [2, 0])) is None)
        self.assertTrue(cache.find((_mpi.COMM_SELF, None, [2, 0])) is None)
        self.assertTrue(cache.find((comm, None, [2, 1])) is None)
        self.assertTrue(cache.find((comm, None)) is None)
        self.assertRaises(ValueError, cache.add, (comm, None, [2, 0]), Value())
        self.assertEqual(1, len(cache.lookup))

        create_key_equality_tuple = cache.create_key_equality_tuple

        def raise_error(*args):
            raise AssertionError("Made the debug message.")

        level = logger.level
        logger.setLevel(_logging.WARNING)
        cache.create_key_equality_tuple = raise_error
        try:
            self.assertTrue(cache.find((comm, None, [2, 0])) is v0)
        finally:
            logger.setLevel(level)
            cache.create_key_equality_tuple = create_key_equality_tuple
        cache.clear()
        self.assertTrue(cache.find((comm, None, [2, 0])) is None)
        self.assertEqual(0, len(cache.lookup))

    def test_lru_cache(self):
        """
        Tests for :obj:`mpi_array.utils.LruCache`.
        """
        cache = LruCache(max_size=2)
        cache["a"] = 1
        cache["b"] = 2
        self.assertEqual(1, cache.get("a"))
        cache["c"] = 3
        self.assertEqual(2, len(cache))
        self.assertTrue("b" not in cache)
        self.assertEqual(None, cache.get("b"))
        self.assertEqual(4, cache.get("b", 4))
        self.assertEqual(1, cache.get("a"))
        self.assertEqual(3, cache.get("c"))
        cache.clear()
        self.assertEqual(0, len(cache))

    def test_agree_on_error(self):
        """
        Tests for :func:`mpi_array.comms.agree_on_error`, the processes without
        an error get one like the error of the lowest rank process which has one (also
        when its message can't be made, which used to raise on that process only).
        """
        world = _mpi.COMM_WORLD

        class UnprintableError(ValueError):

            def __str__(self):
                raise RuntimeError("Raised for the test.")

        self.assertTrue(agree_on_error(world, None) is None)
        for error_type in (ValueError, UnprintableError):
            error = None
            if world.rank == (world.size - 1):
                error = error_type("Raised on the last process.")
            agreed_error = agree_on_error(world, error)
            if error is not None:
                is_ok = agreed_error is error
            else:
                is_ok = isinstance(agreed_error, ValueError)
            self.assertTrue(world.allreduce(is_ok, op=_mpi.LAND))

        error = None
        if (world.rank > 0) or (world.size == 1):
            error = IndexError("Raised on process %s." % (world.rank,))
        agreed_error = agree_on_error(world, error)
        if error is not None:
            is_ok = agreed_error is error
        else:
            is_ok = \
                isinstance(agreed_error, IndexError) \
                and \
                (str(agreed_error) == "Raised on process 1.")
        self.assertTrue(world.allreduce(is_ok, op=_mpi.LAND))


class CartLocaleCommsTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.comms.CartLocaleComms`.
    """

    def test_construct_invalid_dims(self):
        lc = None
        with self.assertRaises(ValueError):
            lc = CartLocaleComms()
        with self.assertRaises(ValueError):
            lc = CartLocaleComms(ndims=None, dims=None)
        with self.assertRaises(ValueError):
            lc = CartLocaleComms(dims=tuple(), ndims=1)
        with self.assertRaises(ValueError):
            lc = CartLocaleComms(dims=tuple([0, 2]), ndims=1)
        with self.assertRaises(ValueError):
            lc = CartLocaleComms(dims=tuple([1, 2]), ndims=3)

        self.assertEqual(None, lc)

    def test_construct_invalid_cart_comm(self):
        cart_comm = _mpi.COMM_WORLD.Create_cart(dims=(_mpi.COMM_WORLD.size,))

        if _mpi.COMM_WORLD.size > 1:
            self.assertRaises(
                ValueError,
                CartLocaleComms,
                ndims=1,
                peer_comm=_mpi.COMM_WORLD,
                cart_comm=cart_comm
            )

    def test_construct_shared(self):
        lc = CartLocaleComms(ndims=1)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))
        self.assertEqual(1, lc.ndim)

        lc = CartLocaleComms(ndims=4)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))
        self.assertEqual(4, lc.ndim)

        lc = CartLocaleComms(dims=(0,))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))
        self.assertEqual(1, lc.ndim)

        lc = CartLocaleComms(dims=(0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))
        self.assertEqual(2, lc.ndim)

        lc = CartLocaleComms(dims=(0, 0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))
        self.assertEqual(3, lc.ndim)

    def test_construct_no_shared(self):
        lc = CartLocaleComms(ndims=1, intra_locale_comm=_mpi.COMM_SELF)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))
        self.assertEqual(1, lc.intra_locale_comm.size)
        self.assertNotEqual(_mpi.COMM_WORLD, _mpi.COMM_NULL)

    def test_alloc_locale_buffer(self):
        lc = CartLocaleComms(ndims=1)
        rma_window_buff = lc.alloc_locale_buffer(shape=(100,), dtype="uint16")
        self.assertEqual(_np.dtype("uint16"), rma_window_buff.dtype)
        self.assertEqual(_np.dtype("uint16").itemsize, rma_window_buff.itemsize)
        self.assertEqual(100 * rma_window_buff.dtype.itemsize, len(rma_window_buff.buffer))

        lc = CartLocaleComms(ndims=1, intra_locale_comm=_mpi.COMM_SELF)
        rma_window_buff = lc.alloc_locale_buffer(shape=(100,), dtype="uint16")
        self.assertEqual(_np.dtype("uint16"), rma_window_buff.dtype)
        self.assertEqual(_np.dtype("uint16").itemsize, rma_window_buff.itemsize)
        self.assertEqual(100 * rma_window_buff.dtype.itemsize, len(rma_window_buff.buffer))


class CreateDistributionTest(_unittest.TestCase):

    """
    Tests for :func:`mpi_array.comms.create_distribution`.
    """

    def test_check_distrib_type(self):
        self.assertEqual(None, check_distrib_type(DT_SLAB))
        self.assertEqual(None, check_distrib_type(DT_BLOCK))
        self.assertEqual(None, check_distrib_type(DT_CLONED))
        self.assertEqual(None, check_distrib_type(DT_SINGLE_LOCALE))
        self.assertRaises(ValueError, check_distrib_type, "not_a_valid_distrib_type")

    def test_check_locale_type(self):
        self.assertEqual(None, check_locale_type(LT_PROCESS))
        self.assertEqual(None, check_locale_type(LT_NODE))
        self.assertRaises(ValueError, check_locale_type, "not_a_valid_locale_type")

    def test_create_locale_comms_invalid_args(self):
        """
        Test that :func:`mpi_array.comms.create_locale_comms` raises exception
        for invalid arguments.
        """

        if _mpi.COMM_WORLD.size > 1:
            self.assertRaises(
                ValueError,
                create_locale_comms,
                locale_type=LT_PROCESS,
                peer_comm=_mpi.COMM_WORLD,
                intra_locale_comm=_mpi.COMM_WORLD
            )

    def check_is_single_locale_distribution(self, distrib):
        """
        Asserts for checking that the :samp:`{distrib}` :obj:`Distribution`
        is single-locale.
        """
        self.assertTrue(isinstance(distrib, _SingleLocaleDistribution))
        gshape = tuple(distrib.globale_extent.shape_n)
        self.assertSequenceEqual(
            gshape,
            tuple(distrib.locale_extents[0].shape)
        )
        self.assertSequenceEqual(
            (0, 0, 0, 0),
            tuple(distrib.locale_extents[0].start_n)
        )
        self.assertSequenceEqual(
            gshape,
            tuple(distrib.locale_extents[0].stop_n)
        )
        self.assertSequenceEqual(
            (0, 0, 0, 0),
            tuple(distrib.globale_extent.start_n)
        )
        self.assertSequenceEqual(
            gshape,
            tuple(distrib.globale_extent.stop_n)
        )

    def test_create_single_locale_distribution(self):
        """
        Tests for :func:`mpi_array.comms.create_single_locale_distribution`.
        """
        candd = \
            create_single_locale_distribution(
                shape=(20, 31, 17, 4),
                locale_type=LT_PROCESS,
                peer_comm=_mpi.COMM_WORLD
            )
        distrib = candd.distribution
        self.check_is_single_locale_distribution(distrib)

    def test_create_distribution_slab(self):
        """
        Tests for :func:`mpi_array.comms.create_distribution`.
        """
        candd = \
            create_distribution(
                shape=(20, 31, 17, 4),
                locale_type=LT_PROCESS,
                distrib_type=DT_SLAB,
                peer_comm=_mpi.COMM_WORLD
            )
        distrib = candd.distribution
        self.assertSequenceEqual(
            (20, 31, 17, 4)[1:],
            tuple(distrib.locale_extents[0].shape)[1:]
        )
        self.assertSequenceEqual(
            (0, 0, 0, 0)[1:],
            tuple(distrib.locale_extents[0].start_n)[1:]
        )
        self.assertSequenceEqual(
            (20, 31, 17, 4)[1:],
            tuple(distrib.locale_extents[0].stop_n)[1:]
        )
        self.assertEqual(candd.locale_comms.num_locales, distrib.num_locales)
        if distrib.num_locales > 1:
            for i in range(1, distrib.num_locales):
                self.assertSequenceEqual(
                    (20, 31, 17, 4)[1:],
                    tuple(distrib.locale_extents[i].shape)[1:]
                )
                self.assertSequenceEqual(
                    (0, 0, 0, 0)[1:],
                    tuple(distrib.locale_extents[i].start_n)[1:]
                )
                self.assertSequenceEqual(
                    (20, 31, 17, 4)[1:],
                    tuple(distrib.locale_extents[i].stop_n)[1:]
                )

        self.assertSequenceEqual(
            (20, 31, 17, 4),
            tuple(distrib.globale_extent.shape)
        )
        self.assertSequenceEqual(
            (0, 0, 0, 0),
            tuple(distrib.globale_extent.start_n)
        )
        self.assertSequenceEqual(
            (20, 31, 17, 4),
            tuple(distrib.globale_extent.stop_n)
        )

    def test_get_cannonical_dims_with_shape(self):
        """
        Tests for :func:`mpi_array.comms.get_cannonical_dims` with the :samp:`shape` argument:
        the chosen numbers of blocks are at most the lengths of the axes, unless the shape
        is too small for the number of blocks.
        """
        self.assertSequenceEqual([4, 4], get_cannonical_dims(16, ndims=2).tolist())
        self.assertSequenceEqual(
            [2, 8],
            get_cannonical_dims(16, ndims=2, shape=(2, 10 ** 6)).tolist()
        )
        self.assertSequenceEqual([1, 3], get_cannonical_dims(3, ndims=2, shape=(1, 9)).tolist())
        self.assertSequenceEqual(
            [1, 2, 3],
            get_cannonical_dims(6, dims=(0, 0, 3), shape=(1, 9, 9)).tolist()
        )
        # Too small for the number of blocks, or the given numbers of blocks.
        self.assertSequenceEqual([2, 2], get_cannonical_dims(4, ndims=2, shape=(1, 1)).tolist())
        self.assertSequenceEqual([4, 1], get_cannonical_dims(4, dims=(0, 1), shape=(2, 5)).tolist())
        self.assertSequenceEqual([2, 2], get_cannonical_dims(4, ndims=2, shape=(0, 0)).tolist())

    def test_create_block_distribution_without_empty_tiles(self):
        """
        Block distributions partition each axis into at most as many blocks as its length,
        when they can, and so do the tiles of the processes of a locale. The partitions
        ignored the shape, e.g. a :samp:`(1, n)` array over two or more locales (or the
        processes of a locale) had empty blocks.
        """
        from .globale_creation import empty as _empty

        world = _mpi.COMM_WORLD
        shape = (1, 8 * world.size)
        candd = create_distribution(shape=shape, locale_type=LT_PROCESS)
        is_ok = all(e.size_n > 0 for e in candd.distribution.locale_extents)
        with _empty(shape, locale_type=LT_NODE) as gary:
            is_ok = is_ok and (gary.rank_view_n.size > 0)
        self.assertTrue(world.allreduce(bool(is_ok), op=_mpi.LAND))

    def test_shape_squeeze(self):
        """
        Tests for :func:`mpi_array.comms.shape_squeeze`, a sequence (e.g. :obj:`tuple`)
        shape argument used to raise :obj:`ValueError` (or give a wrong result).
        """
        for shape in ((1, 5, 1, 7), [1, 5, 1, 7], _np.array((1, 5, 1, 7))):
            self.assertSequenceEqual([5, 7], shape_squeeze(shape).tolist())
        self.assertSequenceEqual([], shape_squeeze((1, 1)).tolist())
        self.assertSequenceEqual([], shape_squeeze(()).tolist())

    def test_reshape_comms_distribution(self):
        """
        Tests for :func:`mpi_array.comms.reshape_comms_distribution`, it shares
        the distribution for the same shape (it used to copy it) and returns :samp:`None`
        for a different shape (reshaping distributions is not implemented). Shapes with
        the same axis lengths, apart from axes of length one, used to
        raise :obj:`AttributeError`.
        """
        gshape = (20, 31)
        for distrib_type in (DT_BLOCK, DT_CLONED):
            candd = \
                create_distribution(
                    shape=gshape,
                    locale_type=LT_PROCESS,
                    distrib_type=distrib_type,
                    peer_comm=_mpi.COMM_WORLD
                )
            reshaped = reshape_comms_distribution(candd, gshape)
            self.assertTrue(reshaped.locale_comms is candd.locale_comms)
            self.assertTrue(reshaped.distribution is candd.distribution)
            self.assertSequenceEqual(gshape, tuple(reshaped.distribution.globale_extent.shape_n))
            self.assertEqual(candd.this_locale, reshaped.this_locale)
            for shape in ((1, 20, 31), (20, 31, 1), (20, 1, 31), (31, 20), (20, 30), (620,)):
                self.assertTrue(reshape_comms_distribution(candd, shape) is None)

    def test_create_distribution_single_locale(self):
        """
        Tests for :func:`mpi_array.comms.create_distribution`.
        """
        candd = \
            create_distribution(
                shape=(20, 31, 17, 4),
                locale_type=LT_PROCESS,
                distrib_type=DT_SINGLE_LOCALE,
                peer_comm=_mpi.COMM_WORLD
            )
        distrib = candd.distribution
        self.assertEqual(candd.locale_comms.num_locales, distrib.num_locales)
        self.check_is_single_locale_distribution(distrib)


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
