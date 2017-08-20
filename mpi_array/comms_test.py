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

from .comms import CartLocaleComms, LocaleComms
from .comms import create_single_locale_distribution, create_locale_comms, create_distribution
from .comms import check_distrib_type, DT_BLOCK, DT_SLAB, DT_CLONED, DT_SINGLE_LOCALE
from .comms import check_locale_type, LT_NODE, LT_PROCESS, get_shared_mem_usage_percent_string
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
            _np.sum(len(i.peer_ranks_per_locale[r]) for r in range(i.num_locales))
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

        if _mpi.COMM_WORLD.size != _mpi.COMM_SELF.size:
            self.assertRaises(
                ValueError,
                LocaleComms,
                _mpi.COMM_SELF,  # peer
                _mpi.COMM_SELF,  # intra
                _mpi.COMM_WORLD  # inter
            )
        if i.intra_locale_comm.size > 1:
            self.assertRaises(
                ValueError,
                LocaleComms,
                i.peer_comm,  # peer
                i.intra_locale_comm,  # intra
                i.peer_comm  # inter
            )


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
