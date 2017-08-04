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


"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401

from .comms import CartLocaleComms, LocaleComms


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class LocaleCommsTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.comms.LocaleComms`.
    """

    def test_construct(self):
        """
        Test :meth:`mpi_array.comms.LocaleComms.__init__`
        """
        i = LocaleComms(peer_comm=_mpi.COMM_WORLD)

        self.assertTrue(i.intra_locale_comm is not None)
        self.assertTrue(i.intra_locale_comm.size >= 1)
        self.assertTrue(i.peer_comm is not None)
        self.assertTrue(i.peer_comm.size >= 1)

        i = LocaleComms()

        self.assertTrue(i.intra_locale_comm is not None)
        self.assertTrue(i.intra_locale_comm.size >= 1)
        self.assertTrue(i.peer_comm is not None)
        self.assertTrue(i.peer_comm.size >= 1)
        i.inter_locale_comm = _mpi.COMM_NULL
        self.assertEqual(_mpi.COMM_NULL, i.inter_locale_comm)
        i.inter_locale_comm = None
        self.assertEqual(None, i.inter_locale_comm)

        self.assertRaises(ValueError, LocaleComms, _mpi.COMM_SELF, _mpi.COMM_SELF, _mpi.COMM_WORLD)


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

        lc = CartLocaleComms(ndims=4)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))

        lc = CartLocaleComms(dims=(0,))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))

        lc = CartLocaleComms(dims=(0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))

        lc = CartLocaleComms(dims=(0, 0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.peer_comm))

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


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
