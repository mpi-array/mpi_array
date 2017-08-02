"""
=============================================
The :mod:`mpi_array.distribution_test` Module
=============================================

Module defining :mod:`mpi_array.distribution` unit-tests.
Execute as::

   python -m mpi_array.distribution_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   LocaleCommsTest - Tests for :obj:`mpi_array.distribution.LocaleComms`.
   CartLocaleExtentTest - Tests for :obj:`mpi_array.distribution.CartLocaleExtent`.
   CartLocaleCommsTest - Tests for :obj:`mpi_array.distribution.CartLocaleComms`.
   BlockPartitionTest - Tests for :obj:`mpi_array.distribution.BlockPartition`.


"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
import array_split as _array_split

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401

from .indexing import IndexingExtent
from .distribution import BlockPartition, CartLocaleComms, LocaleComms
from .distribution import CartLocaleExtent, GlobaleExtent


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class CartLocaleExtentTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.distribution.CartLocaleExtent`.
    """

    def test_construct_attribs(self):
        """
        Assertions for properties.
        """
        de = \
            CartLocaleExtent(
                rank=0,
                inter_locale_rank=0,
                cart_coord=(0,),
                cart_shape=(1,),
                globale_extent=GlobaleExtent(stop=(100,)),
                slice=(slice(0, 100),),
                halo=((10, 10),)
            )

        self.assertEqual(0, de.rank)
        self.assertEqual(0, de.cart_rank)
        self.assertTrue(_np.all(de.cart_coord == (0,)))
        self.assertTrue(_np.all(de.cart_shape == (1,)))
        self.assertTrue(_np.all(de.halo == 0))

        de = \
            CartLocaleExtent(
                rank=56,
                inter_locale_rank=7,
                cart_coord=(7,),
                cart_shape=(8,),
                globale_extent=GlobaleExtent(stop=(640,)),
                slice=(slice(560, 640),),
                halo=((10, 10),)
            )
        self.assertEqual(56, de.rank)
        self.assertEqual(7, de.cart_rank)

    def test_extent_calcs_1d_thick_tiles(self):
        """
        Tests :meth:`mpi_array.distribution.CartLocaleExtent.halo_slab_extent`
        and :meth:`mpi_array.distribution.CartLocaleExtent.no_halo_extent` methods
        when halo size is smaller than the tile size.
        """
        halo = ((10, 10),)
        splt = _array_split.shape_split((300,), axis=(3,), halo=0)
        de = \
            [
                CartLocaleExtent(
                    rank=r,
                    inter_locale_rank=r,
                    cart_coord=(r,),
                    cart_shape=(splt.shape[0],),
                    globale_extent=GlobaleExtent(stop=(300,)),
                    slice=splt[r],
                    halo=halo
                )
                for r in range(0, splt.shape[0])
            ]

        self.assertEqual(0, de[0].cart_rank)
        self.assertTrue(_np.all(de[0].cart_coord == (0,)))
        self.assertTrue(_np.all(de[0].cart_shape == (3,)))
        self.assertTrue(_np.all(de[0].halo == ((0, 10),)))
        self.assertEqual(
            IndexingExtent(start=(0,), stop=(0,)),
            de[0].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(100,), stop=(110,)),
            de[0].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0,), stop=(100,)),
            de[0].no_halo_extent(0)
        )

        self.assertEqual(1, de[1].cart_rank)
        self.assertTrue(_np.all(de[1].cart_coord == (1,)))
        self.assertTrue(_np.all(de[1].cart_shape == (3,)))
        self.assertTrue(_np.all(de[1].halo == ((10, 10),)))
        self.assertEqual(
            IndexingExtent(start=(90,), stop=(100,)),
            de[1].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(200,), stop=(210,)),
            de[1].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(100,), stop=(200,)),
            de[1].no_halo_extent(0)
        )

        self.assertEqual(2, de[2].cart_rank)
        self.assertTrue(_np.all(de[2].cart_coord == (2,)))
        self.assertTrue(_np.all(de[2].cart_shape == (3,)))
        self.assertTrue(_np.all(de[2].halo == ((10, 0),)))
        self.assertEqual(
            IndexingExtent(start=(190,), stop=(200,)),
            de[2].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(300,), stop=(300,)),
            de[2].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(200,), stop=(300,)),
            de[2].no_halo_extent(0)
        )

    def test_extent_calcs_1d_thin_tiles(self):
        """
        Tests :meth:`mpi_array.distribution.CartLocaleExtent.halo_slab_extent`
        and :meth:`mpi_array.distribution.CartLocaleExtent.no_halo_extent` methods
        when halo size is larger than the tile size, 1D fixture.
        """
        halo = ((5, 5),)
        splt = _array_split.shape_split((15,), axis=(5,), halo=0)
        de = \
            [
                CartLocaleExtent(
                    rank=r,
                    inter_locale_rank=r,
                    cart_coord=(r,),
                    cart_shape=(splt.shape[0],),
                    globale_extent=GlobaleExtent(stop=(15,)),
                    slice=splt[r],
                    halo=halo
                )
                for r in range(0, splt.shape[0])
            ]

        self.assertEqual(0, de[0].cart_rank)
        self.assertTrue(_np.all(de[0].cart_coord == (0,)))
        self.assertTrue(_np.all(de[0].cart_shape == (5,)))
        self.assertTrue(_np.all(de[0].halo == ((0, 5),)))
        self.assertEqual(
            IndexingExtent(start=(0,), stop=(0,)),
            de[0].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(3,), stop=(8,)),
            de[0].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0,), stop=(3,)),
            de[0].no_halo_extent(0)
        )

        self.assertEqual(1, de[1].cart_rank)
        self.assertTrue(_np.all(de[1].cart_coord == (1,)))
        self.assertTrue(_np.all(de[1].cart_shape == (5,)))
        self.assertTrue(_np.all(de[1].halo == ((3, 5),)))
        self.assertEqual(
            IndexingExtent(start=(0,), stop=(3,)),
            de[1].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(6,), stop=(11,)),
            de[1].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(3,), stop=(6,)),
            de[1].no_halo_extent(0)
        )

        self.assertEqual(2, de[2].cart_rank)
        self.assertTrue(_np.all(de[2].cart_coord == (2,)))
        self.assertTrue(_np.all(de[2].cart_shape == (5,)))
        self.assertTrue(_np.all(de[2].halo == ((5, 5),)))
        self.assertEqual(
            IndexingExtent(start=(1,), stop=(6,)),
            de[2].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(9,), stop=(14,)),
            de[2].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(6,), stop=(9,)),
            de[2].no_halo_extent(0)
        )

        self.assertEqual(3, de[3].cart_rank)
        self.assertTrue(_np.all(de[3].cart_coord == (3,)))
        self.assertTrue(_np.all(de[3].cart_shape == (5,)))
        self.assertTrue(_np.all(de[3].halo == ((5, 3),)))
        self.assertEqual(
            IndexingExtent(start=(4,), stop=(9,)),
            de[3].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(12,), stop=(15,)),
            de[3].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(9,), stop=(12,)),
            de[3].no_halo_extent(0)
        )

        self.assertEqual(4, de[4].cart_rank)
        self.assertTrue(_np.all(de[4].cart_coord == (4,)))
        self.assertTrue(_np.all(de[4].cart_shape == (5,)))
        self.assertTrue(_np.all(de[4].halo == ((5, 0),)))
        self.assertEqual(
            IndexingExtent(start=(7,), stop=(12,)),
            de[4].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(15,), stop=(15,)),
            de[4].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(12,), stop=(15,)),
            de[4].no_halo_extent(0)
        )

    def test_extent_calcs_2d_thick_tiles(self):
        """
        Tests :meth:`mpi_array.distribution.CartLocaleExtent.halo_slab_extent`
        and :meth:`mpi_array.distribution.CartLocaleExtent.no_halo_extent` methods
        when halo size is smaller than the tile size, 2D fixture.
        """
        halo = ((10, 10), (5, 5))
        splt = _array_split.shape_split((300, 600), axis=(3, 3), halo=0)
        de = \
            [
                CartLocaleExtent(
                    rank=r,
                    inter_locale_rank=r,
                    cart_coord=_np.unravel_index(r, splt.shape),
                    cart_shape=splt.shape,
                    globale_extent=GlobaleExtent(stop=(300, 600)),
                    slice=splt[tuple(_np.unravel_index(r, splt.shape))],
                    halo=halo
                )
                for r in range(0, _np.product(splt.shape))
            ]

        self.assertEqual(0, de[0].cart_rank)
        self.assertTrue(_np.all(de[0].cart_coord == (0, 0)))
        self.assertTrue(_np.all(de[0].cart_shape == (3, 3)))
        self.assertSequenceEqual([[0, 10], [0, 5]], de[0].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(0, 0), stop=(0, 205)),
            de[0].halo_slab_extent(0, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(100, 0), stop=(110, 205)),
            de[0].halo_slab_extent(0, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 0), stop=(110, 0)),
            de[0].halo_slab_extent(1, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 200), stop=(110, 205)),
            de[0].halo_slab_extent(1, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 0), stop=(100, 205)),
            de[0].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 0), stop=(110, 200)),
            de[0].no_halo_extent(1)
        )

        self.assertEqual(1, de[1].cart_rank)
        self.assertTrue(_np.all(de[1].cart_coord == (0, 1)))
        self.assertTrue(_np.all(de[1].cart_shape == (3, 3)))
        self.assertSequenceEqual([[0, 10], [5, 5]], de[1].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(0, 195), stop=(0, 405)),
            de[1].halo_slab_extent(0, de[1].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(100, 195), stop=(110, 405)),
            de[1].halo_slab_extent(0, de[1].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 195), stop=(110, 200)),
            de[1].halo_slab_extent(1, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 400), stop=(110, 405)),
            de[1].halo_slab_extent(1, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 195), stop=(100, 405)),
            de[1].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 200), stop=(110, 400)),
            de[1].no_halo_extent(1)
        )

        self.assertEqual(2, de[2].cart_rank)
        self.assertTrue(_np.all(de[2].cart_coord == (0, 2)))
        self.assertTrue(_np.all(de[2].cart_shape == (3, 3)))
        self.assertSequenceEqual([[0, 10], [5, 0]], de[2].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(0, 395), stop=(0, 600)),
            de[2].halo_slab_extent(0, de[2].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(100, 395), stop=(110, 600)),
            de[2].halo_slab_extent(0, de[2].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 395), stop=(110, 400)),
            de[2].halo_slab_extent(1, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 600), stop=(110, 600)),
            de[2].halo_slab_extent(1, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 395), stop=(100, 600)),
            de[2].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(0, 400), stop=(110, 600)),
            de[2].no_halo_extent(1)
        )

        self.assertEqual(3, de[3].cart_rank)
        self.assertTrue(_np.all(de[3].cart_coord == (1, 0)))
        self.assertTrue(_np.all(de[3].cart_shape == (3, 3)))
        self.assertSequenceEqual([[10, 10], [0, 5]], de[3].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(90, 0), stop=(100, 205)),
            de[3].halo_slab_extent(0, de[3].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(200, 0), stop=(210, 205)),
            de[3].halo_slab_extent(0, de[3].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 0), stop=(210, 0)),
            de[3].halo_slab_extent(1, de[3].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 200), stop=(210, 205)),
            de[3].halo_slab_extent(1, de[3].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(100, 0), stop=(200, 205)),
            de[3].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 0), stop=(210, 200)),
            de[3].no_halo_extent(1)
        )

        self.assertEqual(4, de[4].cart_rank)
        self.assertTrue(_np.all(de[4].cart_coord == (1, 1)))
        self.assertTrue(_np.all(de[4].cart_shape == (3, 3)))
        self.assertSequenceEqual([[10, 10], [5, 5]], de[4].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(90, 195), stop=(100, 405)),
            de[4].halo_slab_extent(0, de[4].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(200, 195), stop=(210, 405)),
            de[4].halo_slab_extent(0, de[4].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 195), stop=(210, 200)),
            de[4].halo_slab_extent(1, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 400), stop=(210, 405)),
            de[4].halo_slab_extent(1, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(100, 195), stop=(200, 405)),
            de[4].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 200), stop=(210, 400)),
            de[4].no_halo_extent(1)
        )

        self.assertEqual(5, de[5].cart_rank)
        self.assertTrue(_np.all(de[5].cart_coord == (1, 2)))
        self.assertTrue(_np.all(de[5].cart_shape == (3, 3)))
        self.assertSequenceEqual([[10, 10], [5, 0]], de[5].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(90, 395), stop=(100, 600)),
            de[5].halo_slab_extent(0, de[5].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(200, 395), stop=(210, 600)),
            de[5].halo_slab_extent(0, de[5].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 395), stop=(210, 400)),
            de[5].halo_slab_extent(1, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 600), stop=(210, 600)),
            de[5].halo_slab_extent(1, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(100, 395), stop=(200, 600)),
            de[5].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(90, 400), stop=(210, 600)),
            de[5].no_halo_extent(1)
        )

        self.assertEqual(6, de[6].cart_rank)
        self.assertTrue(_np.all(de[6].cart_coord == (2, 0)))
        self.assertTrue(_np.all(de[6].cart_shape == (3, 3)))
        self.assertSequenceEqual([[10, 0], [0, 5]], de[6].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(190, 0), stop=(200, 205)),
            de[6].halo_slab_extent(0, de[6].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(300, 0), stop=(300, 205)),
            de[6].halo_slab_extent(0, de[6].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 0), stop=(300, 0)),
            de[6].halo_slab_extent(1, de[6].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 200), stop=(300, 205)),
            de[6].halo_slab_extent(1, de[6].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(200, 0), stop=(300, 205)),
            de[6].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 0), stop=(300, 200)),
            de[6].no_halo_extent(1)
        )

        self.assertEqual(7, de[7].cart_rank)
        self.assertTrue(_np.all(de[7].cart_coord == (2, 1)))
        self.assertTrue(_np.all(de[7].cart_shape == (3, 3)))
        self.assertSequenceEqual([[10, 0], [5, 5]], de[7].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(190, 195), stop=(200, 405)),
            de[7].halo_slab_extent(0, de[7].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(300, 195), stop=(300, 405)),
            de[7].halo_slab_extent(0, de[7].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 195), stop=(300, 200)),
            de[7].halo_slab_extent(1, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 400), stop=(300, 405)),
            de[7].halo_slab_extent(1, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(200, 195), stop=(300, 405)),
            de[7].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 200), stop=(300, 400)),
            de[7].no_halo_extent(1)
        )

        self.assertEqual(8, de[8].cart_rank)
        self.assertTrue(_np.all(de[8].cart_coord == (2, 2)))
        self.assertTrue(_np.all(de[8].cart_shape == (3, 3)))
        self.assertSequenceEqual([[10, 0], [5, 0]], de[8].halo.tolist())
        self.assertEqual(
            IndexingExtent(start=(190, 395), stop=(200, 600)),
            de[8].halo_slab_extent(0, de[8].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(300, 395), stop=(300, 600)),
            de[8].halo_slab_extent(0, de[8].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 395), stop=(300, 400)),
            de[8].halo_slab_extent(1, de[0].LO)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 600), stop=(300, 600)),
            de[8].halo_slab_extent(1, de[0].HI)
        )
        self.assertEqual(
            IndexingExtent(start=(200, 395), stop=(300, 600)),
            de[8].no_halo_extent(0)
        )
        self.assertEqual(
            IndexingExtent(start=(190, 400), stop=(300, 600)),
            de[8].no_halo_extent(1)
        )


class LocaleCommsTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.distribution.LocaleComms`.
    """

    def test_construct(self):
        """
        Test :meth:`mpi_array.distribution.LocaleComms.__init__`
        """
        i = LocaleComms(comm=_mpi.COMM_WORLD)

        self.assertTrue(i.intra_locale_comm is not None)
        self.assertTrue(i.intra_locale_comm.size >= 1)
        self.assertTrue(i.rank_comm is not None)
        self.assertTrue(i.rank_comm.size >= 1)

        i = LocaleComms()

        self.assertTrue(i.intra_locale_comm is not None)
        self.assertTrue(i.intra_locale_comm.size >= 1)
        self.assertTrue(i.rank_comm is not None)
        self.assertTrue(i.rank_comm.size >= 1)
        if i.num_locales <= 1:
            self.assertEqual(None, i.inter_locale_comm)
        i.inter_locale_comm = _mpi.COMM_NULL
        self.assertEqual(_mpi.COMM_NULL, i.inter_locale_comm)
        i.inter_locale_comm = None
        self.assertEqual(None, i.inter_locale_comm)

        self.assertRaises(ValueError, LocaleComms, _mpi.COMM_SELF, _mpi.COMM_SELF, _mpi.COMM_WORLD)


class CartLocaleCommsTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.distribution.CartLocaleComms`.
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

    def test_construct_shared(self):
        lc = CartLocaleComms(ndims=1)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.rank_comm))

        lc = CartLocaleComms(ndims=4)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.rank_comm))

        lc = CartLocaleComms(dims=(0,))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.rank_comm))

        lc = CartLocaleComms(dims=(0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.rank_comm))

        lc = CartLocaleComms(dims=(0, 0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.rank_comm))

    def test_construct_no_shared(self):
        lc = CartLocaleComms(ndims=1, intra_locale_comm=_mpi.COMM_SELF)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, lc.rank_comm))
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


class BlockPartitionTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.distribution.BlockPartition`.
    """

    def setUp(self):
        """
        Initialise self.root_logger.
        """
        self.root_logger = _logging.get_root_logger(__name__ + "." + self.id())
        self.rank_logger = _logging.get_rank_logger(__name__ + "." + self.id())

    def test_construct_single_locale_1d(self):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction.
        """

        distrib = \
            BlockPartition(
                globale_extent=(8,),
                dims=[1, ],
                cart_coord_to_cart_rank={(0,): 0}
            )
        self.assertEqual(1, len(distrib.locale_extents))
        self.assertEqual(GlobaleExtent(stop=(8,)), distrib.globale_extent)
        self.assertEqual(1, distrib.num_locales)
        self.root_logger.info("START " + self.id())
        self.root_logger.info(str(distrib))
        self.root_logger.info("END   " + self.id())
        self.root_logger.info("distrib.locale_extents[0]=\n%s" % (distrib.locale_extents[0],))
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=0,
                globale_extent=GlobaleExtent(stop=(8,)),
                cart_coord=(0,),
                cart_shape=(1,),
                start=(0,),
                stop=(8,)
            ),
            distrib.locale_extents[0]
        )

        distrib = \
            BlockPartition(
                globale_extent=distrib.globale_extent,
                dims=[4, ],
                cart_coord_to_cart_rank={(i,): i for i in range(0, 4)}
            )
        self.assertEqual(4, len(distrib.locale_extents))
        self.assertEqual(GlobaleExtent(stop=(8,)), distrib.globale_extent)
        self.assertEqual(4, distrib.num_locales)
        self.root_logger.info("START " + self.id())
        self.root_logger.info(str(distrib))
        self.root_logger.info("END   " + self.id())

    def do_test_construct_1d_with_halo(self, halo=0):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction.
        """
        distrib = \
            BlockPartition(
                (32,),
                dims=[4, ],
                halo=halo,
                cart_coord_to_cart_rank={(i,): i for i in range(0, 4)}
            )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=0,
                globale_extent=GlobaleExtent(stop=(32,)),
                cart_coord=(0,),
                cart_shape=(4,),
                start=(0,),
                stop=(8,),
                halo=halo
            ),
            distrib.locale_extents[0]
        )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=1,
                globale_extent=GlobaleExtent(stop=(32,)),
                cart_coord=(1,),
                cart_shape=(4,),
                start=(8,),
                stop=(16,),
                halo=halo
            ),
            distrib.locale_extents[1]
        )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=2,
                globale_extent=GlobaleExtent(stop=(32,)),
                cart_coord=(2,),
                cart_shape=(4,),
                start=(16,),
                stop=(24,),
                halo=halo
            ),
            distrib.locale_extents[2]
        )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=3,
                globale_extent=GlobaleExtent(stop=(32,)),
                cart_coord=(3,),
                cart_shape=(4,),
                start=(24,),
                stop=(32,),
                halo=halo
            ),
            distrib.locale_extents[3]
        )

        self.root_logger.info("START " + self.id())
        self.root_logger.info(str(distrib))
        self.root_logger.info("END   " + self.id())

    def test_construct_1d_no_halo(self):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction.
        """
        self.do_test_construct_1d_with_halo(halo=0)

    def test_construct_1d_with_halo(self):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction.
        """
        self.do_test_construct_1d_with_halo(halo=[[2, 4], ])

    def test_construct_1d_empty_tiles(self):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction
        when the partition leads to empty extents.
        """
        halo = 0
        distrib = \
            BlockPartition(
                globale_extent=(slice(0, 2),),
                dims=(4,),
                halo=halo,
                cart_coord_to_cart_rank={(i,): i for i in range(0, 4)}
            )

        self.root_logger.info("START " + self.id())
        self.root_logger.info(str(distrib))
        self.root_logger.info("END   " + self.id())

        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=0,
                globale_extent=GlobaleExtent(stop=(2,)),
                cart_coord=(0,),
                cart_shape=(4,),
                start=(0,),
                stop=(1,),
                halo=halo
            ),
            distrib.locale_extents[0]
        )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=1,
                globale_extent=GlobaleExtent(stop=(2,)),
                cart_coord=(1,),
                cart_shape=(4,),
                start=(1,),
                stop=(2,),
                halo=halo
            ),
            distrib.locale_extents[1]
        )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=2,
                globale_extent=GlobaleExtent(stop=(2,)),
                cart_coord=(2,),
                cart_shape=(4,),
                start=(2,),
                stop=(2,),
                halo=halo
            ),
            distrib.locale_extents[2]
        )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=3,
                globale_extent=GlobaleExtent(stop=(2,)),
                cart_coord=(3,),
                cart_shape=(4,),
                start=(2,),
                stop=(2,),
                halo=halo
            ),
            distrib.locale_extents[3]
        )

    def do_test_construct_2d_with_halo(self, halo=0):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction.
        """
        distrib = \
            BlockPartition(
                globale_extent=(16, 32),
                dims=(2, 4),
                halo=halo,
                cart_coord_to_cart_rank={
                    tuple(_np.unravel_index(i, (2, 4))): i for i in range(0, 8)
                }
            )
        self.root_logger.info("START " + self.id())
        self.root_logger.info(str(distrib))
        self.root_logger.info("END   " + self.id())

        self.assertEqual(8, distrib.num_locales)
        self.assertEqual(8, len(distrib.locale_extents))

        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=0,
                globale_extent=GlobaleExtent(stop=(16, 32,)),
                cart_coord=(0, 0),
                cart_shape=(2, 4),
                start=(0, 0),
                stop=(8, 8),
                halo=halo
            ),
            distrib.locale_extents[0]
        )
        self.assertEqual(
            CartLocaleExtent(
                rank=_mpi.UNDEFINED,
                inter_locale_rank=7,
                globale_extent=GlobaleExtent(stop=(16, 32,)),
                cart_coord=(1, 3),
                cart_shape=(2, 4),
                start=(8, 24),
                stop=(16, 32),
                halo=halo
            ),
            distrib.locale_extents[7]
        )

    def test_construct_2d_no_halo(self):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction.
        """
        self.do_test_construct_2d_with_halo(halo=0)

    def test_construct_2d_with_halo(self):
        """
        Test :obj:`mpi_array.distribution.BlockPartition` construction.
        """
        self.do_test_construct_2d_with_halo(halo=[[1, 2], [3, 4]])

    def do_not_test_recalculate_2d(self):
        """
        Test :meth:`mpi_array.distribution.BlockPartition.recalculate` construction.
        """
        locale_comms = \
            [
                None,
                CartLocaleComms(
                    ndims=2,
                    rank_comm=_mpi.COMM_WORLD,
                    intra_locale_comm=_mpi.COMM_SELF
                )
            ]
        for lc in locale_comms:
            orig_shape = (8 * _mpi.COMM_WORLD.size, 12 * _mpi.COMM_WORLD.size)
            distrib = \
                BlockPartition(
                    orig_shape,
                    locale_comms=lc,
                    halo=((2, 2), (4, 4))
                )

            self.assertSequenceEqual([[2, 2], [4, 4]], distrib.halo.tolist())

            distrib.halo = [[1, 2], [3, 4]]
            self.assertSequenceEqual([[1, 2], [3, 4]], distrib.halo.tolist())

            self.assertSequenceEqual(list(orig_shape), distrib.shape.tolist())
            new_shape = (10 * _mpi.COMM_WORLD.size, 7 * _mpi.COMM_WORLD.size)
            distrib.shape = new_shape
            self.assertSequenceEqual(list(new_shape), distrib.shape.tolist())

            new_shape = (23 * _mpi.COMM_WORLD.size,)
            distrib.recalculate(new_shape, new_halo=5)
            self.assertSequenceEqual(list(new_shape), distrib.shape.tolist())
            self.assertSequenceEqual([[5, 5], ], distrib.halo.tolist())

            new_shape = \
                (23 * _mpi.COMM_WORLD.size, 14 * _mpi.COMM_WORLD.size, 8 * _mpi.COMM_WORLD.size)
            distrib.recalculate(new_shape, new_halo=5)
            self.assertSequenceEqual(list(new_shape), distrib.shape.tolist())
            self.assertSequenceEqual([[5, 5], [5, 5], [5, 5]], distrib.halo.tolist())

            distrib.halo = None
            self.assertSequenceEqual([[0, 0], [0, 0], [0, 0]], distrib.halo.tolist())
            self.assertSequenceEqual(list(new_shape), distrib.shape.tolist())


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
