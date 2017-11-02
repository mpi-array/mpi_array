"""
=========================================
The :mod:`mpi_array.indexing_test` Module
=========================================

Module defining :mod:`mpi_array.indexing` unit-tests.
Execute as::

   python -m mpi_array.indexing_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   IndexingExtentTest - Tests for :obj:`mpi_array.indexing.IndexingExtent`.
   HaloIndexingExtentTest - Tests for :obj:`mpi_array.indexing.IndexingExtent`.


"""
from __future__ import absolute_import

import numpy as _np  # noqa: E402,F401

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .indexing import IndexingExtent, HaloIndexingExtent, calc_intersection_split


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version


class IndexingExtentTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.indexing.IndexingExtentTest`.
    """

    def test_repr(self):
        """
        Test for :samp:`repr(IndexingExtent(start=(1,2,3), stop=(8,9,10)))`.
        """
        ie = IndexingExtent(start=(10,), stop=(32,))
        self.assertNotEqual(None, str(ie))
        self.assertNotEqual("", str(ie))

        self.assertEqual(ie, eval(repr(ie)))

    def test_to_tuple(self):
        """
        Test for :meth:`IndexingExtent.to_tuple`.
        """
        ie = IndexingExtent(start=(10, 15), stop=(32, 66))
        self.assertEqual(ie, IndexingExtent(*(ie.to_tuple())))

    def test_assign_different_dimension_index(self):
        """
        Test for :meth:`IndexingExtent.start = ...`.
        """
        ie = IndexingExtent(start=(10, 15), stop=(32, 66))

        def assign_start():
            ie.start = (1,)

        def assign_stop():
            ie.stop = (1,)

        self.assertRaises(ValueError, assign_start)
        self.assertRaises(ValueError, assign_stop)

    def test_attributes(self):
        """
        Tests :attr:`mpi_array.indexing.IndexingExtent.start`
        and :attr:`mpi_array.indexing.IndexingExtent.stop`
        and :attr:`mpi_array.indexing.IndexingExtent.shape`
        attributes.
        """
        ie = IndexingExtent(start=(10,), stop=(32,))
        self.assertTrue(_np.all(ie.shape == (22,)))
        self.assertTrue(_np.all(ie.start == (10,)))
        self.assertTrue(_np.all(ie.stop == (32,)))

        ie = IndexingExtent((slice(10, 32),))
        self.assertTrue(_np.all(ie.shape == (22,)))
        self.assertTrue(_np.all(ie.start == (10,)))
        self.assertTrue(_np.all(ie.stop == (32,)))

        ie = IndexingExtent(start=(10, 25), stop=(32, 55))
        self.assertTrue(_np.all(ie.shape == (22, 30)))
        self.assertTrue(_np.all(ie.start == (10, 25)))
        self.assertTrue(_np.all(ie.stop == (32, 55)))

        ie = IndexingExtent((slice(10, 32), slice(25, 55)))
        self.assertTrue(_np.all(ie.shape == (22, 30)))
        self.assertTrue(_np.all(ie.start == (10, 25)))
        self.assertTrue(_np.all(ie.stop == (32, 55)))

        ie = IndexingExtent((slice(10, 32), slice(25, 55)))
        ie.start = (5, 6)
        self.assertSequenceEqual([5, 6], ie.start.tolist())

        ie.stop = (11, 12)
        self.assertSequenceEqual([11, 12], ie.stop.tolist())

    def test_intersection_1d(self):
        """
        Tests :meth:`mpi_array.indexing.IndexingExtent.calc_intersection` method, 1D indexing.
        """
        ie0 = IndexingExtent(start=(10,), stop=(32,))
        iei = ie0.calc_intersection(ie0)
        self.assertTrue(_np.all(iei.shape == (22,)))
        self.assertTrue(_np.all(iei.start == (10,)))
        self.assertTrue(_np.all(iei.stop == (32,)))

        ie1 = IndexingExtent(start=(5,), stop=(32,))
        iei = ie0.calc_intersection(ie1)
        self.assertTrue(_np.all(iei.shape == (22,)))
        self.assertTrue(_np.all(iei.start == (10,)))
        self.assertTrue(_np.all(iei.stop == (32,)))

        ie1 = IndexingExtent(start=(10,), stop=(39,))
        iei = ie0.calc_intersection(ie1)
        self.assertTrue(_np.all(iei.shape == (22,)))
        self.assertTrue(_np.all(iei.start == (10,)))
        self.assertTrue(_np.all(iei.stop == (32,)))

        ie1 = IndexingExtent(start=(-5,), stop=(39,))
        iei = ie0.calc_intersection(ie1)
        self.assertTrue(_np.all(iei.shape == (22,)))
        self.assertTrue(_np.all(iei.start == (10,)))
        self.assertTrue(_np.all(iei.stop == (32,)))

        ie1 = IndexingExtent(start=(11,), stop=(31,))
        iei = ie0.calc_intersection(ie1)
        self.assertTrue(_np.all(iei.shape == (20,)))
        self.assertTrue(_np.all(iei.start == (11,)))
        self.assertTrue(_np.all(iei.stop == (31,)))

        ie1 = IndexingExtent(start=(5,), stop=(10,))
        iei = ie0.calc_intersection(ie1)
        self.assertEqual(None, iei)

        ie1 = IndexingExtent(start=(32,), stop=(55,))
        iei = ie0.calc_intersection(ie1)
        self.assertEqual(None, iei)

    def test_intersection_2d(self):
        """
        Tests :meth:`mpi_array.indexing.IndexingExtent.calc_intersection` method, 2D indexing.
        """
        ie0 = IndexingExtent(start=(10, 20), stop=(32, 64))
        iei = ie0.calc_intersection(ie0)
        self.assertSequenceEqual(ie0.shape.tolist(), iei.shape.tolist())
        self.assertSequenceEqual(ie0.start.tolist(), iei.start.tolist())
        self.assertSequenceEqual(ie0.stop.tolist(), iei.stop.tolist())

        ie1 = IndexingExtent(start=(0, 20), stop=(10, 64))
        iei = ie0.calc_intersection(ie1)
        self.assertEqual(None, iei)

        ie1 = IndexingExtent(start=(10, 0), stop=(32, 20))
        iei = ie0.calc_intersection(ie1)
        self.assertEqual(None, iei)

        ie1 = IndexingExtent(start=(0, 0), stop=(10, 20))
        iei = ie0.calc_intersection(ie1)
        self.assertEqual(None, iei)

        ie1 = IndexingExtent(start=(32, 64), stop=(110, 120))
        iei = ie0.calc_intersection(ie1)
        self.assertEqual(None, iei)

        ie1 = IndexingExtent(start=(20, 10), stop=(30, 40))
        iei = ie0.calc_intersection(ie1)
        self.assertSequenceEqual([10, 20], iei.shape.tolist())
        self.assertSequenceEqual([20, 20], iei.start.tolist())
        self.assertSequenceEqual([30, 40], iei.stop.tolist())

        ie1 = IndexingExtent(start=(22, 54), stop=(80, 90))
        iei = ie0.calc_intersection(ie1)
        self.assertSequenceEqual([10, 10], iei.shape.tolist())
        self.assertSequenceEqual([22, 54], iei.start.tolist())
        self.assertSequenceEqual([32, 64], iei.stop.tolist())

    def test_split(self):
        """
        Test for :meth:`mpi_array.indexing.IndexingExtent.split`.
        """
        ie = IndexingExtent(start=(10, 3), stop=(32, 20))

        lo, hi = ie.split(0, 10)
        self.assertTrue(lo is None)
        self.assertTrue(hi is ie)

        lo, hi = ie.split(1, 3)
        self.assertTrue(lo is None)
        self.assertTrue(hi is ie)

        lo, hi = ie.split(0, 32)
        self.assertTrue(lo is ie)
        self.assertTrue(hi is None)

        lo, hi = ie.split(1, 20)
        self.assertTrue(lo is ie)
        self.assertTrue(hi is None)

        lo, hi = ie.split(0, 11)
        self.assertEqual(IndexingExtent(start=(10, 3), stop=(11, 20)), lo)
        self.assertEqual(IndexingExtent(start=(11, 3), stop=(32, 20)), hi)

        lo, hi = ie.split(1, 4)
        self.assertEqual(IndexingExtent(start=(10, 3), stop=(32, 4)), lo)
        self.assertEqual(IndexingExtent(start=(10, 4), stop=(32, 20)), hi)

        lo, hi = ie.split(0, 31)
        self.assertEqual(IndexingExtent(start=(10, 3), stop=(31, 20)), lo)
        self.assertEqual(IndexingExtent(start=(31, 3), stop=(32, 20)), hi)

        lo, hi = ie.split(1, 19)
        self.assertEqual(IndexingExtent(start=(10, 3), stop=(32, 19)), lo)
        self.assertEqual(IndexingExtent(start=(10, 19), stop=(32, 20)), hi)

    def test_calc_intersection_split(self):
        """
        Test for :meth:`mpi_array.indexing.IndexingExtent.calc_intersection_split`.
        """
        ie = IndexingExtent(start=(0, 50), stop=(50, 100))

        other = IndexingExtent(start=(0, 50), stop=(50, 100))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(0, len(leftovers))
        self.assertEqual(intersection, ie)
        self.assertEqual(intersection, other)

        other = IndexingExtent(start=(25, 50), stop=(50, 100))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(intersection, other)
        self.assertEqual(1, len(leftovers))
        self.assertEqual(IndexingExtent(start=(0, 50), stop=(25, 100)), leftovers[0])

        other = IndexingExtent(start=(0, 50), stop=(25, 100))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(intersection, other)
        self.assertEqual(1, len(leftovers))
        self.assertEqual(IndexingExtent(start=(25, 50), stop=(50, 100)), leftovers[0])

        other = IndexingExtent(start=(0, 50), stop=(50, 75))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(intersection, other)
        self.assertEqual(1, len(leftovers))
        self.assertEqual(IndexingExtent(start=(0, 75), stop=(50, 100)), leftovers[0])

        other = IndexingExtent(start=(0, 75), stop=(50, 100))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(intersection, other)
        self.assertEqual(1, len(leftovers))
        self.assertEqual(IndexingExtent(start=(0, 50), stop=(50, 75)), leftovers[0])

        other = IndexingExtent(start=(0, 50), stop=(25, 75))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(intersection, other)
        self.assertEqual(2, len(leftovers))
        self.assertEqual(IndexingExtent(start=(25, 50), stop=(50, 100)), leftovers[0])
        self.assertEqual(IndexingExtent(start=(0, 75), stop=(25, 100)), leftovers[1])

        other = IndexingExtent(start=(25, 75), stop=(50, 100))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(intersection, other)
        self.assertEqual(2, len(leftovers))
        self.assertEqual(IndexingExtent(start=(0, 50), stop=(25, 100)), leftovers[0])
        self.assertEqual(IndexingExtent(start=(25, 50), stop=(50, 75)), leftovers[1])

        other = IndexingExtent(start=(20, 60), stop=(40, 80))
        leftovers, intersection = ie.calc_intersection_split(other)
        self.assertEqual(intersection, other)
        self.assertEqual(4, len(leftovers))
        self.assertEqual(IndexingExtent(start=(0, 50), stop=(20, 100)), leftovers[0])
        self.assertEqual(IndexingExtent(start=(40, 50), stop=(50, 100)), leftovers[1])
        self.assertEqual(IndexingExtent(start=(20, 50), stop=(40, 60)), leftovers[2])
        self.assertEqual(IndexingExtent(start=(20, 80), stop=(40, 100)), leftovers[3])


class HaloIndexingExtentTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.indexing.HaloIndexingExtentTest`.
    """

    def test_repr(self):
        """
        Test for :samp:`repr(HaloIndexingExtent(start=(1,2,3), stop=(8,9,10)))`.
        """
        ie = HaloIndexingExtent(start=(10, 15), stop=(32, 66), halo=((1, 2), (3, 4)))
        self.assertNotEqual(None, str(ie))
        self.assertNotEqual("", str(ie))

        self.assertEqual(ie, eval(repr(ie)))

    def test_to_tuple(self):
        """
        Test for :meth:`HaloIndexingExtent.to_tuple`.
        """
        ie = HaloIndexingExtent(start=(10, 15), stop=(32, 66), halo=((1, 2), (3, 4)))
        self.assertEqual(ie, HaloIndexingExtent(*(ie.to_tuple())))

    def test_attributes(self):
        """
        :obj:`unittest.TestCase` for :obj:`mpi_array.indexing.HaloIndexingExtentTest`
        attributes.
        """

        hie1 = HaloIndexingExtent(start=(10, 0), stop=(32, 20), halo=_np.array(((0, 0), (0, 0))))
        self.assertSequenceEqual([10, 0], hie1.start_n.tolist())
        self.assertSequenceEqual([10, 0], hie1.start_h.tolist())
        self.assertSequenceEqual([32, 20], hie1.stop_n.tolist())
        self.assertSequenceEqual([32, 20], hie1.stop_h.tolist())
        self.assertSequenceEqual([22, 20], hie1.shape_n.tolist())
        self.assertSequenceEqual([22, 20], hie1.shape_h.tolist())
        self.assertEqual(22 * 20, hie1.size_n)
        self.assertEqual(22 * 20, hie1.size_h)

        hie1 = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))
        self.assertSequenceEqual([10, 3], hie1.start_n.tolist())
        self.assertSequenceEqual([9, 0], hie1.start_h.tolist())
        self.assertSequenceEqual([32, 20], hie1.stop_n.tolist())
        self.assertSequenceEqual([34, 24], hie1.stop_h.tolist())
        self.assertSequenceEqual([22, 17], hie1.shape_n.tolist())
        self.assertSequenceEqual([25, 24], hie1.shape_h.tolist())
        self.assertEqual(22 * 17, hie1.size_n)
        self.assertEqual(25 * 24, hie1.size_h)

        ie = HaloIndexingExtent((slice(10, 32), slice(25, 55)))
        ie.start_n = (3, 4)
        self.assertSequenceEqual([3, 4], ie.start_n.tolist())
        self.assertSequenceEqual([3, 4], ie.start.tolist())

        ie.stop_n = (8, 9)
        self.assertSequenceEqual([8, 9], ie.stop_n.tolist())
        self.assertSequenceEqual([8, 9], ie.stop.tolist())

        ie.halo = [[1, 2], [4, 8]]
        self.assertSequenceEqual([[1, 2], [4, 8]], ie.halo.tolist())
        ie.halo = 0
        self.assertSequenceEqual([[0, 0], [0, 0]], ie.halo.tolist())

    def test_globale_and_locale_index_conversion(self):
        """
        Test for :meth:`mpi_array.indexing.HaloIndexingExtent.globale_to_locale_h`,
        and :meth:`mpi_array.indexing.HaloIndexingExtent.locale_to_globale_h`.
        """
        hie = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))
        self.assertSequenceEqual(
            [1, 3],
            list(hie.globale_to_locale_h((10, 3)))
        )
        self.assertSequenceEqual(
            [10, 3],
            list(hie.locale_to_globale_h(hie.globale_to_locale_h((10, 3))))
        )

        self.assertSequenceEqual(
            [0, 0],
            list(hie.globale_to_locale_n((10, 3)))
        )
        self.assertSequenceEqual(
            [10, 3],
            list(hie.locale_to_globale_n(hie.globale_to_locale_n((10, 3))))
        )

    def test_globale_and_locale_extent_conversion(self):
        """
        Test for :meth:`mpi_array.indexing.HaloIndexingExtent.globale_to_locale_h`,
        and :meth:`mpi_array.indexing.HaloIndexingExtent.locale_to_globale_h`.
        """
        hie = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))
        gext = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))
        self.assertEqual(
            HaloIndexingExtent(start=(1, 3), stop=(23, 20), halo=_np.array(((1, 2), (3, 4)))),
            hie.globale_to_locale_extent_h(gext)
        )

        gext = IndexingExtent(start=(10, 3), stop=(32, 20))
        self.assertEqual(
            IndexingExtent(start=(1, 3), stop=(23, 20)),
            hie.globale_to_locale_extent_h(gext)
        )

        lext = HaloIndexingExtent(start=(1, 3), stop=(23, 20), halo=_np.array(((1, 2), (3, 4))))
        self.assertEqual(
            hie.ndim,
            hie.locale_to_globale_extent_h(lext).ndim
        )
        self.assertEqual(
            hie,
            hie.locale_to_globale_extent_h(lext)
        )

        lext = IndexingExtent(start=(1, 3), stop=(23, 20))
        self.assertEqual(
            IndexingExtent(start=hie.start, stop=hie.stop),
            hie.locale_to_globale_extent_h(lext)
        )

    def test_globale_and_locale_slice_conversion(self):
        """
        Test for :meth:`mpi_array.indexing.HaloIndexingExtent.globale_to_locale_slice_h`,
        and :meth:`mpi_array.indexing.HaloIndexingExtent.locale_to_globale_slice_h`.
        """
        hie = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))
        gext = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))
        self.assertEqual(
            HaloIndexingExtent(
                start=(1, 3), stop=(23, 20), halo=_np.array(((1, 2), (3, 4)))
            ).to_slice_h(),
            hie.globale_to_locale_slice_h(gext.to_slice_h())
        )
        self.assertEqual(
            HaloIndexingExtent(
                start=(0, 0), stop=(22, 17), halo=_np.array(((1, 2), (3, 4)))
            ).to_slice_n(),
            hie.globale_to_locale_slice_n(gext.to_slice_n())
        )

        lext = HaloIndexingExtent(start=(1, 3), stop=(23, 20), halo=_np.array(((1, 2), (3, 4))))
        self.assertEqual(
            hie.to_slice_h(),
            hie.locale_to_globale_slice_h(lext.to_slice_h())
        )
        lext = HaloIndexingExtent(start=(0, 0), stop=(22, 17), halo=_np.array(((1, 2), (3, 4))))
        self.assertEqual(
            hie.to_slice_n(),
            hie.locale_to_globale_slice_n(lext.to_slice_n())
        )

    def test_to_slice(self):
        """
        :obj:`unittest.TestCase` for :obj:`mpi_array.indexing.HaloIndexingExtent`
        methods: :samp:`to_slice`, :samp:`to_slice_n`, and :samp:`to_slice_h`.
        """
        hie1 = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))
        self.assertSequenceEqual(
            (slice(10, 32, None), slice(3, 20, None)),
            hie1.to_slice_n()
        )
        self.assertSequenceEqual(
            (slice(10, 32, None), slice(3, 20, None)),
            hie1.to_slice()
        )
        self.assertSequenceEqual(
            (slice(9, 34, None), slice(0, 24, None)),
            hie1.to_slice_h()
        )

    def test_start_stop_shape(self):
        """
        :obj:`unittest.TestCase` for :obj:`mpi_array.indexing.HaloIndexingExtent`
        attributes: :samp:`start`, :samp:`stop`, and :samp:`shape`.
        """
        hie1 = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))

        self.assertSequenceEqual(hie1.start_n.tolist(), hie1.start.tolist())
        self.assertSequenceEqual(hie1.stop_n.tolist(), hie1.stop.tolist())
        self.assertSequenceEqual(hie1.shape_n.tolist(), hie1.shape.tolist())

    def test_calc_intersection_split(self):
        """
        Tests for :obj:`mpi_array.indexing.calc_intersection_split`.
        """
        def update_factory(dst_extent, src_extent, intersection):
            return [(dst_extent, src_extent, intersection), ]

        dst_extent = HaloIndexingExtent(start=(4, 50), stop=(50, 100), halo=4)
        src_extent = HaloIndexingExtent(start=(50, 46), stop=(150, 104), halo=4)
        update_dst_halo = False
        leftovers, updates = \
            calc_intersection_split(
                dst_extent,
                src_extent,
                update_factory,
                update_dst_halo
            )
        self.assertEqual(0, len(updates))
        self.assertEqual(1, len(leftovers))
        self.assertTrue(leftovers[0] is dst_extent)

        dst_extent = HaloIndexingExtent(start=(4, 50), stop=(50, 100), halo=4)
        src_extent = HaloIndexingExtent(start=(50, 46), stop=(150, 104), halo=4)
        update_dst_halo = True
        leftovers, updates = \
            calc_intersection_split(
                dst_extent,
                src_extent,
                update_factory,
                update_dst_halo
            )
        self.assertEqual(1, len(updates))
        self.assertEqual(
            IndexingExtent(start=(50, 46), stop=(54, 104)),
            updates[0][2]
        )
        self.assertEqual(1, len(leftovers))
        self.assertEqual(
            HaloIndexingExtent(start=(0, 46), stop=(50, 104)),
            leftovers[0]
        )

        dst_extent = HaloIndexingExtent(start=(4, 50), stop=(50, 100), halo=4)
        src_extent = HaloIndexingExtent(start=(20, 30), stop=(32, 128), halo=4)
        update_dst_halo = False
        leftovers, updates = \
            calc_intersection_split(
                dst_extent,
                src_extent,
                update_factory,
                update_dst_halo
            )
        self.assertEqual(1, len(updates))
        self.assertEqual(
            IndexingExtent(start=(20, 50), stop=(32, 100)),
            updates[0][2]
        )
        self.assertEqual(2, len(leftovers))
        self.assertEqual(
            HaloIndexingExtent(start=(4, 50), stop=(20, 100), halo=src_extent.halo),
            leftovers[0]
        )
        self.assertEqual(
            HaloIndexingExtent(start=(32, 50), stop=(50, 100), halo=src_extent.halo),
            leftovers[1]
        )


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
