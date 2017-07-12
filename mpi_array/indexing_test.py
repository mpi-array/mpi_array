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

   IndexingExtentTest - Tests for :obj:`mpi_array.indexing.IndexingExtent`.
   HaloIndexingExtentTest - Tests for :obj:`mpi_array.indexing.IndexingExtent`.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from mpi_array.indexing import IndexingExtent, HaloIndexingExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


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
        methods: :samp:`start`, :samp:`stop`, and :samp:`shape`.
        """
        hie1 = HaloIndexingExtent(start=(10, 3), stop=(32, 20), halo=_np.array(((1, 2), (3, 4))))

        self.assertSequenceEqual(hie1.start_n.tolist(), hie1.start.tolist())
        self.assertSequenceEqual(hie1.stop_n.tolist(), hie1.stop.tolist())
        self.assertSequenceEqual(hie1.shape_n.tolist(), hie1.shape.tolist())


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
