"""
=======================================
The :mod:`mpi_array.locale_test` Module
=======================================

Module defining :mod:`mpi_array.locale` unit-tests.
Execute as::

   python -m mpi_array.locale_test


Classes
=======

.. autosummary::
   :toctree: generated/

   LndarrayTest - Tests for :obj:`mpi_array.locale.lndarray`.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array
from array_split.split import shape_factors as _shape_factors
import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from mpi_array.decomposition import CartesianDecomposition, MemAllocTopology, IndexingExtent
import mpi_array.locale

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class SlndarrayTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.locale.slndarray`.
    """

    def test_construct(self):
        """
        Tests :meth:`mpi_array.locale.slndarray.__new__`.
        """
        gshape = (11, 13, 51)
        lary = mpi_array.locale.ones(shape=gshape, dtype="int16")

        slary = lary.slndarray

        # lary.decomp owns data
        self.assertTrue(slary.flags.carray)
        self.assertFalse(slary.flags.owndata)
        self.assertFalse(slary.base.flags.owndata)
        self.assertTrue(slary.base.flags.carray)

        lshape = slary.shape
        bad_lshape = list(lshape)
        bad_lshape[-1] += 1
        self.assertRaises(ValueError, mpi_array.locale.slndarray,
                          shape=bad_lshape, decomp=lary.decomp)

    def test_view(self):
        """
        Tests :meth:`mpi_array.locale.slndarray.__getitem__`.
        """
        gshape = (11, 13, 51)
        lary = mpi_array.locale.ones(shape=gshape, dtype="int16")
        slary = lary.slndarray

        # lary.decomp owns data
        self.assertTrue(slary.flags.carray)
        self.assertFalse(slary.flags.owndata)
        self.assertFalse(slary.base.flags.owndata)
        self.assertTrue(slary.base.flags.carray)

        v = lary[0:slary.shape[0] // 2, 0:slary.shape[1] // 2, 0:slary.shape[2] // 2]
        self.assertTrue(isinstance(v, mpi_array.locale.slndarray))
        self.assertFalse(v.flags.owndata)
        self.assertFalse(v.flags.carray)
        self.assertTrue(isinstance(v.base, mpi_array.locale.slndarray))
        self.assertFalse(v.base.flags.owndata)
        self.assertTrue(v.base.flags.carray)
        self.assertFalse(isinstance(v.base.base, mpi_array.locale.slndarray))
        self.assertTrue(isinstance(v.base.base, _np.ndarray))
        self.assertFalse(v.base.base.flags.owndata)
        self.assertTrue(v.base.base.flags.carray)

    def test_numpy_sum(self):
        """
        Test :func:`numpy.sum` reduction using a :obj:`mpi_array.locale.slndarray`
        as argument.
        """
        gshape = (50, 50, 50)
        lary = mpi_array.locale.ones(shape=gshape, dtype="int32")

        slary = lary.slndarray
        l_sum = _np.sum(lary.rank_view_n)
        self.assertFalse(l_sum.flags.owndata)
        self.assertTrue(l_sum.base.flags.owndata)

        lary.decomp.rank_logger.info("type(slary)=%s", type(slary))
        lary.decomp.rank_logger.info("type(slary.base)=%s", type(slary.base))
        lary.decomp.rank_logger.info("type(l_sum)=%s", type(l_sum))
        lary.decomp.rank_logger.info("type(l_sum.base)=%s", type(l_sum.base))

        g_sum = lary.decomp.rank_comm.allreduce(l_sum, op=_mpi.SUM)

        self.assertEqual(_np.product(gshape), g_sum)


class LndarrayTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.locale.lndarray`.
    """

    def test_construct_arg_checking(self):
        """
        Test for :meth:`mpi_array.locale.lndarray.__new__`.
        """
        self.assertRaises(
            ValueError,
            mpi_array.locale.lndarray,
            shape=None,
            decomp=None,
            dtype="int64"
        )

        self.assertRaises(
            ValueError,
            mpi_array.locale.lndarray,
            shape=(100, 100, 100),
            decomp=CartesianDecomposition(shape=(100, 99, 100)),
            dtype="int64"
        )

        self.assertRaises(
            ValueError,
            mpi_array.locale.lndarray,
            shape=None,
            decomp=CartesianDecomposition(shape=(100,)),
            dtype="int64",
            buffer=[0] * 100
        )

        lary = mpi_array.locale.lndarray(shape=(64, 32, 16), dtype="int32")
        self.assertRaises(
            NotImplementedError,
            lary.__reduce__
        )

    def test_get_and_set_item(self):
        """
        """
        lary = mpi_array.locale.empty(shape=(24, 35, 14, 7), dtype="int64")
        rank_val = lary.decomp.rank_comm.rank + 1
        lary[lary.rank_view_slice_n] = rank_val

        self.assertSequenceEqual(
            list(IndexingExtent(lary.rank_view_slice_n).shape),
            list(lary[lary.rank_view_slice_n].shape)
        )
        self.assertTrue(_np.all(lary[lary.rank_view_slice_n] == rank_val))

        self.assertSequenceEqual(
            list(IndexingExtent(lary.rank_view_slice_h).shape),
            list(lary[lary.rank_view_slice_h].shape)
        )

    def test_empty_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.empty` and :func:`mpi_array.locale.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        decomp = CartesianDecomposition(shape=gshape)

        lary = mpi_array.locale.empty(decomp=decomp, dtype="int64")

        self.assertEqual(_np.dtype("int64"), lary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(lary.decomp.rank_view_slice_n).shape)
        )

        lary1 = mpi_array.locale.empty_like(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(lary1.decomp.rank_view_slice_n).shape)
        )

        ary = mpi_array.locale.empty_like(_np.zeros(lshape, dtype="int64"))
        self.assertEqual(_np.dtype("int64"), ary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(ary.shape)
        )

    def test_empty_non_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.empty` and :func:`mpi_array.locale.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        mat = MemAllocTopology(ndims=1, rank_comm=_mpi.COMM_WORLD, shared_mem_comm=_mpi.COMM_SELF)
        decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat)

        lary = mpi_array.locale.empty(decomp=decomp, dtype="int64")
        self.assertEqual(_np.dtype("int64"), lary.dtype)
        self.assertSequenceEqual(list(lshape), list(lary.shape))
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(lary.decomp.rank_view_slice_n).shape)
        )

        lary1 = mpi_array.locale.empty_like(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        self.assertSequenceEqual(list(lshape), list(lary1.shape))
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(lary1.decomp.rank_view_slice_n).shape)
        )

    def test_zeros_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.zeros` and :func:`mpi_array.locale.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        decomp = CartesianDecomposition(shape=gshape)

        lary = mpi_array.locale.zeros(decomp=decomp, dtype="int64")
        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 0))

        lary1 = mpi_array.locale.zeros_like(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary1 == 0))

    def test_zeros_non_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.zeros` and :func:`mpi_array.locale.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        mat = MemAllocTopology(ndims=1, rank_comm=_mpi.COMM_WORLD, shared_mem_comm=_mpi.COMM_SELF)
        decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat)

        lary = mpi_array.locale.zeros(decomp=decomp, dtype="int64")
        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 0))

        lary1 = mpi_array.locale.zeros_like(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary1 == 0))

    def test_ones_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.ones` and :func:`mpi_array.locale.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        decomp = CartesianDecomposition(shape=gshape)

        lary = mpi_array.locale.ones(decomp=decomp, dtype="int64")
        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 1))

        lary1 = mpi_array.locale.ones_like(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary1 == 1))

    def test_ones_non_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.ones` and :func:`mpi_array.locale.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        mat = MemAllocTopology(ndims=1, rank_comm=_mpi.COMM_WORLD, shared_mem_comm=_mpi.COMM_SELF)
        decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat)

        lary = mpi_array.locale.ones(decomp=decomp, dtype="int64")
        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 1))

        lary1 = mpi_array.locale.ones_like(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary1 == 1))

    def test_copy_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        decomp = CartesianDecomposition(shape=gshape)

        lary = mpi_array.locale.ones(decomp=decomp, dtype="int64")
        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.rank_view_n[...] = lary.decomp.rank_comm.rank

        lary1 = mpi_array.locale.copy(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary1 == lary))

    def test_copy_non_shared_1d(self):
        """
        Test for :func:`mpi_array.locale.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        mat = MemAllocTopology(ndims=1, rank_comm=_mpi.COMM_WORLD, shared_mem_comm=_mpi.COMM_SELF)
        decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat)

        lary = mpi_array.locale.ones(decomp=decomp, dtype="int64")
        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.rank_view_n[...] = lary.decomp.rank_comm.rank

        lary1 = mpi_array.locale.copy(lary)
        self.assertEqual(_np.dtype("int64"), lary1.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary1 == lary))

    def test_views_2d(self):
        """
        Test for :meth:`mpi_array.locale.lndarray.rank_view_n`
        and :meth:`mpi_array.locale.lndarray.rank_view_h`.
        """

        lshape = _np.array((4, 3), dtype="int64")
        gshape = lshape * _shape_factors(_mpi.COMM_WORLD.size, lshape.size)[::-1]

        mats = \
            [
                None,
                MemAllocTopology(
                    ndims=gshape.size,
                    rank_comm=_mpi.COMM_WORLD,
                    shared_mem_comm=_mpi.COMM_SELF
                )
            ]
        for mat in mats:
            decomp = CartesianDecomposition(shape=gshape, halo=2, mem_alloc_topology=mat)

            lary = mpi_array.locale.ones(decomp=decomp, dtype="int64")
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            rank_logger = _logging.get_rank_logger(self.id(), comm=decomp.rank_comm)
            rank_logger.info(
                (
                    "\n========================================================\n" +
                    "lndarray_extent = %s\n" +
                    "rank_view_slice_n          = %s\n" +
                    "rank_view_slice_h          = %s\n" +
                    "rank_view_relative_slice_n = %s\n" +
                    "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                )
                %
                (
                    lary.decomp._lndarray_extent,
                    lary.decomp.rank_view_slice_n,
                    lary.decomp.rank_view_slice_h,
                    lary.decomp.rank_view_relative_slice_n,
                )
            )

            if lary.decomp.shared_mem_comm.rank == 0:
                lary.view_h[...] = -1
            lary.decomp.shared_mem_comm.barrier()

            lary.rank_view_n[...] = lary.decomp.rank_comm.rank
            lary.decomp.shared_mem_comm.barrier()
            if lary.decomp.shared_mem_comm.size > 1:
                self.assertTrue(_np.any(lary.rank_view_h != lary.decomp.rank_comm.rank))
            self.assertSequenceEqual(
                lary.rank_view_h[lary.decomp.rank_view_relative_slice_n].tolist(),
                lary.rank_view_n.tolist()
            )
            self.assertTrue(_np.all(lary.view_n >= 0))

            lary.decomp.rank_comm.barrier()


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
