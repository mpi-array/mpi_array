"""
======================================
The :mod:`mpi_array.local_test` Module
======================================

Module defining :mod:`mpi_array.local` unit-tests.
Execute as::

   python -m mpi_array.local_test


Classes
=======

.. autosummary::
   :toctree: generated/

   LndarrayTest - Tests for :obj:`mpi_array.local.lndarray`.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from mpi_array.decomposition import CartesianDecomposition, MemAllocTopology, IndexingExtent
import mpi_array.local

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class LndarrayTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.local.lndarray`.
    """

    def test_empty_shared_1d(self):
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        decomp = CartesianDecomposition(shape=gshape)

        lary = mpi_array.local.empty(decomp=decomp, dtype="int64")

        self.assertEqual(_np.dtype("int64"), lary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(lary.decomp.rank_view_slice_n).shape)
        )

    def test_empty_non_shared_1d(self):
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        mat = MemAllocTopology(ndims=1, rank_comm=_mpi.COMM_WORLD, shared_mem_comm=_mpi.COMM_SELF)
        decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat)

        lary = mpi_array.local.empty(decomp=decomp, dtype="int64")

        self.assertEqual(_np.dtype("int64"), lary.dtype)
        self.assertSequenceEqual(list(lshape), list(lary.shape))
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(lary.decomp.rank_view_slice_n).shape)
        )

    def test_zeros_shared_1d(self):
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        decomp = CartesianDecomposition(shape=gshape)

        lary = mpi_array.local.zeros(decomp=decomp, dtype="int64")

        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 0))

    def test_zeros_non_shared_1d(self):
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        mat = MemAllocTopology(ndims=1, rank_comm=_mpi.COMM_WORLD, shared_mem_comm=_mpi.COMM_SELF)
        decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat)

        lary = mpi_array.local.zeros(decomp=decomp, dtype="int64")

        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 0))

    def test_ones_shared_1d(self):
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        decomp = CartesianDecomposition(shape=gshape)

        lary = mpi_array.local.ones(decomp=decomp, dtype="int64")

        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 1))

    def test_ones_non_shared_1d(self):
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        mat = MemAllocTopology(ndims=1, rank_comm=_mpi.COMM_WORLD, shared_mem_comm=_mpi.COMM_SELF)
        decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat)

        lary = mpi_array.local.ones(decomp=decomp, dtype="int64")

        self.assertEqual(_np.dtype("int64"), lary.dtype)
        lary.decomp.rank_comm.barrier()
        self.assertTrue(_np.all(lary == 1))

_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
