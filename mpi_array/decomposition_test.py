"""
==============================================
The :mod:`mpi_array.decomposition_test` Module
==============================================

Module defining :mod:`mpi_array.decomposition` unit-tests.
Execute as::

   python -m mpi_array.decomposition_test


Classes
=======

.. autosummary::
   :toctree: generated/

   MemNodeTopologyTest - Test case for :mod:`mpi_array.decomposition.MemNodeTopology`.
   DecompositionTest - Test case for :mod:`mpi_array.decomposition.Decomposition`.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from mpi_array.decomposition import MemNodeTopology, Decomposition

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class MemNodeTopologyTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :mod:`mpi_array.decomposition.MemNodeTopology`.
    """

    def testConstructInvalidDims(self):
        mnt = None
        with self.assertRaises(ValueError):
            mnt = MemNodeTopology()
        with self.assertRaises(ValueError):
            mnt = MemNodeTopology(ndims=None, dims=None)
        with self.assertRaises(ValueError):
            mnt = MemNodeTopology(dims=tuple(), ndims=1)
        with self.assertRaises(ValueError):
            mnt = MemNodeTopology(dims=tuple([0, 2]), ndims=1)
        with self.assertRaises(ValueError):
            mnt = MemNodeTopology(dims=tuple([1, 2]), ndims=3)

        self.assertEqual(None, mnt)

    def testConstructShared(self):
        mnt = MemNodeTopology(ndims=1)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, mnt.rank_comm))

        mnt = MemNodeTopology(ndims=4)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, mnt.rank_comm))

        mnt = MemNodeTopology(dims=(0,))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, mnt.rank_comm))

        mnt = MemNodeTopology(dims=(0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, mnt.rank_comm))

        mnt = MemNodeTopology(dims=(0, 0, 0))
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, mnt.rank_comm))

    def testConstructNoShared(self):
        mnt = MemNodeTopology(ndims=1, shared_mem_comm=_mpi.COMM_SELF)
        self.assertEqual(_mpi.IDENT, _mpi.Comm.Compare(_mpi.COMM_WORLD, mnt.rank_comm))
        self.assertEqual(1, mnt.shared_mem_comm.size)


class DecompositionTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :mod:`mpi_array.decomposition.Decomposition`.
    """

    def testConstruct(self):
        """
        Test :obj:`mpi_array.decomposition.Decomposition` construction.
        """
        decomp = Decomposition((8 * _mpi.COMM_WORLD.size,))
        self.assertNotEqual(None, decomp._mem_node_topology)

        decomp = \
            Decomposition((8 * _mpi.COMM_WORLD.size,))


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
