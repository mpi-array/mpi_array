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

   IndexingExtentTest - Tests for :obj:`mpi_array.decomposition.IndexingExtent`.
   MemNodeTopologyTest - Tests for :obj:`mpi_array.decomposition.MemNodeTopology`.
   DecompositionTest - Tests for :obj:`mpi_array.decomposition.Decomposition`.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from mpi_array.decomposition import IndexingExtent, DecompExtent, MemNodeTopology, Decomposition

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class IndexingExtentTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.decomposition.IndexingExtentTest`.
    """

    def testAttributes(self):
        """
        Tests :attr:`mpi_array.decomposition.IndexingExtent.start`
        and :attr:`mpi_array.decomposition.IndexingExtent.stop`
        and :attr:`mpi_array.decomposition.IndexingExtent.shape`
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

    def testIntersection1d(self):
        """
        Tests :meth:`mpi_array.decomposition.IndexingExtent.calc_intersection` method, 1D indexing.
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


class DecompExtentTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.decomposition.DecompExtent`.
    """

    def testConstructAttribs(self):
        """
        """
        de = \
            DecompExtent(
                cart_rank=0,
                cart_coord=(0,),
                cart_shape=(1,),
                slice=(slice(0, 100),),
                halo=((10, 10),)
            )

        self.assertEqual(0, de.cart_rank)
        self.assertTrue(_np.all(de.cart_coord == (0,)))
        self.assertTrue(_np.all(de.cart_shape == (1,)))
        self.assertTrue(_np.all(de.halo == 0))

    def testExtentCalcs(self):
        """
        """
        de = \
            [
                DecompExtent(
                    cart_rank=r,
                    cart_coord=(r,),
                    cart_shape=(3,),
                    slice=(slice(r * 100, r * 100 + 100),),
                    halo=((10, 10),)
                )
                for r in range(0, 3)
            ]

        self.assertEqual(0, de[0].cart_rank)
        self.assertTrue(_np.all(de[0].cart_coord == (0,)))
        self.assertTrue(_np.all(de[0].cart_shape == (3,)))
        self.assertTrue(_np.all(de[0].halo == ((0, 10),)))

        self.assertEqual(1, de[1].cart_rank)
        self.assertTrue(_np.all(de[1].cart_coord == (1,)))
        self.assertTrue(_np.all(de[1].cart_shape == (3,)))
        self.assertTrue(_np.all(de[1].halo == ((10, 10),)))

        self.assertEqual(2, de[2].cart_rank)
        self.assertTrue(_np.all(de[2].cart_coord == (2,)))
        self.assertTrue(_np.all(de[2].cart_shape == (3,)))
        self.assertTrue(_np.all(de[2].halo == ((10, 0),)))


class MemNodeTopologyTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.decomposition.MemNodeTopology`.
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
        self.assertNotEqual(_mpi.COMM_WORLD, _mpi.COMM_NULL)


class DecompositionTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.decomposition.Decomposition`.
    """

    def testConstruct1d(self):
        """
        Test :obj:`mpi_array.decomposition.Decomposition` construction.
        """
        decomp = Decomposition((8 * _mpi.COMM_WORLD.size,))
        self.assertNotEqual(None, decomp._mem_node_topology)

        mnt = MemNodeTopology(ndims=1, shared_mem_comm=_mpi.COMM_SELF)
        decomp = \
            Decomposition((8 * _mpi.COMM_WORLD.size,), mem_node_topology=mnt)

        root_logger = _logging.get_root_logger(self.id(), comm=decomp.rank_comm)
        root_logger.info("START " + self.id())
        root_logger.info(str(decomp))
        root_logger.info("END   " + self.id())

    def testConstruct1dWithHalo(self):
        """
        Test :obj:`mpi_array.decomposition.Decomposition` construction.
        """
        decomp = Decomposition((8 * _mpi.COMM_WORLD.size,), halo=((2, 4),))
        self.assertNotEqual(None, decomp._mem_node_topology)

        mnt = MemNodeTopology(ndims=1, shared_mem_comm=_mpi.COMM_SELF)
        decomp = \
            Decomposition((8 * _mpi.COMM_WORLD.size,), halo=((2, 4),), mem_node_topology=mnt)

        root_logger = _logging.get_root_logger(self.id(), comm=decomp.rank_comm)
        root_logger.info("START " + self.id())
        root_logger.info(str(decomp))
        root_logger.info("END   " + self.id())

    def testConstruct2d(self):
        """
        Test :obj:`mpi_array.decomposition.Decomposition` construction.
        """
        decomp = Decomposition((8 * _mpi.COMM_WORLD.size, 12 * _mpi.COMM_WORLD.size))
        self.assertNotEqual(None, decomp._mem_node_topology)

        mnt = MemNodeTopology(ndims=2, shared_mem_comm=_mpi.COMM_SELF)
        decomp = \
            Decomposition(
                (8 * _mpi.COMM_WORLD.size, 12 * _mpi.COMM_WORLD.size),
                mem_node_topology=mnt
            )

        root_logger = _logging.get_root_logger(self.id(), comm=decomp.rank_comm)
        root_logger.info("START " + self.id())
        root_logger.info(str(decomp))
        root_logger.info("END   " + self.id())

    def testConstruct2dWithHalo(self):
        """
        Test :obj:`mpi_array.decomposition.Decomposition` construction.
        """
        decomp = \
            Decomposition(
                (8 * _mpi.COMM_WORLD.size, 12 * _mpi.COMM_WORLD.size),
                halo=((2, 2), (4, 4))
            )
        self.assertNotEqual(None, decomp._mem_node_topology)

        mnt = MemNodeTopology(ndims=2, shared_mem_comm=_mpi.COMM_SELF)
        decomp = \
            Decomposition(
                (8 * _mpi.COMM_WORLD.size, 12 * _mpi.COMM_WORLD.size),
                halo=((1, 2), (3, 4)),
                mem_node_topology=mnt
            )

        root_logger = _logging.get_root_logger(self.id(), comm=decomp.rank_comm)
        root_logger.info("START " + self.id())
        root_logger.info(str(decomp))
        root_logger.info("END   " + self.id())


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
