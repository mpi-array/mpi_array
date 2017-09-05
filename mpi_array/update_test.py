"""
=======================================
The :mod:`mpi_array.update_test` Module
=======================================

Module defining :mod:`mpi_array.update` unit-tests.
Execute as::

   python -m mpi_array.update_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   MpiPairExtentUpdateTest - Tests for :obj:`mpi_array.update.MpiPairExtentUpdate`.
   MpiHaloSingleExtentUpdateTest - Tests :obj:`mpi_array.update.MpiHaloSingleExtentUpdate`.
   HalosUpdateTest - Test mpi_array.update.HalosUpdate`.
   UpdatesForRedistributeTest -Tests :obj:`mpi_array.update.UpdatesForRedistribute`.

"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from array_split import shape_split as _shape_split

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401

from .indexing import IndexingExtent
from .distribution import CartLocaleExtent, GlobaleExtent, BlockPartition
from .update import MpiHaloSingleExtentUpdate, HalosUpdate
from .update import MpiPairExtentUpdate, UpdatesForRedistribute

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class MpiPairExtentUpdateTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.distribution.MpiPairExtentUpdate`.
    """

    def setUp(self):
        self.se = \
            CartLocaleExtent(
                peer_rank=0,
                inter_locale_rank=0,
                cart_coord=(0,),
                cart_shape=(2,),
                globale_extent=GlobaleExtent(stop=(100,)),
                slice=(slice(0, 100),),
                halo=((10, 10),)
            )
        self.de = \
            CartLocaleExtent(
                peer_rank=1,
                inter_locale_rank=1,
                cart_coord=(1,),
                cart_shape=(2,),
                globale_extent=GlobaleExtent(stop=(100,)),
                slice=(slice(100, 200),),
                halo=((10, 10),)
            )
        self.due = IndexingExtent(start=(90,), stop=(100,))
        self.sue = IndexingExtent(start=(90,), stop=(100,))

    def test_construct(self):
        """
        Tests for :meth:`mpi_array.distribution.MpiPairExtentUpdate.__init__`.
        """
        se = self.se
        de = self.de
        due = self.due
        sue = self.sue

        u = MpiPairExtentUpdate(de, se, due, sue)
        self.assertTrue(u.dst_extent is de)
        self.assertTrue(u.src_extent is se)
        self.assertTrue(u.dst_update_extent is due)
        self.assertTrue(u.src_update_extent is sue)

    def test_str(self):
        """
        Tests for :meth:`mpi_array.distribution.MpiPairExtentUpdate.__str__`.
        """
        se = self.se
        de = self.de
        due = self.due
        sue = self.sue

        u = MpiPairExtentUpdate(de, se, due, sue)
        self.assertTrue(len(str(u)) > 0)

        u.initialise_data_types(dst_dtype="int32", src_dtype="int32", dst_order="C", src_order="C")
        self.assertTrue(len(str(u)) > 0)

    def test_data_type(self):
        """
        Tests for :meth:`mpi_array.distribution.MpiHaloSingleExtentUpdate.__str__`.
        """
        se = self.se
        de = self.de
        due = self.due
        sue = self.sue

        u = MpiPairExtentUpdate(de, se, due, sue)
        u.initialise_data_types(dst_dtype="int32", src_dtype="int32", dst_order="C", src_order="C")
        self.assertTrue(u.dst_data_type is not None)
        self.assertTrue(isinstance(u.dst_data_type, _mpi.Datatype))
        self.assertTrue(u.src_data_type is not None)
        self.assertTrue(isinstance(u.src_data_type, _mpi.Datatype))

        ddt = u.dst_data_type
        sdt = u.src_data_type
        u.initialise_data_types(dst_dtype="int32", src_dtype="int32", dst_order="C", src_order="C")
        self.assertTrue(u.dst_data_type is ddt)
        self.assertTrue(u.src_data_type is sdt)

        ddt = u.dst_data_type
        sdt = u.src_data_type
        u.initialise_data_types(dst_dtype="int32", src_dtype="int32", dst_order="F", src_order="F")
        self.assertTrue(u.dst_data_type is not ddt)
        self.assertTrue(u.src_data_type is not sdt)


class MpiHaloSingleExtentUpdateTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.distribution.MpiHaloSingleExtentUpdate`.
    """

    def setUp(self):
        self.se = \
            CartLocaleExtent(
                peer_rank=0,
                inter_locale_rank=0,
                cart_coord=(0,),
                cart_shape=(2,),
                globale_extent=GlobaleExtent(stop=(100,)),
                slice=(slice(0, 100),),
                halo=((10, 10),)
            )
        self.de = \
            CartLocaleExtent(
                peer_rank=1,
                inter_locale_rank=1,
                cart_coord=(1,),
                cart_shape=(2,),
                globale_extent=GlobaleExtent(stop=(100,)),
                slice=(slice(100, 200),),
                halo=((10, 10),)
            )
        self.ue = IndexingExtent(start=(90,), stop=(100,))

    def test_construct(self):
        """
        Tests for :meth:`mpi_array.distribution.MpiHaloSingleExtentUpdate.__init__`.
        """
        se = self.se
        de = self.de
        ue = self.ue

        u = MpiHaloSingleExtentUpdate(de, se, ue)
        self.assertTrue(u.dst_extent is de)
        self.assertTrue(u.src_extent is se)
        self.assertTrue(u.update_extent is ue)

    def test_str(self):
        """
        Tests for :meth:`mpi_array.distribution.MpiHaloSingleExtentUpdate.__str__`.
        """
        se = self.se
        de = self.de
        ue = self.ue

        u = MpiHaloSingleExtentUpdate(de, se, ue)
        self.assertTrue(len(str(u)) > 0)

        u.initialise_data_types(dtype="int32", order="C")
        self.assertTrue(len(str(u)) > 0)

    def test_data_type(self):
        """
        Tests for :meth:`mpi_array.distribution.MpiHaloSingleExtentUpdate.__str__`.
        """
        se = self.se
        de = self.de
        ue = self.ue

        u = MpiHaloSingleExtentUpdate(de, se, ue)
        u.initialise_data_types(dtype="int32", order="C")
        self.assertTrue(u.dst_data_type is not None)
        self.assertTrue(isinstance(u.dst_data_type, _mpi.Datatype))
        self.assertTrue(u.src_data_type is not None)
        self.assertTrue(isinstance(u.src_data_type, _mpi.Datatype))

        ddt = u.dst_data_type
        sdt = u.src_data_type
        u.initialise_data_types(dtype="int32", order="C")
        self.assertTrue(u.dst_data_type is ddt)
        self.assertTrue(u.src_data_type is sdt)

        ddt = u.dst_data_type
        sdt = u.src_data_type
        u.initialise_data_types(dtype="int32", order="F")
        self.assertTrue(u.dst_data_type is not ddt)
        self.assertTrue(u.src_data_type is not sdt)


class HalosUpdateTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.distribution.HalosUpdate`.
    """

    def setUp(self):
        self.se = \
            CartLocaleExtent(
                peer_rank=0,
                inter_locale_rank=0,
                cart_coord=(0,),
                cart_shape=(2,),
                globale_extent=GlobaleExtent(stop=(100,)),
                slice=(slice(0, 100),),
                halo=((10, 10),)
            )
        self.de = \
            CartLocaleExtent(
                peer_rank=1,
                inter_locale_rank=1,
                cart_coord=(1,),
                cart_shape=(2,),
                globale_extent=GlobaleExtent(stop=(100,)),
                slice=(slice(100, 200),),
                halo=((10, 10),)
            )
        self.ue = IndexingExtent(start=(90,), stop=(100,))

    def test_construct(self):
        """
        Tests for :meth:`mpi_array.distribution.HalosUpdate.__init__`.
        """
        rank_to_extent_dict = \
            {
                self.se.cart_rank: self.se,
                self.de.cart_rank: self.de
            }
        hu = HalosUpdate(self.de.cart_rank, rank_to_extent_dict)
        self.assertEqual(1, len(hu.updates_per_axis))
        self.assertEqual(2, len(hu.updates_per_axis[0]))
        self.assertEqual(0, len(hu.updates_per_axis[0][1]))
        self.assertEqual(1, len(hu.updates_per_axis[0][0]))
        self.assertTrue(self.de is hu.updates_per_axis[0][0][0].dst_extent)
        self.assertTrue(self.se is hu.updates_per_axis[0][0][0].src_extent)
        self.assertEqual(self.ue, hu.updates_per_axis[0][0][0].update_extent)


class UpdatesForRedistributeTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.update.UpdatesForRedistribute`.
    """

    def setUp(self):
        """
        Sets :samp:`self.rank_logger` attribute.
        """
        self.rank_logger = _logging.get_rank_logger(self.id())

    def test_slab_to_block(self):
        """
        Tests :obj:`mpi_array.update.UpdatesForRedistribute` by calling
        the :obj:`mpi_array.update.UpdatesForRedistribute.check_updates` method.
        """
        num_peer_ranks = 64
        num_peer_ranks_per_node = 4
        num_nodes = num_peer_ranks // num_peer_ranks_per_node

        gshape = (num_peer_ranks * 16, num_peer_ranks * 16)

        d_proc_blck_dims = _shape_split(gshape, num_peer_ranks).shape
        d_proc_blck_cart_ranks = _np.array(range(0, num_peer_ranks))
        coords = _np.array(_np.unravel_index(d_proc_blck_cart_ranks, tuple(d_proc_blck_dims))).T
        coords = [tuple(c) for c in coords]
        d_proc_blck_cc2cr = {coords[cart_rank]: cart_rank for cart_rank in d_proc_blck_cart_ranks}
        d_proc_blck_ilr2pr = _np.array(range(0, num_peer_ranks))
        d_proc_blck = \
            BlockPartition(
                gshape,
                d_proc_blck_dims,
                d_proc_blck_cc2cr,
                inter_locale_rank_to_peer_rank=d_proc_blck_ilr2pr
            )
        d_proc_blck_prpl = \
            _np.arange(0, d_proc_blck.num_locales).reshape((d_proc_blck.num_locales, 1))
        d_proc_blck.peer_ranks_per_locale = d_proc_blck_prpl

        d_node_slab_dims = _shape_split(gshape, num_nodes, axis=0).shape
        d_node_slab_cart_ranks = _np.array(range(0, num_nodes))
        coords = _np.array(_np.unravel_index(d_node_slab_cart_ranks, tuple(d_node_slab_dims))).T
        coords = [tuple(c) for c in coords]
        d_node_slab_cc2cr = {coords[cart_rank]: cart_rank for cart_rank in d_node_slab_cart_ranks}
        d_node_slab_ilr2pr = _np.array(range(0, num_nodes))
        d_node_slab_ilr2pr *= num_peer_ranks_per_node
        d_node_slab = \
            BlockPartition(
                gshape,
                d_node_slab_dims,
                d_node_slab_cc2cr,
                inter_locale_rank_to_peer_rank=d_node_slab_ilr2pr
            )
        d_node_slab_prpl = \
            _np.arange(0, num_peer_ranks).reshape(
                (d_node_slab.num_locales, num_peer_ranks_per_node)
            )
        d_node_slab.peer_ranks_per_locale = d_node_slab_prpl

        class RankTranslator(object):

            def dst_to_src(self, ranks):
                return _np.array(ranks, copy=True)

            def src_to_dst(self, ranks):
                return _np.array(ranks, copy=True)

        self.rank_logger.info("BEG: u4r = UpdatesForRedistribute(d_proc_blck, d_node_slab)...")
        u4r = UpdatesForRedistribute(d_proc_blck, d_node_slab)
        self.rank_logger.info("END: UpdatesForRedistribute(d_proc_blck, d_node_slab).")
        self.rank_logger.info("BEG: u4r.check_updates()...")
        u4r.check_updates()
        self.rank_logger.info("END: u4r.check_updates()...")

        self.rank_logger.info(
            "BEG: u4r = UpdatesForRedistribute(d_proc_blck, d_node_slab, RankTranslator())..."
        )
        u4r = UpdatesForRedistribute(d_proc_blck, d_node_slab, RankTranslator())
        self.rank_logger.info(
            "END: u4r = UpdatesForRedistribute(d_proc_blck, d_node_slab, RankTranslator())."
        )
        self.rank_logger.info("BEG: u4r.check_updates()...")
        u4r.check_updates()
        self.rank_logger.info("END: u4r.check_updates()...")

        self.rank_logger.info("BEG: u4r = UpdatesForRedistribute(d_node_slab, d_proc_blck)...")
        u4r = UpdatesForRedistribute(d_node_slab, d_proc_blck)
        self.rank_logger.info("END: u4r = UpdatesForRedistribute(d_node_slab, d_proc_blck).")
        self.rank_logger.info("BEG: u4r.check_updates()...")
        u4r.check_updates()
        self.rank_logger.info("END: u4r.check_updates()...")

        self.rank_logger.info(
            "BEG: u4r = UpdatesForRedistribute(d_node_slab, d_proc_blck, RankTranslator())..."
        )
        u4r = UpdatesForRedistribute(d_node_slab, d_proc_blck, RankTranslator())
        self.rank_logger.info(
            "END: u4r = UpdatesForRedistribute(d_node_slab, d_proc_blck, RankTranslator())..."
        )
        self.rank_logger.info("BEG: u4r.check_updates()...")
        u4r.check_updates()
        self.rank_logger.info("END: u4r.check_updates()...")


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
