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


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array

import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from mpi_array.indexing import IndexingExtent
from mpi_array.distribution import CartLocaleExtent, GlobaleExtent
from mpi_array.update import MpiHaloSingleExtentUpdate, HalosUpdate
from mpi_array.update import MpiPairExtentUpdate

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


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


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
