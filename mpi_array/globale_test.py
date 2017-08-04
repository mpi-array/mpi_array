"""
========================================
The :mod:`mpi_array.globale_test` Module
========================================

Module defining :mod:`mpi_array.globale` unit-tests.
Execute as::

   python -m mpi_array.globale_test

and with parallelism::

   mpirun -n  2 python -m mpi_array.globale_test
   mpirun -n  4 python -m mpi_array.globale_test
   mpirun -n 27 python -m mpi_array.globale_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   GndarrayTest - Tests for :obj:`mpi_array.globale.gndarray`.


"""
from __future__ import absolute_import

import mpi_array.globale
import mpi4py.MPI as _mpi
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .comms import create_distribution, LT_PROCESS, LT_NODE, DT_SLAB, DT_BLOCK
from .distribution import IndexingExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class GndarrayTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.globale.gndarray`.
    """

    def test_attr(self):
        """
        Test various attributes of :obj:`mpi_array.globale.gndarray`.
        """
        halos = [0, 3]
        for halo in halos:
            gshape = (50, 17, 23)
            cand = create_distribution(gshape, locale_type=LT_PROCESS, halo=halo)
            locale_comms = cand.locale_comms
            gary = mpi_array.globale.zeros(gshape, comms_and_distrib=cand, dtype="int8")
            locale_comms.rank_logger.info("all zero gary.rank_view_h = %s" % (gary.rank_view_h,))
            rank_val = locale_comms.peer_comm.rank + 1
            gary.rank_view_n[...] = rank_val
            cand.locale_comms.rank_logger.info(
                "rank_val gary.rank_view_h = %s" %
                (gary.rank_view_h,))

            self.assertEqual(gary.dtype, _np.dtype("int8"))
            self.assertSequenceEqual(list(gary.shape), list(gshape))
            self.assertTrue(gary.comms_and_distrib is not None)
            self.assertTrue(gary.lndarray_proxy is not None)
            self.assertTrue(isinstance(gary.lndarray_proxy, mpi_array.locale.LndarrayProxy))
            self.assertEqual("C", gary.order)
            self.assertTrue(gary.rank_logger is not None)
            self.assertTrue(isinstance(gary.rank_logger, _logging.Logger))
            self.assertTrue(gary.root_logger is not None)
            self.assertTrue(isinstance(gary.root_logger, _logging.Logger))
            self.assertTrue(_np.all(gary.rank_view_n == rank_val))
            if _np.any(gary.rank_view_h.shape > gary.rank_view_n.shape):
                cand.locale_comms.rank_logger.info("gary.rank_view_h = %s" % (gary.rank_view_h,))
                self.assertTrue(
                    _np.all(_np.where(gary.rank_view_h == rank_val, 0, gary.rank_view_h) == 0)
                )

    def test_get_item_and_set_item(self):
        """
        Test the :meth:`mpi_array.globale.gndarray.__getitem__`
        and :meth:`mpi_array.globale.gndarray.__setitem__` methods.
        """
        gary = mpi_array.globale.zeros((20, 20, 20), dtype="int8")
        gary[1, 2, 8] = 22
        gary[1:10, 2:4, 4:8]
        gary[...] = 19
        gary[:] = 101

    def test_update(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.update`, 1D and 2D distribution.
        """

        halo = 4
        for lshape in ((100, 200), (1000,), ):
            gshape = (_mpi.COMM_WORLD.size * lshape[0],) + lshape[1:]
            cand_lt_process = \
                create_distribution(
                    shape=gshape,
                    distrib_type=DT_SLAB,
                    axis=0,
                    locale_type=LT_PROCESS,
                    halo=halo
                )

            lshape = (10000,) + lshape[1:]
            gshape = (cand_lt_process.locale_comms.num_locales * lshape[0],) + lshape[1:]
            halo = 4
            cand_lt_node = \
                create_distribution(
                    shape=gshape,
                    distrib_type=DT_SLAB,
                    axis=0,
                    locale_type=LT_NODE,
                    halo=halo
                )

            cand_list = [cand_lt_process, cand_lt_node]

            for cand in cand_list:
                gary = mpi_array.globale.empty(comms_and_distrib=cand, dtype="int32")
                self.assertEqual(_np.dtype("int32"), gary.dtype)

                if gary.locale_comms.have_valid_inter_locale_comm:
                    cart_rank_val = gary.locale_comms.cart_comm.rank + 1
                    gary.lndarray_proxy.view_h[...] = 0
                    gary.lndarray_proxy.view_n[...] = cart_rank_val

                    if gary.locale_comms.cart_comm.size > 1:
                        if gary.locale_comms.cart_comm.rank == 0:
                            self.assertTrue(_np.all(gary.lndarray_proxy[-halo:] == 0))
                        elif (
                            gary.locale_comms.inter_locale_comm.rank
                            ==
                            (gary.locale_comms.inter_locale_comm.size - 1)
                        ):
                            self.assertTrue(_np.all(gary.lndarray_proxy[0:halo] == 0))
                        else:
                            self.assertTrue(_np.all(gary.lndarray_proxy[0:halo] == 0))
                            self.assertTrue(_np.all(gary.lndarray_proxy[-halo:] == 0))

                gary.update()

                if gary.locale_comms.have_valid_inter_locale_comm:

                    self.assertTrue(_np.all(gary.lndarray_proxy.view_n[...] == cart_rank_val))

                    if gary.locale_comms.cart_comm.size > 1:
                        if gary.locale_comms.cart_comm.rank == 0:
                            self.assertTrue(
                                _np.all(gary.lndarray_proxy[-halo:] == (cart_rank_val + 1))
                            )
                        elif (
                            gary.locale_comms.cart_comm.rank
                            ==
                            (gary.locale_comms.cart_comm.size - 1)
                        ):
                            self.assertTrue(
                                _np.all(gary.lndarray_proxy[0:halo] == (cart_rank_val - 1))
                            )
                        else:
                            self.assertTrue(
                                _np.all(
                                    gary.lndarray_proxy[0:halo]
                                    ==
                                    (cart_rank_val - 1)
                                )
                            )
                            self.assertTrue(
                                _np.all(
                                    gary.lndarray_proxy[-halo:]
                                    ==
                                    (cart_rank_val + 1)
                                )
                            )
                gary.locale_comms.intra_locale_comm.barrier()

    def test_empty_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.empty` and :func:`mpi_array.globale.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape)

        gary = mpi_array.globale.empty(comms_and_distrib=cand, dtype="int64")

        self.assertEqual(_np.dtype("int64"), gary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(gary.lndarray_proxy.rank_view_slice_n).shape)
        )

        gary1 = mpi_array.globale.empty_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(gary1.lndarray_proxy.rank_view_slice_n).shape)
        )

        ary = mpi_array.globale.empty_like(_np.zeros(lshape, dtype="int64"))
        self.assertEqual(_np.dtype("int64"), ary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(ary.shape)
        )

    def test_empty_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.empty` and :func:`mpi_array.globale.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary = mpi_array.globale.empty(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        self.assertSequenceEqual(list(lshape), list(gary.lndarray_proxy.shape))
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(gary.lndarray_proxy.rank_view_slice_n).shape)
        )

        gary1 = mpi_array.globale.empty_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        self.assertSequenceEqual(list(lshape), list(gary1.lndarray_proxy.shape))
        self.assertSequenceEqual(
            list(lshape),
            list(IndexingExtent(gary1.lndarray_proxy.rank_view_slice_n).shape)
        )

    def test_zeros_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.zeros` and :func:`mpi_array.globale.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_NODE)

        gary = mpi_array.globale.zeros(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 0).all())

        gary1 = mpi_array.globale.zeros_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 0).all())

    def test_zeros_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.zeros` and :func:`mpi_array.globale.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary = mpi_array.globale.zeros(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 0).all())

        gary1 = mpi_array.globale.zeros_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 0).all())

    def test_ones_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.ones` and :func:`mpi_array.globale.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape)

        gary = mpi_array.globale.ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 1).all())

        gary1 = mpi_array.globale.ones_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 1).all())

    def test_ones_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.ones` and :func:`mpi_array.globale.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary = mpi_array.globale.ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 1).all())

        gary1 = mpi_array.globale.ones_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 1).all())

    def test_copy_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(gshape)

        gary = mpi_array.globale.ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.rank_view_n[...] = gary.locale_comms.peer_comm.rank

        gary1 = mpi_array.globale.copy(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == gary).all())

    def test_copy_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(gshape, locale_type=LT_PROCESS)

        gary = mpi_array.globale.ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.rank_view_n[...] = gary.locale_comms.peer_comm.rank

        gary1 = mpi_array.globale.copy(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == gary).all())

    def test_all(self):
        """
        Tests for :meth:`mpi_array.globale.gndarray.all`.
        """
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary0 = mpi_array.globale.zeros(comms_and_distrib=cand, dtype="int64")
        gary1 = mpi_array.globale.ones(comms_and_distrib=cand, dtype="int64")
        self.assertFalse((gary0 == gary1).all())

    def do_test_copyto(self, halo=0, dst_dtype="int32", src_dtype="int32"):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        lshape = (128, 128)
        gshape = (_mpi.COMM_WORLD.size * lshape[0], _mpi.COMM_WORLD.size * lshape[1])
        cand_src = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_SLAB,
                axis=0,
                locale_type=LT_PROCESS,
                halo=halo
            )

        gary_src = mpi_array.globale.zeros(comms_and_distrib=cand_src, dtype=src_dtype)
        rank_val = gary_src.comms_and_distrib.this_locale.inter_locale_rank + 1
        gary_src.rank_view_n[...] = rank_val
        gary_src.update()

        cand_dst = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_SLAB,
                axis=1,
                locale_type=LT_PROCESS,
                halo=halo
            )
        gary_dst = mpi_array.globale.zeros(comms_and_distrib=cand_dst, dtype=dst_dtype)
        gary_dst.update()
        self.assertTrue(_np.all(gary_dst.lndarray_proxy.lndarray[...] == 0))

        if gary_src.locale_comms.peer_comm.size <= 1:
            self.assertSequenceEqual(gary_src.lndarray_proxy.shape, gary_dst.lndarray_proxy.shape)
        else:
            self.assertTrue(
                _np.any(_np.array(gary_src.lndarray_proxy.shape) != gary_dst.lndarray_proxy.shape)
            )

        mpi_array.globale.copyto(gary_dst, gary_src)

        for le0 in gary_src.distribution.locale_extents:
            intersection_extent = le0.calc_intersection(gary_dst.lndarray_proxy.locale_extent)
            rank_val = le0.cart_rank + 1
            self.assertNotEqual(0, rank_val)
            locale_slice = \
                gary_dst.lndarray_proxy.locale_extent.globale_to_locale_extent_h(
                    intersection_extent
                ).to_slice()
            self.assertTrue(_np.all(_np.array(intersection_extent.shape) > 0))
            self.assertTrue(_np.all(gary_dst.lndarray_proxy[locale_slice] == rank_val))

    def test_copyto_no_halo(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto(halo=0)

    def test_copyto_halo(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto(halo=4)

    def do_test_copyto_different_locale_types(
        self,
        halo=0,
        node_slab_dtype="int32",
        proc_blok_dtype="int32"
    ):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        lshape = (128, 128)
        gshape = (_mpi.COMM_WORLD.size * lshape[0], _mpi.COMM_WORLD.size * lshape[1])

        cand_node_slab = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_SLAB,
                axis=0,
                locale_type=LT_NODE,
                halo=halo
            )

        cand_proc_blok = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_BLOCK,
                locale_type=LT_PROCESS,
                halo=halo
            )

        gary_node_slab = \
            mpi_array.globale.zeros(comms_and_distrib=cand_node_slab, dtype=node_slab_dtype)
        rank_val = gary_node_slab.comms_and_distrib.this_locale.inter_locale_rank + 1
        gary_node_slab.rank_view_n[...] = rank_val
        gary_node_slab.update()
        gary_proc_blok = \
            mpi_array.globale.zeros(comms_and_distrib=cand_proc_blok, dtype=proc_blok_dtype)

        mpi_array.globale.copyto(gary_proc_blok, gary_node_slab, casting="unsafe")
        gary_node_slab0 = mpi_array.globale.zeros_like(gary_node_slab)
        mpi_array.globale.copyto(gary_node_slab0, gary_proc_blok, casting="unsafe")

        gary_node_slab0.update()

        gary_proc_blok.locale_comms.rank_logger.info(
            "num diffs = %s",
            _np.sum(
                gary_node_slab.rank_view_n[...] != gary_node_slab0.rank_view_n[...],
                dtype="int64"
            )
        )
        self.assertTrue(
            _np.all(gary_node_slab.rank_view_h[...] == gary_node_slab0.rank_view_h[...])
        )
        del gary_node_slab0, gary_node_slab, gary_proc_blok

        # Try in copyto in opposite order

        gary_node_slab = \
            mpi_array.globale.zeros(comms_and_distrib=cand_node_slab, dtype=node_slab_dtype)

        gary_proc_blok = \
            mpi_array.globale.zeros(comms_and_distrib=cand_proc_blok, dtype=proc_blok_dtype)

        rank_val = gary_proc_blok.comms_and_distrib.this_locale.inter_locale_rank + 1
        gary_proc_blok.rank_view_n[...] = rank_val
        gary_proc_blok.update()

        mpi_array.globale.copyto(gary_node_slab, gary_proc_blok, casting="unsafe")
        gary_proc_blok0 = mpi_array.globale.zeros_like(gary_proc_blok)
        mpi_array.globale.copyto(gary_proc_blok0, gary_node_slab, casting="unsafe")

        gary_proc_blok0.update()

        gary_proc_blok.locale_comms.rank_logger.info(
            "num diffs = %s",
            _np.sum(
                gary_proc_blok.rank_view_n[...] != gary_proc_blok0.rank_view_n[...],
                dtype="int64"
            )
        )
        self.assertTrue(
            _np.all(gary_proc_blok.rank_view_h[...] == gary_proc_blok0.rank_view_h[...])
        )

    def test_copyto_different_locale_types_no_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_different_locale_types(
            halo=0,
            node_slab_dtype="int32",
            proc_blok_dtype="int32"
        )

    def test_copyto_different_locale_types_wt_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_different_locale_types(
            halo=4,
            node_slab_dtype="int32",
            proc_blok_dtype="int32"
        )

    def test_copyto_different_locale_types_no_halo_diff_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_different_locale_types(
            halo=0,
            node_slab_dtype="uint16",
            proc_blok_dtype="int64"
        )

    def test_copyto_different_locale_types_wt_halo_diff_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_different_locale_types(
            halo=8,
            node_slab_dtype="float32",
            proc_blok_dtype="uint64"
        )

    def test_copyto_arg_check(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        gary = mpi_array.globale.zeros(shape=(10, 10, 10), dtype="uint64")

        self.assertRaises(ValueError, mpi_array.globale.copyto, gary, [1, ])
        self.assertRaises(ValueError, mpi_array.globale.copyto, [1, ], gary)
        self.assertRaises(ValueError, mpi_array.globale.copyto, [1, ], [1, ])


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
