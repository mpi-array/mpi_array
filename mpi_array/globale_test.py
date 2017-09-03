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

import unittest as _builtin_unittest
import mpi_array.globale
import mpi4py.MPI as _mpi
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .comms import create_distribution, LT_PROCESS, LT_NODE, DT_SLAB, DT_BLOCK
from . import globale_creation as _globale_creation

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
            gary = _globale_creation.zeros(gshape, comms_and_distrib=cand, dtype="int32")
            rank_val = locale_comms.peer_comm.rank + 1
            gary.rank_view_n[...] = rank_val

            self.assertEqual(gary.dtype, _np.dtype("int32"))
            self.assertSequenceEqual(list(gary.shape), list(gshape))
            self.assertTrue(gary.comms_and_distrib is not None)
            self.assertTrue(gary.lndarray_proxy is not None)
            self.assertTrue(isinstance(gary.lndarray_proxy, mpi_array.locale.LndarrayProxy))
            self.assertEqual("C", gary.order)
            self.assertTrue(gary.rank_logger is not None)
            self.assertTrue(isinstance(gary.rank_logger, _logging.Logger))
            self.assertTrue(gary.root_logger is not None)
            self.assertTrue(isinstance(gary.root_logger, _logging.Logger))
            gary.rank_logger.info(
                "halo=%s, gary.rank_view_n.shape=%s",
                halo,
                gary.rank_view_n.shape
            )
            gary.rank_logger.info(
                "gary.rank_view_n=\n%s",
                gary.rank_view_n
            )
            self.assertEqual(
                _np.product(gary.rank_view_n.shape),
                _np.sum((gary.rank_view_n == rank_val).astype("uint32"), dtype="int64")
            )
            if _np.any(gary.rank_view_h.shape > gary.rank_view_n.shape):
                cand.locale_comms.rank_logger.info("gary.rank_view_h = %s" % (gary.rank_view_h,))
                self.assertEqual(
                    0,
                    _np.sum(
                        _np.where(gary.rank_view_h == rank_val, 0, gary.rank_view_h),
                        dtype="int64"
                    )
                )

    def test_get_item_and_set_item(self):
        """
        Test the :meth:`mpi_array.globale.gndarray.__getitem__`
        and :meth:`mpi_array.globale.gndarray.__setitem__` methods.
        """
        gary = _globale_creation.zeros((20, 20, 20), dtype="int8")
        gary[1, 2, 8] = 22
        gary[1:10, 2:4, 4:8]
        gary[...] = 19
        gary[:] = 101

    def test_update(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.update`, 1D and 2D distribution.
        """

        halo = 4
        for lshape in ((100,), (10, 20), ):
            gshape = (_mpi.COMM_WORLD.size * lshape[0],) + lshape[1:]
            cand_lt_process = \
                create_distribution(
                    shape=gshape,
                    distrib_type=DT_SLAB,
                    axis=0,
                    locale_type=LT_PROCESS,
                    halo=halo
                )

            gshape = (cand_lt_process.locale_comms.num_locales * lshape[0],) + lshape[1:]
            if len(lshape) > 1:
                lshape = (lshape[0],) + (cand_lt_process.locale_comms.num_locales * lshape[1],)

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
                gary = _globale_creation.empty(comms_and_distrib=cand, dtype="int32")
                gary.rank_logger.debug("gary.shape=%s", gary.shape)
                gary.rank_logger.debug(
                    "gary.locale_comms.num_locales=%s",
                    gary.locale_comms.num_locales
                )
                gary.rank_logger.debug(
                    "gary.locale_comms.dims=%s",
                    gary.locale_comms.dims
                )

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

    def test_all(self):
        """
        Tests for :meth:`mpi_array.globale.gndarray.all`.
        """
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary0 = _globale_creation.zeros(comms_and_distrib=cand, dtype="int64")
        gary1 = _globale_creation.ones(comms_and_distrib=cand, dtype="int64")
        self.assertFalse((gary0 == gary1).all())

    def do_test_copyto_same_locale_types(self, halo=0, dst_dtype="int32", src_dtype="int32"):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        lshape = (16, 16)
        gshape = (_mpi.COMM_WORLD.size * lshape[0], _mpi.COMM_WORLD.size * lshape[1])
        locale_type = LT_PROCESS
        if _mpi.COMM_WORLD.size >= 128:
            locale_type = LT_NODE
        cand_slab_ax0 = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_SLAB,
                axis=0,
                locale_type=locale_type,
                halo=halo
            )

        gary_slb_ax0 = _globale_creation.zeros(comms_and_distrib=cand_slab_ax0, dtype=src_dtype)
        rank_val = gary_slb_ax0.comms_and_distrib.this_locale.inter_locale_rank + 1
        gary_slb_ax0.rank_view_n[...] = rank_val

        cand_slb_ax1 = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_SLAB,
                axis=1,
                locale_type=locale_type,
                halo=halo
            )
        gary_slb_ax1 = _globale_creation.zeros(comms_and_distrib=cand_slb_ax1, dtype=dst_dtype)
        self.assertTrue(_np.all(gary_slb_ax1.lndarray_proxy.lndarray[...] == 0))

        if gary_slb_ax0.locale_comms.peer_comm.size <= 1:
            self.assertSequenceEqual(
                gary_slb_ax0.lndarray_proxy.shape, gary_slb_ax1.lndarray_proxy.shape
            )
        else:
            self.assertTrue(
                _np.any(
                    _np.array(gary_slb_ax0.lndarray_proxy.shape)
                    !=
                    gary_slb_ax1.lndarray_proxy.shape
                )
            )

        mpi_array.globale.copyto(gary_slb_ax1, gary_slb_ax0)
        gary_slb_ax0_0 = mpi_array.zeros_like(gary_slb_ax0)
        mpi_array.globale.copyto(gary_slb_ax0_0, gary_slb_ax1)

        gary_slb_ax0.locale_comms.rank_logger.info(
            "num diffs = %s",
            _np.sum(
                gary_slb_ax0.rank_view_n[...] != gary_slb_ax0_0.rank_view_n[...],
                dtype="int64"
            )
        )
        self.assertTrue(
            _np.all(gary_slb_ax0.rank_view_n[...] == gary_slb_ax0_0.rank_view_n[...])
        )

        del gary_slb_ax0, gary_slb_ax0_0, gary_slb_ax1

        gary_slb_ax0 = _globale_creation.zeros(comms_and_distrib=cand_slab_ax0, dtype=src_dtype)

        gary_slb_ax1 = _globale_creation.zeros(comms_and_distrib=cand_slb_ax1, dtype=dst_dtype)
        rank_val = gary_slb_ax1.comms_and_distrib.this_locale.inter_locale_rank + 1
        gary_slb_ax1.rank_view_n[...] = rank_val

        mpi_array.globale.copyto(gary_slb_ax0, gary_slb_ax1)
        gary_slb_ax1_1 = mpi_array.zeros_like(gary_slb_ax1)
        mpi_array.globale.copyto(gary_slb_ax1_1, gary_slb_ax0)

        gary_slb_ax1.locale_comms.rank_logger.info(
            "num diffs = %s",
            _np.sum(
                gary_slb_ax1.rank_view_n[...] != gary_slb_ax1_1.rank_view_n[...],
                dtype="int64"
            )
        )
        self.assertTrue(
            _np.all(gary_slb_ax1.rank_view_n[...] == gary_slb_ax1_1.rank_view_n[...])
        )

    def test_copyto_same_locale_types_no_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_same_locale_types(halo=0)

    def test_copyto_same_locale_types_wt_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_same_locale_types(halo=4)

    def do_test_copyto_diff_locale_types(
        self,
        halo=0,
        node_slab_dtype="int32",
        proc_blok_dtype="int32"
    ):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        lshape = (16, 16)
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
            _globale_creation.zeros(comms_and_distrib=cand_node_slab, dtype=node_slab_dtype)
        rank_val = gary_node_slab.comms_and_distrib.this_locale.inter_locale_rank + 1
        gary_node_slab.rank_view_n[...] = rank_val
        # gary_node_slab.update()
        gary_proc_blok = \
            _globale_creation.zeros(comms_and_distrib=cand_proc_blok, dtype=proc_blok_dtype)

        mpi_array.globale.copyto(gary_proc_blok, gary_node_slab, casting="unsafe")
        gary_node_slab0 = _globale_creation.zeros_like(gary_node_slab)
        mpi_array.globale.copyto(gary_node_slab0, gary_proc_blok, casting="unsafe")

        # gary_node_slab0.update()

        gary_proc_blok.locale_comms.rank_logger.info(
            "num diffs = %s",
            _np.sum(
                gary_node_slab.rank_view_n[...] != gary_node_slab0.rank_view_n[...],
                dtype="int64"
            )
        )
        self.assertTrue(
            _np.all(gary_node_slab.rank_view_n[...] == gary_node_slab0.rank_view_n[...])
        )
        del gary_node_slab0, gary_node_slab, gary_proc_blok

        # Try in copyto in opposite order

        gary_node_slab = \
            _globale_creation.zeros(comms_and_distrib=cand_node_slab, dtype=node_slab_dtype)

        gary_proc_blok = \
            _globale_creation.zeros(comms_and_distrib=cand_proc_blok, dtype=proc_blok_dtype)

        rank_val = gary_proc_blok.comms_and_distrib.this_locale.inter_locale_rank + 1
        gary_proc_blok.rank_view_n[...] = rank_val
        # gary_proc_blok.update()

        mpi_array.globale.copyto(gary_node_slab, gary_proc_blok, casting="unsafe")
        gary_proc_blok0 = _globale_creation.zeros_like(gary_proc_blok)
        mpi_array.globale.copyto(gary_proc_blok0, gary_node_slab, casting="unsafe")

        # gary_proc_blok0.update()

        gary_proc_blok.locale_comms.rank_logger.info(
            "num diffs = %s",
            _np.sum(
                gary_proc_blok.rank_view_n[...] != gary_proc_blok0.rank_view_n[...],
                dtype="int64"
            )
        )
        self.assertTrue(
            _np.all(gary_proc_blok.rank_view_n[...] == gary_proc_blok0.rank_view_n[...])
        )

    def test_copyto_diff_locale_types_no_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_diff_locale_types(
            halo=0,
            node_slab_dtype="int32",
            proc_blok_dtype="int32"
        )

    def test_copyto_diff_locale_types_wt_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_diff_locale_types(
            halo=4,
            node_slab_dtype="int32",
            proc_blok_dtype="int32"
        )

    def test_copyto_diff_locale_types_no_halo_diff_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_diff_locale_types(
            halo=0,
            node_slab_dtype="uint16",
            proc_blok_dtype="int64"
        )

    def test_copyto_diff_locale_types_wt_halo_diff_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_diff_locale_types(
            halo=8,
            node_slab_dtype="float32",
            proc_blok_dtype="uint64"
        )

    def test_copyto_arg_check(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        gary = _globale_creation.zeros(shape=(10, 10, 10), dtype="uint64")

        self.assertRaises(ValueError, mpi_array.globale.copyto, gary, [1, ])
        self.assertRaises(ValueError, mpi_array.globale.copyto, [1, ], gary)
        self.assertRaises(ValueError, mpi_array.globale.copyto, [1, ], [1, ])


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
