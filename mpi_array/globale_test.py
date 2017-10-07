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

import mpi4py.MPI as _mpi
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import globale as _globale
from . import locale as _locale
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
            self.assertTrue(isinstance(gary.lndarray_proxy, _locale.LndarrayProxy))
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

    def test_construct_with_structured_array_dtype(self):
        """
        Construct a
        a :obj:`mpi_array.globale.gndarray`
        `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_.
        """
        gshape = (15, 20, 11, 7)
        dt = _np.dtype([('float64', 'f8'), ('u32str', 'U32'), ('int32', 'i4'), ('u1', 'u1')])
        gary = _globale_creation.empty(shape=gshape, dtype=dt)
        self.assertEqual(dt, gary.dtype)
        self.assertSequenceEqual(gshape, tuple(gary.shape))

    def do_test_construct_empty_locale_extent(self, locale_type=LT_PROCESS, halo=0):
        """
        Test constructing a :obj:`mpi_array.globale.gndarray` when locale extent is empty.
        """
        # test empty gndarray
        halo_1d = halo
        if _np.array(halo_1d).ndim > 1:
            halo_1d = halo[0]

        gshp = (0, )
        gary = \
            _globale_creation.empty(
                shape=gshp, dtype="int32", locale_type=locale_type, halo=halo_1d
            )
        self.assertSequenceEqual(gshp, tuple(gary.shape))
        self.assertEqual(0, _np.product(gary.lndarray_proxy.locale_extent.shape_n))
        self.assertEqual(0, gary.lndarray_proxy.lndarray.size)
        self.assertEqual(0, gary.view_n.size)

        gshp = (15, 20, 11, 0)
        gary = \
            _globale_creation.empty(shape=gshp, dtype="int32", locale_type=locale_type, halo=halo)
        self.assertSequenceEqual(gshp, tuple(gary.shape))
        self.assertEqual(0, _np.product(gary.lndarray_proxy.locale_extent.shape_n))
        self.assertEqual(0, gary.lndarray_proxy.lndarray.size)
        self.assertEqual(0, gary.view_n.size)

        num_locales = gary.locale_comms.num_locales

        # test non-empty gndarray but some empty locale extents
        if num_locales > 1:
            gshp = (15, 20, 11, num_locales // 2)
            dims = (1, 1, 1, 0)
            gary = \
                _globale_creation.ones(
                    shape=gshp,
                    dtype="int32",
                    locale_type=locale_type,
                    distrib_type=DT_BLOCK,
                    dims=dims,
                    halo=halo
                )
            self.assertSequenceEqual(gshp, tuple(gary.shape))
            self.assertTrue(
                _np.any(
                    _np.array(
                        tuple(
                            _np.product(locale_extent.shape_n)
                            for locale_extent in gary.distribution.locale_extents
                        )
                    )
                )
            )
            if _np.product(gary.lndarray_proxy.locale_extent.shape_n) == 0:
                gary.rank_logger.debug(
                    "gary.lndarray_proxy.locale_extent.shape_n=%s",
                    gary.lndarray_proxy.locale_extent.shape_n
                )
                gary.rank_logger.debug(
                    "gary.lndarray_proxy.locale_extent.start_n=%s",
                    gary.lndarray_proxy.locale_extent.start_n
                )
                gary.rank_logger.debug(
                    "gary.lndarray_proxy.locale_extent.stop_n=%s",
                    gary.lndarray_proxy.locale_extent.stop_n
                )
                gary.rank_logger.debug(
                    "gary.lndarray_proxy.locale_extent.shape_h=%s",
                    gary.lndarray_proxy.locale_extent.shape_h
                )
                gary.rank_logger.debug(
                    "gary.lndarray_proxy.locale_extent.start_h=%s",
                    gary.lndarray_proxy.locale_extent.start_h
                )
                gary.rank_logger.debug(
                    "gary.lndarray_proxy.locale_extent.stop_h=%s",
                    gary.lndarray_proxy.locale_extent.stop_h
                )
                gary.rank_logger.debug(
                    "gary.lndarray_proxy.locale_extent.halo=%s",
                    gary.lndarray_proxy.locale_extent.halo
                )
                self.assertTrue(_np.all(gary.lndarray_proxy.locale_extent.halo == 0))
                self.assertSequenceEqual(
                    tuple(gary.lndarray_proxy.locale_extent.stop_h),
                    tuple(gary.lndarray_proxy.locale_extent.stop_n),
                )
                self.assertSequenceEqual(
                    tuple(gary.lndarray_proxy.locale_extent.start_h),
                    tuple(gary.lndarray_proxy.locale_extent.start_n),
                )

                self.assertEqual(0, gary.view_n.size)
                self.assertEqual(0, gary.lndarray_proxy.lndarray.size)

    def test_construct_empty_locale_extent_locale_type_process(self):
        """
        Test constructing a :obj:`mpi_array.globale.gndarray` when locale extent is empty.
        """
        self.do_test_construct_empty_locale_extent(locale_type=LT_PROCESS, halo=0)
        self.do_test_construct_empty_locale_extent(
            locale_type=LT_PROCESS,
            halo=[[1, 2], [3, 4], [4, 3], [2, 1]]
        )

    def test_construct_empty_locale_extent_locale_type_node(self):
        """
        Test constructing a :obj:`mpi_array.globale.gndarray` when locale extent is empty.
        """
        self.do_test_construct_empty_locale_extent(locale_type=LT_NODE, halo=0)
        self.do_test_construct_empty_locale_extent(
            locale_type=LT_NODE,
            halo=[[1, 2], [3, 4], [4, 3], [2, 1]]
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
        Test for :meth:`mpi_array.globale.gndarray.update`, 1D and 2D shaped data
        with 1D distribution.
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

    def test_update_block(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.update`, block 2D distribution.
        """

        halo = (2, 0, 4)
        lshape = (10, 12, 8)
        shape_factor = max([1, int(_np.floor(_np.power(_mpi.COMM_WORLD.size, 1.0 / 3.0)))])
        gshape = \
            (
                shape_factor * lshape[0],
                shape_factor * lshape[1],
                shape_factor * lshape[2],
            )
        cand_lt_process = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_BLOCK,
                locale_type=LT_PROCESS,
                halo=halo
            )
        gary = _globale_creation.zeros(comms_and_distrib=cand_lt_process, dtype="int32")
        if gary.locale_comms.have_valid_inter_locale_comm:
            inter_locale_rank_val = gary.locale_comms.inter_locale_comm.rank + 1
            gary.lndarray_proxy.view_n[...] = inter_locale_rank_val
        gary.locale_comms.peer_comm.barrier()

        gary.update()

        LO = gary.lndarray_proxy.LO
        HI = gary.lndarray_proxy.HI
        lhalo = gary.lndarray_proxy.locale_extent.halo
        for axis in range(gary.ndim):
            for dir in [LO, HI]:
                halo_slab = gary.lndarray_proxy.locale_extent.halo_slab_extent(axis, dir)
                halo_slab_num_elems = _np.product(halo_slab.shape)
                self.assertTrue((lhalo[axis, dir] == 0) or (halo_slab_num_elems > 0))
                halo_slab = gary.lndarray_proxy.locale_extent.globale_to_locale_extent_h(halo_slab)
                self.assertTrue(
                    _np.all(
                        _np.logical_and(
                            gary.lndarray_proxy[halo_slab.to_slice()] != inter_locale_rank_val,
                            gary.lndarray_proxy[halo_slab.to_slice()] != 0
                        )
                    )
                )

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

    def do_test_copyto_same_locale_types(
        self,
        halo=0,
        dst_dtype="int32",
        src_dtype="int32",
        locale_type=LT_PROCESS
    ):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        if (_mpi.COMM_WORLD.size > 128) and (locale_type == LT_PROCESS):
            # Skip testing for LT_PROCESS when have large number of processes, slow...
            return
        lshape = (16, 16)
        gshape = (_mpi.COMM_WORLD.size * lshape[0], _mpi.COMM_WORLD.size * lshape[1])
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

        if gary_slb_ax0.locale_comms.num_locales <= 1:
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

        _globale.copyto(gary_slb_ax1, gary_slb_ax0)
        gary_slb_ax0_0 = _globale_creation.zeros_like(gary_slb_ax0)
        _globale.copyto(gary_slb_ax0_0, gary_slb_ax1)

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

        _globale.copyto(gary_slb_ax0, gary_slb_ax1)
        gary_slb_ax1_1 = _globale_creation.zeros_like(gary_slb_ax1)
        _globale.copyto(gary_slb_ax1_1, gary_slb_ax0)

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

    def test_copyto_same_node_locale_types_no_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_same_locale_types(halo=0, locale_type=LT_NODE)

    def test_copyto_same_node_locale_types_wt_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_same_locale_types(halo=4, locale_type=LT_NODE)

    def test_copyto_same_process_locale_types_no_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_same_locale_types(halo=0, locale_type=LT_PROCESS)

    def test_copyto_same_process_locale_types_wt_halo_same_dtype(self):
        """
        Tests for :func:`mpi_array.globale.copyto`.
        """
        self.do_test_copyto_same_locale_types(halo=4, locale_type=LT_PROCESS)

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

        _globale.copyto(gary_proc_blok, gary_node_slab, casting="unsafe")
        gary_node_slab0 = _globale_creation.zeros_like(gary_node_slab)
        gary_proc_blok.rank_logger.debug('*' * 60)
        _globale.copyto(gary_node_slab0, gary_proc_blok, casting="unsafe")

        # gary_node_slab0.update()

        diff_msk = gary_node_slab.view_n[...] != gary_node_slab0.view_n[...]
        num_diffs = int(_np.sum(diff_msk, dtype="int64"))
        if num_diffs > 0:
            max_elem_at_diffs = _np.max(gary_node_slab0.view_n[diff_msk])
            min_elem_at_diffs = _np.min(gary_node_slab0.view_n[diff_msk])
        else:
            max_elem_at_diffs = None
            min_elem_at_diffs = None
        gary_proc_blok.locale_comms.rank_logger.info(
            "num diffs = %s", (num_diffs,)
        )
        self.assertEqual(
            0,
            num_diffs,
            msg=(
                (
                    "gary_node_slab.view_n[...] != gary_node_slab0.view_n[...], "
                    +
                    "num different elements = %s, gary_node_slab.view_n[...].shape=%s"
                    +
                    ", min_elem_at_diffs=%s, max_elem_at_diffs=%s"
                )
                %
                (num_diffs, gary_node_slab.view_n[...].shape, min_elem_at_diffs, max_elem_at_diffs)
            )
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

        _globale.copyto(gary_node_slab, gary_proc_blok, casting="unsafe")
        gary_proc_blok0 = _globale_creation.zeros_like(gary_proc_blok)
        _globale.copyto(gary_proc_blok0, gary_node_slab, casting="unsafe")

        # gary_proc_blok0.update()

        diff_msk = gary_proc_blok.view_n[...] != gary_proc_blok0.view_n[...]
        num_diffs = int(_np.sum(diff_msk, dtype="int64"))
        if num_diffs > 0:
            max_elem_at_diffs = _np.max(gary_proc_blok0.view_n[diff_msk])
            min_elem_at_diffs = _np.min(gary_proc_blok0.view_n[diff_msk])
        else:
            max_elem_at_diffs = None
            min_elem_at_diffs = None
        gary_proc_blok.locale_comms.rank_logger.info(
            "num diffs = %s", num_diffs
        )
        self.assertEqual(
            0,
            num_diffs,
            msg=(
                (
                    "gary_proc_blok.view_n[...] != gary_proc_blok0.view_n[...], "
                    +
                    "num different elements = %s, gary_proc_blok.view_n[...].shape=%s"
                    +
                    ", min_elem_at_diffs=%s, max_elem_at_diffs=%s"
                )
                %
                (num_diffs, gary_proc_blok.view_n[...].shape, min_elem_at_diffs, max_elem_at_diffs)
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

        self.assertRaises(ValueError, _globale.copyto, gary, [1, ])
        self.assertRaises(ValueError, _globale.copyto, [1, ], gary)
        self.assertRaises(ValueError, _globale.copyto, [1, ], [1, ])


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
