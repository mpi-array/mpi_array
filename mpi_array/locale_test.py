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
   :template: autosummary/inherits_TestCase_class.rst

   WinLndarrayTest - Tests for :obj:`mpi_array.locale.win_lndarray`.
   LndarrayTest - Tests for :obj:`mpi_array.locale.lndarray`.
   LndarrayProxyTest - Tests for :obj:`mpi_array.locale.LndarrayProxy`.


"""
from __future__ import absolute_import

import weakref as _weakref
from array_split.split import shape_factors as _shape_factors
import mpi4py.MPI as _mpi
import numpy as _np  # noqa: E402,F401
from . import locale as _locale

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .comms import create_distribution, LT_PROCESS, LT_NODE
from .distribution import IndexingExtent, LocaleExtent
from .distribution import GlobaleExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class WinLndarrayTest(_unittest.TestCase):

    """
    Tests for :obj:`mpi_array.locale.win_lndarray`.
    """

    def test_construct_with_invalid_comm(self):
        """
        Tests that ValueError is raised for invalid communicator argument
        passed to :obj:`mpi_array.locale.win_lndarray` constructor.
        """
        comm = None
        self.assertRaises(ValueError, _locale.win_lndarray, shape=(100,), comm=comm)

        comm = _mpi.COMM_NULL
        self.assertRaises(ValueError, _locale.win_lndarray, shape=(100,), comm=comm)

    def test_construct(self):
        """
        Tests for :obj:`mpi_array.locale.win_lndarray` construction.
        """
        comm = _mpi.COMM_SELF
        ary = _locale.win_lndarray(shape=(10, 10, 10), dtype="int32", comm=comm)
        self.assertTrue(ary.comm is comm)
        self.assertTrue(ary.win is not None)

        comm = _mpi.COMM_WORLD
        if (_mpi.VERSION >= 3) and (comm.size > 1):
            comm = comm.Split_type(_mpi.COMM_TYPE_SHARED, key=comm.rank)
            if comm.size > 1:
                ary = _locale.win_lndarray(shape=(comm.size, 10, 10), dtype="int32", comm=comm)
                self.assertTrue(ary.comm is comm)
                self.assertTrue(ary.win is not None)
                ary[comm.rank] = comm.rank + 1
                comm.barrier()
                for r in range(0, comm.size):
                    self.assertTrue(_np.all(ary[r] == (r + 1)))

    def test_construct_with_buffer(self):
        """
        Tests :obj:`mpi_array.locale.win_lndarray` construction with a :samp:`buffer`
        argument (used to raise :obj:`UnboundLocalError`), no window is allocated.
        """
        buffer = bytearray(10 * 4)
        ary = _locale.win_lndarray(shape=(10,), dtype="int32", buffer=buffer, comm=_mpi.COMM_SELF)
        self.assertTrue(ary.win is None)
        ary[...] = _np.arange(10)
        self.assertSequenceEqual(list(range(10)), _np.frombuffer(buffer, dtype="int32").tolist())
        ary.free()
        self.assertSequenceEqual(list(range(10)), ary.tolist())

    def test_free(self):
        """
        Tests :meth:`mpi_array.locale.win_lndarray.free` (used to raise :obj:`ValueError`
        for any non-empty array) and the :samp:`with` statement. Freeing twice is allowed,
        and leaving a :samp:`with` block because of an exception skips the (collective) free.
        """
        world = _mpi.COMM_WORLD
        comms = [_mpi.COMM_SELF, world.Split_type(_mpi.COMM_TYPE_SHARED, key=world.rank)]
        for comm in comms:
            ary = _locale.win_lndarray(shape=(comm.size, 10), dtype="int32", comm=comm)
            win = ary.win
            self.assertNotEqual(_mpi.WIN_NULL, win)
            ary.free()
            self.assertEqual(_mpi.WIN_NULL, win)
            ary.free()

            with _locale.win_lndarray(shape=(comm.size, 10), dtype="int32", comm=comm) as ary:
                win = ary.win
                ary[comm.rank] = comm.rank
                comm.barrier()
                self.assertSequenceEqual(list(range(comm.size)), ary[:, 0].tolist())
                comm.barrier()
            self.assertEqual(_mpi.WIN_NULL, win)

            with self.assertRaises(RuntimeError):
                with _locale.win_lndarray(shape=(comm.size, 10), dtype="int32", comm=comm) as ary:
                    win = ary.win
                    raise RuntimeError("Raised on all processes.")
            self.assertNotEqual(_mpi.WIN_NULL, win)
            ary.free()
            self.assertEqual(_mpi.WIN_NULL, win)
        comms[1].Free()

    def test_derived_arrays_do_not_free_window(self):
        """
        Only the array which allocated the window frees it. A copy, an :samp:`astype`
        result or a ufunc output has its own memory and no window, a view shares the
        window. Their :meth:`mpi_array.locale.win_lndarray.free` used to free the window
        of the original array.
        """
        with _locale.win_lndarray(shape=(4,), dtype="float64", comm=_mpi.COMM_SELF) as ary:
            ary[...] = 1
            view = ary[1:]
            self.assertTrue(view.win is ary.win)
            self.assertTrue(view.comm is ary.comm)
            derived = (ary + 1, ary.copy(), ary.astype("float32"))
            for derived_ary in derived:
                self.assertTrue(derived_ary.win is None)
            for derived_ary in (view,) + derived:
                derived_ary.free()
                with derived_ary:
                    pass
            self.assertNotEqual(_mpi.WIN_NULL, ary.win)
            self.assertSequenceEqual([1.0, 1.0, 1.0, 1.0], ary.tolist())
        self.assertEqual(_mpi.WIN_NULL, ary.win)


class LndarrayTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.locale.lndarray`.
    """

    def test_construct(self):
        """
        Tests :meth:`mpi_array.locale.lndarray.__new__`.
        """
        gshape = (11, 13, 51)
        with _locale.ones(shape=gshape, dtype="int16") as lary:

            slary = lary.lndarray

            # MPI windows own buffer data
            self.assertTrue(slary.flags.carray)
            self.assertFalse(slary.flags.owndata)
            self.assertFalse(slary.base.flags.owndata)
            self.assertTrue(slary.base.flags.carray)

            lshape = slary.shape

        bad_lshape = list(lshape)
        bad_lshape[-1] += 1
        self.assertRaises(
            ValueError,
            _locale.lndarray,
            shape=lshape
        )

    def test_view(self):
        """
        Tests :meth:`mpi_array.locale.lndarray.__getitem__`.
        """
        gshape = (11, 13, 51)
        with _locale.ones(shape=gshape, dtype="int16") as lary:
            slary = lary.lndarray

            # MPI windows own buffer data
            self.assertTrue(slary.flags.carray)
            self.assertFalse(slary.flags.owndata)
            self.assertFalse(slary.base.flags.owndata)
            self.assertTrue(slary.base.flags.carray)

            v = lary[0:slary.shape[0] // 2, 0:slary.shape[1] // 2, 0:slary.shape[2] // 2]
            self.assertTrue(isinstance(v, _locale.lndarray))
            self.assertFalse(v.flags.owndata)
            self.assertFalse(
                ((v.size > 0) and (v.shape[0] > 1) and (v.shape[1] > 1))
                and
                v.flags.carray,
                "v.size=%s, v.shape=%s, v.flags.carray=%s" % (v.size, v.shape, v.flags.carray)
            )
            self.assertTrue(isinstance(v.base, _locale.lndarray))
            self.assertFalse(v.base.flags.owndata)
            self.assertTrue(v.base.flags.carray)
            self.assertFalse(isinstance(v.base.base, _locale.lndarray))
            self.assertTrue(isinstance(v.base.base, _np.ndarray))
            self.assertFalse(v.base.base.flags.owndata)
            self.assertTrue(v.base.base.flags.carray)

    def test_numpy_sum(self):
        """
        Test :func:`numpy.sum` reduction using a :obj:`mpi_array.locale.lndarray`
        as argument.
        """
        gshape = (50, 50, 50)
        comms_and_distrib = create_distribution(gshape)
        with \
                _locale.ones(shape=gshape, comms_and_distrib=comms_and_distrib, dtype="int32") \
                as lary:

            slary = lary.lndarray
            l_sum = _np.sum(lary.rank_view_n)
            self.assertFalse(l_sum.flags.owndata)
            self.assertTrue(l_sum.base.flags.owndata)

            rank_logger = _logging.get_rank_logger(__name__ + self.id())
            rank_logger.info("type(slary)=%s", type(slary))
            rank_logger.info("type(slary.base)=%s", type(slary.base))
            rank_logger.info("type(l_sum)=%s", type(l_sum))
            rank_logger.info("type(l_sum.base)=%s", type(l_sum.base))

            g_sum = comms_and_distrib.locale_comms.peer_comm.allreduce(l_sum, op=_mpi.SUM)

            self.assertEqual(_np.prod(gshape), g_sum)


class LndarrayProxyTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.locale.LndarrayProxy`.
    """

    def test_construct_arg_checking(self):
        """
        Test for :meth:`mpi_array.locale.LndarrayProxy.__new__`.
        """
        self.assertRaises(
            ValueError,
            _locale.LndarrayProxy,
            shape=None,
            locale_extent=None,
            dtype="int64"
        )

        gshape = (10, 11, 12, 13, 5)
        bad_lshape = gshape
        bad_lshape = (bad_lshape[0] - 1,) + bad_lshape[1:]
        self.assertRaises(
            ValueError,
            _locale.LndarrayProxy,
            shape=bad_lshape,
            locale_extent=LocaleExtent(
                peer_rank=_mpi.COMM_WORLD.rank,
                inter_locale_rank=_mpi.COMM_WORLD.rank,
                globale_extent=GlobaleExtent(stop=gshape),
                start=(0, 0, 0, 0, 0),
                stop=gshape
            )
        )

    def test_fill(self):
        """
        Test for :meth:`mpi_array.locale.LndarrayProxy.fill`.
        """
        comms_and_distrib = create_distribution(shape=(24, 35, 14, 7))
        with \
                _locale.empty(
                    shape=(24, 35, 14, 7),
                    comms_and_distrib=comms_and_distrib,
                    dtype="int64"
                ) as lary:

            rank_val = comms_and_distrib.locale_comms.peer_comm.rank + 1
            lary.fill(rank_val)
            self.assertTrue(_np.all(lary[lary.rank_view_slice_n] == rank_val))

    def test_get_and_set_item(self):
        """
        """
        comms_and_distrib = create_distribution(shape=(24, 35, 14, 7))
        with \
                _locale.empty(
                    shape=(24, 35, 14, 7),
                    comms_and_distrib=comms_and_distrib,
                    dtype="int64"
                ) as lary:

            rank_val = comms_and_distrib.locale_comms.peer_comm.rank + 1
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
        Test for :func:`_locale.empty` and :func:`_locale.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape)

        with _locale.empty(comms_and_distrib=cand, dtype="int64") as lary:

            self.assertEqual(_np.dtype("int64"), lary.dtype)
            self.assertSequenceEqual(
                list(lshape),
                list(IndexingExtent(lary.intra_partition.rank_view_slice_n).shape)
            )

            with _locale.empty_like(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                self.assertSequenceEqual(
                    list(lshape),
                    list(IndexingExtent(lary1.intra_partition.rank_view_slice_n).shape)
                )

        ary = _locale.empty_like(_np.zeros(lshape, dtype="int64"))
        self.assertEqual(_np.dtype("int64"), ary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(ary.shape)
        )

    def test_empty_non_shared_1d(self):
        """
        Test for :func:`_locale.empty` and :func:`_locale.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        with _locale.empty(comms_and_distrib=cand, dtype="int64") as lary:
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            self.assertSequenceEqual(list(lshape), list(lary.shape))
            self.assertSequenceEqual(
                list(lshape),
                list(IndexingExtent(lary.intra_partition.rank_view_slice_n).shape)
            )

            with _locale.empty_like(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                self.assertSequenceEqual(list(lshape), list(lary1.shape))
                self.assertSequenceEqual(
                    list(lshape),
                    list(IndexingExtent(lary1.intra_partition.rank_view_slice_n).shape)
                )

    def test_zeros_shared_1d(self):
        """
        Test for :func:`_locale.zeros` and :func:`_locale.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape)

        with _locale.zeros(comms_and_distrib=cand, dtype="int64") as lary:
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            cand.locale_comms.peer_comm.barrier()
            self.assertTrue(_np.all(lary == 0))

            with _locale.zeros_like(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                cand.locale_comms.peer_comm.barrier()
                self.assertTrue(_np.all(lary1 == 0))

    def test_zeros_non_shared_1d(self):
        """
        Test for :func:`_locale.zeros` and :func:`_locale.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        with _locale.zeros(comms_and_distrib=cand, dtype="int64") as lary:
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            cand.locale_comms.peer_comm.barrier()
            self.assertTrue(_np.all(lary == 0))

            with _locale.zeros_like(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                cand.locale_comms.peer_comm.barrier()
                self.assertTrue(_np.all(lary1 == 0))

    def test_ones_shared_1d(self):
        """
        Test for :func:`_locale.ones` and :func:`_locale.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape)

        with _locale.ones(comms_and_distrib=cand, dtype="int64") as lary:
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            cand.locale_comms.peer_comm.barrier()
            self.assertTrue(_np.all(lary == 1))

            with _locale.ones_like(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                cand.locale_comms.peer_comm.barrier()
                self.assertTrue(_np.all(lary1 == 1))

    def test_ones_non_shared_1d(self):
        """
        Test for :func:`_locale.ones` and :func:`_locale.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        with _locale.ones(comms_and_distrib=cand, dtype="int64") as lary:
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            cand.locale_comms.peer_comm.barrier()
            self.assertTrue(_np.all(lary == 1))

            with _locale.ones_like(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                cand.locale_comms.peer_comm.barrier()
                self.assertTrue(_np.all(lary1 == 1))

    def test_like_functions_dtype(self):
        """
        The :samp:`dtype` argument of :func:`mpi_array.locale.empty_like`,
        :func:`mpi_array.locale.zeros_like` and :func:`mpi_array.locale.ones_like`
        gives the type of the new array, it used to be ignored.
        """
        cand = create_distribution(shape=(10 * _mpi.COMM_WORLD.size,))
        with _locale.zeros(comms_and_distrib=cand, dtype="float64") as lary:
            for like_func, value in \
                    ((_locale.empty_like, None), (_locale.zeros_like, 0), (_locale.ones_like, 1)):
                with like_func(lary, dtype="int16") as like_lary:
                    self.assertEqual(_np.dtype("int16"), like_lary.dtype)
                    self.assertSequenceEqual(lary.shape, like_lary.shape)
                    if value is not None:
                        cand.locale_comms.intra_locale_comm.barrier()
                        self.assertTrue(_np.all(like_lary.view_n == value))
                with like_func(lary) as like_lary:
                    self.assertEqual(_np.dtype("float64"), like_lary.dtype)
                cand.locale_comms.intra_locale_comm.barrier()

    def test_copy_shared_1d(self):
        """
        Test for :func:`_locale.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape)

        with _locale.ones(comms_and_distrib=cand, dtype="int64") as lary:
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            lary.rank_view_n[...] = cand.locale_comms.peer_comm.rank

            with _locale.copy(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                cand.locale_comms.peer_comm.barrier()
                self.assertTrue(_np.all(lary1 == lary))

    def test_copy_non_shared_1d(self):
        """
        Test for :func:`_locale.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = create_distribution(shape=gshape, locale_type=LT_PROCESS)

        with _locale.ones(comms_and_distrib=cand, dtype="int64") as lary:
            self.assertEqual(_np.dtype("int64"), lary.dtype)
            lary.rank_view_n[...] = cand.locale_comms.peer_comm.rank

            with _locale.copy(lary) as lary1:
                self.assertEqual(_np.dtype("int64"), lary1.dtype)
                cand.locale_comms.peer_comm.barrier()
                self.assertTrue(_np.all(lary1 == lary))

    def do_test_views_2d(self, halo=0):
        """
        Test for :meth:`mpi_array.locale.LndarrayProxy.rank_view_n`
        and :meth:`mpi_array.locale.LndarrayProxy.rank_view_h`.
        """

        lshape = _np.array((4, 3), dtype="int64")
        gshape = lshape * _shape_factors(_mpi.COMM_WORLD.size, lshape.size)[::-1]

        cands = \
            [
                create_distribution(shape=gshape, locale_type=LT_NODE, halo=halo),
                create_distribution(shape=gshape, locale_type=LT_PROCESS, halo=halo)
            ]
        for cand in cands:
            with _locale.ones(comms_and_distrib=cand, dtype="int64") as lary:
                self.assertEqual(_np.dtype("int64"), lary.dtype)
                rank_logger = _logging.get_rank_logger(self.id(), comm=cand.locale_comms.peer_comm)
                rank_logger.info(
                    (
                        "\n========================================================\n" +
                        "locale_extent = %s\n" +
                        "rank_view_slice_n          = %s\n" +
                        "rank_view_slice_h          = %s\n" +
                        "rank_view_relative_slice_n = %s\n" +
                        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
                    )
                    %
                    (
                        lary.locale_extent,
                        lary.intra_partition.rank_view_slice_n,
                        lary.intra_partition.rank_view_slice_h,
                        lary.intra_partition.rank_view_relative_slice_n,
                    )
                )

                if cand.locale_comms.intra_locale_comm.rank == 0:
                    lary.view_h[...] = -1
                cand.locale_comms.intra_locale_comm.barrier()

                lary.rank_view_n[...] = cand.locale_comms.peer_comm.rank
                cand.locale_comms.intra_locale_comm.barrier()
                if _np.any(lary.halo > 0) and (cand.locale_comms.intra_locale_comm.size > 1):
                    self.assertTrue(_np.any(lary.rank_view_h != cand.locale_comms.peer_comm.rank))
                self.assertSequenceEqual(
                    lary.rank_view_h[lary.intra_partition.rank_view_relative_slice_n].tolist(),
                    lary.rank_view_n.tolist()
                )
                self.assertTrue(_np.all(lary.view_n >= 0))

                cand.locale_comms.peer_comm.barrier()

    def test_views_2d_no_halo(self):
        """
        """
        self.do_test_views_2d(halo=0)

    def test_views_2d_halo(self):
        """
        """
        self.do_test_views_2d(halo=[[1, 2], [3, 4]])

    def test_intra_partition_dims(self):
        """
        The :samp:`intra_partition_dims` argument of :func:`mpi_array.locale.empty` gives
        the partitioning of the locale array into the tiles of the processes of the
        locale. It used to be replaced by zeros (any partitioning).
        """
        cand = create_distribution(shape=(16, 12), locale_type=LT_NODE)
        intra_locale_size = cand.locale_comms.intra_locale_comm.size
        for dims in ((intra_locale_size, 1), (1, intra_locale_size)):
            with \
                    _locale.empty(
                        comms_and_distrib=cand,
                        dtype="int32",
                        intra_partition_dims=dims
                    ) as lary:
                self.assertSequenceEqual(dims, tuple(lary.intra_partition_dims))
                for axis in range(2):
                    if dims[axis] == 1:
                        self.assertEqual(lary.view_n.shape[axis], lary.rank_view_n.shape[axis])

    def test_view_keeps_array_alive(self):
        """
        A view of an :obj:`mpi_array.locale.LndarrayProxy` which has its own memory
        (an :attr:`mpi_array.locale.LndarrayProxy.rma_window_buffer`) keeps the proxy alive,
        it used to be freed (:meth:`mpi_array.locale.LndarrayProxy.__del__`) while
        the view existed.
        """
        cand = create_distribution(shape=(10 * _mpi.COMM_WORLD.size,))
        lary = _locale.ones(comms_and_distrib=cand, dtype="int64")
        # All processes of the locale have filled their tile.
        cand.locale_comms.intra_locale_comm.barrier()
        lary_ref = _weakref.ref(lary)
        rma_window_buffer = lary.rma_window_buffer
        self.assertTrue(lary.memory_owner is lary)
        view = lary.view_n[1:]
        del lary
        self.assertTrue(lary_ref() is not None)
        self.assertNotEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)
        is_ok = bool(_np.all(view == 1))
        self.assertTrue(cand.locale_comms.peer_comm.allreduce(is_ok, op=_mpi.LAND))
        del view
        self.assertTrue(lary_ref() is None)
        self.assertEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)

    def test_exit_with_exception_skips_free(self):
        """
        Leaving a :samp:`with` block because of an exception skips the (collective)
        free of the window memory, see :meth:`mpi_array.locale.LndarrayProxy.__exit__`.
        """
        cand = create_distribution(shape=(10 * _mpi.COMM_WORLD.size,))
        with _locale.zeros(comms_and_distrib=cand, dtype="int64") as lary:
            rma_window_buffer = lary.rma_window_buffer
            with self.assertRaises(RuntimeError):
                with lary:
                    raise RuntimeError("Raised on all processes.")
            self.assertTrue(lary.rma_window_buffer is rma_window_buffer)
            self.assertNotEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)

        self.assertTrue(lary.rma_window_buffer is None)
        self.assertEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)

    def test_del_skips_free_when_aborting(self):
        """
        Garbage collection (:meth:`mpi_array.locale.LndarrayProxy.__del__`) frees the
        window memory, except when mpi4py is going to abort the job
        (:func:`mpi_array.comms.can_free_in_del`).
        """
        cand = create_distribution(shape=(10 * _mpi.COMM_WORLD.size,))
        lary = _locale.zeros(comms_and_distrib=cand, dtype="int64")
        rma_window_buffer = lary.rma_window_buffer
        orig_can_free_in_del = _locale._can_free_in_del
        _locale._can_free_in_del = lambda: False
        try:
            lary.__del__()
        finally:
            _locale._can_free_in_del = orig_can_free_in_del
        self.assertTrue(lary.rma_window_buffer is rma_window_buffer)
        self.assertNotEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)

        lary.__del__()
        self.assertTrue(lary.rma_window_buffer is None)
        self.assertEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
