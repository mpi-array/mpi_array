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
   CommLoggerTest - Tests for :obj:`mpi_array.globale.CommLogger`.
   PackageTest - Tests for the names of the :mod:`mpi_array` package.


"""
from __future__ import absolute_import

import os.path as _os_path
import weakref as _weakref
import mpi4py.MPI as _mpi
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import globale as _globale
from . import locale as _locale
from .indexing import IndexingExtent as _IndexingExtent
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .comms import create_distribution, LT_PROCESS, LT_NODE, DT_SLAB, DT_BLOCK, DT_CLONED
from .comms import DT_SINGLE_LOCALE
from . import comms as _comms
from . import globale_creation as _globale_creation

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class GndarrayTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.globale.gndarray`.
    """

    def setUp(self):
        """
        """
        _np.random.seed(534219812)

    def test_attr(self):
        """
        Test various attributes of :obj:`mpi_array.globale.gndarray`.
        """
        halos = [0, 3]
        for halo in halos:
            gshape = (50, 17, 23)
            cand = create_distribution(gshape, locale_type=LT_PROCESS, halo=halo)
            locale_comms = cand.locale_comms
            with _globale_creation.zeros(gshape, comms_and_distrib=cand, dtype="int32") as gary:
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
                    _np.prod(gary.rank_view_n.shape),
                    _np.sum((gary.rank_view_n == rank_val).astype("uint32"), dtype="int64")
                )
                if _np.any(gary.rank_view_h.shape > gary.rank_view_n.shape):
                    cand.locale_comms.rank_logger.info(
                        "gary.rank_view_h = %s" %
                        (gary.rank_view_h,))
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
        with _globale_creation.empty(shape=gshape, dtype=dt) as gary:
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
        with \
                _globale_creation.empty(
                    shape=gshp, dtype="int32", locale_type=locale_type, halo=halo_1d
                ) as gary:

            self.assertSequenceEqual(gshp, tuple(gary.shape))
            self.assertEqual(0, _np.prod(gary.lndarray_proxy.locale_extent.shape_n))
            self.assertEqual(0, gary.lndarray_proxy.lndarray.size)
            self.assertEqual(0, gary.view_n.size)

        gshp = (15, 20, 11, 0)
        with \
                _globale_creation.empty(
                    shape=gshp,
                    dtype="int32",
                    locale_type=locale_type,
                    halo=halo
                ) \
                as gary:

            self.assertSequenceEqual(gshp, tuple(gary.shape))
            self.assertEqual(0, _np.prod(gary.lndarray_proxy.locale_extent.shape_n))
            self.assertEqual(0, gary.lndarray_proxy.lndarray.size)
            self.assertEqual(0, gary.view_n.size)

            num_locales = gary.locale_comms.num_locales

        # test non-empty gndarray but some empty locale extents
        if num_locales > 1:
            gshp = (15, 20, 11, num_locales // 2)
            dims = (1, 1, 1, 0)
            with \
                    _globale_creation.ones(
                        shape=gshp,
                        dtype="int32",
                        locale_type=locale_type,
                        distrib_type=DT_BLOCK,
                        dims=dims,
                        halo=halo
                    ) as gary:
                self.assertSequenceEqual(gshp, tuple(gary.shape))
                self.assertTrue(
                    _np.any(
                        _np.array(
                            tuple(
                                _np.prod(locale_extent.shape_n)
                                for locale_extent in gary.distribution.locale_extents
                            )
                        )
                    )
                )
                if _np.prod(gary.lndarray_proxy.locale_extent.shape_n) == 0:
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
        and :meth:`mpi_array.globale.gndarray.__setitem__` methods
        raise :obj:`NotImplementedError` (they used to silently do nothing
        and return :samp:`None`), and the array is unchanged.
        """
        with _globale_creation.zeros((20, 20, 20), dtype="int8") as gary:
            def setitem(index, value):
                gary[index] = value

            self.assertRaises(NotImplementedError, setitem, (1, 2, 8), 22)
            self.assertRaises(NotImplementedError, setitem, Ellipsis, 19)
            self.assertRaises(NotImplementedError, setitem, slice(None), 101)
            self.assertRaises(NotImplementedError, gary.__getitem__, (slice(1, 10), 2, 4))
            self.assertRaises(NotImplementedError, gary.__getitem__, Ellipsis)
            self.assertTrue((gary == 0).all())

    def test_readme_quick_start(self):
        """
        Runs the ``quickstart.py`` code block from :samp:`README.rst`
        (the code asserts the resulting element values).
        """
        readme_file_name = \
            _os_path.realpath(_os_path.join(_os_path.dirname(__file__), "..", "README.rst"))
        if not _os_path.exists(readme_file_name):
            self.skipTest("No README.rst at %s." % readme_file_name)
        with open(readme_file_name, "rt") as readme_file:
            lines = readme_file.read().split("\n")
        # The code block follows the first "::" line from the
        # paragraph which introduces the ``quickstart.py`` script.
        intro = [i for i in range(len(lines)) if "``quickstart.py`` script" in lines[i]]
        self.assertTrue(len(intro) > 0)
        start = intro[0]
        while not lines[start].rstrip().endswith("::"):
            start += 1
        code_lines = []
        for line in lines[start + 1:]:
            if (len(line.strip()) > 0) and (not line.startswith("   ")):
                break
            code_lines.append(line[3:])
        code = "\n".join(code_lines)
        self.assertTrue("mpia.zeros" in code)
        exec(compile(code, readme_file_name + ":quickstart.py", "exec"), dict())

    def test_copyto_permuted_cart_comm(self):
        """
        Block distribution with a :samp:`cart_comm` whose ranks are ordered differently
        from the :samp:`inter_locale_comm` ranks (as :samp:`Create_cart(reorder=True)`
        is allowed to do). The :samp:`peer_ranks_per_locale` table was in
        inter-locale rank order while the locale extents are in cart rank order,
        so :func:`mpi_array.globale.copyto` copied the wrong blocks.
        Tests single process locales and (if possible) two process locales.
        A default block array is created first, then arrays with new :samp:`cart_comm`
        objects which are only passed on the inter-locale processes, so the processes
        of a locale get different communicator cache hit/miss results (which must not
        deadlock).
        """
        world = _mpi.COMM_WORLD
        npy_ary = _np.arange(8 * world.size * 3, dtype="float64").reshape(8 * world.size, 3)
        intra_locale_comms = [_mpi.COMM_SELF, ]
        if (world.size % 2) == 0:
            intra_locale_comms.append(world.Split(world.rank // 2, world.rank))
        for intra_locale_comm in intra_locale_comms:
            kwargs = dict(peer_comm=world, intra_locale_comm=intra_locale_comm)
            inter_locale_comm = _comms.LocaleComms(**kwargs).inter_locale_comm
            with \
                    _globale_creation.zeros(
                        npy_ary.shape, distrib_type=DT_BLOCK, dims=(0, 1), **kwargs
                    ) as default_ary:
                for not_inter_locale_cart_comm in (_mpi.COMM_NULL, None):
                    cart_comm = not_inter_locale_cart_comm
                    if inter_locale_comm != _mpi.COMM_NULL:
                        # Reverse the rank order.
                        reversed_comm = \
                            inter_locale_comm.Split(
                                0,
                                inter_locale_comm.size - 1 - inter_locale_comm.rank
                            )
                        cart_comm = \
                            reversed_comm.Create_cart(
                                [reversed_comm.size, 1], [False, False], reorder=False
                            )
                        reversed_comm.Free()
                    with \
                            _globale_creation.asarray(
                                npy_ary, distrib_type=DT_BLOCK, dims=(0, 1), cart_comm=cart_comm,
                                **kwargs
                            ) as src, \
                            _globale_creation.zeros(
                                npy_ary.shape, distrib_type=DT_CLONED, **kwargs
                            ) as dst:

                        # This process belongs to the locale of its own locale extent.
                        self.assertTrue(
                            world.rank
                            in
                            src.distribution.peer_ranks_per_locale[src.this_locale[0]]
                        )
                        _globale.copyto(dst, src)
                        self.assertTrue(_np.all(dst.view_n == npy_ary))

                        _globale.copyto(default_ary, src)
                        _globale.copyto(dst, default_ary)
                        self.assertTrue(_np.all(dst.view_n == npy_ary))

                        with _globale_creation.zeros_like(src) as src_copy:
                            _globale.copyto(src_copy, dst)
                            self.assertTrue((src_copy == src).all())

    def test_copyto_padded_struct_dtype(self):
        """
        Redistribution (:func:`mpi_array.globale.copyto`) of an array with a structured
        dtype which has padding (MPI datatype extent used to be smaller than the itemsize).
        """
        # Packed fields (11 bytes) padded to an itemsize of 16 bytes.
        dtyp = \
            _np.dtype(
                {
                    "names": ["m0", "m1", "m2"],
                    "formats": ["int8", "float64", "int16"],
                    "offsets": [0, 1, 9],
                    "itemsize": 16
                }
            )
        npy_ary = _np.zeros((4 * _mpi.COMM_WORLD.size + 3, 5), dtype=dtyp)
        npy_ary["m0"] = (_np.arange(npy_ary.size) % 100).reshape(npy_ary.shape)
        npy_ary["m1"] = _np.arange(npy_ary.size).reshape(npy_ary.shape) + 0.5
        npy_ary["m2"] = -_np.arange(npy_ary.size).reshape(npy_ary.shape)
        for locale_type in (LT_PROCESS, LT_NODE):
            with \
                    _globale_creation.asarray(
                        npy_ary, locale_type=locale_type, distrib_type=DT_BLOCK
                    ) as src, \
                    _globale_creation.empty(
                        npy_ary.shape, dtype=dtyp, locale_type=locale_type, distrib_type=DT_CLONED
                    ) as dst:
                _globale.copyto(dst, src)
                self.assertSequenceEqual(npy_ary.tolist(), dst.view_n.tolist())

    def test_locales_of_different_sizes(self):
        """
        Locales with different numbers of processes (e.g. locales of 1 and 2 processes
        at 3 processes). Array creation used to deadlock: a single process locale
        allocated its memory with a collective over all processes, while the processes
        of the other locales allocated shared memory over their own locale.
        Tests creation, halo update, :func:`mpi_array.globale.copyto`, a ufunc
        which fetches remote input elements, and the functions which create an array
        like another one (they used to raise :obj:`ValueError`, the partitioning of each
        locale array was checked against the sizes of all locales).
        """
        world = _mpi.COMM_WORLD
        node_comm = world.Split_type(_mpi.COMM_TYPE_SHARED, key=world.rank)
        # Locale i of a node has (up to) i + 1 processes.
        locale_index = 0
        while ((locale_index + 1) * (locale_index + 2)) // 2 <= node_comm.rank:
            locale_index += 1
        intra_locale_comm = node_comm.Split(locale_index, node_comm.rank)
        node_comm.Free()
        kwargs = dict(peer_comm=world, intra_locale_comm=intra_locale_comm)

        npy_ary = _np.arange(10 * world.size + 3, dtype="float64")
        with \
                _globale_creation.zeros(
                    npy_ary.shape, distrib_type=DT_BLOCK, halo=1, **kwargs
                ) as blk, \
                _globale_creation.zeros(npy_ary.shape, distrib_type=DT_CLONED, **kwargs) as cln, \
                _globale_creation.asarray(
                    npy_ary, distrib_type=DT_SINGLE_LOCALE, **kwargs
                ) as sgl:

            lndarray_proxy = blk.lndarray_proxy
            locale_extent = lndarray_proxy.locale_extent
            if lndarray_proxy.rank_view_n.size > 0:
                rank_slice = \
                    locale_extent.locale_to_globale_slice_h(lndarray_proxy.rank_view_slice_n)
                blk.rank_view_n[...] = npy_ary[rank_slice]
            blk.update()
            ok = _np.all(lndarray_proxy.lndarray == npy_ary[locale_extent.to_slice_h()])

            _globale.copyto(cln, blk)
            ok = ok and _np.all(cln.view_n == npy_ary)

            # The result has the block distribution, the locales (except locale 0)
            # fetch their elements of sgl from locale 0.
            with (blk + sgl) as result:
                _globale.copyto(cln, result)
                ok = ok and _np.all(cln.view_n == 2 * npy_ary)

            with \
                    _globale_creation.zeros_like(blk) as zeros, \
                    blk.copy() as blk_copy, \
                    _globale_creation.array(blk, dtype="int32") as int_blk:
                ok = ok and (zeros == 0).all() and (blk_copy == blk).all()
                ok = ok and (int_blk == blk).all()
            mantissa, exponent = _np.frexp(blk)
            with mantissa, exponent:
                ok = ok and (_np.ldexp(mantissa, exponent) == blk).all()

            self.assertTrue(world.allreduce(bool(ok), op=_mpi.LAND))

    def test_exit_with_exception_skips_free(self):
        """
        Leaving a :samp:`with` block because of an exception skips the (collective) free.
        When only some processes raised, these processes used to wait in the collective
        free for processes which never reached it, so the job hung instead of the
        exception propagating.
        """
        world = _mpi.COMM_WORLD
        with _globale_creation.zeros((10 * world.size,), dtype="int32") as gary:
            gary.initialise_windows()
            rma_window_buffer = gary.rma_window_buffer

            with self.assertRaises(RuntimeError):
                with gary:
                    raise RuntimeError("Raised on all processes.")
            not_freed = \
                (gary.rma_window_buffer is rma_window_buffer) \
                and \
                (rma_window_buffer.peer_win != _mpi.WIN_NULL)
            self.assertTrue(world.allreduce(not_freed, op=_mpi.LAND))

            if world.rank == 0:
                with self.assertRaises(RuntimeError):
                    with gary:
                        raise RuntimeError("Raised on rank 0 only.")
            world.barrier()
            gary.fill(1)
            self.assertTrue((gary == 1).all())

        self.assertTrue(gary.rma_window_buffer is None)
        self.assertEqual(_mpi.WIN_NULL, rma_window_buffer.peer_win)

    def test_del_skips_free_when_aborting(self):
        """
        Garbage collection (:meth:`mpi_array.globale.gndarray.__del__`) frees the array,
        except when mpi4py is going to abort the job (:func:`mpi_array.comms.can_free_in_del`).
        At exit, a process with an unhandled exception used to wait in the collective
        free for the other processes, so :samp:`python -m mpi4py` couldn't abort the job.
        The end of a program is tested by :mod:`mpi_array.exit_scenarios`.
        """
        self.assertTrue(_comms.can_free_in_del())
        gary = _globale_creation.zeros((10 * _mpi.COMM_WORLD.size,), dtype="int32")
        rma_window_buffer = gary.rma_window_buffer
        orig_can_free_in_del = _globale._can_free_in_del
        _globale._can_free_in_del = lambda: False
        try:
            gary.__del__()
        finally:
            _globale._can_free_in_del = orig_can_free_in_del
        self.assertTrue(gary.rma_window_buffer is rma_window_buffer)
        self.assertNotEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)

        gary.__del__()
        self.assertTrue(gary.rma_window_buffer is None)
        self.assertEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)

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
                with _globale_creation.empty(comms_and_distrib=cand, dtype="int32") as gary:
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
        with _globale_creation.zeros(comms_and_distrib=cand_lt_process, dtype="int32") as gary:
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
                    halo_slab_num_elems = _np.prod(halo_slab.shape)
                    self.assertTrue((lhalo[axis, dir] == 0) or (halo_slab_num_elems > 0))
                    halo_slab = gary.lndarray_proxy.locale_extent.globale_to_locale_extent_h(
                        halo_slab)
                    self.assertTrue(
                        _np.all(
                            _np.logical_and(
                                gary.lndarray_proxy[halo_slab.to_slice()] != inter_locale_rank_val,
                                gary.lndarray_proxy[halo_slab.to_slice()] != 0
                            )
                        )
                    )

    def test_update_fetches_each_halo_once(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.update` with block distributions
        over 2 and 3 dimensions. The halo elements (including those of the corners) must
        equal the elements of the neighbouring locales, and each halo slab is fetched
        with one RMA :samp:`Get`. The update of each axis used to fetch the halos of
        every axis.
        """

        class CountingWin(object):

            def __init__(self, win):
                self.win = win
                self.num_gets = 0

            @property
            def group(self):
                return self.win.group

            def Fence(self, *args, **kwargs):
                return self.win.Fence(*args, **kwargs)

            def Get(self, *args, **kwargs):
                self.num_gets += 1
                return self.win.Get(*args, **kwargs)

        halo = 2
        for ndim in (2, 3):
            # Each axis has at most size tiles, so tiles have at least halo elements.
            gshape = (max(12, halo * _mpi.COMM_WORLD.size),) * ndim
            npy_ary = _np.arange(_np.prod(gshape), dtype="int64").reshape(gshape)
            with \
                    _globale_creation.zeros(
                        gshape,
                        dtype="int64",
                        distrib_type=DT_BLOCK,
                        locale_type=LT_PROCESS,
                        halo=halo
                    ) as gary:
                extent = gary.lndarray_proxy.locale_extent
                slice_n = tuple(slice(extent.start_n[a], extent.stop_n[a]) for a in range(ndim))
                slice_h = tuple(slice(extent.start_h[a], extent.stop_h[a]) for a in range(ndim))
                gary.fill_h(-1)
                gary.lndarray_proxy.view_n[...] = npy_ary[slice_n]
                counting_win = CountingWin(gary.halo_updater._inter_locale_win)
                gary.halo_updater._inter_locale_win = counting_win

                gary.update()

                # The tiles are at least as big as the halo, so each halo slab
                # is a part of one neighbouring tile.
                num_halo_slabs = int(_np.sum(extent.halo > 0))
                is_ok = \
                    (
                        (min(extent.shape_n) >= halo)
                        and
                        _np.all(gary.lndarray_proxy.view_h == npy_ary[slice_h])
                        and
                        (counting_win.num_gets == num_halo_slabs)
                    )
                gary.rank_logger.debug(
                    "num_gets=%s, num_halo_slabs=%s", counting_win.num_gets, num_halo_slabs
                )
                self.assertTrue(gary.locale_comms.peer_comm.allreduce(is_ok, op=_mpi.LAND))

    def test_get_outside_of_array(self):
        """
        :meth:`mpi_array.globale.gndarray.locale_get` and
        :meth:`mpi_array.globale.gndarray.peer_rank_get` of a region which is not
        within the array raise :obj:`ValueError` on every process, before allocating
        memory. They used to raise :obj:`AttributeError`, only on the inter-locale
        processes for :samp:`locale_get` (the other processes of a locale waited).
        """
        size = _mpi.COMM_WORLD.size
        for locale_type in (LT_PROCESS, LT_NODE):
            with \
                    _globale_creation.ones(
                        (10 * size,), locale_type=locale_type, distrib_type=DT_BLOCK
                    ) as gary:
                gary.initialise_windows()
                gary.inter_locale_barrier()
                for get in (gary.locale_get, gary.peer_rank_get):
                    self.assertRaises(ValueError, get, (slice(5, 10 * size + 5),))
                    self.assertRaises(ValueError, get, start=(-1,), stop=(5,))
                # A view of the locale array or fetched (win_lndarray) memory.
                locale_ary = gary.locale_get((slice(0, 10 * size),))
                self.assertTrue(_np.all(locale_ary == 1))
                gary.inter_locale_barrier()
                gary.intra_locale_barrier()
                locale_ary.free()

    def test_view_keeps_array_alive(self):
        """
        A view of the locale memory of a gndarray keeps the array (which frees the memory)
        alive, e.g. a view of a temporary, :samp:`(a + 1).view_n`. The temporary used to be freed
        (:meth:`mpi_array.globale.gndarray.__del__`) as soon as the view was made, and reading
        the view read freed memory (which could segfault). The array is freed when the last
        view is released. Empty views and copies (of the array type) keep it alive too:
        the size of a view, and whether NumPy makes a view or a copy, can differ between
        processes, and so would the point where the array is (collectively) freed.
        """
        def locale_get_locale_extent(gary):
            extent = gary.lndarray_proxy.locale_extent
            return gary.locale_get(start=extent.start_n, stop=extent.stop_n)

        # (view function, whether the view excludes the halo)
        get_views = \
            (
                (lambda gary: gary.view_n, True),
                (lambda gary: gary.view_h, False),
                (lambda gary: gary.rank_view_n, True),
                (lambda gary: gary.rank_view_h, False),
                (lambda gary: gary.lndarray_proxy.view_n, True),
                (lambda gary: gary.lndarray_proxy.rank_view_partition_h, False),
                (lambda gary: gary.lndarray_proxy[...], False),
                (lambda gary: gary.view_n[1:], True),
                (lambda gary: gary.view_n[5:5], True),
                (lambda gary: gary.lndarray_proxy[5:5], True),
                (lambda gary: _np.asarray(gary.view_n), True),
                (locale_get_locale_extent, True),
            )
        size = _mpi.COMM_WORLD.size
        for locale_type in (LT_PROCESS, LT_NODE):
            with \
                    _globale_creation.ones(
                        (100 * size,), locale_type=locale_type, distrib_type=DT_BLOCK, halo=2
                    ) as gary:
                peer_comm = gary.locale_comms.peer_comm
                for get_view, is_no_halo_view in get_views:
                    tmp_gary = gary + 1
                    tmp_gary_ref = _weakref.ref(tmp_gary)
                    rma_window_buffer = tmp_gary.rma_window_buffer
                    view = get_view(tmp_gary)
                    del tmp_gary
                    is_alive = \
                        (
                            (tmp_gary_ref() is not None)
                            and
                            (rma_window_buffer.intra_locale_win != _mpi.WIN_NULL)
                        )
                    self.assertTrue(peer_comm.allreduce(is_alive, op=_mpi.LAND))
                    if is_no_halo_view:
                        self.assertTrue(peer_comm.allreduce(bool(_np.all(view == 2)), op=_mpi.LAND))
                    sub_view = view[...]
                    del view
                    self.assertTrue(tmp_gary_ref() is not None)
                    del sub_view
                    # The last view is released, the array is freed (no reference cycle).
                    self.assertTrue(tmp_gary_ref() is None)
                    self.assertEqual(_mpi.WIN_NULL, rma_window_buffer.intra_locale_win)

                # Copies of the array type keep the array alive too (whether NumPy
                # makes a view or a copy can depend on the layout of a process's tile).
                # NumPy copies (numpy.ndarray) don't.
                tmp_gary = gary + 1
                tmp_gary_ref = _weakref.ref(tmp_gary)
                view_n = tmp_gary.view_n
                copies = \
                    (
                        view_n.copy(),
                        view_n + 1,
                        view_n.astype("int32"),
                        tmp_gary.lndarray_proxy[[2, 3]],  # advanced indexing copies
                    )
                npy_copies = (_np.array(view_n), view_n.tolist())
                del tmp_gary, view_n
                self.assertTrue(tmp_gary_ref() is not None)
                self.assertTrue(_np.all(copies[0] == 2))
                self.assertTrue(_np.all(copies[1] == 3))
                self.assertTrue(_np.all(copies[2] == 2))
                self.assertTrue(_np.all(copies[3] == 2))
                del copies
                self.assertTrue(tmp_gary_ref() is None)
                self.assertTrue(_np.all(npy_copies[0] == 2))

    def test_view_lifetime_on_all_processes(self):
        """
        A view which is empty on some processes (and not on others) keeps the array
        alive on every process, the array is then freed (collectively) at the same point
        on all of them. Views of views used to keep it alive only when not empty. And
        arrays derived from a view keep it alive whether NumPy makes them views or copies,
        which depends on the layout of a process's tile.
        """
        size = _mpi.COMM_WORLD.size
        with \
                _globale_creation.ones(
                    (2 * size + 1,), locale_type=LT_NODE, distrib_type=DT_BLOCK
                ) as gary:
            tmp_gary = gary + 1
            tmp_gary_ref = _weakref.ref(tmp_gary)
            rma_window_buffer = tmp_gary.rma_window_buffer
            # Empty on the processes whose tile has fewer than 3 elements.
            view = tmp_gary.rank_view_n[2:]
            del tmp_gary
            is_alive = \
                (
                    (tmp_gary_ref() is not None)
                    and
                    (rma_window_buffer.intra_locale_win != _mpi.WIN_NULL)
                )
            self.assertTrue(gary.locale_comms.peer_comm.allreduce(is_alive, op=_mpi.LAND))
            del view
            self.assertTrue(tmp_gary_ref() is None)

        # The tiles of a 2-D locale array have different layouts: ravel (and reshape) make a
        # view of a contiguous tile, and a copy of a non-contiguous one.
        with _globale_creation.ones((3, 4), locale_type=LT_NODE, distrib_type=DT_BLOCK) as gary:
            for derive in (lambda ary: ary.ravel(), lambda ary: ary.reshape(-1)):
                tmp_gary = gary + 1
                tmp_gary_ref = _weakref.ref(tmp_gary)
                derived = derive(tmp_gary.rank_view_n)
                del tmp_gary
                is_alive = tmp_gary_ref() is not None
                self.assertTrue(gary.locale_comms.peer_comm.allreduce(is_alive, op=_mpi.LAND))
                del derived
                self.assertTrue(tmp_gary_ref() is None)

    def test_get_keeps_array_alive_on_all_processes(self):
        """
        :meth:`mpi_array.globale.gndarray.locale_get` and
        :meth:`mpi_array.globale.gndarray.peer_rank_get` return a view on the processes
        whose locale has the region and fetched memory on the others. Both keep the
        array alive, also through arrays made from them (e.g. :samp:`numpy.asarray`), so
        that it is (collectively) freed at the same point on every process. The fetched
        memory didn't keep it alive, then :samp:`numpy.asarray` of it didn't, and the free
        mismatched (a hang).
        """
        size = _mpi.COMM_WORLD.size
        conversions = \
            (
                lambda ary: ary,
                lambda ary: _np.asarray(ary),
                lambda ary: ary.view(_np.ndarray),
                lambda ary: ary[1:],
            )
        with \
                _globale_creation.ones(
                    (8 * size,), locale_type=LT_PROCESS, distrib_type=DT_BLOCK
                ) as gary:
            peer_comm = gary.locale_comms.peer_comm
            for get_name in ("locale_get", "peer_rank_get"):
                for convert in conversions:
                    tmp_gary = gary + 1
                    tmp_gary.initialise_windows()
                    tmp_gary.inter_locale_barrier()
                    tmp_gary_ref = _weakref.ref(tmp_gary)
                    # The region is local on the process of the first locale only.
                    locale_ary = getattr(tmp_gary, get_name)((slice(0, 4),))
                    converted = convert(locale_ary)
                    # Remote processes have finished fetching before the arrays are freed.
                    peer_comm.barrier()
                    is_ok = bool(_np.all(converted == 2))
                    locale_ary.free()
                    del locale_ary, tmp_gary
                    is_ok = is_ok and (tmp_gary_ref() is not None)
                    self.assertTrue(peer_comm.allreduce(is_ok, op=_mpi.LAND))
                    del converted
                    self.assertTrue(tmp_gary_ref() is None)

    def test_all(self):
        """
        Tests for :meth:`mpi_array.globale.gndarray.all`.
        """
        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)

        with \
                _globale_creation.zeros(
                    shape=gshape, locale_type=LT_PROCESS, dtype="int64"
                ) as gary0, \
                _globale_creation.ones(
                    shape=gshape, locale_type=LT_PROCESS, dtype="int64"
                ) \
                as gary1:

            self.assertFalse((gary0 == gary1).all())

    def do_test_peer_rank_get(self, locale_type=LT_PROCESS):
        """
        Test for :meth:`mpi_array.globale.gndarray.peer_rank_get`.
        """
        lshape = (11, 12)
        f = int(_np.ceil(_np.sqrt(_mpi.COMM_WORLD.size)))
        gshape = (f * lshape[0] + 1, f * lshape[1] + 3)

        npy_ary = _np.random.uniform(low=1.5, high=2.9, size=gshape)
        with \
                _globale_creation.asarray(npy_ary, distrib_type=DT_BLOCK, locale_type=locale_type) \
                as gnd_ary:

            # Wait for all procs to finish copying.
            gnd_ary.initialise_windows()
            gnd_ary.locale_comms.peer_comm.barrier()

            num_locales = gnd_ary.distribution.num_locales

            fetch_extent = \
                gnd_ary.distribution.locale_extents[_np.random.randint(low=0, high=num_locales)]

            fetch_portion_start = \
                (fetch_extent.start_n + fetch_extent.stop_n) // 2 - fetch_extent.shape_n // 4
            fetch_portion_stop = fetch_portion_start + fetch_extent.shape_n // 2
            fetch_portion_extent = _IndexingExtent(
                start=fetch_portion_start, stop=fetch_portion_stop)
            fetch_portion_slice = fetch_portion_extent.to_slice()

            gnd_ary.rank_logger.debug("fetch_portion_extent=%s", fetch_portion_extent)
            fetch_ary = gnd_ary.peer_rank_get(slice=fetch_portion_slice)

            self.assertTrue(
                _np.all(fetch_ary == npy_ary[fetch_portion_slice])
            )
            # Wait for all procs to finish fetching.
            gnd_ary.locale_comms.peer_comm.barrier()

    def test_peer_rank_get_locale_type_process(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.peer_rank_get`.
        """
        locale_type = LT_PROCESS
        self.do_test_peer_rank_get(locale_type=locale_type)

    def test_peer_rank_get_locale_type_node(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.peer_rank_get`.
        """
        locale_type = LT_NODE
        self.do_test_peer_rank_get(locale_type=locale_type)

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

        cand_slb_ax1 = \
            create_distribution(
                shape=gshape,
                distrib_type=DT_SLAB,
                axis=1,
                locale_type=locale_type,
                halo=halo
            )

        with \
                _globale_creation.zeros(comms_and_distrib=cand_slab_ax0, dtype=src_dtype) \
                as gary_slb_ax0, \
                _globale_creation.zeros(comms_and_distrib=cand_slb_ax1, dtype=dst_dtype) \
                as gary_slb_ax1:

            rank_val = gary_slb_ax0.comms_and_distrib.this_locale.inter_locale_rank + 1
            gary_slb_ax0.rank_view_n[...] = rank_val

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
            with _globale_creation.zeros_like(gary_slb_ax0) as gary_slb_ax0_0:
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
        with \
            _globale_creation.zeros(comms_and_distrib=cand_slab_ax0, dtype=src_dtype)\
            as gary_slb_ax0, \
            _globale_creation.zeros(comms_and_distrib=cand_slb_ax1, dtype=dst_dtype) \
            as gary_slb_ax1: \

            rank_val = gary_slb_ax1.comms_and_distrib.this_locale.inter_locale_rank + 1
            gary_slb_ax1.rank_view_n[...] = rank_val

            _globale.copyto(gary_slb_ax0, gary_slb_ax1)
            with _globale_creation.zeros_like(gary_slb_ax1) as gary_slb_ax1_1:
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

        with \
                _globale_creation.zeros(comms_and_distrib=cand_node_slab, dtype=node_slab_dtype)\
                as gary_node_slab, \
                _globale_creation.zeros(comms_and_distrib=cand_proc_blok, dtype=proc_blok_dtype)\
                as gary_proc_blok:

            rank_val = gary_node_slab.comms_and_distrib.this_locale.inter_locale_rank + 1
            gary_node_slab.rank_view_n[...] = rank_val
            # gary_node_slab.update()

            _globale.copyto(gary_proc_blok, gary_node_slab, casting="unsafe")
            with _globale_creation.zeros_like(gary_node_slab) as gary_node_slab0:
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
                        (
                            num_diffs,
                            gary_node_slab.view_n[...].shape,
                            min_elem_at_diffs,
                            max_elem_at_diffs
                        )
                    )
                )

        # Try in copyto in opposite order
        with \
                _globale_creation.zeros(comms_and_distrib=cand_node_slab, dtype=node_slab_dtype)\
                as gary_node_slab, \
                _globale_creation.zeros(comms_and_distrib=cand_proc_blok, dtype=proc_blok_dtype)\
                as gary_proc_blok:

            rank_val = gary_proc_blok.comms_and_distrib.this_locale.inter_locale_rank + 1
            gary_proc_blok.rank_view_n[...] = rank_val
            # gary_proc_blok.update()

            _globale.copyto(gary_node_slab, gary_proc_blok, casting="unsafe")

            with _globale_creation.zeros_like(gary_proc_blok) as gary_proc_blok0:
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
                        (
                            num_diffs,
                            gary_proc_blok.view_n[...].shape,
                            min_elem_at_diffs, max_elem_at_diffs
                        )
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
        with _globale_creation.zeros(shape=(10, 10, 10), dtype="uint64") as gary:

            self.assertRaises(ValueError, _globale.copyto, gary, [1, ])
            self.assertRaises(ValueError, _globale.copyto, [1, ], gary)
            self.assertRaises(ValueError, _globale.copyto, [1, ], [1, ])

    def test_copyto_synchronises_twice(self):
        """
        :func:`mpi_array.globale.copyto` synchronises the processes twice (barriers),
        before the source is read and after the destination is written, and copies
        the elements. It used to synchronise three times, also between the copies
        within locales and the fetches from other locales.
        """
        updater_type = _globale.RmaRedistributeUpdater
        barrier = updater_type.barrier
        num_barriers = []

        def counting_barrier(updater):
            num_barriers.append(1)
            barrier(updater)

        shape = (8 * _mpi.COMM_WORLD.size, 8)
        for locale_type in (LT_PROCESS, LT_NODE):
            with \
                    _globale_creation.empty(shape, locale_type=locale_type, dims=(0, 1)) \
                    as dst, \
                    _globale_creation.empty(shape, locale_type=locale_type, dims=(1, 0)) \
                    as src:
                src.rank_view_n[...] = src.this_locale.inter_locale_rank + 1
                src.intra_locale_barrier()
                del num_barriers[:]
                updater_type.barrier = counting_barrier
                try:
                    _globale.copyto(dst, src)
                finally:
                    updater_type.barrier = barrier
                self.assertEqual(2, len(num_barriers))
                expected = _np.zeros(shape)
                for extent in src.distribution.locale_extents:
                    expected[extent.to_slice_n()] = extent.inter_locale_rank + 1
                self.assertTrue(
                    dst.locale_comms.peer_comm.allreduce(
                        bool(
                            _np.all(
                                dst.view_n
                                ==
                                expected[dst.lndarray_proxy.locale_extent.to_slice_n()]
                            )
                        ),
                        op=_mpi.LAND
                    )
                )


class CommLoggerTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.globale.CommLogger`.
    """

    def test_construct(self):
        """
        The :obj:`mpi_array.globale.CommLogger` constructor arguments set the loggers,
        they used to be ignored.
        """
        rank_logger = _logging.get_rank_logger(self.id())
        root_logger = _logging.get_root_logger(self.id())
        comm_logger = _globale.CommLogger(rank_logger=rank_logger, root_logger=root_logger)
        self.assertTrue(comm_logger.rank_logger is rank_logger)
        self.assertTrue(comm_logger.root_logger is root_logger)
        comm_logger = _globale.CommLogger()
        self.assertTrue(comm_logger.rank_logger is None)
        self.assertTrue(comm_logger.root_logger is None)


class PackageTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for the names of the :mod:`mpi_array` package.
    """

    def test_names(self):
        """
        The :mod:`mpi_array` package has :func:`mpi_array.globale.copyto`, and its
        :samp:`__all__` lists the public functions, classes and ufuncs. It used to
        miss :samp:`copyto`, and to list the loop variable :samp:`s`, :samp:`absolute_import`
        and the sub-modules.
        """
        import mpi_array
        self.assertTrue(mpi_array.copyto is _globale.copyto)
        self.assertTrue(mpi_array.add is _np.add)
        for name in ("s", "absolute_import"):
            self.assertFalse(hasattr(mpi_array, name))
        for name in \
                (
                    "gndarray", "free_all", "copyto", "dtype", "empty", "empty_like", "zeros",
                    "zeros_like", "ones", "ones_like", "full", "full_like", "eye", "identity",
                    "array", "asarray", "asanyarray", "copy", "add", "multiply", "frexp"
                ):
            self.assertIn(name, mpi_array.__all__)
        for name in ("globale", "comms", "locale", "logging"):
            self.assertNotIn(name, mpi_array.__all__)
        self.assertEqual(len(set(mpi_array.__all__)), len(mpi_array.__all__))
        for name in mpi_array.__all__:
            self.assertTrue(hasattr(mpi_array, name), name)


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
