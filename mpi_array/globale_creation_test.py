"""
=================================================
The :mod:`mpi_array.globale_creation_test` Module
=================================================

Module for testing creation/factory functions which
generate instances of :mod:`mpi_array.globale.gndarray`.
Execute as::

   python -m mpi_array.globale_creation_test

and with parallelism::

   mpirun -n  2 python -m mpi_array.globale_creation_test
   mpirun -n  4 python -m mpi_array.globale_creation_test
   mpirun -n 27 python -m mpi_array.globale_creation_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   GndarrayCreationTest - Tests for :func:`mpi_array.globale.gndarray` creation functions.
"""
from __future__ import absolute_import

import numpy as _np
import mpi4py.MPI as _mpi

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .globale import gndarray as _gndarray
from .globale_creation import asarray as _asarray, asanyarray as _asanyarray
from .globale_creation import empty as _empty, zeros as _zeros, ones as _ones, copy as _copy
from .globale_creation import empty_like as _empty_like, zeros_like as _zeros_like
from .globale_creation import ones_like as _ones_like, full as _full, array as _array
from .globale_creation import full_like as _full_like
from .globale import copyto as _copyto
from . import locale as _locale
from .comms import create_distribution as _create_distribution, LT_PROCESS, LT_NODE, DT_CLONED
from .comms import DT_BLOCK
from .indexing import IndexingExtent as _IndexingExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class GndarrayCreationTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :func:`mpi_array.globale.gndarray` instance generation.
    """

    def test_asarray_with_scalar(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asarray`.
        """

        sary0 = _asarray(5.0)
        self.assertTrue(sary0.__class__ is _gndarray)
        self.assertEqual(_np.dtype("float64"), sary0)
        self.assertEqual(0, sary0.ndim)
        self.assertSequenceEqual((), sary0.shape)
        self.assertTrue(sary0.locale_comms.peer_comm is _mpi.COMM_WORLD)

        sary1 = _asarray(sary0)
        self.assertTrue(sary1 is sary0)

    def test_order(self):
        """
        Only C order arrays are implemented. Fortran order arrays used to be
        created, but their elements were scrambled by redistribution (copyto),
        so :samp:`order='F'` raises :obj:`NotImplementedError`.
        """
        npy_ary = _np.arange(24, dtype="float64").reshape(4, 6)
        for order in ("F", "f"):
            self.assertRaises(NotImplementedError, _empty, (4, 6), order=order)
            self.assertRaises(NotImplementedError, _zeros, (4, 6), order=order)
            self.assertRaises(NotImplementedError, _ones, (4, 6), order=order)
            self.assertRaises(NotImplementedError, _full, (4, 6), 3, order=order)
            self.assertRaises(NotImplementedError, _array, npy_ary, order=order)
            self.assertRaises(NotImplementedError, _asarray, npy_ary, order=order)
            self.assertRaises(NotImplementedError, _locale.empty, (4, 6), order=order)
            with _zeros((4, 6)) as gary:
                self.assertRaises(NotImplementedError, _empty_like, gary, order=order)
                self.assertRaises(NotImplementedError, _zeros_like, gary, order=order)
                self.assertRaises(NotImplementedError, gary.copy, order=order)
        self.assertRaises(ValueError, _empty, (4, 6), order="X")

        # Redistribution of C order arrays, with same and different dtypes.
        for order in ("C", "c", "K", "A", None):
            with \
                    _asarray(
                        npy_ary, order=order, locale_type=LT_PROCESS, distrib_type=DT_BLOCK
                    ) as src:
                self.assertEqual("C", src.order)
                for dtype in ("float64", "float32"):
                    with \
                            _zeros(
                                npy_ary.shape,
                                dtype=dtype,
                                locale_type=LT_PROCESS,
                                distrib_type=DT_CLONED
                            ) as dst:
                        _copyto(dst, src)
                        self.assertTrue(_np.all(dst.view_n == npy_ary.astype(dtype)))

    def test_positional_arguments(self):
        """
        The creation functions take the :samp:`shape`, :samp:`dtype` and :samp:`order`
        positional arguments in the same order as NumPy. :func:`mpi_array.globale_creation.ones`
        used to take :samp:`comms_and_distrib` before :samp:`order`.
        """
        for gary, value in \
                (
                    (_empty((4, 6), "int32", "C"), None),
                    (_zeros((4, 6), "int32", "C"), 0),
                    (_ones((4, 6), "int32", "C"), 1),
                    (_full((4, 6), 3, "int32", "C"), 3),
                ):
            with gary:
                self.assertEqual(_np.dtype("int32"), gary.dtype)
                self.assertSequenceEqual((4, 6), gary.shape)
                if value is not None:
                    self.assertTrue((gary == value).all())

    def test_asarray_with_tuple(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asarray`.
        """

        tary0 = _asarray(_np.linspace(100.0, 200.0, 101).tolist())
        tary0.rank_logger.debug("tary0.num_locales = %s" % (tary0.num_locales,))
        self.assertTrue(tary0.__class__ is _gndarray)
        self.assertEqual(_np.dtype("float64"), tary0)
        self.assertTrue(tary0.locale_comms.peer_comm is _mpi.COMM_WORLD)

        tary1 = _asarray(tary0)
        self.assertTrue(tary1 is tary0)

    def test_asarray_with_subclass(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asarray`.
        """

        class GndarraySubclass(_gndarray):
            pass

        candd = _create_distribution(shape=(8, 32, 32, 32))
        lndarray_proxy, rma_window_buffer = \
            _locale.empty(
                comms_and_distrib=candd,
                dtype="int8",
                order='C',
                return_rma_window_buffer=True
            )

        ary_subclass = GndarraySubclass(candd, rma_window_buffer, lndarray_proxy)
        self.assertTrue(ary_subclass.__class__ is not _gndarray)
        self.assertTrue(isinstance(ary_subclass, _gndarray))
        ary_subclass.fill(3)
        asary0 = _asarray(ary_subclass)
        self.assertTrue(asary0.__class__ is _gndarray)
        # A copy: a second gndarray which owned (and freed) the same windows was returned.
        self.assertTrue(asary0.rma_window_buffer is not ary_subclass.rma_window_buffer)
        self.assertTrue(asary0.lndarray_proxy is not ary_subclass.lndarray_proxy)
        self.assertTrue((asary0 == 3).all())
        asary0.free()
        self.assertTrue((ary_subclass == 3).all())
        # array() (copy=True) used to free the windows of ary_subclass.
        with _array(ary_subclass) as ary0:
            self.assertTrue(ary0.__class__ is _gndarray)
            self.assertTrue((ary0 == 3).all())
        self.assertTrue((ary_subclass == 3).all())
        ary_subclass.free()

    def test_array_of_gndarray(self):
        """
        :func:`mpi_array.globale_creation.array` of a :obj:`mpi_array.globale.gndarray`.
        The result has type :samp:`dtype` (it used to be ignored), and with :samp:`copy=False`
        it is the input array when no conversion is needed.
        """
        gshape = (8 * _mpi.COMM_WORLD.size, 5)
        npy_ary = _np.arange(_np.prod(gshape), dtype="float64").reshape(gshape) + 0.5
        with _asarray(npy_ary, locale_type=LT_PROCESS, distrib_type=DT_BLOCK, halo=1) as gary:
            for copy in (True, False):
                with _array(gary, dtype="int32", copy=copy) as int_gary:
                    self.assertTrue(int_gary is not gary)
                    self.assertEqual(_np.dtype("int32"), int_gary.dtype)
                    self.assertSequenceEqual(
                        gary.distribution.halo.tolist(),
                        int_gary.distribution.halo.tolist()
                    )
                    self.assertTrue((int_gary == npy_ary.astype("int32")).all())
            with _array(gary) as copy_gary:
                self.assertTrue(copy_gary is not gary)
                self.assertTrue((copy_gary == npy_ary).all())
            self.assertTrue(_array(gary, copy=False) is gary)
            self.assertTrue(_asarray(gary, dtype="float64") is gary)
            self.assertTrue(_asanyarray(gary) is gary)
            self.assertRaises(NotImplementedError, _array, gary, ndmin=3)

    def test_asanyarray_with_tuple(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asanyarray`.
        """

        tary0 = _asanyarray(_np.linspace(100.0, 200.0, 101).tolist())
        tary0.rank_logger.debug("tary0.num_locales = %s" % (tary0.num_locales,))
        self.assertTrue(tary0.__class__ is _gndarray)
        self.assertEqual(_np.dtype("float64"), tary0)
        self.assertTrue(tary0.locale_comms.peer_comm is _mpi.COMM_WORLD)

        tary1 = _asanyarray(tary0)
        self.assertTrue(tary1 is tary0)

    def test_asanyarray_with_subclass(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asanyarray`.
        """
        class GndarraySubclass(_gndarray):
            pass
        candd = _create_distribution(shape=(8, 32, 32, 32))
        lndarray_proxy, rma_window_buffer = \
            _locale.empty(
                comms_and_distrib=candd,
                dtype="int8",
                order='C',
                return_rma_window_buffer=True
            )

        ary_subclass = GndarraySubclass(candd, rma_window_buffer, lndarray_proxy)
        self.assertTrue(ary_subclass.__class__ is not _gndarray)
        self.assertTrue(isinstance(ary_subclass, _gndarray))
        asanyary0 = _asanyarray(ary_subclass)
        self.assertTrue(asanyary0.__class__ is GndarraySubclass)
        self.assertTrue(asanyary0 is ary_subclass)

    def test_empty_scalar(self):
        """
        Test for :func:`mpi_array.globale.empty` and :func:`mpi_array.globale.empty_like`.
        """
        gary = \
            _empty(
                shape=(),
                dtype="float64",
                locale_type=LT_PROCESS,
                distrib_type=DT_CLONED
            )
        gary.lndarray_proxy[...] = 4

        self.assertEqual(4, gary.lndarray_proxy.lndarray)

    def test_empty_shared_1d(self):
        """
        Test for :func:`mpi_array.globale.empty` and :func:`mpi_array.globale.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(shape=gshape)

        gary = _empty(comms_and_distrib=cand, dtype="int64")

        self.assertEqual(_np.dtype("int64"), gary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(_IndexingExtent(gary.lndarray_proxy.rank_view_slice_n).shape)
        )

        gary1 = _empty_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(_IndexingExtent(gary1.lndarray_proxy.rank_view_slice_n).shape)
        )

        ary = _empty_like(_np.zeros(lshape, dtype="int64"))
        self.assertEqual(_np.dtype("int64"), ary.dtype)
        self.assertSequenceEqual(
            list(lshape),
            list(ary.shape)
        )

    def test_empty_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale_creation.empty`
        and :func:`mpi_array.globale_creation.empty_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary = _empty(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        self.assertSequenceEqual(list(lshape), list(gary.lndarray_proxy.shape))
        self.assertSequenceEqual(
            list(lshape),
            list(_IndexingExtent(gary.lndarray_proxy.rank_view_slice_n).shape)
        )

        gary1 = _empty_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        self.assertSequenceEqual(list(lshape), list(gary1.lndarray_proxy.shape))
        self.assertSequenceEqual(
            list(lshape),
            list(_IndexingExtent(gary1.lndarray_proxy.rank_view_slice_n).shape)
        )

    def test_zeros_shared_1d(self):
        """
        Test for :func:`mpi_array.globale_creation.zeros`
        and :func:`mpi_array.globale_creation.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(shape=gshape, locale_type=LT_NODE)

        gary = _zeros(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 0).all())

        gary1 = _zeros_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 0).all())

    def test_zeros_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale_creation.zeros`
        and :func:`mpi_array.globale_creation.zeros_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary = _zeros(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 0).all())

        gary1 = _zeros_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 0).all())

    def test_ones_shared_1d(self):
        """
        Test for :func:`mpi_array.globale_creation.ones`
        and :func:`mpi_array.globale_creation.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(shape=gshape)

        gary = _ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 1).all())

        gary1 = _ones_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 1).all())

    def test_ones_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale_creation.ones`
        and :func:`mpi_array.globale_creation.ones_like`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(shape=gshape, locale_type=LT_PROCESS)

        gary = _ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary == 1).all())

        gary1 = _ones_like(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == 1).all())

    def test_like_functions_dtype(self):
        """
        The :samp:`dtype` argument of :func:`mpi_array.globale_creation.empty_like`,
        :func:`mpi_array.globale_creation.zeros_like`, :func:`mpi_array.globale_creation.ones_like`
        and :func:`mpi_array.globale_creation.full_like` gives the type of the new array,
        it used to be ignored.
        """
        like_funcs_and_values = \
            (
                (_empty_like, (), None),
                (_zeros_like, (), 0),
                (_ones_like, (), 1),
                (_full_like, (3,), 3),
            )
        gshape = (8 * _mpi.COMM_WORLD.size, 5)
        with \
                _zeros(
                    gshape,
                    dtype="float64",
                    locale_type=LT_PROCESS,
                    distrib_type=DT_BLOCK,
                    halo=1
                ) as gary:
            for like_func, args, value in like_funcs_and_values:
                for ary in (gary, _np.zeros(gshape, dtype="float64"), [[0.0] * 5] * gshape[0]):
                    with like_func(ary, *args, dtype="int16") as like_gary:
                        self.assertEqual(_np.dtype("int16"), like_gary.dtype)
                        self.assertSequenceEqual(gshape, like_gary.shape)
                        if value is not None:
                            self.assertTrue((like_gary == value).all())
                        if ary is gary:
                            self.assertTrue(
                                _np.all(like_gary.distribution.halo == gary.distribution.halo)
                            )
                    with like_func(ary, *args) as like_gary:
                        self.assertEqual(_np.dtype("float64"), like_gary.dtype)

    def test_copy_shared_1d(self):
        """
        Test for :func:`mpi_array.globale_creation.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(gshape)

        gary = _ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.rank_view_n[...] = gary.locale_comms.peer_comm.rank

        gary1 = _copy(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == gary).all())

    def test_copy_non_shared_1d(self):
        """
        Test for :func:`mpi_array.globale_creation.copy`.
        """

        lshape = (10,)
        gshape = (_mpi.COMM_WORLD.size * lshape[0],)
        cand = _create_distribution(gshape, locale_type=LT_PROCESS)

        gary = _ones(comms_and_distrib=cand, dtype="int64")
        self.assertEqual(_np.dtype("int64"), gary.dtype)
        gary.rank_view_n[...] = gary.locale_comms.peer_comm.rank

        gary1 = _copy(gary)
        self.assertEqual(_np.dtype("int64"), gary1.dtype)
        gary.locale_comms.peer_comm.barrier()
        self.assertTrue((gary1 == gary).all())


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
