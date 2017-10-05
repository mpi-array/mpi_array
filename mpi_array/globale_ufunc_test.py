"""
==============================================
The :mod:`mpi_array.globale_ufunc_test` Module
==============================================

Module defining :mod:`mpi_array.globale` unit-tests.
Execute as::

   python -m mpi_array.globale_ufunc_test

and with parallelism::

   mpirun -n  2 python -m mpi_array.globale_ufunc_test
   mpirun -n  4 python -m mpi_array.globale_ufunc_test
   mpirun -n 27 python -m mpi_array.globale_ufunc_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   UfuncResultTypeTest - Tests for :func:`mpi_array.globale_ufunc.ufunc_result_type` function.
   BroadcastShapeTest - Tests for :func:`mpi_array.globale_ufunc.broadcast_shape` function.
   GndarrayUfuncTest - Tests for :func:`mpi_array.globale_ufunc.gndarray_array_ufunc` function.
"""
from __future__ import absolute_import

import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .comms import LT_NODE, LT_PROCESS, DT_CLONED, DT_SINGLE_LOCALE  # , DT_BLOCK, DT_SLAB
from . import comms as _comms
from .globale_ufunc import broadcast_shape, ufunc_result_type
from .globale import gndarray as _gndarray
from .globale_creation import ones as _ones, zeros as _zeros, asarray as _asarray
from .globale import copyto as _copyto

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class UfuncResultTypeTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :func:`mpi_array.globale_ufunc.ufunc_result_type`.
    """

    def test_single_output(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_ufunc.ufunc_result_type`,
        with single output array.
        """

        rank_logger = _logging.get_rank_logger(self.id())

        uft = ['ff->f', 'll->l', 'cc->c', 'fl->b', 'dd->d']

        inputs = (_np.array([1, 2], dtype='l'), _np.array([3, 4], dtype='l'))
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('l'),), dtypes)

        inputs = (_np.array([1, 2], dtype='f'), _np.array([3, 4], dtype='f'))
        outputs = (_np.array([0, 0], dtype='f'),)
        dtypes = ufunc_result_type(uft, inputs, outputs)
        self.assertSequenceEqual((_np.dtype('f'),), dtypes)
        outputs = (_np.array([0, 0], dtype='d'),)
        dtypes = ufunc_result_type(uft, inputs, outputs)
        self.assertSequenceEqual((_np.dtype('d'),), dtypes)
        outputs = (_np.array([0, 0], dtype='b'),)
        self.assertRaises(ValueError, ufunc_result_type, uft, inputs, outputs)

        inputs = (_np.array([1, 2], dtype='f'), _np.array([3, 4], dtype='l'))
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        self.assertSequenceEqual((_np.dtype('b'),), dtypes)

        inputs = (_np.array([1, 2], dtype='f'), 5.0)
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('f'),), dtypes)

        inputs = (_np.array([1, 2], dtype='f'), 5.0e150)
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('d'),), dtypes)

        inputs = (_np.array([1, 2], dtype='complex128'), 5.0e150)
        outputs = None
        self.assertRaises(
            ValueError,
            ufunc_result_type, uft, inputs, outputs
        )

    def test_tuple_input(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_ufunc.ufunc_result_type`,
        with single output array.
        """

        rank_logger = _logging.get_rank_logger(self.id())

        uft = ['ff->f', 'll->l', 'cc->c', 'fl->b', 'dd->d']

        inputs = (_np.array((1, 2, 3, 4), dtype='int32'), (1, 2, 3, 4))
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('l'),), dtypes)

    def test_multiple_output(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_ufunc.ufunc_result_type`,
        with multiple output arrays.
        """

        rank_logger = _logging.get_rank_logger(self.id())

        uft = ['eee->eBl', 'fff->fBl', 'ddd->dBl', ]

        inputs = (_np.array([1, 2], dtype='e'), _np.array([3, 4], dtype='f'), 4.0)
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('f'), _np.dtype('B'), _np.dtype('l')), dtypes)

        inputs = (_np.array([1, 2], dtype='e'), _np.array([3, 4], dtype='f'), 4.0)
        outputs = (_np.array([1, 2], dtype='f'),)
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('f'), _np.dtype('B'), _np.dtype('l')), dtypes)

        inputs = (_np.array([1, 2], dtype='e'), _np.array([3, 4], dtype='f'), 4.0)
        outputs = (_np.array([1, 2], dtype='d'), _np.array([1, 2], dtype='i'))
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('d'), _np.dtype('i'), _np.dtype('l')), dtypes)

        inputs = (_np.array([1, 2], dtype='e'), _np.array([3, 4], dtype='f'), 4.0)
        outputs = (_np.array([1, 2], dtype='d'), _np.array([1, 2], dtype='b'))
        self.assertRaises(
            ValueError,
            ufunc_result_type, uft, inputs, outputs
        )

        inputs = (_np.array([1, 2], dtype='e'), _np.array([3, 4], dtype='f'), 4.0)
        outputs = \
            (
                _np.array([1, 2], dtype='d'),
                _np.array([1, 2], dtype='i'),
                _np.array([1, 2], dtype='i')
            )
        self.assertRaises(
            ValueError,
            ufunc_result_type, uft, inputs, outputs
        )

    def test_example(self):
        import numpy as np
        import mpi_array as mpia
        inp = (
            np.zeros((10, 10, 10), dtype='float16'),
            16.0,
            mpia.zeros((10, 10, 10), dtype='float32'),
        )
        dtypes = ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp)
        self.assertSequenceEqual((_np.dtype('float32'), _np.dtype('bool')), dtypes)
        out = (mpia.zeros((10, 10, 10), dtype="float64"),)
        dtypes = ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp, outputs=out)
        self.assertSequenceEqual((_np.dtype('float64'), _np.dtype('bool')), dtypes)
        out += (mpia.zeros((10, 10, 10), dtype="uint16"),)
        dtypes = ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp, outputs=out)
        self.assertSequenceEqual((_np.dtype('float64'), _np.dtype('uint16')), dtypes)


class BroadcastShapeTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :func:`mpi_array.globale_ufunc.broadcast_shape`.
    """

    def test_non_broadcastable(self):
        """
        Test that :func:`mpi_array.globale_ufunc.broadcast_shape` raises
        a :obj:`ValueError` if shapes are not broadcastable.
        """

        self.assertRaises(
            ValueError,
            broadcast_shape,
            (4,),
            (5,)
        )
        self.assertRaises(
            ValueError,
            broadcast_shape,
            (5,),
            (6,)
        )
        self.assertRaises(
            ValueError,
            broadcast_shape,
            (5, 5),
            (6, 5)
        )
        self.assertRaises(
            ValueError,
            broadcast_shape,
            (5, 6),
            (5, 5)
        )
        self.assertRaises(
            ValueError,
            broadcast_shape,
            (5, 6, 7),
            (5, 1, 1),
            (5, 6, 1),
            (1, 1, 7),
            (1, 1, 6)
        )

    def test_broadcastable(self):
        """
        Asserts for variety of broadcastable shapes.
        """
        self.assertSequenceEqual((), broadcast_shape(()))
        self.assertSequenceEqual((), broadcast_shape((), ()))
        self.assertSequenceEqual((), broadcast_shape((), (), ()))

        self.assertSequenceEqual((1,), broadcast_shape((), (), (1, )))
        self.assertSequenceEqual((4, ), broadcast_shape((4, ), (), (1, )))
        self.assertSequenceEqual((1, 4), broadcast_shape((4, ), (1, 1), (1, )))
        self.assertSequenceEqual((4, 5), broadcast_shape((5, ), (1, 5), (4, 1)))


class GndarrayUfuncTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.globale_ufunc`.
    """

    def setUp(self):
        """
        Initialise :func:`numpy.random.seed`.
        """
        _np.random.seed(1531796312)
        self.rank_logger = _logging.get_rank_logger(self.id())

    def compare_results(self, npy_result_ary, mpi_result_ary):
        """
        """
        mpi_cln_npy_result_ary = _asarray(npy_result_ary)
        mpi_cln_mpi_result_ary = \
            _zeros(
                shape=mpi_result_ary.shape,
                dtype=mpi_result_ary.dtype,
                locale_type=LT_NODE,
                distrib_type=DT_CLONED
            )
        _copyto(dst=mpi_cln_mpi_result_ary, src=mpi_result_ary)
        self.assertTrue(
            _np.all(
                mpi_cln_npy_result_ary.lndarray_proxy.lndarray
                ==
                mpi_cln_mpi_result_ary.lndarray_proxy.lndarray
            )
        )

    def convert_func_args_to_gndarrays(self, converter, func_args):
        """
        """
        return [converter(arg) if isinstance(arg, _np.ndarray) else arg for arg in func_args]

    def do_cloned_distribution_test(self, func, *func_args):
        """
        """
        def converter(np_ary):
            return _asarray(np_ary, locale_type=LT_PROCESS)

        mpi_func_args = self.convert_func_args_to_gndarrays(converter, func_args)
        mpi_result_ary = func(*mpi_func_args)
        npy_result_ary = func(*func_args)
        self.compare_results(npy_result_ary, mpi_result_ary)

    def do_single_locale_distribution_test(self, func, *func_args):
        """
        """
        class Converter:

            def __init__(self, inter_locale_rank=0):
                self.inter_locale_rank = inter_locale_rank

            def __call__(self, np_ary):
                gndary = \
                    _asarray(
                        np_ary,
                        locale_type=LT_PROCESS,
                        distrib_type=DT_SINGLE_LOCALE,
                        inter_locale_rank=self.inter_locale_rank
                    )
                num_locales = gndary.locale_comms.num_locales
                self.inter_locale_rank = ((self.inter_locale_rank + 1) % num_locales)
                return gndary

        mpi_func_args = self.convert_func_args_to_gndarrays(Converter(), func_args)
        mpi_result_ary = func(*mpi_func_args)
        npy_result_ary = func(*func_args)
        self.compare_results(npy_result_ary, mpi_result_ary)

    def do_block_distribution_test(self, func, *func_args):
        """
        """
        pass

    def do_multi_distribution_tests(self, func, *func_args):
        """
        """
        self.do_cloned_distribution_test(func, *func_args)
        self.do_single_locale_distribution_test(func, *func_args)
        self.do_block_distribution_test(func, *func_args)

    def test_umath_multiply(self):
        gshape = (99, 99, 5)
        npy_ary = _np.random.uniform(low=0.5, high=1.75, size=gshape)
        cln_ary = _asarray(npy_ary)

        self.assertTrue(_np.all(npy_ary == cln_ary.lndarray_proxy.lndarray))

        def multiply(a, b):
            return a * b

        self.do_multi_distribution_tests(multiply, npy_ary, 1.0 / 3.0)

    def test_umath(self):
        """
        """
        a = _ones((32, 48), dtype="int32", locale_type=_comms.LT_PROCESS)
        b = _ones(a.shape, dtype="int32", locale_type=_comms.LT_PROCESS)

        c = a + b

        self.assertTrue(isinstance(c, _gndarray))
        self.assertTrue((c == 2).all())

        c *= 2
        self.assertTrue((c == 4).all())

    def test_umath_broadcast(self):
        """
        """
        a = _ones((64, 64, 4), dtype="int32", locale_type=_comms.LT_PROCESS)
        b = _ones(a.shape, dtype="int32", locale_type=_comms.LT_PROCESS)

        c = a + b

        c *= (2, 2, 2, 2)
        self.assertTrue((c == 4).all())

        a = _ones((8, 64, 24), dtype="int32", locale_type=_comms.LT_PROCESS)
        b = _ones(a.shape, dtype="int32", locale_type=_comms.LT_PROCESS)

        c = a + b

        self.assertTrue(isinstance(c, _gndarray))
        self.assertTrue((c == 2).all())

        c *= (_np.ones(tuple(c.shape[1:]), dtype=c.dtype) * 2)
        self.assertTrue((c == 4).all())

    def test_umath_distributed_broadcast(self):
        a = _ones((64, 64, 4), dtype="int32", locale_type=_comms.LT_PROCESS)
        b = _ones(a.shape, dtype="int32", locale_type=_comms.LT_PROCESS)

        c = a + b

        twos = _ones(tuple(a.shape[1:]), dtype=c.dtype, locale_type=_comms.LT_PROCESS, dims=(0, 1))
        twos.fill_h(2)

        c *= twos
        self.assertTrue((c == 4).all())


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
