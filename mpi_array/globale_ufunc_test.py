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
from .globale_ufunc import broadcast_shape, ufunc_result_type
from .globale import gndarray as _gndarray, zeros as _zeros, ones as _ones

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
            TypeError,
            ufunc_result_type, uft, inputs, outputs
        )

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

    def test_add(self):
        """
        """
        a = _zeros((32, 48), dtype="int32")
        b = _ones((32, 48), dtype="int32")

        c = a + b

        self.assertTrue(isinstance(c, _gndarray))
        self.assertTrue((c == b).all())


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
