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
   ToGndarrayConverter - Base class for :obj:`numpy.ndarray` to :obj:`mpi_array.globale.gndarray`.
"""
from __future__ import absolute_import

import time as _time
import mpi4py.MPI as _mpi
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .comms import LT_NODE, LT_PROCESS, DT_CLONED, DT_SINGLE_LOCALE, DT_BLOCK  # , DT_SLAB
from . import comms as _comms
from . import distribution as _distribution
from . import update as _update
from .globale_ufunc import broadcast_shape, ufunc_result_type, get_extents
from .globale_ufunc import check_equivalent_inter_locale_comms
from .globale import gndarray as _gndarray
from .locale import win_lndarray as _win_lndarray
from .globale_creation import ones as _ones, zeros as _zeros, asarray as _asarray
from .globale_creation import empty as _empty, zeros_like as _zeros_like
from .globale_creation import full as _full, array as _array
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

        # Python scalars are weakly typed (NEP 50), the result does
        # not depend on the value (5.0e150 overflows float32).
        inputs = (_np.array([1, 2], dtype='f'), 5.0e150)
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        rank_logger.debug("dtypes=%s", dtypes)
        self.assertSequenceEqual((_np.dtype('f'),), dtypes)

        inputs = (_np.array([1, 2], dtype='f'), _np.asarray(5.0))
        outputs = None
        dtypes = ufunc_result_type(uft, inputs, outputs)
        self.assertSequenceEqual((_np.dtype('d'),), dtypes)

        inputs = (_np.array([1, 2], dtype='complex128'), 5.0e150)
        outputs = None
        self.assertRaises(
            ValueError,
            ufunc_result_type, uft, inputs, outputs
        )

        bool_uft = ['??->?', 'bb->b', 'll->l', 'ff->f', 'dd->d']
        dtypes = ufunc_result_type(bool_uft, (True, False))
        self.assertSequenceEqual((_np.dtype('?'),), dtypes)
        dtypes = ufunc_result_type(bool_uft, (True, 1))
        self.assertSequenceEqual((_np.dtype('l'),), dtypes)
        dtypes = ufunc_result_type(bool_uft, (True, 1.0))
        self.assertSequenceEqual((_np.dtype('d'),), dtypes)

    def test_type_strings_numpy_parity(self):
        """
        :func:`mpi_array.globale_ufunc.ufunc_result_type` with :attr:`numpy.ufunc.types`
        gives the same result types as the :obj:`numpy.ufunc` call for array and
        Python scalar inputs (for ufuncs without a ufunc specific type resolver).
        """
        cases = \
            (
                (_np.add, (_np.zeros((2,), dtype="uint16"), 1)),
                (_np.add, (_np.zeros((2,), dtype="int8"), 2.5)),
                (_np.add, (_np.zeros((2,), dtype="bool"), 1)),
                (_np.add, (_np.zeros((2,), dtype="int16"), 2j)),
                (_np.add, (_np.zeros((2,), dtype="float32"), 2j)),
                (_np.add, (_np.zeros((2,), dtype="float32"), 2.0)),
                (_np.add, (_np.zeros((2,), dtype="float16"), _np.zeros((2,), dtype="float32"))),
                (_np.multiply, (3, _np.zeros((2,), dtype="int16"))),
                (_np.power, (_np.zeros((2,), dtype="uint16"), 1.0 / 3.0)),
            )
        for ufunc, inputs in cases:
            self.assertSequenceEqual(
                (ufunc(*inputs).dtype,),
                ufunc_result_type(ufunc.types, inputs),
                "%s%s" % (ufunc.__name__, tuple(getattr(i, "dtype", i) for i in inputs))
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
                _np.array([1, 2], dtype='uint16')
            )
        self.assertRaises(
            ValueError,
            ufunc_result_type, uft, inputs, outputs
        )

    def test_numpy_ufunc(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_ufunc.ufunc_result_type`,
        with a :obj:`numpy.ufunc`. Result types must match those of the :obj:`numpy.ufunc` call.
        """
        cases = \
            (
                (_np.add, (_np.zeros((2,), dtype="uint16"), 1)),
                (_np.add, (_np.zeros((2,), dtype="int8"), 2.5)),
                (_np.add, (_np.zeros((2,), dtype="float32"), 2.0)),
                (_np.add, (_np.zeros((2,), dtype="float32"), 2.0j)),
                (_np.add, (_np.zeros((2,), dtype="int32"), _np.asarray(2))),
                (_np.add, (_np.zeros((2,), dtype="bool"), 1)),
                (_np.multiply, (3, _np.zeros((2,), dtype="int16"))),
                (_np.frexp, (_np.zeros((2,), dtype="float32"),)),
                (_np.divmod, (_np.zeros((2,), dtype="int16"), 3)),
            )
        for ufunc, inputs in cases:
            expected = ufunc(*inputs)
            if not isinstance(expected, tuple):
                expected = (expected,)
            self.assertSequenceEqual(
                tuple(e.dtype for e in expected),
                ufunc_result_type(ufunc, inputs, casting="same_kind")
            )

        # Out of range Python int, as for the numpy.ufunc call.
        self.assertRaises(
            OverflowError,
            ufunc_result_type,
            _np.add,
            (_np.zeros((2,), dtype="int16"), 70000),
            None,
            "same_kind"
        )

        inputs = (_np.zeros((2,), dtype="float32"), _np.zeros((2,), dtype="float32"))
        self.assertSequenceEqual(
            (_np.dtype("float64"),),
            ufunc_result_type(_np.add, inputs, (_np.zeros((2,), dtype="float64"),))
        )
        self.assertRaises(
            TypeError,
            ufunc_result_type,
            _np.add,
            inputs,
            (_np.zeros((2,), dtype="int8"),),
            "same_kind"
        )
        self.assertSequenceEqual(
            (_np.dtype("float32"), _np.dtype("int32")),
            ufunc_result_type(
                _np.frexp,
                (_np.zeros((2,), dtype="float32"),),
                (None, None),
                "same_kind"
            )
        )

    def test_example(self):
        import numpy as np
        import mpi_array as mpia
        try:
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
        finally:
            inp[2].free()
            out[0].free()
            out[1].free()


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


class ToGndarrayConverter(object):

    """
    Base class for converting :obj:`numpy.ndarray` objects
    to :obj:`mpi_array.globale.gndarray` objects.
    """

    def __init__(self, **kwargs):
        """
        The :samp:`kwargs` are passed directly to the :func:`mpi_array.globale_creation.asarray`
        function in :meth:`__call__`.
        """
        self.kwargs = kwargs

    def __call__(self, npy_ary):
        """
        Converts the :samp:`{npy_ary}` to a :obj:`mpi_array.globale.gndarray` instance.

        :type npy_ary: :obj:`numpy.ndarray`
        :param npy_ary: Array converted to :obj:`mpi_array.globale.gndarray`.
           This array is assumed to be identical on all peer-rank MPI processes.
        :rtype: :obj:`mpi_array.globale.gndarray`
        :return: The :samp:`{npy_ary}` converted to a :obj:`mpi_array.globale.gndarray` instance.
        """
        if "halo" not in self.kwargs.keys():
            halo = _np.random.randint(low=1, high=4, size=(npy_ary.ndim, 2))
            gnd_ary = _asarray(npy_ary, halo=halo, **self.kwargs)
        else:
            gnd_ary = _asarray(npy_ary, **self.kwargs)

        return gnd_ary


class GetExtentsTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :func:`mpi_array.globale_ufunc.get_extents`.
    """

    def setUp(self):
        """
        """
        locale_comms, inter_locale_rank_to_peer_rank, this_locale = _comms.create_locale_comms()
        self.comms = locale_comms
        self.locale_info = this_locale
        del inter_locale_rank_to_peer_rank

    def test_scalar(self):
        """
        Test :func:`mpi_array.globale_ufunc.get_extents` with scalar input.
        """
        l, g = get_extents(5.0, self.locale_info)
        self.assertTrue(isinstance(l, _distribution.ScalarLocaleExtent))
        self.assertTrue(isinstance(g, _distribution.ScalarGlobaleExtent))


class GndarrayUfuncTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.globale_ufunc`.
    """

    def setUp(self):
        """
        Initialise :func:`numpy.random.seed`.
        """
        _np.random.seed(1531796312)
        self._rank_logger = _logging.get_rank_logger(self.id())
        with _asarray(_np.zeros((100,))) as gary:
            self.num_node_locales = gary.num_locales

    @property
    def rank_logger(self):
        """
        A :obj:`logging.Logger` object.
        """
        return self._rank_logger

    def compare_results(self, mpi_cln_npy_result_ary, mpi_result_ary):
        """
        Asserts that all elements of
        the :obj:`mpi_array.globale.gndarray` :samp:`{mpi_cln_npy_result_ary}`
        equal all elements of the  :obj:`mpi_array.globale.gndarray` :samp:`{mpi_result_ary}`.

        :type mpi_cln_npy_result_ary: :obj:`mpi_array.globale.gndarray`
        :param mpi_cln_npy_result_ary: The result returned by :samp:`{func}(*{func_args})`
            converted to a cloned-distribution :obj:`mpi_array.globale.gndarray`.
        :type mpi_result_ary: :obj:`mpi_array.globale.gndarray`
        :param mpi_result_ary: The result array
           from :meth:`mpi_array.globale.gndarray.__array_ufunc__` execution.
        """
        with \
                _zeros(
                    shape=mpi_result_ary.shape,
                    dtype=mpi_result_ary.dtype,
                    locale_type=LT_NODE,
                    distrib_type=DT_CLONED
                ) as mpi_cln_mpi_result_ary:

            _copyto(dst=mpi_cln_mpi_result_ary, src=mpi_result_ary)

            self.assertSequenceEqual(
                tuple(mpi_cln_npy_result_ary.shape),
                tuple(mpi_cln_mpi_result_ary.shape)
            )
            # All processes agree, so that a difference on one locale fails on all of them
            # (and they all leave the with statement).
            is_equal = \
                bool(
                    _np.all(
                        mpi_cln_npy_result_ary.lndarray_proxy.view_n
                        ==
                        mpi_cln_mpi_result_ary.lndarray_proxy.view_n
                    )
                )
            self.assertTrue(
                mpi_cln_mpi_result_ary.locale_comms.peer_comm.allreduce(is_equal, op=_mpi.LAND)
            )

    def convert_func_args_to_gndarrays(self, converter, func_args):
        """

        :type converter: :obj:`ToGndarrayConverter`
        :param converter: Used to convert the :obj:`numpy.ndarray` instances
           of :samp:`{func_args}` to :obj:`mpi_array.globale.gndarray` instances.
        :type func_args: sequence of :obj:`numpy.ndarray` or array-like objects
        :param func_args: Sequence of array-like objects.
           Can be comprised of misture of :obj:`numpy.ndarray` instances, scalars
           or (broadcastable) sequences (e.g. tuple of scalars) elements.
        :rtype: :obj:`list`
        :return: The :samp:`{func_args}` list with :obj:`numpy.ndarray` instances
           converted to :obj:`mpi_array.globale.gndarray` instances.
        """
        return [converter(arg) if isinstance(arg, _np.ndarray) else arg for arg in func_args]

    def do_convert_execute_and_compare(self, mpi_cln_npy_result_ary, converter, func, *func_args):
        """
        Compares the result of :samp:`{func}` called
        with :samp:`self.convert_func_args_to_gndarrays({converter}, {func_args})`
        converted arguments with the :samp:`{mpi_cln_npy_result_ary}` array (which should
        have been produced by
        calling :samp:`mpi_array.globale_creation.asarray({func}(*func_args))`).

        :type mpi_cln_npy_result_ary: :obj:`mpi_array.globale.gndarray`
        :param mpi_cln_npy_result_ary: The result returned by :samp:`{func}(*{func_args})`
            converted to a cloned-distribution :obj:`mpi_array.globale.gndarray`.
        :type converter: :obj:`ToGndarrayConverter`
        :param converter: Used to convert the :obj:`numpy.ndarray` instances
           of :samp:`{func_args}` to :samp:`mpi_array.globale.gndarray` instances.
        :type func: callable
        :param func: Function which computes a new array from the :samp:`*{func_args}`
            arguments and for arguments converted with :samp:`{converter}`.
        :type func_args: sequence of :obj:`numpy.ndarray` or array-like objects
        :param func_args: The arguments for the :samp:`{func}` function.
           Can be comprised of :obj:`numpy.ndarray`, scalars or broadcastable
           sequence (e.g. tuple of scalars) elements.

        .. seealso: :meth:`convert_func_args_to_gndarrays`

        """
        mpi_func_args = self.convert_func_args_to_gndarrays(converter, func_args)
        with func(*mpi_func_args) as mpi_result_ary:
            if mpi_cln_npy_result_ary is None:
                mpi_cln_npy_result_ary = func(*func_args)
            self.compare_results(mpi_cln_npy_result_ary, mpi_result_ary)
        for arg in mpi_func_args:
            if hasattr(arg, "free"):
                arg.free()

    def do_cloned_distribution_test(self, mpi_cln_npy_result_ary, func, *func_args):
        """
        Converts :obj:`numpy.ndarray` elements of :samp:`func_args`
        to :obj:`mpi_array.globale.gndarray` instances distributed as
        the :attr:`mpi_array.comms.DT_CLONED` distribution type.

        :type mpi_cln_npy_result_ary: :obj:`mpi_array.globale.gndarray`
        :param mpi_cln_npy_result_ary: The result returned by :samp:`{func}(*{func_args})`
            converted to a cloned-distribution :obj:`mpi_array.globale.gndarray`.
        :type func: callable
        :param func: Function which computes a new array from the :samp:`*{func_args}`
            arguments.
        :type func_args: sequence of :obj:`numpy.ndarray` or array-like objects
        :param func_args: The arguments for the :samp:`{func}` function.
           Can be comprised of :obj:`numpy.ndarray`, scalars or broadcastable
           sequence (e.g. tuple of scalars) elements.

        .. seealso: :meth:`do_convert_execute_and_compare`, :meth:`compare_results`
        """
        converter = ToGndarrayConverter(locale_type=LT_PROCESS, distrib_type=DT_CLONED)
        self.do_convert_execute_and_compare(mpi_cln_npy_result_ary, converter, func, *func_args)

        converter = ToGndarrayConverter(locale_type=LT_NODE, distrib_type=DT_CLONED, halo=0)
        self.do_convert_execute_and_compare(mpi_cln_npy_result_ary, converter, func, *func_args)

    def do_single_locale_distribution_test(self, mpi_cln_npy_result_ary, func, *func_args):
        """
        Converts :obj:`numpy.ndarray` elements of :samp:`func_args`
        to :obj:`mpi_array.globale.gndarray` instances distributed as
        the :attr:`mpi_array.comms.DT_SINGLE_LOCALE` distribution type.

        :type mpi_cln_npy_result_ary: :obj:`mpi_array.globale.gndarray`
        :param mpi_cln_npy_result_ary: The result returned by :samp:`{func}(*{func_args})`
            converted to a cloned-distribution :obj:`mpi_array.globale.gndarray`.
        :type func: callable
        :param func: Function which computes a new array from the :samp:`*{func_args}`
            arguments.
        :type func_args: sequence of :obj:`numpy.ndarray` or array-like objects
        :param func_args: The arguments for the :samp:`{func}` function.
           Can be comprised of :obj:`numpy.ndarray`, scalars or broadcastable
           sequence (e.g. tuple of scalars) elements.

        .. seealso: :meth:`do_convert_execute_and_compare`
        """
        class Converter(ToGndarrayConverter):

            def __call__(self, np_ary):
                gndary = ToGndarrayConverter.__call__(self, np_ary)
                num_locales = gndary.locale_comms.num_locales
                self.kwargs["inter_locale_rank"] = \
                    ((self.kwargs["inter_locale_rank"] + 1) % num_locales)
                return gndary

        converter = \
            Converter(locale_type=LT_PROCESS, distrib_type=DT_SINGLE_LOCALE, inter_locale_rank=0)
        self.do_convert_execute_and_compare(mpi_cln_npy_result_ary, converter, func, *func_args)

        converter = \
            Converter(locale_type=LT_NODE, distrib_type=DT_SINGLE_LOCALE, inter_locale_rank=0)
        self.do_convert_execute_and_compare(mpi_cln_npy_result_ary, converter, func, *func_args)

    def do_block_distribution_test(self, mpi_cln_npy_result_ary, func, *func_args):
        """
        Converts :obj:`numpy.ndarray` elements of :samp:`func_args`
        to :obj:`mpi_array.globale.gndarray` instances distributed as
        the :attr:`mpi_array.comms.DT_BLOCK` distribution type.

        :type mpi_cln_npy_result_ary: :obj:`mpi_array.globale.gndarray`
        :param mpi_cln_npy_result_ary: The result returned by :samp:`{func}(*{func_args})`
            converted to a cloned-distribution :obj:`mpi_array.globale.gndarray`.
        :type func: callable
        :param func: Function which computes a new array from the :samp:`*{func_args}`
            arguments.
        :type func_args: sequence of :obj:`numpy.ndarray` or array-like objects
        :param func_args: The arguments for the :samp:`{func}` function.
           Can be comprised of :obj:`numpy.ndarray`, scalars or broadcastable
           sequence (e.g. tuple of scalars) elements.

        .. seealso: :meth:`do_convert_execute_and_compare`

        """
        class Converter(ToGndarrayConverter):

            def __init__(self, **kwargs):
                ToGndarrayConverter.__init__(self, **kwargs)
                self.dims = None
                self.axis = 1

            def __call__(self, np_ary):
                if self.dims is None:
                    self.kwargs["dims"] = tuple(_np.zeros((np_ary.ndim,), dtype="int64"))
                gndary = ToGndarrayConverter.__call__(self, np_ary)
                self.axis = min([self.axis, np_ary.ndim])
                self.kwargs["dims"] = _np.ones((np_ary.ndim,), dtype="int64")
                self.kwargs["dims"][self.axis] = 0
                self.kwargs["dims"] = tuple(self.kwargs["dims"])
                self.dims = self.kwargs["dims"]
                self.axis = ((self.axis + 1) % np_ary.ndim)

                return gndary

        converter = Converter(locale_type=LT_PROCESS, distrib_type=DT_BLOCK)
        self.do_convert_execute_and_compare(mpi_cln_npy_result_ary, converter, func, *func_args)

        converter = Converter(locale_type=LT_NODE, distrib_type=DT_BLOCK)
        self.do_convert_execute_and_compare(mpi_cln_npy_result_ary, converter, func, *func_args)

    def do_multi_distribution_tests(self, func, *func_args):
        """
        Compares result of :samp:`{func}` called with :obj:`numpy.ndarray`
        arguments and result of :samp:`{func}` called with :obj:`mpi_array.globale.gndarray`
        arguments.
        Executes :samp:`{func}(*{func_args})` and compares the result
        with :samp:`{func}(*self.convert_func_args_to_gndarrays(converter, {func_args}))`,
        where multiple versions instances of :samp:`converter` are used to generate
        different distributions for the :obj:`mpi_array.globale.gndarray` :samp:`{func}`
        arguments.

        :type func: callable
        :param func: Function which computes a new array from the :samp:`*{func_args}`
            arguments.
        :type func_args: sequence of :obj:`numpy.ndarray` or array-like objects
        :param func_args: The arguments for the :samp:`{func}` function.
           Can be comprised of :obj:`numpy.ndarray`, scalars or broadcastable
           sequence (e.g. tuple of scalars) elements.

        .. seealso: :meth:`do_cloned_distribution_test`, :meth:`do_single_locale_distribution_test`
           , :meth:`do_block_distribution_test` and :meth:`do_convert_execute_and_compare`
        """
        with _asarray(func(*func_args)) as mpi_cln_npy_result_ary:
            self.do_cloned_distribution_test(mpi_cln_npy_result_ary, func, *func_args)
            self.do_single_locale_distribution_test(mpi_cln_npy_result_ary, func, *func_args)
            self.do_block_distribution_test(mpi_cln_npy_result_ary, func, *func_args)

    def test_umath_multiply(self):
        """
        Asserts that binary ufunc multiplication (:obj:`numpy.multiply`) computation
        for :obj:`mpi_array.globale.gndarray` arguments produces same results as
        for :obj:`numpy.ndarray` arguments. Tries various argument combinations
        and different distribution types for the :obj:`mpi_array.globale.gndarray`
        arguments.
        """
        per_axis_size_factor = int(_np.floor(_np.sqrt(float(self.num_node_locales))))
        gshape0 = (41 * per_axis_size_factor + 1, 43 * per_axis_size_factor + 3, 5)
        npy_ary0 = _np.random.uniform(low=0.5, high=1.75, size=gshape0)

        with _asarray(npy_ary0) as cln_ary:
            self.assertTrue(_np.all(npy_ary0 == cln_ary.lndarray_proxy.lndarray))

        def multiply(ary0, ary1):
            return ary0 * ary1

        self.do_multi_distribution_tests(multiply, npy_ary0, 1.0 / 3.0)
        self.do_multi_distribution_tests(multiply, npy_ary0, (0.1, 0.3, 0.5, 0.7, 1.9))
        self.do_multi_distribution_tests(multiply, npy_ary0, npy_ary0)

        gshape1 = gshape0[0:2] + (1,)
        npy_ary1 = _np.random.uniform(low=-0.5, high=2.9, size=gshape1)
        self.do_multi_distribution_tests(multiply, npy_ary0, npy_ary1)
        self.do_multi_distribution_tests(multiply, npy_ary1, npy_ary0)

        npy_ary0 = _np.random.uniform(low=-0.5, high=2.9, size=(gshape0[0], 1, gshape0[2]))
        npy_ary1 = _np.random.uniform(low=-0.5, high=2.9, size=(1, gshape0[1], gshape0[2]))
        self.do_multi_distribution_tests(multiply, npy_ary1, npy_ary0)

    def do_test_umath(self, halo=0, gshape=(32, 48)):
        """
        Test binary op for a :obj:`mpi_array.globale.gndarray` object
        and a scalar.
        """
        with _ones(gshape, dtype="int32", locale_type=_comms.LT_PROCESS, halo=halo) as c:
            # if True:
            #    c = _ones(gshape, dtype="int32", locale_type=_comms.LT_PROCESS, halo=halo)
            c_orig_halo = c.distribution.halo

            self.assertTrue(isinstance(c, _gndarray))
            self.assertTrue((c == 1).all())

            c *= 2
            self.assertTrue((c == 2).all())
            self.assertTrue(_np.all(c.distribution.halo == c_orig_halo))

            with (c + 2) as d:
                self.assertTrue(isinstance(d, _gndarray))
                self.assertEqual(c.dtype, d.dtype)
                self.assertTrue((d == 4).all())
                self.assertTrue(_np.all(d.distribution.halo == c_orig_halo))

    def test_umath_no_halo(self):
        """
        Test binary op for a :obj:`mpi_array.globale.gndarray` object
        and a scalar.
        """
        self.do_test_umath(halo=0)

    def test_umath_halo(self):
        """
        Test binary op for a :obj:`mpi_array.globale.gndarray` object
        and a scalar, test halo is preserved.
        """
        self.do_test_umath(halo=[[1, 2], [3, 4]])

    def do_test_umath_broadcast(self, halo=0, dims=(0, 0, 0)):
        """
        Test binary op for a :obj:`mpi_array.globale.gndarray` objects
        and an *array-like* object which requires requiring broadcast to result shape.
        """
        with \
                _ones(
                    (61, 55, 3),
                    dtype="int32",
                    locale_type=_comms.LT_PROCESS,
                    distrib_type=_comms.DT_BLOCK,
                    dims=dims,
                    halo=halo
                ) as c:
            c_orig_halo = c.distribution.halo

            with (c * (2, 2, 2)) as d:

                self.assertTrue(isinstance(d, _gndarray))
                self.assertEqual(_np.asarray((2, 2, 2)).dtype, d.dtype)
                self.assertSequenceEqual(tuple(c.shape), tuple(d.shape))
                self.assertSequenceEqual(d.distribution.halo.tolist(), c_orig_halo.tolist())
                self.assertTrue((d.view_n == 2).all())
                self.assertTrue((d == 2).all())

    def test_umath_broadcast_no_halo(self):
        """
        Test binary op for a :obj:`mpi_array.globale.gndarray` objects
        and an *array-like* object which requires requiring broadcast to result shape.
        """
        self.do_test_umath_broadcast(halo=0, dims=(0, 0, 0))
        self.do_test_umath_broadcast(halo=0, dims=(1, 1, 0))

    def test_umath_broadcast_halo(self):
        """
        Test binary op for a :obj:`mpi_array.globale.gndarray` objects
        and an *array-like* object which requires requiring broadcast to result shape.
        """
        self.do_test_umath_broadcast(halo=[[1, 2], [3, 4], [2, 1]], dims=(0, 0, 0))
        self.do_test_umath_broadcast(halo=[[1, 2], [3, 4], [2, 1]], dims=(1, 1, 0))

    def do_test_umath_broadcast_upsized_result(
        self,
        halo_a=0,
        halo_b=0,
        dims_a=(0, 0),
        dims_b=(0, 0, 0)
    ):
        """
        Test binary op for two :obj:`mpi_array.globale.gndarray` objects
        with the resulting :obj:`mpi_array.globale.gndarray` object having
        different (larger) shape than that of both inputs.
        """
        with \
                _ones(
                    (19, 3),
                    dtype="int32",
                    locale_type=_comms.LT_PROCESS,
                    distrib_type=_comms.DT_BLOCK,
                    dims=dims_a,
                    halo=halo_a
                ) as a, \
                _ones(
                    (23, 1, 3),
                    dtype="int32",
                    locale_type=_comms.LT_PROCESS,
                    distrib_type=_comms.DT_BLOCK,
                    dims=dims_b,
                    halo=halo_b
                ) as b:

            with (a + b) as d:

                self.assertTrue(isinstance(d, _gndarray))
                self.assertSequenceEqual((b.shape[0], a.shape[0], 3), tuple(d.shape))
                self.assertTrue((d == 2).all())

    def test_umath_broadcast_upsized_result(self):
        """
        Test binary op for two :obj:`mpi_array.globale.gndarray` objects
        with the resulting :obj:`mpi_array.globale.gndarray` object having
        different (larger) shape than that of both inputs.
        """
        self.do_test_umath_broadcast_upsized_result(
            halo_a=0,
            halo_b=0,
            dims_a=(0, 0),
            dims_b=(0, 0, 0)
        )
        self.do_test_umath_broadcast_upsized_result(
            halo_a=0,
            halo_b=0,
            dims_a=(0, 1),
            dims_b=(0, 0, 1)
        )
        self.do_test_umath_broadcast_upsized_result(
            halo_a=[[1, 2], [2, 1]],
            halo_b=0,
            dims_a=(0, 1),
            dims_b=(0, 0, 1)
        )
        self.do_test_umath_broadcast_upsized_result(
            halo_a=0,
            halo_b=[[1, 2], [3, 4], [2, 1]],
            dims_a=(0, 1),
            dims_b=(0, 0, 1)
        )
        self.do_test_umath_broadcast_upsized_result(
            halo_a=[[1, 2], [2, 1]],
            halo_b=[[1, 2], [3, 4], [2, 1]],
            dims_a=(0, 1),
            dims_b=(0, 0, 1)
        )

    def do_test_umath_distributed_broadcast(self, halo_a=0, halo_b=0):
        """
        Test binary op for two :obj:`mpi_array.globale.gndarray` objects
        which requires remote fetch of data when broadcasting to result shape.
        """
        with \
                _ones(
                    (61, 53, 5), dtype="int32", locale_type=_comms.LT_PROCESS, halo=halo_a
                ) as a, \
                _ones(a.shape, dtype="int32", locale_type=_comms.LT_PROCESS, halo=halo_b) as b:
            a_orig_halo = a.distribution.halo
            b_orig_halo = b.distribution.halo

            with (a + b) as c:

                self.assertTrue(isinstance(c, _gndarray))
                self.assertTrue((c == 2).all())
                self.assertSequenceEqual(c.distribution.halo.tolist(), a_orig_halo.tolist())

                with \
                        _ones(
                            tuple(a.shape[1:]),
                            dtype=c.dtype,
                            locale_type=_comms.LT_PROCESS,
                            dims=(0, 1),
                            halo=b_orig_halo[1:]
                        ) as twos:

                    twos.fill_h(2)

                    with (a * twos) as d:
                        self.assertTrue(isinstance(d, _gndarray))
                        self.assertSequenceEqual(tuple(a.shape), tuple(d.shape))
                        self.assertTrue((d == 2).all())

    def test_umath_distributed_broadcast_no_halo(self):
        """
        Test binary op for two :obj:`mpi_array.globale.gndarray` objects
        which requires remote fetch of data when broadcasting to result shape.
        """
        self.do_test_umath_distributed_broadcast(halo_a=0, halo_b=0)

    def test_umath_distributed_broadcast_halo(self):
        """
        Test binary op for two :obj:`mpi_array.globale.gndarray` objects
        which requires remote fetch of data when broadcasting to result shape.
        Ghost elements added to arrays.
        """
        self.do_test_umath_distributed_broadcast(halo_a=[[1, 2], [3, 4], [2, 1]], halo_b=0)
        self.do_test_umath_distributed_broadcast(halo_a=0, halo_b=[[1, 2], [3, 4], [2, 1]])
        self.do_test_umath_distributed_broadcast(
            halo_a=[[2, 1], [4, 3], [1, 2]],
            halo_b=[[1, 2], [3, 4], [2, 1]]
        )

    def test_umath_broadcast_new_output_distribution(self):
        """
        Ufuncs whose output shape differs from the shape of every gndarray input
        create a new distribution for the output. With locales of more than one
        process this could fail with mismatched collectives (and hang): the inter-locale
        processes passed their cartesian communicator to the (cached) communicators
        creation, the other processes passed :attr:`mpi4py.MPI.COMM_NULL`, so the cache
        could have an entry for some processes of a locale but not for the others.
        """
        for shape_a, shape_b in (((8, 1), (1, 8)), ((8, 1, 1), (1, 8, 1)), ((1, 8), (8, 1))):
            with \
                    _ones(shape_a, locale_type=LT_NODE, distrib_type=DT_BLOCK) as a, \
                    _ones(shape_b, locale_type=LT_NODE, distrib_type=DT_BLOCK) as b:
                with (a * b) as c:
                    self.assertSequenceEqual(_np.broadcast_shapes(shape_a, shape_b), c.shape)
                    self.assertTrue((c == 1).all())

    def test_output_shares_input_distribution(self):
        """
        The output of a ufunc which has the shape of a gndarray input shares the
        distribution (object) of that input, the distribution used to be copied. Copies
        between them then use the cached updates
        (:obj:`mpi_array.globale.RedistributeUpdatesCache`).
        """
        for distrib_type in (DT_BLOCK, DT_CLONED):
            with _ones((8, 13), locale_type=LT_PROCESS, distrib_type=distrib_type) as a:
                with (a + 1) as b, _np.multiply(a, b) as c:
                    self.assertTrue(b.comms_and_distrib.distribution is a.distribution)
                    self.assertTrue(c.comms_and_distrib.distribution is a.distribution)
                    self.assertTrue(b.locale_comms is a.locale_comms)
                    self.assertTrue((c == 2).all())

    def test_umath_broadcast_to_more_dims(self):
        """
        Ufuncs whose output has more axes (of length one) than the gndarray input
        with the most similar shape, e.g. inputs of shape :samp:`(41, 43)` and :samp:`(1, 1, 1)`
        give a :samp:`(1, 41, 43)` output. Creating the output used to raise :obj:`AttributeError`.
        """
        npy_ary = _np.random.uniform(low=0.5, high=1.75, size=(41, 43))
        npy_ones = _np.ones((1, 1, 1))

        def add_numpy_array(ary):
            return ary + npy_ones

        self.do_multi_distribution_tests(add_numpy_array, npy_ary)
        with \
                _ones((41, 43), locale_type=LT_NODE, distrib_type=DT_BLOCK) as a, \
                _ones((1, 1, 1), locale_type=LT_NODE, distrib_type=DT_BLOCK) as b:
            with (a + b) as c:
                self.assertSequenceEqual((1, 41, 43), c.shape)
                self.assertTrue((c == 2).all())

    def test_umath_zero_dim_gndarray_input(self):
        """
        Ufuncs of a 0-d gndarray (each locale has a copy of its element) and gndarrays
        with more dimensions. Checking whether remote data was needed raised :obj:`ValueError`
        for a 2-D output (all axes of the output were compared with the zero axes of the
        0-d input), and the 0-d gndarray itself was passed to the per-tile ufunc call.
        """
        npy_arys = \
            (
                _np.random.uniform(low=0.5, high=1.75, size=(41,)).astype("float32"),
                _np.random.uniform(low=0.5, high=1.75, size=(41, 43)).astype("float32"),
            )
        for distrib_type in (DT_CLONED, DT_SINGLE_LOCALE):
            for locale_type in (LT_PROCESS, LT_NODE):
                for value in (_np.float32(2.5), _np.float64(2.5)):
                    npy_value = _np.asarray(value)
                    for npy_ary in npy_arys:
                        with \
                                _asarray(
                                    value, distrib_type=distrib_type, locale_type=locale_type
                                ) as zero_dim, \
                                _asarray(
                                    npy_ary, locale_type=LT_PROCESS, distrib_type=DT_BLOCK
                                ) as gary:
                            for func in (_np.add, _np.multiply):
                                with \
                                        func(gary, zero_dim) as result, \
                                        _asarray(func(npy_ary, npy_value)) as cln_npy_result:
                                    self.assertEqual(cln_npy_result.dtype, result.dtype)
                                    self.compare_results(cln_npy_result, result)
                                with \
                                        func(zero_dim, gary) as result, \
                                        _asarray(func(npy_value, npy_ary)) as cln_npy_result:
                                    self.assertEqual(cln_npy_result.dtype, result.dtype)
                                    self.compare_results(cln_npy_result, result)

    def test_zero_dim_gndarrays(self):
        """
        Filling 0-d gndarrays and ufuncs with 0-d outputs, including in-place ones. Filling
        indexed the locale array with :samp:`()`, which gives a NumPy scalar (a copy),
        so e.g. :samp:`ones(())` was left uninitialised, and ufuncs with a 0-d output
        raised :obj:`TypeError`.
        """
        world = _mpi.COMM_WORLD
        size = world.size

        def all_equal(gary, value):
            gary.intra_locale_barrier()
            is_equal = bool(gary.lndarray_proxy.lndarray == value)
            return world.allreduce(is_equal, op=_mpi.LAND)

        for locale_type in (LT_PROCESS, LT_NODE):
            for distrib_type in (DT_CLONED, DT_SINGLE_LOCALE):
                kwargs = dict(locale_type=locale_type, distrib_type=distrib_type)
                with \
                        _ones((), **kwargs) as one, \
                        _full((), 5.0, **kwargs) as five, \
                        _zeros((), **kwargs) as seven, \
                        _asarray(_np.float64(2.5), **kwargs) as zero_dim, \
                        _ones((4 * size,), locale_type=LT_PROCESS, distrib_type=DT_BLOCK) as gary:
                    seven.fill(7.0)
                    self.assertTrue(all_equal(one, 1.0))
                    self.assertTrue(all_equal(five, 5.0))
                    self.assertTrue(all_equal(seven, 7.0))
                    with (zero_dim + 1) as result:
                        self.assertSequenceEqual((), result.shape)
                        self.assertTrue(all_equal(result, 3.5))
                    with (zero_dim * five) as result:
                        self.assertTrue(all_equal(result, 12.5))
                    with (gary + five) as result:
                        self.assertTrue((result == 6.0).all())
                    for i in range(3):
                        _np.add(zero_dim, 1.0, out=zero_dim)
                    self.assertTrue(all_equal(zero_dim, 5.5))
                    with zero_dim.copy() as copy:
                        self.assertTrue(all_equal(copy, 5.5))
                    with _array(zero_dim, dtype="int32") as int_zero_dim:
                        self.assertTrue(all_equal(int_zero_dim, 5))

    def test_umath_numpy_array_inputs(self):
        """
        Ufuncs of :obj:`numpy.ndarray` and :obj:`mpi_array.globale.gndarray` inputs.
        The output gets the distribution of a gndarray input. A :obj:`numpy.ndarray`
        input used to be chosen when it came first (or was closest to the result shape),
        and creating the output raised :obj:`AttributeError`.
        """
        gshape = (41, 43)
        npy_ary0 = _np.random.uniform(low=0.5, high=1.75, size=gshape)
        npy_ary1 = _np.random.uniform(low=0.5, high=1.75, size=gshape)
        npy_col = _np.random.uniform(low=0.5, high=1.75, size=(gshape[0], 1))
        npy_row = _np.random.uniform(low=0.5, high=1.75, size=(1, gshape[1]))

        def add_numpy_array_first(ary):
            return npy_ary1 + ary

        def multiply_numpy_array_closest(ary):
            # The (1, 43) numpy array is closer to the (41, 43) result shape
            # than the (41, 1) gndarray.
            return npy_row * ary

        self.do_multi_distribution_tests(add_numpy_array_first, npy_ary0)
        self.do_multi_distribution_tests(multiply_numpy_array_closest, npy_col)

        with \
                _asarray(
                    npy_ary0, locale_type=LT_PROCESS, distrib_type=DT_BLOCK, halo=2
                ) as gary:
            with _np.add(npy_ary1, gary) as gary_sum:
                self.assertSequenceEqual(
                    gary.distribution.halo.tolist(),
                    gary_sum.distribution.halo.tolist()
                )
                self.assertSequenceEqual(
                    gary.lndarray_proxy.locale_extent.to_tuple(),
                    gary_sum.lndarray_proxy.locale_extent.to_tuple()
                )

    def test_python_scalar_dtype_parity(self):
        """
        Result dtypes of :obj:`mpi_array.globale.gndarray` and Python scalar ufunc
        operations match those of :obj:`numpy.ndarray` (Python scalars are weakly
        typed, NEP 50), and in-place operations keep the array dtype.
        """
        gshape = (16, 9)
        cases = \
            (
                ("uint16", 1),
                ("int8", 2.5),
                ("float32", 2.0),
                ("float32", 2.0j),
                ("int32", 2),
                ("bool", 1),
                ("int16", _np.int64(2)),
                ("int16", _np.asarray(2)),
            )
        for dtype, scalar in cases:
            npy_result = _np.ones(gshape, dtype=dtype) + scalar
            with _ones(gshape, dtype=dtype, locale_type=LT_PROCESS) as gary:
                with (gary + scalar) as gary_result:
                    self.assertEqual(npy_result.dtype, gary_result.dtype)
                    self.assertTrue((gary_result == npy_result.flat[0]).all())

        with _ones(gshape, dtype="uint16", locale_type=LT_PROCESS) as gary:
            gary += 1
            self.assertEqual(_np.dtype("uint16"), gary.dtype)
            self.assertTrue((gary == 2).all())

    def test_python_scalar_out_of_range(self):
        """
        A Python :obj:`int` which is out of range for the array dtype raises :obj:`OverflowError`
        (like numpy) on every process, including processes whose locale has no elements
        (they skip the per-tile numpy call, so used to return normally while the other
        processes raised, and the processes then deadlocked).
        """
        world = _mpi.COMM_WORLD
        shape = (max((1, world.size - 1)),)
        with _ones(shape, dtype="int16", locale_type=LT_PROCESS, distrib_type=DT_BLOCK) as gary:
            self.assertRaises(OverflowError, lambda: gary + 70000)
            with self.assertRaises(OverflowError):
                gary += 70000
            self.assertTrue((gary == 1).all())

    def test_ufunc_write_after_read(self):
        """
        A ufunc which fetches remote input elements must not return on any locale
        before all locales have finished fetching. Otherwise a locale can overwrite
        an input while another locale is still reading it (:samp:`c = b + a; a += 1`).
        A delay is injected into the remote fetch on all ranks except rank 0.
        """
        world = _mpi.COMM_WORLD
        shape = (100 * world.size,)
        orig_do_locale_rma_update = _update.RmaUpdateExecutor.do_locale_rma_update

        def delayed_do_locale_rma_update(executor, updates):
            if world.rank != 0:
                _time.sleep(1.0)
            return orig_do_locale_rma_update(executor, updates)

        with \
                _zeros(shape, locale_type=LT_PROCESS, distrib_type=DT_SINGLE_LOCALE) as a, \
                _zeros(shape, locale_type=LT_PROCESS, distrib_type=DT_BLOCK) as b:
            a.fill(1.0)
            _update.RmaUpdateExecutor.do_locale_rma_update = delayed_do_locale_rma_update
            try:
                # c has the block distribution of b, all locales except
                # locale 0 fetch their part of a from locale 0.
                with (b + a) as c:
                    a += 1.0
                    self.assertTrue((c == 1.0).all())
                    self.assertTrue((a == 2.0).all())
            finally:
                _update.RmaUpdateExecutor.do_locale_rma_update = orig_do_locale_rma_update

    def test_ufunc_frees_fetched_inputs(self):
        """
        The memory which a ufunc allocates for the input elements it fetches from
        remote locales (:meth:`mpi_array.globale.gndarray.locale_get`) is freed before
        the ufunc returns, it used to be leaked on every call. Tests single process
        locales and locales of (up to) two processes, which allocate shared memory.
        """
        world = _mpi.COMM_WORLD
        shape = (100 * world.size,)
        orig_locale_get = _gndarray.locale_get
        fetched_arys = []

        def recording_locale_get(gary, *args, **kwargs):
            locale_ary = orig_locale_get(gary, *args, **kwargs)
            if isinstance(locale_ary, _win_lndarray):
                fetched_arys.append(locale_ary)
            return locale_ary

        node_comm = world.Split_type(_mpi.COMM_TYPE_SHARED, key=world.rank)
        intra_locale_comms = \
            (_mpi.COMM_SELF, node_comm.Split(node_comm.rank // 2, node_comm.rank))
        node_comm.Free()
        for intra_locale_comm in intra_locale_comms:
            kwargs = dict(peer_comm=world, intra_locale_comm=intra_locale_comm)
            with \
                    _ones(shape, distrib_type=DT_SINGLE_LOCALE, **kwargs) as a, \
                    _ones(shape, distrib_type=DT_BLOCK, **kwargs) as b:
                _gndarray.locale_get = recording_locale_get
                try:
                    # c has the block distribution of b, all locales except
                    # locale 0 fetch their part of a from locale 0.
                    for i in range(3):
                        with (b + a) as c:
                            self.assertTrue((c == 2).all())
                finally:
                    _gndarray.locale_get = orig_locale_get

                if a.num_locales > 1:
                    self.assertGreater(world.allreduce(len(fetched_arys), op=_mpi.SUM), 0)
                all_freed = all(locale_ary.win == _mpi.WIN_NULL for locale_ary in fetched_arys)
                self.assertTrue(world.allreduce(all_freed, op=_mpi.LAND))
                del fetched_arys[:]

    def test_ufunc_outputs_of_different_dtypes(self):
        """
        Each output of a ufunc gets its own type, e.g. :obj:`numpy.frexp` gives
        a float mantissa and an int exponent. The outputs after the first used to get
        the type of the first output.
        """
        npy_ary = _np.linspace(0.25, 100.0, 64 * _mpi.COMM_WORLD.size)
        npy_mantissa, npy_exponent = _np.frexp(npy_ary)
        for locale_type in (LT_PROCESS, LT_NODE):
            with _asarray(npy_ary, locale_type=locale_type, distrib_type=DT_BLOCK) as gary:
                mantissa, exponent = _np.frexp(gary)
                with mantissa, exponent:
                    self.assertEqual(npy_mantissa.dtype, mantissa.dtype)
                    self.assertEqual(npy_exponent.dtype, exponent.dtype)
                    with _asarray(npy_mantissa) as cln_mantissa:
                        self.compare_results(cln_mantissa, mantissa)
                    with _asarray(npy_exponent) as cln_exponent:
                        self.compare_results(cln_exponent, exponent)

    def test_ufunc_casting_arg(self):
        """
        Test ufunc with casting argument.
        """
        self.rank_logger.info("Calling ufunc with 'casting' kwarg.")
        with \
                _ones(
                    (64, 32, 100),
                    dtype="float32",
                    locale_type=_comms.LT_PROCESS,
                    dims=(0, 0, 0),
                    halo=2
                ) as a:

            _np.add(a, 1, casting="same_kind", out=a)
            self.assertTrue((a == 2).all())

    def test_check_equivalent_inter_locale_comms(self):

        with \
                _empty((50, 50, 50), locale_type=LT_NODE) as gary0, \
                _empty((50, 50, 50), locale_type=LT_PROCESS) as gary1:

            if gary1.distribution.num_locales > 1:
                self.assertRaises(
                    ValueError,
                    check_equivalent_inter_locale_comms,
                    (gary0, gary1)
                )

    def test_ufunc_multiple_out_arrays(self):
        """
        Ufunc with several output arrays given as :samp:`out` on locales of more than one
        process. The processes which are not inter-locale processes have :samp:`COMM_NULL`
        inter-locale communicators, checking that the outputs have equivalent
        communicators compared them, which raised an MPI error on those processes.
        """
        npy_ary = _np.linspace(0.25, 100.0, 64 * _mpi.COMM_WORLD.size)
        npy_mantissa, npy_exponent = _np.frexp(npy_ary)
        for locale_type in (LT_NODE, LT_PROCESS):
            with \
                    _asarray(npy_ary, locale_type=locale_type, distrib_type=DT_BLOCK) as gary, \
                    _zeros_like(gary) as mantissa, \
                    _zeros_like(gary, dtype=npy_exponent.dtype) as exponent:
                check_equivalent_inter_locale_comms((gary, mantissa, exponent))
                outputs = _np.frexp(gary, out=(mantissa, exponent))
                self.assertTrue(outputs[0] is mantissa)
                self.assertTrue(outputs[1] is exponent)
                with _asarray(npy_mantissa) as cln_mantissa:
                    self.compare_results(cln_mantissa, mantissa)
                with _asarray(npy_exponent) as cln_exponent:
                    self.compare_results(cln_exponent, exponent)

    def test_ufunc_out_with_none(self):
        """
        Ufunc with :samp:`out` arrays and :samp:`None` entries, a new output array is
        created for each :samp:`None` entry. They used to raise :obj:`AttributeError`.
        """
        npy_ary = _np.linspace(0.25, 100.0, 64 * _mpi.COMM_WORLD.size)
        npy_mantissa, npy_exponent = _np.frexp(npy_ary)
        for locale_type in (LT_NODE, LT_PROCESS):
            with \
                    _asarray(npy_ary, locale_type=locale_type, distrib_type=DT_BLOCK) as gary, \
                    _zeros_like(gary) as mantissa, \
                    _zeros_like(gary, dtype=npy_exponent.dtype) as exponent:
                for out in ((None, exponent), (mantissa, None), (None, None)):
                    outputs = _np.frexp(gary, out=out)
                    for i in range(2):
                        if out[i] is not None:
                            self.assertTrue(outputs[i] is out[i])
                    with \
                            _asarray(npy_mantissa) as cln_mantissa, \
                            _asarray(npy_exponent) as cln_exponent:
                        self.assertEqual(npy_mantissa.dtype, outputs[0].dtype)
                        self.assertEqual(npy_exponent.dtype, outputs[1].dtype)
                        self.compare_results(cln_mantissa, outputs[0])
                        self.compare_results(cln_exponent, outputs[1])
                    for i in range(2):
                        if out[i] is None:
                            outputs[i].free()
                with _np.add(gary, 1.0, out=(None,)) as gary_sum:
                    with _asarray(npy_ary + 1.0) as cln_sum:
                        self.compare_results(cln_sum, gary_sum)

    def test_check_equivalent_inter_locale_comms_on_all_processes(self):
        """
        :func:`mpi_array.globale_ufunc.check_equivalent_inter_locale_comms` raises
        :obj:`ValueError` on every process when the inter-locale communicators aren't
        equivalent, also on the processes which aren't inter-locale processes
        (:samp:`COMM_NULL` communicators). Those used to carry on (e.g. into the ufunc)
        while the inter-locale processes raised. Here the communicators have the same
        processes in a different order (:attr:`mpi4py.MPI.SIMILAR`), the locales have
        up to two processes.
        """
        world = _mpi.COMM_WORLD
        node_comm = world.Split_type(_mpi.COMM_TYPE_SHARED, key=world.rank)
        intra_locale_comm = node_comm.Split(node_comm.rank // 2, node_comm.rank)
        node_comm.Free()
        color = 0 if intra_locale_comm.rank == 0 else _mpi.UNDEFINED
        inter_locale_comm = world.Split(color, world.rank)
        reversed_inter_locale_comm = world.Split(color, -world.rank)
        kwargs = \
            dict(
                locale_type=LT_NODE,
                distrib_type=DT_CLONED,
                peer_comm=world,
                intra_locale_comm=intra_locale_comm
            )
        npy_ary = _np.linspace(0.25, 100.0, 16)
        with \
                _asarray(npy_ary, inter_locale_comm=inter_locale_comm, **kwargs) as gary, \
                _zeros((16,), inter_locale_comm=inter_locale_comm, **kwargs) as mantissa, \
                _zeros(
                    (16,), dtype="int32", inter_locale_comm=reversed_inter_locale_comm, **kwargs
                ) as exponent:
            # With a single locale the communicators are the same (CONGRUENT).
            expect_raise = gary.num_locales > 1
            for func in \
                    (
                        lambda: check_equivalent_inter_locale_comms((mantissa, exponent)),
                        lambda: _np.frexp(gary, out=(mantissa, exponent)),
                    ):
                try:
                    func()
                    raised = False
                except ValueError:
                    raised = True
                self.assertTrue(world.allreduce(raised == expect_raise, op=_mpi.LAND))

    def test_ufunc_out_broadcast(self):
        """
        Ufunc with :samp:`out` arrays to which the inputs broadcast, as for NumPy (the outputs
        aren't broadcast). Checking the :samp:`out` shapes used to compare them with the
        broadcast shape of the inputs only, and raised :obj:`ValueError`.
        """
        size = _mpi.COMM_WORLD.size
        for locale_type in (LT_PROCESS, LT_NODE):
            kwargs = dict(locale_type=locale_type, distrib_type=DT_BLOCK)
            with \
                    _ones((3,), **kwargs) as row, \
                    _zeros((2 * size, 3), **kwargs) as out_2d, \
                    _zeros((3,), **kwargs) as out_1d, \
                    _asarray(_np.float64(2.5), locale_type=locale_type) as zero_dim:
                self.assertTrue(_np.add(row, 1, out=out_2d) is out_2d)
                self.assertTrue((out_2d == 2).all())
                self.assertTrue(_np.add(zero_dim, 1, out=out_1d) is out_1d)
                self.assertTrue((out_1d == 3.5).all())
                self.assertTrue(_np.add(1.0, 2.0, out=out_1d) is out_1d)
                self.assertTrue((out_1d == 3).all())

    def test_ufunc_errors_on_all_processes(self):
        """
        A ufunc error of the per-tile call on some processes of a locale only (one process
        of a locale computes a 0-d output, processes with empty tiles compute nothing)
        is raised on all processes of the locale, the others used to wait in a barrier.
        The errors are agreed within each locale (the arrays are cloned, every locale
        computes the same elements). An :samp:`out` array of the wrong shape raises
        :obj:`ValueError` on every process.
        """
        world = _mpi.COMM_WORLD

        def raises_on_all(func):
            try:
                func()
                raised = False
            except ValueError:
                raised = True
            return world.allreduce(raised, op=_mpi.LAND)

        kwargs = dict(locale_type=LT_NODE, distrib_type=DT_BLOCK)
        cloned = dict(locale_type=LT_NODE, distrib_type=DT_CLONED)
        with \
                _asarray(_np.int64(2), **cloned) as zero_dim, \
                _asarray(_np.array([2], dtype="int64"), **cloned) as one_element, \
                _zeros((), **cloned) as zero_dim_float, \
                _ones((8,), **kwargs) as gary, \
                _zeros((4,), **kwargs) as wrong_shape:
            # Integers to negative integer powers are not allowed.
            self.assertTrue(raises_on_all(lambda: zero_dim ** -1))
            self.assertTrue(raises_on_all(lambda: one_element ** -1))
            self.assertTrue(raises_on_all(lambda: _np.add(gary, 1.0, out=zero_dim_float)))
            self.assertTrue(raises_on_all(lambda: _np.add(gary, 1.0, out=wrong_shape)))
            self.assertTrue((gary == 1).all())

    def test_gufunc_not_implemented(self):
        """
        Generalised ufuncs (e.g. :obj:`numpy.matmul`) used to be computed tile by tile,
        giving wrong results with more than one locale. They must raise :obj:`TypeError`.
        """
        npy_a = _np.arange(16, dtype="float64").reshape(4, 4)
        npy_b = npy_a.T + 1.0
        for distrib_type in (DT_CLONED, DT_BLOCK):
            with \
                    _asarray(npy_a, locale_type=LT_PROCESS, distrib_type=distrib_type) as a, \
                    _asarray(npy_b, locale_type=LT_PROCESS, distrib_type=distrib_type) as b:
                self.assertRaises(TypeError, lambda: a @ b)
                self.assertRaises(TypeError, lambda: npy_a @ b)
                self.assertRaises(TypeError, _np.matmul, a, b)
                self.assertRaises(TypeError, _np.matmul, a, npy_b)
                if hasattr(_np, "vecdot"):
                    self.assertRaises(TypeError, _np.vecdot, a, b)

    def test_not_implemented(self):
        uf = _np.add
        with _empty((50, 50, 50), locale_type=LT_NODE) as gary0:

            method_args = \
                (
                    ("reduce", (gary0,)),
                    ("accumulate", (gary0,)),
                    ("reduceat", (gary0, [0, 1])),
                    ("at", (gary0, [0, 1], 5.0)),
                    ("outer", (gary0, 5.0)),
                )
            for method, args in method_args:
                self.assertRaises(
                    TypeError,
                    getattr(uf, method),
                    *args
                )


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
