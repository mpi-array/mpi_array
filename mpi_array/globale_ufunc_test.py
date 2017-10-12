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

import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .comms import LT_NODE, LT_PROCESS, DT_CLONED, DT_SINGLE_LOCALE, DT_BLOCK  # , DT_SLAB
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
                _np.array([1, 2], dtype='uint16')
            )
        self.assertRaises(
            ValueError,
            ufunc_result_type, uft, inputs, outputs
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
            self.assertTrue(
                _np.all(
                    mpi_cln_npy_result_ary.lndarray_proxy.view_n
                    ==
                    mpi_cln_mpi_result_ary.lndarray_proxy.view_n
                )
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

        converter = ToGndarrayConverter(locale_type=LT_NODE, distrib_type=DT_CLONED)
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
                _ones((61, 53, 5), dtype="int32", locale_type=_comms.LT_PROCESS, halo=halo_a) as a,\
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


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
