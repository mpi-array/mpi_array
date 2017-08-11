"""
=========================================
The :mod:`mpi_array.globale_ufunc` Module
=========================================

Defines :obj:`numpy.ufunc` functions for :obj:`mpi_array.globale.gndarray`.

Classes
=======

.. autosummary::
   :toctree: generated/

   GndarrayArrayUfuncExecutor - Creates :obj:`gndarray` outputs and forwards to `numpy.ufunc`.

Functions
=========

.. autosummary::
   :toctree: generated/

   ufunc_result_type - Like :func:`numpy.result_type`.
   broadcast_shape - Calculates broadcast shape from sequence of shape arguments.
   gndarray_array_ufunc - A :obj:`numpy.ndarray` like distributed array.


"""

from __future__ import absolute_import

import numpy as _np
import copy as _copy

from .license import license as _license, copyright as _copyright, version as _version
from . import logging as _logging  # noqa: E402,F401
from . import globale_creation as _globale_creation

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def ufunc_result_type(ufunc_types, inputs, outputs=None, casting="safe"):
    """
    Like :obj:`numpy.result_type`, but
    handles :obj:`mpi_array.globale.gndarray` in the :samp:`{inputs}`
    and handles multuple :samp:`{outputs}`.

    :type ufunc_types: sequence of `str`
    :param ufunc_types: The :attr:`numpy.ufunc.types` attribute,
       e.g. :samp:`['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', ..., 'mm->m', 'mM->M', 'OO->O']`.
    :type inputs: sequence of :obj:`object`
    :param inputs: The inputs (e.g. :obj:`numpy.ndarray`, scalars
       or :obj:`mpi_array.globale.gndarray`) to a :obj:`numpy.ufunc` call.
    :type outputs: :samp:`None` or sequence of :obj:`object`
    :param outputs: The output arrays these are explicitly checked casting correctness.
    :type casting: :obj:`str` :samp:`{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}`
    :param casting: Casting mode. See :func:`numpy.can_cast`.
    :rtype: :obj:`tuple` of :obj:`numpy.dtype`
    :return: A tuple of :obj:`numpy.dtype` indicating the output types produced for
       the given inputs.
    :raises ValueError: If the the inputs (and outputs) cannot be cast to an
       appropriate element of :samp:`{ufunc_types}`.

    Example::

       >>> import numpy as np
       >>> import mpi_array as mpia
       >>> inp = (
       ... np.zeros((10,10,10), dtype='float16'),
       ... 16.0,
       ... mpia.zeros((10,10,10), dtype='float32'),
       ... )
       >>> ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp)
       (dtype('float32'), dtype('bool'))
       >>> out = (mpia.zeros((10,10,10), dtype="float64"),)
       >>> ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp, outputs=out)
       (dtype('float64'), dtype('bool'))
       >>> out += (mpia.zeros((10, 10, 10), dtype="uint16"),)
       >>> ufunc_result_type(['eee->e?', 'fff->f?', 'ddd->d?'], inputs=inp, outputs=out)
       (dtype('float64'), dtype('uint16'))
    """
    result_dtypes = None
    ufunc_in_types = tuple(in2out_str.split("->")[0] for in2out_str in ufunc_types)
    ufunc_in_dtypes = \
        _np.asarray(
            tuple(
                tuple(_np.dtype(c) for c in ufunc_in_types[i])
                for i in range(len(ufunc_in_types))
            )
        )
    ufunc_out_types = tuple(in2out_str.split("->")[1] for in2out_str in ufunc_types)
    ufunc_out_dtypes = \
        _np.asarray(
            tuple(
                tuple(_np.dtype(c) for c in ufunc_out_types[i])
                for i in range(len(ufunc_out_types))
            )
        )

    in_dtypes = \
        _np.asarray(
            tuple(
                input.dtype
                if hasattr(input, "dtype") else _np.asarray(input).dtype
                for input in inputs
            )
        )
    out_dtypes = None
    if (outputs is not None) and (len(outputs) > 0):
        out_dtypes = \
            _np.asarray(
                tuple(
                    output.dtype
                    if hasattr(output, "dtype") else _np.asarray(output).dtype
                    for output in outputs
                )
            )

    idx = None
    idxs = _np.where(_np.logical_and.reduce(ufunc_in_dtypes == in_dtypes, axis=1))
    if len(idxs) > 0 and len(idxs[0]) > 0:
        idx = idxs[0][0]

    if idx is None:
        in_scalars_and_dtypes = \
            _np.where(
                _np.asarray(tuple(dt.ndim <= 0 for dt in in_dtypes)),
                inputs,
                in_dtypes
            )
        idxs = \
            _np.where(
                _np.asarray(
                    tuple(
                        _np.all(
                            tuple(
                                _np.can_cast(
                                    in_scalars_and_dtypes[j],
                                    ufunc_in_dtypes[i, j],
                                    casting=casting
                                )
                                for j in range(ufunc_in_dtypes.shape[1])
                            )
                        )
                        for i in range(ufunc_in_dtypes.shape[0])
                    )
                )
            )
        if len(idxs) > 0 and len(idxs[0]) > 0:
            idx = idxs[0][0]

    if idx is not None:
        ufunc_out_dtypes_for_in = ufunc_out_dtypes[idx]
        if (
            (out_dtypes is not None)
            and
            _np.any(ufunc_out_dtypes_for_in[:len(out_dtypes)] != out_dtypes)
        ):
            if (
                _np.any(
                    tuple(
                        not _np.can_cast(ufunc_out_dtypes_for_in[i], out_dtypes[i], casting=casting)
                        for i in range(len(out_dtypes))
                    )
                )
            ):
                raise ValueError(
                    "Could not cast ufunc-output-types %s to desired output-types = %s."
                    %
                    (
                        tuple(ufunc_out_dtypes_for_in),
                        tuple(out_dtypes)
                    )
                )
        if out_dtypes is None:
            out_dtypes = _np.array((), dtype='O')
        result_dtypes = \
            tuple(
                out_dtypes.tolist()
                +
                ufunc_out_dtypes_for_in[len(out_dtypes):].tolist()
            )
    else:
        raise ValueError(
            "Could not cast inputs types = %s to ufunc types=\n%s"
            %
            (in_dtypes, ufunc_in_dtypes, )
        )

    return result_dtypes


def broadcast_shape(*shape_args):
    """
    Returns
    the :mod:`numpy` `broadcast <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    shape for the give shape arguments.

    :type shape1, shape2, ...: sequence of `int`
    :param shape1, shape2, ...: Array shapes to be broadcast.
    :rtype: sequence of `int`
    :return: The broadcast shape.

    Examples::

        >>> broadcast_shape((4,), (4,))
        (4,)
        >>> broadcast_shape((4, 1), (1, 5))
        (4, 5)
        >>> broadcast_shape((4, 1, 3, 7), (1, 8, 1, 7))
        (4, 8, 3, 7)
        >>> broadcast_shape((3, 7), ())
        (3, 7)
    """
    ndim = _np.max(tuple(len(shape) for shape in shape_args))

    bcast_shape = ()
    if ndim > 0:
        ndim_shapes = \
            _np.asarray(tuple((1,) * (ndim - len(shape)) + tuple(shape) for shape in shape_args))
        bcast_shape = _np.amax(ndim_shapes, axis=0)

        if (_np.any(_np.logical_and(ndim_shapes != 1, ndim_shapes != bcast_shape))):
            raise ValueError(
                "shape mismatch - objects cannot be broadcast to a single shape:\n%s"
                %
                (shape_args,)
            )

        bcast_shape = tuple(bcast_shape)

    return bcast_shape


class GndarrayArrayUfuncExecutor(object):
    """
    """

    def __init__(self, array_like_obj, ufunc, method, *inputs, **kwargs):
        """
        """
        self._array_like_obj = array_like_obj
        self._ufunc = ufunc
        self._method = method
        self._inputs = inputs
        self._kwargs = kwargs
        self._outputs = None
        if "out" in self._kwargs.keys():
            self._outputs = self._kwargs["out"]
        self._casting = None
        if "casting" in self._kwargs.keys():
            self._casting = self._kwargs["casting"]

    @property
    def peer_comm(self):
        """
        """
        return self._array_like_obj.locale_comms.peer_comm

    @property
    def intra_locale_comm(self):
        """
        """
        return self._array_like_obj.locale_comms.intra_locale_comm

    @property
    def inter_locale_comm(self):
        """
        """
        return self._array_like_obj.locale_comms.inter_locale_comm

    @property
    def ufunc(self):
        """
        """
        return self._ufunc

    @property
    def outputs(self):
        """
        """
        return self._outputs

    @property
    def inputs(self):
        """
        """
        return self._inputs

    @property
    def casting(self):
        """
        """
        return self._casting

    @property
    def method(self):
        """
        """
        return self._method

    def get_inputs_shapes(self):
        """
        """
        return \
            tuple(
                input.shape
                if hasattr(input, "shape") else
                _np.asarray(
                    input,
                    peer_comm=self.peer_comm,
                    intra_locale_comm=self.intra_locale_comm,
                    inter_locale_comm=self.inter_locale_comm
                ).shape
                for input in self._inputs
            )

    def get_best_match_input(self, result_shape):
        """
        """
        return None

    def execute___call__(self):
        """
        """
        result_shape = broadcast_shape(*(self.get_inputs_shapes()))
        result_types = ufunc_result_type(self.ufunc.types, self.inputs, self.outputs, self.casting)
        outputs = self.outputs

        best_match_input = self.get_best_match_input(result_shape)
        if outputs is None:
            outputs = ()

        if best_match_input is not None:
            outputs = \
                (
                    outputs
                    +
                    tuple(
                        _globale_creation.empty_like(best_match_input, dtype=result_types[i])
                        for i in range(len(outputs), len(result_types))
                    )
                )
        else:
            outputs = \
                (
                    outputs
                    +
                    tuple(
                        _globale_creation.empty(result_shape, dtype=result_types[i])
                        for i in range(len(outputs), len(result_types))
                    )
                )

        kwargs = _copy.copy(self._kwargs)
        inputs = tuple(input.lndarray_proxy.lndarray for input in self.inputs)
        kwargs["out"] = tuple(output.lndarray_proxy.lndarray for output in outputs)

        self.ufunc.__call__(*(inputs), **kwargs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def execute_accumulate(self):
        """
        """
        return NotImplemented()

    def execute_reduce(self):
        """
        """
        return NotImplemented()

    def execute_reduceat(self):
        """
        """
        return NotImplemented()

    def execute_at(self):
        """
        """
        return NotImplemented()

    def execute_outer(self):
        """
        """
        return NotImplemented()

    def execute(self):
        """
        Perform the ufunc operation.
        """
        return getattr(self, "execute_" + self.method)()


gndarray_ufunc_executor_factory = GndarrayArrayUfuncExecutor


def gndarray_array_ufunc(array_like_obj, ufunc, method, *inputs, **kwargs):
    """
    The implementation for  :meth:`mpi_array.globale.gndarray.__array_ufunc__`.
    """
    ufunc_executor = \
        gndarray_ufunc_executor_factory(
            array_like_obj,
            ufunc,
            method,
            *inputs,
            **kwargs
        )

    return ufunc_executor.execute()


__all__ = [s for s in dir() if not s.startswith('_')]
