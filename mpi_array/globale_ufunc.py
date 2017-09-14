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
   shape_extend_dims - Prepend ones to 1D *shape* sequence to make it a specified dimension.
   gndarray_array_ufunc - A :obj:`numpy.ndarray` like distributed array.


"""

from __future__ import absolute_import

import numpy as _np
import copy as _copy

from .license import license as _license, copyright as _copyright, version as _version
from . import logging as _logging  # noqa: E402,F401
from . import globale_creation as _globale_creation
from . import comms as _comms

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def ufunc_result_type(ufunc_types, inputs, outputs=None, casting="safe"):
    """
    Like :obj:`numpy.result_type`, but
    handles :obj:`mpi_array.globale.gndarray` in the :samp:`{inputs}`
    and handles multiple :samp:`{outputs}` cases.

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


def shape_extend_dims(ndim, shape):
    """
    Returns :obj:`shape` pre-prepended with ones so returned 1D array has length :samp:`{ndim}`.

    :type ndim: :obj:`int`
    :param ndim: Length of returned 1D sequence.
    :type shape: sequence of :obj:`object`
    :param shape: Length of returned 1D sequence.
    :rtype: :obj:`tuple`
    :return: Sequence pre-pended with one elements so that sequence length equals :samp:`{ndim}`.

    Example::

       >>> shape_extend_dims(5, (3, 1, 5))
       (1, 1, 3, 1, 5)
       >>> shape_extend_dims(3, (3, 1, 5))
       (3, 1, 5)
       >>> shape_extend_dims(1, (3, 1, 5))
       (3, 1, 5)

    """
    return (1,) * (ndim - len(shape)) + tuple(shape)


class GndarrayArrayUfuncExecutor(object):

    """
    Instances execute a ufunc for a :obj:`mpi_array.globale.gndarray`.
    Takes care of creating outputs, fetching required parts of inputs
    and forwarding call to :obj:`numpy.ufunc` instance to perform
    the computation.
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
        else:
            self._casting = "safe"

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
                _np.asarray(input).shape
                for input in self._inputs
            )

    def get_best_match_input(self, result_shape):
        """
        """
        best_input = None
        result_shape = _np.array(result_shape, dtype="int64")
        input_shapes = self.get_inputs_shapes()
        are_same_shape = \
            _np.array(
                tuple(
                    (len(result_shape) == len(in_shape)) and _np.all(result_shape == in_shape)
                    for in_shape in input_shapes
                )
            )
        if _np.any(are_same_shape):
            best_input = self._inputs[_np.where(are_same_shape)[0][0]]
        else:
            input_shapes = \
                _np.array(
                    tuple(
                        _np.array(shape_extend_dims(len(result_shape), in_shape))
                        for in_shape in input_shapes
                    ),
                    dtype="int64"
                )
            d = input_shapes - result_shape
            d *= d
            d = d.sum(axis=1)
            best_input = self._inputs(_np.argmin(d))

        return best_input

    def create_outputs(self, outputs, result_shape, result_types):
        """
        Returns list of output :obj:`mpi_array.globale.gndarray` instances.

        :type outputs: :samp:`None` or :obj:`tuple` of :obj:`mpi_array.globale.gndarray`
        :param outputs: Output arrays passed in as the :samp:`out` argument
           of the :obj:`numpy.ufunc`.
        :type result_shape: sequence of :obj:`int`
        :param result_shape: The shape of all output arrays.
        :type result_types: sequence of :samp:`numpy.dtype`
        :param result_types: The :samp:`dtype` of each output array. Note
            that this is the list for all outputs including any
            in the :samp:`outputs` argument. This determines the
            number of output arrays.
        :rtype: :obj:`list` of :obj:`mpi_array.globale.gndarray`
        :return: A list of length :samp:`len(result_types)` elements,
           each element is a :obj:`mpi_array.globale.gndarray`.
        """

        template_output_gary = None
        if (outputs is not None) and (len(outputs) > 0):
            template_output_gary = outputs[-1]
        else:
            best_match_input = self.get_best_match_input(result_shape)
            if best_match_input is not None:
                comms_distrib = \
                    _comms.reshape_comms_distribution(
                        best_match_input.comms_and_distrib,
                        result_shape
                    )
                template_output_gary = \
                    _globale_creation.empty(
                        result_shape,
                        comms_and_distrib=comms_distrib,
                        dtype=result_types[0]
                    )
            else:
                template_output_gary = \
                    _globale_creation.empty(
                        result_shape,
                        dtype=result_types[0],
                        peer_comm=self._array_like_obj.locale_comms.peer_comm,
                        intra_locale_comm=self._array_like_obj.locale_comms.intra_locale_comm,
                        inter_locale_comm=self._array_like_obj.locale_comms.inter_locale_comm
                    )
            outputs = (template_output_gary,)
        outputs = \
            (
                outputs
                +
                tuple(
                    _globale_creation.empty_like(template_output_gary, dtype=result_types[i])
                    for i in range(len(outputs), len(result_types))
                )
            )

        return outputs

    def execute___call__(self):
        """
        """
        result_shape = broadcast_shape(*(self.get_inputs_shapes()))
        result_types = ufunc_result_type(self.ufunc.types, self.inputs, self.outputs, self.casting)

        outputs = self.create_outputs(self.outputs, result_shape, result_types)

        kwargs = _copy.copy(self._kwargs)
        inputs = \
            tuple(
                input.lndarray_proxy.lndarray if hasattr(input, "lndarray_proxy") else input
                for input in self.inputs
            )
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


#: Factory for generating instance of :obj:`GndarrayArrayUfuncExecutor`.
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
