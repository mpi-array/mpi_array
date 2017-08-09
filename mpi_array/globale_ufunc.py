"""
=========================================
The :mod:`mpi_array.globale_ufunc` Module
=========================================

Defines :obj:`numpy.ufunc` functions for :obj:`mpi_array.globale.gndarray`.

Functions
=========

.. autosummary::
   :toctree: generated/

   gndarray_array_ufunc - A :obj:`numpy.ndarray` like distributed array.


"""

from __future__ import absolute_import

import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
# from . import globale as _globale
from . import logging as _logging  # noqa: E402,F401

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def ufunc_result_type(ufunc_types, inputs, outputs, casting="safe"):
    """

    :rtype: :obj:`tuple` of :obj:`numpy.dtype`
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
        list(
            input.dtype
            if hasattr(input, "dtype") else _np.asarray(input).dtype
            for input in inputs
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
        raise TypeError(
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
        ndim_shapes = _np.asarray(tuple((1,) * (ndim - len(shape)) + shape for shape in shape_args))
        bcast_shape = _np.amax(ndim_shapes, axis=0)

        if (_np.any(_np.logical_and(ndim_shapes != 1, ndim_shapes != bcast_shape))):
            raise ValueError(
                "shape mismatch - objects cannot be broadcast to a single shape:\n%s"
                %
                (shape_args,)
            )

        bcast_shape = tuple(bcast_shape)

    return bcast_shape


def gndarray_array_ufunc(self, ufunc, method, *inputs, **kwargs):
    """
    The implementation for  :meth:`mpi_array.globale.gndarray.__array_ufunc__`.
    """
    rank_logger = self.rank_logger

    rank_logger.debug("gndarray_array_ufunc: ufunc=%s")


__all__ = [s for s in dir() if not s.startswith('_')]
