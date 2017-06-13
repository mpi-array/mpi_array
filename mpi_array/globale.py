"""
==================================
The :mod:`mpi_array.global` Module
==================================

Defines :obj:`gndarray` class and factory functions for
creating multi-dimensional distributed arrays (Partitioned Global Address Space).

Classes
=======

.. autosummary::
   :toctree: generated/

   gndarray - A :obj:`numpy.ndarray` like distributed array.

Factory Functions
=================

.. autosummary::
   :toctree: generated/

   empty - Create uninitialised array.
   empty_like - Create uninitialised array same size/shape as another array.
   zeros - Create zero-initialised array.
   zeros_like - Create zero-initialised array same size/shape as another array.
   ones - Create one-initialised array.
   ones_like - Create one-initialised array same size/shape as another array.
   copy - Create a replica of a specified array.


"""

from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources
import numpy as _np
from mpi_array.locale import lndarray as _lndarray
from mpi_array.decomposition import CartesianDecomposition as _CartDecomp

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


class gndarray(object):
    """
    A distributed array with :obj:`numpy.ndarray` API.
    """

    def __new__(
        cls,
        shape=None,
        dtype=_np.dtype("float64"),
        order=None,
        decomp=None
    ):
        """
        Construct, at least one of :samp:{shape} or :samp:`decomp` should
        be specified (i.e. at least one should not be :samp:`None`).

        :type shape: :samp:`None` or sequence of :obj:`int`
        :param shape: **Global** shape of the array. If :samp:`None`
           global array shape is taken as :samp:`{decomp}.shape`.
        :type dtype: :obj:`numpy.dtype`
        :param dtype: Data type for elements of the array.
        :type strides: :samp:`None` or sequence of :obj:`int`
        :param strides: Strides of data in memory.
        :type order: {:samp:`C`, :samp:`F`} or :samp:`None`
        :param order: Row-major (C-style) or column-major (Fortran-style) order.
        :type decomp: :obj:`mpi_array.decomposition.Decomposition`
        :param decomp: Array decomposition info and used to allocate (possibly)
           shared memory.
        """
        if (shape is not None) and (decomp is None):
            decomp = _CartDecomp(shape)
        elif (shape is not None) and (decomp is not None) and (_np.any(decomp.shape != shape)):
            raise ValueError(
                "Got inconsistent shape argument=%s and decomp.shape=%s, should be the same."
                %
                (shape, decomp.shape)
            )
        elif (shape is None) and (decomp is None):
            raise ValueError(
                (
                    "Got None for both shape argument and decomp argument,"
                    +
                    " at least one should be provided."
                )
            )
        self = object.__new__(cls)
        self._lndarray = _lndarray(shape=shape, dtype=dtype, order=order, decomp=decomp)
        return self

    def __getitem__(self, i):
        """
        """
        self.rank_logger.debug("__getitem__: i=%s", i)
        return None

    def __setitem__(self, i, v):
        """
        """
        self.rank_logger.debug("__setitem__: i=%s, v=%s", i, v)

    @property
    def rank_view_n(self):
        return self._lndarray.rank_view_n

    @property
    def rank_view_h(self):
        return self._lndarray.rank_view_h

    @property
    def rank_logger(self):
        """
        """
        return self._lndarray.decomp.rank_logger

    @property
    def root_logger(self):
        """
        """
        return self._lndarray.decomp.root_logger

    def update(self):
        """
        """


def empty(shape=None, dtype="float64", decomp=None, order='C'):
    """
    Creates array of uninitialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`gndarray`
    :return: Newly created array with uninitialised elements.
    """
    ary = gndarray(shape=shape, dtype=dtype, decomp=decomp, order=order)

    return ary


def empty_like(ary, dtype=None):
    """
    Return a new array with the same shape and type as a given array.

    :type ary: :obj:`numpy.ndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :samp:`type(ary)`
    :return: Array of uninitialized (arbitrary) data with the same shape and type as :samp:`{a}`.
    """
    if dtype is None:
        dtype = ary.dtype
    if (isinstance(ary, gndarray)):
        ret_ary = empty(dtype=ary.dtype, decomp=ary.decomp)
    else:
        ret_ary = _np.empty_like(ary, dtype=dtype)

    return ret_ary


def zeros(shape=None, dtype="float64", decomp=None, order='C'):
    """
    Creates array of zero-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`gndarray`
    :return: Newly created array with zero-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view_n[...] = _np.zeros(1, dtype)

    return ary


def zeros_like(ary, *args, **kwargs):
    """
    Return a new zero-initialised array with the same shape and type as a given array.

    :type ary: :obj:`gndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`gndarray`
    :return: Array of zero-initialized data with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.rank_view_n[...] = _np.zeros(1, ary.dtype)

    return ary


def ones(shape=None, dtype="float64", decomp=None, order='C'):
    """
    Creates array of one-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`gndarray`
    :return: Newly created array with one-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view_n[...] = _np.ones(1, dtype)

    return ary


def ones_like(ary, *args, **kwargs):
    """
    Return a new one-initialised array with the same shape and type as a given array.

    :type ary: :obj:`gndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`gndarray`
    :return: Array of one-initialized data with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.rank_view_n[...] = _np.ones(1, ary.dtype)

    return ary


def copy(ary):
    """
    Return an array copy of the given object.

    :type ary: :obj:`gndarray`
    :param ary: Array to copy.
    :rtype: :obj:`gndarray`
    :return: A copy of :samp:`ary`.
    """
    ary_out = empty_like(ary)
    ary_out.rank_view_n[...] = ary.rank_view_n[...]

    return ary_out


__all__ = [s for s in dir() if not s.startswith('_')]
