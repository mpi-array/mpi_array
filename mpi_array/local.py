"""
=================================
The :mod:`mpi_array.local` Module
=================================

Defines :obj:`lndarray` class and factory functions for
creating multi-dimensional arrays where memory is allocated
using :meth:`mpi4py.MPI.Win.Allocate_shared` or :meth:`mpi4py.MPI.Win.Allocate`.

Classes
=======

..
   Special template for mpi_array.local.lndarray to avoid numpydoc
   documentation style sphinx warnings/errors from numpy.ndarray inheritance.

.. autosummary::
   :toctree: generated/
   :template: autosummary/lndarray_class.rst

   lndarray - Sub-class of :obj:`numpy.ndarray` which uses MPI allocated memory.

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

Utilities
=========

.. autosummary::
   :toctree: generated/

   NdarrayMetaData - Strides, offset and order info.

"""

from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources
import numpy as _np
from mpi_array.decomposition import CartesianDecomposition as _CartDecomp
__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


class NdarrayMetaData(object):
    """
    Encapsulates, strides, offset and order argument of :meth:`lndarray.__new__`.
    """

    def __init__(self, offset, strides, order):
        """
        Construct.

        :type offset: :samp:`None` or :obj:`int`
        :param offset: Offset of array data in buffer.
        :type strides: :samp:`None` or sequence of :obj:`int`
        :param strides: Strides of data in memory.
        :type order: {:samp:`C`, :samp:`F`} or :samp:`None`
        :param order: Row-major (C-style) or column-major (Fortran-style) order.
        """
        object.__init__(self)
        self._strides = strides
        self._offset = offset
        self._order = order


class lndarray(_np.ndarray):
    """
    Sub-class of :obj:`np.ndarray` for use with the :obj:`mpi4py` parallel processing.
    Allocates a shared memory buffer using :func:`mpi4py.MPI.Win.Allocate_shared`.
    (if available, otherwise uses :func:`mpi4py.MPI.Win.Allocate`).
    """

    def __new__(
        cls,
        shape,
        dtype=_np.dtype("float64"),
        buffer=None,
        offset=0,
        strides=None,
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
        :type buffer: :samp:`None` or :obj:`memoryview`
        :param buffer: The sequence of bytes providing array element storage.
           If :samp:`None`, a buffer is allocated using :samp:`{decomp}.alloc_local_buffer`.
        :type offset: :samp:`None` or :obj:`int`
        :param offset: Offset of array data in buffer, i.e where array begins in buffer
           (in buffer bytes).
        :type strides: :samp:`None` or sequence of :obj:`int`
        :param strides: Strides of data in memory.
        :type order: {:samp:`C`, :samp:`F`} or :samp:`None`
        :param order: Row-major (C-style) or column-major (Fortran-style) order.
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

        if buffer is None:
            buffer, itemsize, shape = decomp.alloc_local_buffer(dtype)
        else:
            if not isinstance(buffer, memoryview):
                raise ValueError(
                    "Got buffer type=%s which is not an instance of %s"
                    %
                    (
                        type(buffer),
                        memoryview
                    )
                )

            if decomp is None:
                raise ValueError("Got None value for decomp with non-None buffer = %s" % (buffer,))

        self = \
            _np.ndarray.__new__(
                cls,
                shape,
                dtype,
                buffer,
                offset,
                strides,
                order
            )
        self._md = NdarrayMetaData(offset=offset, strides=strides, order=order)
        self._decomp = decomp

        return self

    def __array_finalize__(self, obj):
        """
        Sets :attr:`md` and :attr:`decomp` attributes for :samp:`{self}`
        from :samp:`{obj}` if required.

        :type obj: :obj:`object` or :samp:`None`
        :param obj: Object from which attributes are set.
        """
        if obj is None:
            return

        self._md = getattr(obj, 'md', None)
        self._decomp = getattr(obj, 'decomp', None)

    @property
    def md(self):
        """
        Meta-data object of type :obj:`NdarrayMetaData`.
        """
        return self._md

    @property
    def decomp(self):
        """
        Decomposition object (e.g. :obj:`mpi_array.decomposition.CartesianDecomposition`)
        describing distribution of the array across memory nodes.
        """
        return self._decomp

    @property
    def rank_view_n(self):
        """
        A tile view of the array for MPI rank :samp:`self.decomp.rank_comm.rank`.
        """
        return self[self.decomp.rank_view_slice_n]

    @property
    def rank_view_h(self):
        """
        A tile view (including halo elements) of the array for MPI
        rank :samp:`self.decomp.rank_comm.rank`.
        """
        return self[self.decomp.rank_view_slice_h]

    def __reduce_ex__(self, protocol):
        """
        Pickle *reference* to shared memory.
        """
        raise NotImplementedError("Cannot pickle objects of type %s" % type(self))
        # return ndarray, (self.shape, self.dtype, self.mp_Array,
        #                 self.strides, self.offset, self.order)

    def __reduce__(self):
        return self.__reduce_ex__(self, 0)


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
    :rtype: :obj:`lndarray`
    :return: Newly created array with uninitialised elements.
    """
    ary = lndarray(shape=shape, dtype=dtype, decomp=decomp, order=order)

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
    if (isinstance(ary, lndarray)):
        ret_ary = empty(dtype=ary.dtype, decomp=ary.decomp)
    else:
        ret_ary = _np.empty_like(ary, dtype=dtype)

    return ret_ary


def zeros(shape, dtype="float64", decomp=None, order='C'):
    """
    Creates array of zero-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`lndarray`
    :return: Newly created array with zero-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view_n()[...] = _np.zeros(1, dtype)

    return ary


def zeros_like(ary, *args, **kwargs):
    """
    Return a new zero-initialised array with the same shape and type as a given array.

    :type ary: :obj:`numpy.ndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :samp:`type(ary)`
    :return: Array of zero-initialized data with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.rank_view_n[...] = _np.zeros(1, ary.dtype)

    return ary


def ones(shape, dtype="float64", decomp=None, order='C'):
    """
    Creates array of one-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`lndarray`
    :return: Newly created array with one-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view()[...] = _np.ones(1, dtype)

    return ary


def ones_like(ary, *args, **kwargs):
    """
    Return a new one-initialised array with the same shape and type as a given array.

    :type ary: :obj:`numpy.ndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :samp:`type(ary)`
    :return: Array of one-initialized data with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.rank_view_n[...] = _np.ones(1, ary.dtype)

    return ary


def copy(ary):
    """
    Return an array copy of the given object.

    :type ary: `numpy.ndarray`
    :param ary: Array to copy.
    :rtype: :samp:`type(ary)`
    :return: A copy of :samp:`ary`.
    """
    ary_out = empty_like(ary)
    ary_out.rank_view_n[...] = ary.rank_view_n[...]

    return ary_out


__all__ = [s for s in dir() if not s.startswith('_')]
