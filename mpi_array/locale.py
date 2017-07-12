"""
==================================
The :mod:`mpi_array.locale` Module
==================================

Defines :obj:`lndarray` class and factory functions for
creating multi-dimensional arrays where memory is allocated
using :meth:`mpi4py.MPI.Win.Allocate_shared` or :meth:`mpi4py.MPI.Win.Allocate`.

Classes
=======

..
   Special template for mpi_array.locale.lndarray to avoid numpydoc
   documentation style sphinx warnings/errors from numpy.ndarray inheritance.

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_ndarray_class.rst

   slndarray - Sub-class of :obj:`numpy.ndarray` which uses MPI allocated memory.

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   lndarray - Thin container for :obj:`slndarray` which provides convenience views.


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

    @property
    def order(self):
        return self._order


class slndarray(_np.ndarray):

    """
    Sub-class of :obj:`numpy.ndarray` which uses :obj:`mpi4py.MPI.Win` instances
    to allocate buffer memory.
    Allocates a shared memory buffer using :func:`mpi4py.MPI.Win.Allocate_shared`.
    (if available, otherwise uses :func:`mpi4py.MPI.Win.Allocate`).
    """

    def __new__(
        cls,
        shape=None,
        dtype=_np.dtype("float64"),
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        gshape=None,
        decomp=None
    ):
        """
        Construct, at least one of :samp:{shape} or :samp:`decomp` should
        be specified (i.e. at least one should not be :samp:`None`).

        :type shape: :samp:`None` or sequence of :obj:`int`
        :param shape: **Local** shape of the array, this parameter is ignored.
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
        :param gshape: **Global** shape of the array. If :samp:`None`
           global array shape is taken as :samp:`{decomp}.shape`.
        :type decomp: :obj:`mpi_array.decomposition.Decomposition`
        :param decomp: Array decomposition info and used to allocate (possibly)
           shared memory via :meth:`mpi_array.decomposition.Decomposition.allocate_local_buffer`.
        """
        if (gshape is not None) and (decomp is not None) and (_np.any(decomp.shape != gshape)):
            raise ValueError(
                "Got inconsistent gshape argument=%s and decomp.shape=%s, should be the same."
                %
                (gshape, decomp.shape)
            )
        elif (gshape is None) and (decomp is None):
            raise ValueError(
                (
                    "Got None for both gshape argument and decomp argument,"
                    +
                    " at least one should be provided."
                )
            )

        if (shape is not None) and (_np.any(decomp.lndarray_extent.shape != shape)):
            raise ValueError(
                (
                    "Got inconsistent shape argument=%s and decomp.lndarray_extent.shape=%s, " +
                    "should be the same."
                )
                %
                (shape, decomp.lndarray_extent.shape)
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

        return self

    def __array_finalize__(self, obj):
        """
        Sets :attr:`md` attribute for :samp:`{self}`
        from :samp:`{obj}` if required.

        :type obj: :obj:`object` or :samp:`None`
        :param obj: Object from which attributes are set.
        """
        if obj is None:
            return

        self._md = getattr(obj, '_md', None)

    @property
    def md(self):
        """
        Meta-data object of type :obj:`NdarrayMetaData`.
        """
        return self._md


class lndarray(object):

    """
    Thin container for :obj:`slndarray` instances.
    Adds the :attr:`decomp` attribute to keep track
    of decomposition.
    """

    def __new__(
        cls,
        shape=None,
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
        :type decomp: :obj:`mpi_array.decomposition.Decomposition`
        :param decomp: Array decomposition info and used to allocate (possibly)
           shared memory via :meth:`mpi_array.decomposition.Decomposition.allocate_local_buffer`.
        """

        self = object.__new__(cls)
        if (shape is not None) and (decomp is None):
            decomp = _CartDecomp(shape)
        self._slndarray = \
            slndarray(
                shape=None,
                dtype=dtype,
                buffer=buffer,
                offset=offset,
                strides=strides,
                order=order,
                gshape=shape,
                decomp=decomp
            )
        self._decomp = decomp
        return self

    def __getitem__(self, *args, **kwargs):
        """
        Return slice/item from :attr:`slndarray` array.
        """
        return self._slndarray.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """
        Set slice/item in :attr:`slndarray` array.
        """
        self._slndarray.__setitem__(*args, **kwargs)

    def __eq__(self, other):
        """
        """
        if isinstance(other, lndarray):
            return self._slndarray == other._slndarray
        else:
            return self._slndarray == other

    @property
    def slndarray(self):
        """
        An :obj:`slndarray` instance containing array data in (potentially)
        shared memory.
        """
        return self._slndarray

    @property
    def md(self):
        """
        Meta-data object of type :obj:`NdarrayMetaData`.
        """
        return self._slndarray.md

    @property
    def decomp(self):
        """
        Decomposition object (e.g. :obj:`mpi_array.decomposition.CartesianDecomposition`)
        describing distribution of the array across memory nodes.
        """
        return self._decomp

    @property
    def dtype(self):
        """
        A :obj:`numpy.dtype` object describing the element type of this array.
        """
        return self._slndarray.dtype

    @property
    def shape(self):
        """
        The shape of the locale array (including halo), i.e. :samp:`self.slndarray.shape`.
        """
        return self._slndarray.shape

    @property
    def rank_view_n(self):
        """
        A tile view of the array for MPI rank :samp:`self.decomp.rank_comm.rank`.
        """
        return self._slndarray[self.decomp.rank_view_slice_n]

    @property
    def rank_view_h(self):
        """
        A tile view (including halo elements) of the array for MPI
        rank :samp:`self.decomp.rank_comm.rank`.
        """
        return self._slndarray[self.decomp.rank_view_slice_h]

    @property
    def rank_view_slice_n(self):
        """
        Sequence of :obj:`slice` objects used to generate :attr:`rank_view_n`.
        """
        return self.decomp.rank_view_slice_n

    @property
    def rank_view_slice_h(self):
        """
        Sequence of :obj:`slice` objects used to generate :attr:`rank_view_h`.
        """
        return self.decomp.rank_view_slice_h

    @property
    def view_n(self):
        """
        View of entire array without halo.
        """
        return self._slndarray[self.decomp.lndarray_view_slice_n]

    @property
    def view_h(self):
        """
        The entire :obj:`lndarray` view including halo (i.e. :samp:{self}).
        """
        return self._slndarray.view()

    def __reduce__(self):
        """
        Pickle.
        """
        raise NotImplementedError("Cannot pickle objects of type %s" % type(self))


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
    :rtype: :obj:`lndarray`
    :return: Newly created array with zero-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view_n[...] = _np.zeros(1, dtype)

    return ary


def zeros_like(ary, *args, **kwargs):
    """
    Return a new zero-initialised array with the same shape and type as a given array.

    :type ary: :obj:`lndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`lndarray`
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
    :rtype: :obj:`lndarray`
    :return: Newly created array with one-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view_n[...] = _np.ones(1, dtype)

    return ary


def ones_like(ary, *args, **kwargs):
    """
    Return a new one-initialised array with the same shape and type as a given array.

    :type ary: :obj:`lndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`lndarray`
    :return: Array of one-initialized data with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.rank_view_n[...] = _np.ones(1, ary.dtype)

    return ary


def copy(ary):
    """
    Return an array copy of the given object.

    :type ary: :obj:`lndarray`
    :param ary: Array to copy.
    :rtype: :obj:`lndarray`
    :return: A copy of :samp:`ary`.
    """
    ary_out = empty_like(ary)
    ary_out.rank_view_n[...] = ary.rank_view_n[...]

    return ary_out


__all__ = [s for s in dir() if not s.startswith('_')]
