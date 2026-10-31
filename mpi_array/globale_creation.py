"""
============================================
The :mod:`mpi_array.globale_creation` Module
============================================

Defines :obj:`mpi_array.globale.gndarray` creation functions.

Ones and zeros
==============

.. autosummary::
   :toctree: generated/

   empty - Create uninitialised array.
   empty_like - Create uninitialised array same size/shape as another array.
   eye - Return 2D array with ones on diagonal and zeros elsewhere.
   identity - Return identity array.
   ones - Create one-initialised array.
   ones_like - Create one-initialised array same size/shape as another array.
   zeros - Create zero-initialised array.
   zeros_like - Create zero-initialised array same size/shape as another array.
   full - Create *fill value* initialised array.
   full_like - Create *fill value* initialised array same size/shape as another array.


From existing data
==================

.. autosummary::
   :toctree: generated/

   array - Returns :obj:`mpi_array.globale.gndarray` equivalent of input.
   asarray - Returns :obj:`mpi_array.globale.gndarray` equivalent of input.
   asanyarray - Returns :obj:`mpi_array.globale.gndarray` equivalent of input.
   copy - Create a replica of a specified array.

"""

from __future__ import absolute_import

import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from . import logging as _logging  # noqa: E402,F401
from . import locale as _locale
from . import comms as _comms
from . import globale as _globale

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def empty(
    shape=None,
    dtype="float64",
    order='C',
    comms_and_distrib=None,
    intra_partition_dims=None,
    **kwargs
):
    """
    Creates array of uninitialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type order: :samp:`{'C', 'F'}`
    :param order: Only :samp:`'C'` implemented, :samp:`'F'`
       raises :obj:`NotImplementedError` (see :func:`mpi_array.locale.check_order`).
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`mpi_array.comms.CommsAndDistribution`
    :param comms_and_distrib: The communicators and distribution of the array
       (e.g. the :samp:`comms_and_distrib` of another array), :samp:`{shape}` is then
       ignored. If :samp:`None`, they are created by :func:`mpi_array.comms.create_distribution`
       from :samp:`{shape}` and :samp:`{kwargs}` (e.g. :samp:`locale_type`, :samp:`distrib_type`,
       :samp:`dims` and :samp:`halo`).
    :type intra_partition_dims: :samp:`None` or sequence of :obj:`int`
    :param intra_partition_dims: The number of tiles (one for each process of a locale)
       along each axis of the locale arrays, see :func:`mpi_array.locale.empty`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Newly created array with uninitialised elements.
    """
    order = _locale.check_order(order)
    if comms_and_distrib is None:
        comms_and_distrib = _comms.create_distribution(shape, **kwargs)
    lndarray_proxy, rma_window_buffer = \
        _locale.empty(
            comms_and_distrib=comms_and_distrib,
            dtype=dtype,
            order=order,
            return_rma_window_buffer=True,
            intra_partition_dims=intra_partition_dims
        )
    ary = \
        _globale.gndarray(
            comms_and_distrib=comms_and_distrib,
            rma_window_buffer=rma_window_buffer,
            lndarray_proxy=lndarray_proxy
        )

    return ary


def empty_like(ary, dtype=None, order='K', subok=True, **kwargs):
    """
    Return a new array with the same shape and type as a given array.

    :type ary: :obj:`mpi_array.globale.gndarray` or array-like
    :param ary: Copy attributes from this array. A :obj:`mpi_array.globale.gndarray` gives
       an array with the same communicators and distribution, otherwise the array is
       created from the shape of :samp:`{ary}` (and :samp:`{kwargs}`, see :func:`empty`).
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :type order: :samp:`{'C', 'F', 'A', or 'K'}`
    :param order: Only C order is implemented: :samp:`'C'`, :samp:`'A'` and :samp:`'K'` all give
        a C order array, :samp:`'F'` raises :obj:`NotImplementedError`.
        Overrides the memory layout of the result. :samp:`'C'` means C-order, :samp:`'F'`
        means F-order, :samp:`'A'` means :samp:`'F'` if a is Fortran
        contiguous, :samp:`'C'` otherwise. :samp:`'K'` means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Array of uninitialized (arbitrary) data with the same shape and type as :samp:`{ary}`.
    """
    if not isinstance(ary, _globale.gndarray):
        ary = _np.asanyarray(ary)
    if dtype is None:
        dtype = ary.dtype
    order = _locale.check_order(order)

    if (isinstance(ary, _globale.gndarray)):
        ret_ary = \
            empty(
                dtype=dtype,
                comms_and_distrib=ary.comms_and_distrib,
                order=order,
                intra_partition_dims=ary.lndarray_proxy.requested_intra_partition_dims
            )
    else:
        ret_ary = empty(ary.shape, dtype=dtype, order=order, **kwargs)

    return ret_ary


def full(
    shape=None,
    fill_value=0,
    *args,
    **kwargs
):
    """
    Return a new array of given shape and type, filled with :samp:`fill_value`.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type fill_value: scalar
    :param fill_value: Fill value.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type order: :samp:`{'C', 'F'}`
    :param order: Only :samp:`'C'` implemented, see :func:`empty`.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`mpi_array.comms.CommsAndDistribution`
    :param comms_and_distrib: The communicators and distribution of the array,
       see :func:`empty`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Newly created array with all elements (including the halo) set
       to :samp:`{fill_value}`.
    """
    ary = empty(shape, *args, **kwargs)
    ary.fill_h(ary.dtype.type(fill_value))

    return ary


def full_like(ary, fill_value, *args, **kwargs):
    """
    Return a new array with the same shape and type as a given array,
    filled with :samp:`{fill_value}`.

    :type ary: :obj:`mpi_array.globale.gndarray` or array-like
    :param ary: Copy attributes from this array, see :func:`empty_like`.
    :type fill_value: scalar
    :param fill_value: Fill value.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :type order: :samp:`{'C', 'F', 'A', or 'K'}`
    :param order: Only C order is implemented, see :func:`empty_like`.
        Overrides the memory layout of the result. :samp:`'C'` means C-order, :samp:`'F'`
        means F-order, :samp:`'A'` means :samp:`'F'` if a is Fortran
        contiguous, :samp:`'C'` otherwise. :samp:`'K'` means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Array with all elements (including the halo) set to :samp:`{fill_value}`,
       with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.fill_h(ary.dtype.type(fill_value))

    return ary


def zeros(shape=None, dtype="float64", order='C', comms_and_distrib=None, **kwargs):
    """
    Creates array of zero-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type order: :samp:`{'C', 'F'}`
    :param order: Only :samp:`'C'` implemented, see :func:`empty`.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`mpi_array.comms.CommsAndDistribution`
    :param comms_and_distrib: The communicators and distribution of the array,
       see :func:`empty`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Newly created array with zero-initialised elements.
    """
    return \
        full(
            shape=shape,
            fill_value=0,
            dtype=dtype,
            order=order,
            comms_and_distrib=comms_and_distrib,
            **kwargs
        )


def zeros_like(ary, *args, **kwargs):
    """
    Return a new zero-initialised array with the same shape and type as a given array.

    :type ary: :obj:`mpi_array.globale.gndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :type order: :samp:`{'C', 'F', 'A', or 'K'}`
    :param order: Only C order is implemented, see :func:`empty_like`.
        Overrides the memory layout of the result. :samp:`'C'` means C-order, :samp:`'F'`
        means F-order, :samp:`'A'` means :samp:`'F'` if a is Fortran
        contiguous, :samp:`'C'` otherwise. :samp:`'K'` means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Array of zero-initialized data with the same shape and type as :samp:`{ary}`.
    """
    return full_like(ary, 0, *args, **kwargs)


def ones(shape=None, dtype="float64", order='C', comms_and_distrib=None, **kwargs):
    """
    Creates array of one-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type order: :samp:`{'C', 'F'}`
    :param order: Only :samp:`'C'` implemented, see :func:`empty`.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`mpi_array.comms.CommsAndDistribution`
    :param comms_and_distrib: The communicators and distribution of the array,
       see :func:`empty`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Newly created array with one-initialised elements.
    """
    return \
        full(
            shape=shape,
            fill_value=1,
            dtype=dtype,
            order=order,
            comms_and_distrib=comms_and_distrib,
            **kwargs
        )


def ones_like(ary, *args, **kwargs):
    """
    Return a new one-initialised array with the same shape and type as a given array.

    :type ary: :obj:`mpi_array.globale.gndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :type order: :samp:`{'C', 'F', 'A', or 'K'}`
    :param order: Only C order is implemented, see :func:`empty_like`.
        Overrides the memory layout of the result. :samp:`'C'` means C-order, :samp:`'F'`
        means F-order, :samp:`'A'` means :samp:`'F'` if a is Fortran
        contiguous, :samp:`'C'` otherwise. :samp:`'K'` means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Array of one-initialized data with the same shape and type as :samp:`{ary}`.
    """
    return full_like(ary, 1, *args, **kwargs)


def eye(N, M=None, k=0, dtype=float):
    """
    Not implemented.
    Return a 2-D array with ones on the diagonal and zeros elsewhere.
    """
    raise NotImplementedError()


def identity(n, dtype=None):
    """
    Not implemented.
    Return the identity array.
    """
    raise NotImplementedError()


def copy(ary, **kwargs):
    """
    Return an array copy of the given object.

    :type ary: :obj:`mpi_array.globale.gndarray`
    :param ary: Array to copy.
    :type order: :samp:`{'C', 'F'}`
    :param order: Only :samp:`'C'` implemented.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory. Defaults to :samp:`'C'`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: A copy of :samp:`{ary}`.
    """
    return ary.copy(**kwargs)


def array(a, dtype=None, copy=True, order='K', subok=False, ndmin=0, **kwargs):
    """
    Create a :obj:`mpi_array.globale.gndarray` from an existing *array-like* object.
    Creating the array is collective over its peer communicator.

    An :samp:`{a}` which isn't a :obj:`mpi_array.globale.gndarray` must be the same on all
    processes: each process copies the elements of its own part of the new array from its
    :samp:`{a}`. The new array is then distributed as given by :samp:`{kwargs}` (see
    :func:`empty`), it is cloned (:attr:`mpi_array.comms.DT_CLONED`, every locale has all the
    elements) when they don't give the :samp:`distrib_type`. A copy of
    a :obj:`mpi_array.globale.gndarray` has its distribution (:samp:`{kwargs}` are ignored).

    :type a: array_like
    :param a: An array, any object exposing the array interface, an object
       whose :samp:`__array__` method returns an array, or any (nested) sequence.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: The desired data-type for the array. If not given,
       then the type will be determined as the minimum type required to hold the
       objects in the sequence.
    :type copy: :obj:`bool`
    :param copy: If :samp:`True`, then the object is copied.
       Otherwise, a copy will only be made if :samp:`__array__` returns a copy,
       if :samp:`a` is a nested sequence, or if a copy is needed to satisfy any of
       the other requirements (:samp:`dtype`, :samp:`order`, etc.).
       A sub-class instance :samp:`{a}` is copied when :samp:`{subok}` is :samp:`False`,
       the returned array never shares the memory (windows) of :samp:`{a}`.
    :type order: :samp:`{'K', 'A', 'C', 'F'}`
    :param order: Only C order is implemented: :samp:`'C'`, :samp:`'A'` and :samp:`'K'` all give
        a C order array, :samp:`'F'` raises :obj:`NotImplementedError`.
        Specify the memory layout of the array.
    :type subok: :obj:`bool`
    :param subok: If :samp:`True`, then sub-classes will be passed-through, otherwise the
        returned array will be forced to be a base-class array.
    :type ndmin: int
    :param ndmin: Specifies the minimum number of dimensions that the resulting array should have.
        Ones will be pre-pended to the shape as needed to meet this requirement.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: An array object satisfying the specified requirements.

    .. seealso:: :func:`asarray`, :func:`asanyarray`
    """
    order = _locale.check_order(order)

    if isinstance(a, _globale.gndarray):
        if a.ndim < ndmin:
            raise NotImplementedError(
                "Got ndmin=%s for a gndarray with ndim=%s, reshaping a gndarray is not implemented."
                %
                (ndmin, a.ndim)
            )
        dtype = a.dtype if dtype is None else _np.dtype(dtype)
        if (
            (not copy)
            and
            (subok or (a.__class__ is _globale.gndarray))
            and
            (dtype == a.dtype)
        ):
            ret_ary = a
        else:
            # A new array, a second gndarray with the windows of a would also free them.
            ret_ary = empty_like(a, dtype=dtype, order=order)
            ret_ary.lndarray_proxy.rank_view_partition_h[...] = \
                a.lndarray_proxy.rank_view_partition_h[...]
            ret_ary.intra_locale_barrier()
    else:
        if "distrib_type" not in kwargs.keys() or kwargs["distrib_type"] is None:
            kwargs["distrib_type"] = _comms.DT_CLONED
        np_ary = _np.array(a, dtype=dtype, order=order, copy=None, subok=True, ndmin=ndmin)
        ret_ary = \
            empty(
                shape=np_ary.shape,
                dtype=np_ary.dtype,
                order=order,
                **kwargs
            )
        if (ret_ary.ndim == 0) and (ret_ary.locale_comms.have_valid_inter_locale_comm):
            ret_ary.lndarray_proxy.lndarray[...] = np_ary
        else:
            locale_rank_view_slice_n = ret_ary.lndarray_proxy.rank_view_slice_n
            if len(locale_rank_view_slice_n) > 0:
                globale_rank_view_slice_n = \
                    ret_ary.lndarray_proxy.locale_extent.locale_to_globale_slice_h(
                        locale_rank_view_slice_n
                    )
                ret_ary.lndarray_proxy.lndarray[locale_rank_view_slice_n] =\
                    np_ary[globale_rank_view_slice_n]

        ret_ary.intra_locale_barrier()

    return ret_ary


def asarray(a, dtype=None, order=None, **kwargs):
    """
    Converts :samp:`{a}` (potentially via a copy)
    to a :obj:`mpi_array.globale.gndarray`.
    The :samp:`{kwargs}` are as for the :func:`mpi_array.comms.create_distribution` function
    and determine the distribution for the
    returned :obj:`mpi_array.globale.gndarray`.

    :type a: scalar, :obj:`tuple`, :obj:`list`, :obj:`numpy.ndarray`, etc
    :param a: Object converted to a :obj:`mpi_array.globale.gndarray`.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: The :obj:`numpy.dtype` for the returned :obj:`mpi_array.globale.gndarray`.
    :type order: :samp:`{'C', 'F'}`
    :param order: Only :samp:`'C'` implemented.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory. Defaults to :samp:`'C'`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: The object :samp:`{a}` converted to an instance
       of :obj:`mpi_array.globale.gndarray`.

    .. seealso:: :func:`array`, :func:`asanyarray`
    """
    return array(a, dtype, copy=False, order=order, **kwargs)


def asanyarray(a, dtype=None, order=None, **kwargs):
    """
    Convert the input to an ndarray, but pass :obj:`mpi_array.globale.gndarray` subclasses through.

    :type a: scalar, :obj:`tuple`, :obj:`list`, :obj:`numpy.ndarray`, etc
    :param a: Object converted to a :obj:`mpi_array.globale.gndarray`.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: The :obj:`numpy.dtype` for the returned :obj:`mpi_array.globale.gndarray`.
    :type order: :samp:`{'C', 'F'}`
    :param order: Only :samp:`'C'` implemented.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory. Defaults to :samp:`'C'`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: The object :samp:`{a}` converted to an instance
       of :obj:`mpi_array.globale.gndarray`.

    .. seealso:: :func:`array`, :func:`asarray`
    """
    return array(a, dtype, copy=False, order=order, subok=True, **kwargs)


__all__ = [s for s in dir() if not s.startswith('_')]
