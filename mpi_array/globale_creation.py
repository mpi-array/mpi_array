"""
============================================
The :mod:`mpi_array.globale_creation` Module
============================================

Defines :obj:`mpi_array.globale.gndarray` creation functions.

Functions
=========

.. autosummary::
   :toctree: generated/

   asarray - Returns :obj:`mpi_array.globale.gndarray` equivalent of input.
   empty - Create uninitialised array.
   empty_like - Create uninitialised array same size/shape as another array.
   ones - Create one-initialised array.
   ones_like - Create one-initialised array same size/shape as another array.
   zeros - Create zero-initialised array.
   zeros_like - Create zero-initialised array same size/shape as another array.
   full - Create *fill value* initialised array.
   full_like - Create *fill value* initialised array same size/shape as another array.
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
    :type order: :samp:{'C', 'F'}
    :param order: Only :samp:`C` implemented.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`numpy.dtype`
    :param comms_and_distrib: Data type of array elements.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Newly created array with uninitialised elements.
    """
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


def empty_like(ary, dtype=None, order='K', subok=True):
    """
    Return a new array with the same shape and type as a given array.

    :type ary: :obj:`numpy.ndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :type order: :samp:`{‘C’, ‘F’, ‘A’, or ‘K’}`
    :param order: Only :samp:`'K'` implemented.
        Overrides the memory layout of the result. :samp:`'C'` means C-order, :samp:`'F'`
        means F-order, :samp:`'A'` means :samp:`'F'` if a is Fortran
        contiguous, :samp:`'C'` otherwise. :samp:`'K'` means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :samp:`type(ary)`
    :return: Array of uninitialized (arbitrary) data with the same shape and type as :samp:`{ary}`.
    """
    if dtype is None:
        dtype = ary.dtype
    if (isinstance(ary, _globale.gndarray)):
        if order == 'K':
            order = 'C'
        ret_ary = \
            empty(
                dtype=ary.dtype,
                comms_and_distrib=ary.comms_and_distrib,
                order=order,
                intra_partition_dims=ary.lndarray_proxy.intra_partition_dims
            )
    else:
        ret_ary = _np.empty_like(ary, dtype=dtype, order=order, subok=subok)

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
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type order: :samp:{'C', 'F'}
    :param order: Only :samp:`C` implemented.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`numpy.dtype`
    :param comms_and_distrib: Data type of array elements.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Newly created array with uninitialised elements.
    """
    ary = empty(shape, *args, **kwargs)
    ary.fill_h(ary.dtype.type(fill_value))

    return ary


def full_like(ary, fill_value, *args, **kwargs):
    """
    Return a new array with the same shape and type as a given array.

    :type ary: :obj:`numpy.ndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :type order: :samp:`{‘C’, ‘F’, ‘A’, or ‘K’}`
    :param order: Only :samp:`'K'` implemented.
        Overrides the memory layout of the result. :samp:`'C'` means C-order, :samp:`'F'`
        means F-order, :samp:`'A'` means :samp:`'F'` if a is Fortran
        contiguous, :samp:`'C'` otherwise. :samp:`'K'` means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :samp:`type(ary)`
    :return: Array of uninitialized (arbitrary) data with the same shape and type as :samp:`{ary}`.
    """
    if (isinstance(ary, _globale.gndarray)):
        ary = empty_like(ary, *args, **kwargs)
        ary.fill_h(ary.dtype.type(fill_value))
    else:
        ary = _np.full_like(ary, fill_value, *args, **kwargs)

    return ary


def zeros(shape=None, dtype="float64", order='C', comms_and_distrib=None, **kwargs):
    """
    Creates array of zero-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type order: :samp:{'C', 'F'}
    :param order: Only :samp:`C` implemented.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`numpy.dtype`
    :param comms_and_distrib: Data type of array elements.
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
    :type order: :samp:`{‘C’, ‘F’, ‘A’, or ‘K’}`
    :param order: Only :samp:`C` implemented.
        Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order,
        ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Array of zero-initialized data with the same shape and type as :samp:`{ary}`.
    """
    return full_like(ary, 0, *args, **kwargs)


def ones(shape=None, dtype="float64", comms_and_distrib=None, order='C', **kwargs):
    """
    Creates array of one-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type order: :samp:{'C', 'F'}
    :param order: Only :samp:`C` implemented.
       Whether to store multi-dimensional data in row-major (C-style)
       or column-major (Fortran-style) order in memory.
    :type comms_and_distrib: :obj:`numpy.dtype`
    :param comms_and_distrib: Data type of array elements.
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
    :type order: :samp:`{‘C’, ‘F’, ‘A’, or ‘K’}`
    :param order: Only :samp:`C` implemented.
        Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order,
        ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout
        of :samp:`{ary}` as closely as possible.
    :type subok: :obj:`bool`
    :param subok: Ignored.
       If True, then the newly created array will use the sub-class type of :samp:`{ary}`,
       otherwise it will be a base-class array. Defaults to True.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: Array of one-initialized data with the same shape and type as :samp:`{ary}`.
    """
    return full_like(ary, 1, *args, **kwargs)


def copy(ary, **kwargs):
    """
    Return an array copy of the given object.

    :type ary: :obj:`mpi_array.globale.gndarray`
    :param ary: Array to copy.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: A copy of :samp:`{ary}`.
    """
    return ary.copy()


def asarray(a, dtype=None, order=None, **kwargs):
    """
    Converts :samp:`{a}` (potentially via a copy)
    to a :obj:`mpi_array.globale.gndarray`.
    The :samp:`{kwargs}` are as for the :func:`mpi_array.comms.create_distributon` function
    and determine the distribution for the
    returned :obj:`mpi_array.globale.gndarray`.

    :type a: scalar, :obj:`tuple`, :obj:`list`, :obj:`numpy.ndarray`, etc
    :param a: Object converted to a :obj:`mpi_array.globale.gndarray`.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: The :obj:`numpy.dtype` for the returned :obj:`mpi_array.globale.gndarray`.
    :rtype: :obj:`mpi_array.globale.gndarray`
    :return: The object :obj:`a` converted to an instance
       of :obj:`mpi_array.globale.gndarray`.
    """
    if hasattr(a, "__class__") and (a.__class__ is _globale.gndarray):
        ret_ary = a
    elif isinstance(a, _globale.gndarray):
        ret_ary =\
            _globale.gndarray(
                comms_and_distrib=a.comms_and_distrib,
                rma_window_buffer=a.rma_window_buffer,
                lndarray_proxy=a.lndarray_proxy
            )
    else:
        if "distrib_type" not in kwargs.keys() or kwargs["distrib_type"] is None:
            kwargs["distrib_type"] = _comms.DT_CLONED
        np_ary = _np.asanyarray(a, dtype, order)
        ret_ary = \
            empty(
                np_ary.shape,
                dtype=np_ary.dtype,
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


__all__ = [s for s in dir() if not s.startswith('_')]
