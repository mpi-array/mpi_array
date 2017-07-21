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
   PartitionViewSlices - Container for per-rank slices for created locale extent array views.

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
from mpi_array.distribution import create_distribution, LocaleExtent as _LocaleExtent
from mpi_array.distribution import HaloSubExtent as _HaloSubExtent
from mpi_array.distribution import IndexingExtent as _IndexingExtent
import array_split as _array_split
from array_split.split import convert_halo_to_array_form as _convert_halo_to_array_form
import collections as _collections

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
        order=None
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
        :type decomp: :obj:`mpi_array.distribution.Decomposition`
        :param decomp: Array distribution info and used to allocate (possibly)
           shared memory via :meth:`mpi_array.distribution.Decomposition.allocate_local_buffer`.
        """

        if buffer is not None:
            if not isinstance(buffer, memoryview):
                raise ValueError(
                    "Got buffer type=%s which is not an instance of %s"
                    %
                    (
                        type(buffer),
                        memoryview
                    )
                )
        else:
            raise ValueError("Got buffer=None, require buffer allocated from LocaleComms.")

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


PartitionViewSlices = \
    _collections.namedtuple(
        "PartitionViewSlices",
        [
            "rank_view_slice_n",
            "rank_view_slice_h",
            "rank_view_relative_slice_n",
            "rank_view_partition_slice_h",
            "lndarray_view_slice_n"
        ]
    )


class lndarray(object):

    """
    Thin container for :obj:`slndarray` instances.
    Adds the :attr:`decomp` attribute to keep track
    of distribution.
    """

    def __new__(
        cls,
        shape=None,
        dtype=_np.dtype("float64"),
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        intra_locale_rank=None,
        intra_locale_size=0,
        intra_partition_dims=None,
        locale_extent=None,
        halo=None,
        comms_and_distrib=None
    ):
        """
        Initialise, at least one of :samp:{shape} or :samp:`locale_extent` should
        be specified (i.e. at least one should not be :samp:`None`).

        :type shape: :samp:`None` or sequence of :obj:`int`
        :param shape: Shape of the array apportioned to this locale. If :samp:`None`
           shape is taken as :samp:`{locale_extent}.shape_h`.
        :type dtype: :obj:`numpy.dtype`
        :param dtype: Data type for elements of the array.
        :type buffer: :obj:`memoryview`
        :param buffer: The sequence of bytes providing array element storage.
           Must be specified.
        :type offset: :samp:`None` or :obj:`int`
        :param offset: Offset of array data in buffer, i.e where array begins in buffer
           (in buffer bytes).
        :type strides: :samp:`None` or sequence of :obj:`int`
        :param strides: Strides of data in memory.
        :type order: {:samp:`C`, :samp:`F`} or :samp:`None`
        :param order: Row-major (C-style) or column-major (Fortran-style) order.
        :type locale_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param locale_extent: The array extent to be allocated on this locale.
        """

        self = object.__new__(cls)
        if locale_extent is None or (not isinstance(locale_extent, _LocaleExtent)):
            raise ValueError(
                "Got locale_extent=%s, expecting instance of type %s"
                %
                (locale_extent, _LocaleExtent)
            )
        if (shape is not None) and (not _np.all(locale_extent.shape_h == shape)):
            raise ValueError(
                "Got conflicting locale shape: shape=%s, locale_extent.shape_n=%s"
                %
                (shape, locale_extent.shape_h)
            )

        self._slndarray = \
            slndarray(
                shape=locale_extent.shape_h,
                dtype=dtype,
                buffer=buffer,
                offset=offset,
                strides=strides,
                order=order
            )
        self._intra_locale_rank = intra_locale_rank
        self._intra_locale_size = intra_locale_size
        self._intra_partition_dims = intra_partition_dims
        self._locale_extent = locale_extent
        self._halo = _convert_halo_to_array_form(halo, self._locale_extent.ndim)
        self._intra_partition_dims = _np.zeros_like(locale_extent.shape_h)
        self._intra_partition_dims, self._intra_partition = \
            self.calculate_intra_partition(
                intra_locale_size=self._intra_locale_size,
                intra_locale_dims=self._intra_partition_dims,
                intra_locale_rank=self._intra_locale_rank,
                extent=self._locale_extent,
                halo=self._halo
            )
        self.comms_and_distrib = comms_and_distrib

        return self

    def calculate_intra_partition(
            self,
            intra_locale_size,
            intra_locale_dims,
            intra_locale_rank,
            extent,
            halo
    ):
        """
        Splits :samp:`{extent}` into :samp:`self.intra_locale_size` number
        of tiles.
        """
        intra_locale_dims = \
            _array_split.split.calculate_num_slices_per_axis(
                intra_locale_dims,
                intra_locale_size
            )
        ndim = extent.ndim
        if extent.size_n > 0:

            shape_splitter = \
                _array_split.ShapeSplitter(
                    array_shape=extent.shape_n,
                    axis=intra_locale_dims,
                    halo=0,
                    array_start=extent.start_n
                )
            split = shape_splitter.calculate_split()
            rank_extent = \
                _HaloSubExtent(
                    globale_extent=extent,
                    slice=split.flatten()[intra_locale_rank],
                    halo=halo
                )
            # Convert rank_extent_n and rank_extent_h from global-indices
            # to local-indices
            rank_extent = extent.globale_to_locale_extent_h(rank_extent)

            rank_h_relative_extent_n = \
                _IndexingExtent(
                    start=rank_extent.start_n - rank_extent.start_h,
                    stop=rank_extent.start_n - rank_extent.start_h + rank_extent.shape_n,
                )

            rank_view_slice_n = rank_extent.to_slice_n()
            rank_view_slice_h = rank_extent.to_slice_h()
            rank_view_relative_slice_n = rank_h_relative_extent_n.to_slice()
            rank_view_partition_h = rank_view_slice_n
            if _np.any(extent.halo > 0):
                shape_splitter = \
                    _array_split.ShapeSplitter(
                        array_shape=extent.shape_h,
                        axis=intra_locale_dims,
                        halo=0,
                    )
                split = shape_splitter.calculate_split()
                rank_view_partition_h = split.flatten()[intra_locale_rank]
            lndarray_view_slice_n = extent.globale_to_locale_extent_h(extent).to_slice_n()
        else:
            rank_view_slice_n = tuple([slice(0, 0) for i in range(ndim)])
            rank_view_slice_h = rank_view_slice_n
            rank_view_relative_slice_n = rank_view_slice_n
            rank_view_partition_h = rank_view_slice_n
            lndarray_view_slice_n = rank_view_slice_n

        return \
            (
                intra_locale_dims,
                PartitionViewSlices(
                    rank_view_slice_n,
                    rank_view_slice_h,
                    rank_view_relative_slice_n,
                    rank_view_partition_h,
                    lndarray_view_slice_n
                )
            )

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
    def intra_partition(self):
        """
        A :obj:`PartitionViewSlices` containing slices for this rank (of :samp:`rank_comm`).
        """
        return self._intra_partition

    @property
    def intra_partition_dims(self):
        """
        A sequence of integers indicating the number of partitions
        along each axis which determines the per-rank views of the locale extent array.
        """
        return self._intra_partition_dims

    @property
    def locale_extent(self):
        """
        A :obj:`LocaleExtent` describing the portion of the array assigned to this locale.
        """
        return self._locale_extent

    @property
    def halo(self):
        """
        The number of ghost cells for intra locale partitioning of the extent.
        This is an upper bound on the per-rank partitions, with the halo possibly
        trimmed by the halo extent (due to being on globale boundary).
        """
        return self._halo

    @property
    def md(self):
        """
        Meta-data object of type :obj:`NdarrayMetaData`.
        """
        return self._slndarray.md

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
        A tile view of the array for this rank of :samp:`rank_comm`.
        """
        return self._slndarray[self._intra_partition.rank_view_slice_n]

    @property
    def rank_view_h(self):
        """
        A tile view (including halo elements) of the array for this rank of :samp:`rank_comm`.
        """
        return self._slndarray[self._intra_partition.rank_view_slice_h]

    @property
    def rank_view_slice_n(self):
        """
        Sequence of :obj:`slice` objects used to generate :attr:`rank_view_n`.
        """
        return self._intra_partition.rank_view_slice_n

    @property
    def rank_view_slice_h(self):
        """
        Sequence of :obj:`slice` objects used to generate :attr:`rank_view_h`.
        """
        return self._intra_partition.rank_view_slice_h

    @property
    def rank_view_partition_h(self):
        """
        Rank tile view from the paritioning of entire :samp:`self._slndarray`
        (i.e. partition of halo array). Same as :samp:`self.rank_view_n` when
        halo is zero.
        """
        return self._slndarray[self._intra_partition.rank_view_partition_slice_h]

    @property
    def view_n(self):
        """
        View of entire array without halo.
        """
        return self._slndarray[self._intra_partition.lndarray_view_slice_n]

    @property
    def view_h(self):
        """
        The entire :obj:`lndarray` view including halo (i.e. :samp:{self}).
        """
        return self._slndarray.view()

    def fill(self, value):
        """
        Fill the array with a scalar value (excludes ghost elements).

        :type value: scalar
        :param value: All non-ghost elements are assigned this value.
        """
        self._slndarray.rank_view_n.fill(value)

    def fill_h(self, value):
        """
        Fill the array with a scalar value (including ghost elements).

        :type value: scalar
        :param value: All elements (including ghost elements) are assigned this value.
        """
        self._slndarray[self._intra_partition.rank_view_partition_slice_h].fill(value)


def empty(
    shape=None,
    dtype="float64",
    comms_and_distrib=None,
    order='C',
    return_rma_window_buffer=False,
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
    :type comms_and_distrib: :obj:`numpy.dtype`
    :param comms_and_distrib: Data type of array elements.
    :rtype: :obj:`lndarray`
    :return: Newly created array with uninitialised elements.
    """
    if comms_and_distrib is None:
        comms_and_distrib = create_distribution(shape=shape, **kwargs)

    intra_locale_rank = comms_and_distrib.locale_comms.intra_locale_comm.rank
    intra_locale_size = comms_and_distrib.locale_comms.intra_locale_comm.size
    locale_extent = \
        comms_and_distrib.distribution.locale_extents[
            comms_and_distrib.this_locale.inter_locale_rank
        ]

    rma_window_buffer = \
        comms_and_distrib.locale_comms.alloc_locale_buffer(
            shape=locale_extent.shape_h,
            dtype=dtype
        )

    ret = \
        lndarray(
            shape=rma_window_buffer.shape,
            buffer=rma_window_buffer.buffer,
            dtype=dtype,
            order=order,
            intra_locale_rank=intra_locale_rank,
            intra_locale_size=intra_locale_size,
            intra_partition_dims=intra_partition_dims,
            locale_extent=locale_extent,
            halo=comms_and_distrib.distribution.halo,
            comms_and_distrib=comms_and_distrib
        )

    if return_rma_window_buffer:
        ret = ret, rma_window_buffer
    return ret


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
        ret_ary = \
            empty(
                dtype=ary.dtype,
                comms_and_distrib=ary.comms_and_distrib,
                order=ary.md.order,
                intra_partition_dims=ary.intra_partition_dims
            )
    else:
        ret_ary = _np.empty_like(ary, dtype=dtype)

    return ret_ary


def zeros(shape=None, dtype="float64", comms_and_distrib=None, order='C', **kwargs):
    """
    Creates array of zero-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type comms_and_distrib: :obj:`numpy.dtype`
    :param comms_and_distrib: Data type of array elements.
    :rtype: :obj:`lndarray`
    :return: Newly created array with zero-initialised elements.
    """
    ary = empty(shape, dtype=dtype, comms_and_distrib=comms_and_distrib, order=order, **kwargs)
    ary.fill_h(ary.dtype.type(0))

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
    ary.fill_h(ary.dtype.type(0))

    return ary


def ones(shape=None, dtype="float64", comms_and_distrib=None, order='C', **kwargs):
    """
    Creates array of one-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type comms_and_distrib: :obj:`numpy.dtype`
    :param comms_and_distrib: Data type of array elements.
    :rtype: :obj:`lndarray`
    :return: Newly created array with one-initialised elements.
    """
    ary = empty(shape, dtype=dtype, comms_and_distrib=comms_and_distrib, order=order, **kwargs)
    ary.fill_h(ary.dtype.type(1))

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
    ary.fill_h(ary.dtype.type(1))

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
