"""
==================================
The :mod:`mpi_array.locale` Module
==================================

Defines :obj:`LndarrayProxy` class and factory functions for
creating multi-dimensional arrays where memory is allocated
using :meth:`mpi4py.MPI.Win.Allocate_shared` or :meth:`mpi4py.MPI.Win.Allocate`.

Classes
=======

..
   Special template for mpi_array.locale.LndarrayProxy to avoid numpydoc
   documentation style sphinx warnings/errors from numpy.ndarray inheritance.

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_ndarray_class.rst

   lndarray - Sub-class of :obj:`numpy.ndarray` which uses MPI allocated memory buffer.

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   LndarrayProxy - Thin container for :obj:`lndarray` which provides convenience views.
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

import sys as _sys
import numpy as _np
import mpi4py.MPI as _mpi
import array_split as _array_split
from array_split.split import convert_halo_to_array_form as _convert_halo_to_array_form
import collections as _collections

from .license import license as _license, copyright as _copyright, version as _version
from .comms import create_distribution
from .distribution import LocaleExtent as _LocaleExtent
from .distribution import HaloSubExtent as _HaloSubExtent
from .distribution import IndexingExtent as _IndexingExtent
from .utils import log_shared_memory_alloc as _log_shared_memory_alloc
from .utils import log_memory_alloc as _log_memory_alloc
from . import logging as _logging


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class NdarrayMetaData(object):

    """
    Encapsulates, strides, offset and order argument of :meth:`LndarrayProxy.__new__`.
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


class win_lndarray(_np.ndarray):

    """
    Sub-class of :obj:`numpy.ndarray` which allocates buffer using
    MPI window allocated memory.
    """

    def __new__(
        cls,
        shape,
        dtype=_np.dtype("float64"),
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        comm=None,
        root_rank=0
    ):
        """
        Construct. Allocates shared-memory (:func:`mpi4py.MPI.Win.Allocated_shared`)
        buffer when :samp:`{comm}.size > 1`. Uses :func:`mpi4py.MPI.Win.Allocate`
        to allocate buffer when :samp:`{comm}.size == 1`.

        :type shape: :samp:`None` or sequence of :obj:`int`
        :param shape: **Local** shape of the array, this parameter is ignored.
        :type dtype: :obj:`numpy.dtype`
        :param dtype: Data type for elements of the array.
        :type buffer: :obj:`buffer`
        :param buffer: The sequence of bytes providing array element storage.
           Raises :obj:`ValueError` if :samp:`{buffer} is None`.
        :type offset: :samp:`None` or :obj:`int`
        :param offset: Offset of array data in buffer, i.e where array begins in buffer
           (in buffer bytes).
        :type strides: :samp:`None` or sequence of :obj:`int`
        :param strides: Strides of data in memory.
        :type order: {:samp:`C`, :samp:`F`} or :samp:`None`
        :param order: Row-major (C-style) or column-major (Fortran-style) order.
        :type comm: :obj:`mpi4py.Comm`
        :param comm: Communicator used for allocating MPI window memory.
        :type root_rank: :obj:`int`
        :param root_rank: Rank of root process which allocates the shared memory.
        """
        dtype = _np.dtype(dtype)
        if comm is None:
            raise ValueError("Got comm is None, require comm to be a valid mpi4py.MPI.Comm object")
        if comm is _mpi.COMM_NULL:
            raise ValueError(
                "Got comm is COMM_NULL, require comm to be a valid mpi4py.MPI.Comm object"
            )
        if buffer is None:
            num_rank_bytes = 0
            rank_shape = shape
            if comm.rank == root_rank:
                num_rank_bytes = int(_np.product(rank_shape) * dtype.itemsize)
            else:
                rank_shape = tuple(_np.zeros_like(rank_shape))

            logger = _logging.get_rank_logger(__name__ + "." + cls.__name__)

            if (_mpi.VERSION >= 3) and (comm.size > 1):
                _log_shared_memory_alloc(
                    logger.debug, "BEG: ", num_rank_bytes, rank_shape, dtype
                )
                win = \
                    _mpi.Win.Allocate_shared(
                        num_rank_bytes,
                        dtype.itemsize,
                        comm=comm
                    )
                _log_shared_memory_alloc(
                    logger.debug, "END: ", num_rank_bytes, rank_shape, dtype
                )
                buf_isize_pair = win.Shared_query(0)
                buffer = buf_isize_pair[0]
            else:
                _log_memory_alloc(
                    logger.debug, "BEG: ", num_rank_bytes, rank_shape, dtype
                )
                win = _mpi.Win.Allocate(num_rank_bytes, dtype.itemsize, comm=comm)
                _log_memory_alloc(
                    logger.debug, "END: ", num_rank_bytes, rank_shape, dtype
                )
                buffer = win.memory
        buffer = _np.array(buffer, dtype='B', copy=False)
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
        self._comm = comm
        self._win = win

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

        self._comm = getattr(obj, '_comm', None)
        self._win = getattr(obj, '_win', None)

    @property
    def comm(self):
        """
        The :obj:`mpi4py.MPI.Comm` communicator which was collectively used to allocate
        the buffer (memory) for this array.
        """
        return self._comm

    @property
    def win(self):
        """
        The :obj:`mpi4py.MPI.Win` window which was created when allocating
        the buffer (memory) for this array.
        """
        return self._win

    def free(self):
        """
        Collective (over all processes in :attr:`comm`) free the MPI window
        and associated memory buffer.
        """
        self.shape = tuple(_np.zeros_like(self.shape))
        if self._win is not None:
            self._win.Free()

    def __enter__(self):
        """
        For use with :samp:`with` contexts.
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        For use with :samp:`with` contexts.
        """
        self.free()
        return False


class lndarray(_np.ndarray):

    """
    Sub-class of :obj:`numpy.ndarray` which requires :samp:`{buffer}` to
    be specified for instantiation.
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
        :type buffer: :obj:`buffer`
        :param buffer: The sequence of bytes providing array element storage.
           Raises :obj:`ValueError` if :samp:`{buffer} is None`.
        :type offset: :samp:`None` or :obj:`int`
        :param offset: Offset of array data in buffer, i.e where array begins in buffer
           (in buffer bytes).
        :type strides: :samp:`None` or sequence of :obj:`int`
        :param strides: Strides of data in memory.
        :type order: {:samp:`C`, :samp:`F`} or :samp:`None`
        :param order: Row-major (C-style) or column-major (Fortran-style) order.
        """

        if buffer is None:
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

    def free(self):
        """
        Release reference to buffer, and zero-ise :samp:`self.shape`.
        """
        pass


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
if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
    PartitionViewSlices.__doc__ =\
        """
        Stores multiple :obj:`tuple`-of-:obj:`slice` objects indicating
        the slice (tile) of the :obj:`lndarray` on which a :samp:`intra_locale_comm`
        rank MPI process operates.
        """
    PartitionViewSlices.rank_view_slice_n.__doc__ =\
        """
        Slice indicating tile of the non-halo array.
        """
    PartitionViewSlices.rank_view_slice_h.__doc__ =\
        """
        The slice :attr:`rank_view_slice_n` with halo added.
        """
    PartitionViewSlices.rank_view_relative_slice_n.__doc__ =\
        """
        *Relative* slice which can be used to remove the
        halo elements from a view generated using :attr:`rank_view_slice_h`.
        """
    PartitionViewSlices.rank_view_partition_slice_h.__doc__ =\
        """
        Slice indicating tile of the halo array.
        """
    PartitionViewSlices.lndarray_view_slice_n.__doc__ =\
        """
        Slice for generating a view of a :obj:`lndarray` with
        the halo removed.
        """

#: Cache for locale array partitioning
_intra_partition_cache = _collections.defaultdict(lambda: None)


class LndarrayProxy(object):

    """
    Proxy for :obj:`lndarray` instances. Provides :samp:`peer_rank`
    views of the array for parallelism.
    """

    #: The "low index" indices.
    LO = _LocaleExtent.LO

    #: The "high index" indices.
    HI = _LocaleExtent.HI

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
        comms_and_distrib=None,
        rma_window_buffer=None
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
           Must be specified (not :samp:`None`).
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

        # initialise these members before potential exceptions
        # because they are referenced in self.free (via self.__del__).
        self._lndarray = None
        self.rma_window_buffer = None

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

        self._lndarray = \
            lndarray(
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
        self.rma_window_buffer = rma_window_buffer

        return self

    def free(self):
        """
        Release locale array memory and assign :samp:`None` to self attributes.
        """
        if self._lndarray is not None:
            self._lndarray.free()
            self._lndarray = None
        self._intra_locale_rank = None
        self._intra_locale_size = None
        self._intra_partition_dims = None
        self._locale_extent = None
        self._halo = None
        self._intra_partition_dims = None
        self._intra_partition = None
        self.comms_and_distrib = None
        if self.rma_window_buffer is not None:
            self.rma_window_buffer.free()
            self.rma_window_buffer = None

    def __del__(self):
        """
        Calls :meth:`free`.
        """
        self.free()

    def __enter__(self):
        """
        For use with :samp:`with` contexts.
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        For use with :samp:`with` contexts.
        """
        self.free()
        return False

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
        global _intra_partition_cache
        key = \
            (
                intra_locale_size,
                tuple(intra_locale_dims),
                intra_locale_rank,
                extent.to_tuple(),
                tuple(tuple(row) for row in halo.tolist())
            )
        partition_pair = _intra_partition_cache[key]

        if partition_pair is None:
            ndim = extent.ndim
            rank_view_slice_n = tuple()
            rank_view_slice_h = rank_view_slice_n
            rank_view_relative_slice_n = rank_view_slice_n
            rank_view_partition_h = rank_view_slice_n
            lndarray_view_slice_n = rank_view_slice_n

            if ndim > 0:
                intra_locale_dims = \
                    _array_split.split.calculate_num_slices_per_axis(
                        intra_locale_dims,
                        intra_locale_size
                    )
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

            partition_pair = \
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
            _intra_partition_cache[key] = partition_pair

        return partition_pair

    def __getitem__(self, *args, **kwargs):
        """
        Return slice/item from :attr:`lndarray` array.
        """
        return self._lndarray.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """
        Set slice/item in :attr:`lndarray` array.
        """
        self._lndarray.__setitem__(*args, **kwargs)

    def __eq__(self, other):
        """
        """
        if isinstance(other, LndarrayProxy):
            return self._lndarray == other._lndarray
        else:
            return self._lndarray == other

    @property
    def lndarray(self):
        """
        An :obj:`lndarray` instance containing array data in (potentially)
        shared memory.
        """
        return self._lndarray

    @property
    def intra_partition(self):
        """
        A :obj:`PartitionViewSlices` containing slices for this rank (of :samp:`peer_comm`).
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
        return self._lndarray.md

    @property
    def dtype(self):
        """
        A :obj:`numpy.dtype` object describing the element type of this array.
        """
        return self._lndarray.dtype

    @property
    def shape(self):
        """
        The shape of the locale array (including halo), i.e. :samp:`self.lndarray.shape`.
        """
        return self._lndarray.shape

    @property
    def rank_view_n(self):
        """
        A tile view of the array for this rank of :samp:`peer_comm`.
        """
        return self._lndarray[self._intra_partition.rank_view_slice_n]

    @property
    def rank_view_h(self):
        """
        A tile view (including halo elements) of the array for this rank of :samp:`peer_comm`.
        """
        return self._lndarray[self._intra_partition.rank_view_slice_h]

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
        Rank tile view from the paritioning of entire :samp:`self._lndarray`
        (i.e. partition of halo array). Same as :samp:`self.rank_view_n` when
        halo is zero.
        """
        return self._lndarray[self._intra_partition.rank_view_partition_slice_h]

    @property
    def view_n(self):
        """
        View of entire array without halo.
        """
        return self._lndarray[self._intra_partition.lndarray_view_slice_n]

    @property
    def view_h(self):
        """
        The entire :obj:`LndarrayProxy` view including halo (i.e. :samp:{self}).
        """
        return self._lndarray.view()

    def fill(self, value):
        """
        Fill the array with a scalar value (excludes ghost elements).

        :type value: scalar
        :param value: All non-ghost elements are assigned this value.
        """
        self._lndarray[self._intra_partition.rank_view_slice_n].fill(value)

    def fill_h(self, value):
        """
        Fill the array with a scalar value (including ghost elements).

        :type value: scalar
        :param value: All elements (including ghost elements) are assigned this value.
        """
        self._lndarray[self._intra_partition.rank_view_partition_slice_h].fill(value)


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
    :rtype: :obj:`LndarrayProxy`
    :return: Newly created array with uninitialised elements.
    """
    if comms_and_distrib is None:
        comms_and_distrib = create_distribution(shape=shape, **kwargs)

    intra_locale_rank = comms_and_distrib.locale_comms.intra_locale_comm.rank
    intra_locale_size = comms_and_distrib.locale_comms.intra_locale_comm.size
    locale_extent = \
        comms_and_distrib.distribution.get_extent_for_rank(
            inter_locale_rank=comms_and_distrib.this_locale.inter_locale_rank
        )

    rma_window_buffer = \
        comms_and_distrib.locale_comms.alloc_locale_buffer(
            shape=locale_extent.shape_h,
            dtype=dtype
        )

    kwargs = dict()
    if not return_rma_window_buffer:
        kwargs = {
            "comms_and_distrib": comms_and_distrib,
            "rma_window_buffer": rma_window_buffer,
        }
    ret = \
        LndarrayProxy(
            shape=rma_window_buffer.shape,
            buffer=rma_window_buffer.buffer,
            dtype=dtype,
            order=order,
            intra_locale_rank=intra_locale_rank,
            intra_locale_size=intra_locale_size,
            intra_partition_dims=intra_partition_dims,
            locale_extent=locale_extent,
            halo=comms_and_distrib.distribution.halo,
            **kwargs
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
    if (isinstance(ary, LndarrayProxy)):
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
    :rtype: :obj:`LndarrayProxy`
    :return: Newly created array with zero-initialised elements.
    """
    ary = empty(shape, dtype=dtype, comms_and_distrib=comms_and_distrib, order=order, **kwargs)
    ary.fill_h(ary.dtype.type(0))

    return ary


def zeros_like(ary, *args, **kwargs):
    """
    Return a new zero-initialised array with the same shape and type as a given array.

    :type ary: :obj:`LndarrayProxy`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`LndarrayProxy`
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
    :rtype: :obj:`LndarrayProxy`
    :return: Newly created array with one-initialised elements.
    """
    ary = empty(shape, dtype=dtype, comms_and_distrib=comms_and_distrib, order=order, **kwargs)
    ary.fill_h(ary.dtype.type(1))

    return ary


def ones_like(ary, *args, **kwargs):
    """
    Return a new one-initialised array with the same shape and type as a given array.

    :type ary: :obj:`LndarrayProxy`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`LndarrayProxy`
    :return: Array of one-initialized data with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.fill_h(ary.dtype.type(1))

    return ary


def copy(ary):
    """
    Return an array copy of the given object.

    :type ary: :obj:`LndarrayProxy`
    :param ary: Array to copy.
    :rtype: :obj:`LndarrayProxy`
    :return: A copy of :samp:`ary`.
    """
    ary_out = empty_like(ary)
    ary_out.rank_view_n[...] = ary.rank_view_n[...]

    return ary_out


__all__ = [s for s in dir() if not s.startswith('_')]
