"""
==================================
The :mod:`mpi_array.update` Module
==================================

Helper classes for calculating sub-extent intersections in order
to perform remote array element copying/updates.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   ExtentAndRegion - container for locale extent and update region (sub-extent).
   MpiExtentAndRegion - Provides MPI datatype creation.
   ExtentUpdate - Base class for describing a sub-extent update.
   PairExtentUpdate - Describes sub-extent source and sub-extent destination.
   MpiPairExtentUpdate - Like :obj:`PairExtentUpdate`, with MPI data types.
   MpiPairExtentUpdateDifferentDtypes - Over-rides :meth:`MpiPairExtentUpdate.do_get`.
   HaloSingleExtentUpdate - Describes sub-extent for halo region update.
   MpiHaloSingleExtentUpdate - Like :obj:`HaloSingleExtentUpdate`, with MPI data types.
   UpdatesForRedistribute - Calculate sequence of overlapping extents between two distributions.
   MpiUpdatesForRedistribute - Extends :obj:`UpdatesForRedistribute` with MPI data types.
   UpdatesForGet - The updates which fetch a region of an array.
   MpiUpdatesForGet - Extends :obj:`UpdatesForGet` with MPI data types.
   HalosUpdate - The halo updates of a locale.
   MpiHalosUpdate - Extends :obj:`HalosUpdate` with MPI data types.
   ProcessUpdates - The updates of a process for a copy between arrays.
   split_region - Splits a region into pieces along an axis.
   pair_extent_update_copyto - Copies the region of an update between arrays.
   raise_not_in_source_distribution - Raises for regions not in a distribution.
   RmaUpdateExecutor - Execute updates using one-sided RMA fetch.
"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi
import collections as _collections
import copy as _copy
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from .indexing import HaloIndexingExtent, IndexingExtent
from .indexing import calc_intersection_split as _calc_intersection_split
from . import types as _types

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class ExtentAndRegion:

    """
    Container for :obj:`mpi_array.distribution.LocaleExtent` and
    an update region (:obj:`mpi_array.indexing.IndexingExtent`).
    """

    def __init__(self, locale_extent, region_extent=None):
        """
        Initialise.

        :type locale_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param locale_extent: The extent of the locale array.
        :type region_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param region_extent: The update region (globale indices).
        """
        self._locale_extent = locale_extent
        self._region_extent = region_extent

    @property
    def locale_extent(self):
        """
        The :obj:`mpi_array.distribution.LocaleExtent` of the locale array.
        """
        return self._locale_extent

    @property
    def region_extent(self):
        """
        The update region, an :obj:`mpi_array.indexing.IndexingExtent` (globale indices).
        """
        return self._region_extent

    @region_extent.setter
    def region_extent(self, region):
        self._region_extent = region


class ExtentUpdate(object):

    """
    Source and destination indexing info for updating a sub-extent region.
    """

    def __init__(self, dst_extent_info, src_extent_info):
        """
        Initialise.

        :type dst_extent_info: :obj:`ExtentAndRegion`
        :param dst_extent_info: Info containing locale extent which is to receive region update.
        :type src_extent_info: :obj:`ExtentAndRegion`
        :param src_extent_info: Info containing locale extent from which the region update is read.
        """
        object.__init__(self)
        self._dst = dst_extent_info
        self._src = src_extent_info

    @property
    def dst_extent(self):
        """
        The locale :obj:`mpi_array.distribution.LocaleExtent` which is to receive sub-array update.
        """
        return self._dst.locale_extent

    @property
    def src_extent(self):
        """
        The locale :obj:`mpi_array.distribution.CartLocaleExtent` from which the sub-array
        update is read.
        """
        return self._src.locale_extent


def pair_extent_update_copyto(peu, dst_array, src_array, casting):
    """
    Copies the :samp:`{peu}.src_update_extent` region from :samp:`{src_array}`
    to the :samp:`{peu}.dst_update_extent` region of :samp:`{dst_array}`

    :type peu: :obj:`PairExtentUpdate`
    :param peu: Object describing extent of :samp:`dst_array` and :samp:`src_array`
       and the source and destination regions.
    :type dst_array: :obj:`numpy.ndarray`
    :param dst_array: Destination for copy.
    :type src_array: :obj:`numpy.ndarray`
    :param src_array: Source for copy.
    :type casting: :obj:`str`
    :param casting: Indicates casting regime, see :func:`numpy.copyto`.
    """
    src_slice = peu.src_extent.globale_to_locale_slice_h(peu.src_update_extent.to_slice())
    dst_slice = peu.dst_extent.globale_to_locale_slice_h(peu.dst_update_extent.to_slice())
    _np.copyto(dst_array[dst_slice], src_array[src_slice], casting=casting)


def raise_not_in_source_distribution(dst_extents, src_distrib):
    """
    Raises :obj:`ValueError` for the regions :samp:`{dst_extents}` of
    the destination, which no locale extent of :samp:`{src_distrib}` provides.

    :type dst_extents: sequence of :obj:`mpi_array.distribution.LocaleExtent`
    :param dst_extents: The destination regions which are not in the source distribution.
    :type src_distrib: :obj:`mpi_array.distribution.Distribution`
    :param src_distrib: The source distribution.
    :raises ValueError: Always.
    """
    raise ValueError(
        "Got destination regions which are not in the source distribution "
        "(globale extent start=%s, stop=%s):\n%s"
        %
        (
            src_distrib.globale_extent.start_n.tolist(),
            src_distrib.globale_extent.stop_n.tolist(),
            "\n".join(
                "start=%s, stop=%s" % (extent.start_h.tolist(), extent.stop_h.tolist())
                for extent in dst_extents
            )
        )
    )


def split_region(region, num_pieces):
    """
    Splits :samp:`{region}` into :samp:`{num_pieces}` regions along one axis: the first axis
    with at least :samp:`{num_pieces}` elements (the longest axis if none has). For C order
    arrays the pieces of a region which spans the trailing axes are contiguous.

    :type region: :obj:`mpi_array.indexing.IndexingExtent`
    :param region: The region to split.
    :type num_pieces: :obj:`int`
    :param num_pieces: The number of pieces.
    :rtype: :obj:`list`
    :return: The :samp:`{num_pieces}` pieces, :obj:`mpi_array.indexing.IndexingExtent`
       objects, :samp:`None` for the empty pieces.

    Example::

       >>> from mpi_array.indexing import IndexingExtent
       >>> split_region(IndexingExtent(start=(0, 0), stop=(2, 9)), 3)
       [IndexingExtent(start=(0, 0), stop=(2, 3)), IndexingExtent(start=(0, 3), stop=(2, 6)), \
IndexingExtent(start=(0, 6), stop=(2, 9))]
    """
    shape = _np.asarray(region.shape)
    if (num_pieces <= 1) or (len(shape) == 0):
        return [region, ] + [None, ] * (num_pieces - 1)
    long_axes = _np.flatnonzero(shape >= num_pieces)
    axis = long_axes[0] if len(long_axes) > 0 else int(_np.argmax(shape))
    bounds = region.start[axis] + (shape[axis] * _np.arange(num_pieces + 1)) // num_pieces
    pieces = []
    for i in range(num_pieces):
        piece = None
        if bounds[i] < bounds[i + 1]:
            start = _np.array(region.start, copy=True)
            stop = _np.array(region.stop, copy=True)
            start[axis] = bounds[i]
            stop[axis] = bounds[i + 1]
            piece = IndexingExtent(start=start, stop=stop)
        pieces.append(piece)
    return pieces


ProcessUpdates = _collections.namedtuple("ProcessUpdates", ["cpy2_updates", "rget_updates"])
ProcessUpdates.__doc__ = \
    """
    The updates of a process for a copy between arrays (its share of the updates of its
    locale, see :meth:`MpiUpdatesForRedistribute.create_process_updates`).
    """
ProcessUpdates.cpy2_updates.__doc__ = \
    "The :obj:`list` of updates which the process copies directly from the source locale array."
ProcessUpdates.rget_updates.__doc__ = \
    "The :obj:`list` of updates which the process fetches from remote source locales."


class PairExtentUpdate(ExtentUpdate):

    """
    Source and destination indexing info for updating a sub-extent region.
    """

    def __init__(self, dst_extent, src_extent, dst_update_extent, src_update_extent):
        ExtentUpdate.__init__(
            self,
            ExtentAndRegion(dst_extent, dst_update_extent),
            ExtentAndRegion(src_extent, src_update_extent)
        )

    def copyto(self, dst_array, src_array, casting):
        """
        Copies the :attr:`src_update_extent` region from :samp:`{src_array}`
        to the :attr:`dst_update_extent` region of :samp:`{dst_array}`
        """
        pair_extent_update_copyto(self, dst_array, src_array, casting)

    @property
    def dst_update_extent(self):
        """
        The locale sub-extent (:obj:`mpi_array.indexing.IndexingExtent`) to be updated.
        """
        return self._dst.region_extent

    @property
    def src_update_extent(self):
        """
        The locale sub-extent (:obj:`mpi_array.indexing.IndexingExtent`) from which the update
        is read.
        """
        return self._src.region_extent


class MpiExtentAndRegion(ExtentAndRegion):

    """
    Extends :obj:`ExtentAndRegion` with the MPI data type of the region.
    """

    _mpi_data_type = None
    _owns_mpi_data_type = False

    def __init__(
        self,
        locale_extent,
        region_extent,
        dtype=None,
        order=None,
        mpi_data_type=None,
        mpi_order=None
    ):
        ExtentAndRegion.__init__(self, locale_extent, region_extent)
        self._dtype = dtype
        self._parent_mpi_data_type = None
        self._order = order
        self._mpi_data_type = mpi_data_type
        #: Only the data types created by :meth:`initialise_mpi_data_type` are freed.
        self._owns_mpi_data_type = False
        self._mpi_order = mpi_order

    def __del__(self):
        """
        Frees the MPI data type (:meth:`free_mpi_data_type`).
        """
        self.free_mpi_data_type()

    def free_mpi_data_type(self):
        """
        Frees the MPI data type created by :meth:`initialise_mpi_data_type`. This is
        a local MPI call (no communication), communication which uses the data type
        completes normally.
        """
        if self._owns_mpi_data_type and (not _mpi.Is_finalized()):
            self._mpi_data_type.Free()
        self._owns_mpi_data_type = False
        self._mpi_data_type = None
        self._dtype = None

    def create_data_type(self, dtype, order="C"):
        """
        Creates (and commits) the MPI sub-array data type of the region of the locale array.

        :type dtype: :obj:`numpy.dtype`
        :param dtype: The element type.
        :type order: :obj:`str`
        :param order: The memory layout, :samp:`"C"` or :samp:`"F"`.
        :rtype: :obj:`tuple`
        :return: The :samp:`(mpi_data_type, mpi_order, parent_mpi_data_type)` triple, the
           parent is the data type of the elements.
        """
        mpi_order = _mpi.ORDER_C
        if order.upper() == "F":
            mpi_order = _mpi.ORDER_FORTRAN

        parent_mpi_data_type = _types.to_datatype(dtype)
        mpi_data_type = \
            parent_mpi_data_type.Create_subarray(
                self.locale_extent.shape_h,
                self.region_extent.shape,
                self.locale_extent.globale_to_locale_h(self.region_extent.start),
                order=mpi_order
            )
        mpi_data_type.Commit()

        return mpi_data_type, mpi_order, parent_mpi_data_type

    def initialise_mpi_data_type(self, dtype, order):
        """
        Creates the MPI data type of the region for the :samp:`{dtype}` element type
        and :samp:`{order}` memory layout, unless it was created for them already
        (a data type created for others is freed).
        """
        dtype = _np.dtype(dtype)
        order = order.lower()
        if (
            (self._dtype is None)
            or
            (self._dtype != dtype)
            or
            (self._order != order)
        ):
            self.free_mpi_data_type()
            self._mpi_data_type, self._mpi_order, self._parent_mpi_data_type = \
                self.create_data_type(dtype, order)
            self._owns_mpi_data_type = True
            self._dtype = dtype
            self._order = order

    @property
    def mpi_data_type(self):
        """
        The :obj:`mpi4py.MPI.Datatype` of the region (see :meth:`initialise_mpi_data_type`),
        :samp:`None` before it is created.
        """
        return self._mpi_data_type


class MpiPairExtentUpdate(ExtentUpdate):

    """
    Source and destination indexing info for updating a sub-extent region
    (like :obj:`PairExtentUpdate`).
    Extends :obj:`ExtentUpdate` with API to create :obj:`mpi4py.MPI.Datatype`
    instances (using :meth:`mpi4py.MPI.Datatype.Create_subarray`) for convenient
    transfer of sub-array data.
    """

    def __init__(self, dst_extent, src_extent, dst_update_extent, src_update_extent):
        self._casting = "same_kind"
        ExtentUpdate.__init__(
            self,
            MpiExtentAndRegion(dst_extent, dst_update_extent),
            MpiExtentAndRegion(src_extent, src_update_extent)
        )
        self._str_format = \
            (
                "%8s, %20s, %20s, %20s, %20s, %20s, %20s, %16s, "
                +
                "%8s, %20s, %20s, %20s, %20s, %20s, %20s, %16s"
            )
        self._header_str = \
            (
                self._str_format
                %
                (
                    "dst rank",
                    "dst ext  glb start",
                    "dst ext  glb stop ",
                    "dst updt loc start",
                    "dst updt loc stop ",
                    "dst updt glb start",
                    "dst updt glb stop ",
                    "dst MPI datatype",
                    "src rank",
                    "src ext  glb start",
                    "src ext  glb stop ",
                    "src updt loc start",
                    "src updt loc stop ",
                    "src updt glb start",
                    "src updt glb stop ",
                    "src MPI datatype",
                )
            )

    def initialise_data_types(self, dst_dtype, src_dtype, dst_order, src_order):
        """
        Assigns new instances of `mpi4py.MPI.Datatype` for
        the :attr:`dst_data_type` and :attr:`src_data_type`
        attributes. Only creates new instances when
        the :samp:`{dst_dtype}`, :samp:`{src_dtype}` or :samp:`{order}`
        do not match existing instances.

        :type dst_dtype: :obj:`numpy.dtype`
        :param dst_dtype: The array element type of the array which is to receive data.
        :type src_dtype: :obj:`numpy.dtype`
        :param src_dtype: The array element type of the array from which data is copied.
        :type order: :obj:`str`
        :param order: Array memory layout, :samp:`"C"` for C array,
           or :samp:`"F"` for fortran array.
        """
        self._dst.initialise_mpi_data_type(dst_dtype, dst_order)
        self._src.initialise_mpi_data_type(src_dtype, src_order)

    def copyto(self, dst_array, src_array, casting):
        """
        Copies the :attr:`src_update_extent` region from :samp:`{src_array}`
        to the :attr:`dst_update_extent` region of :samp:`{dst_array}`
        """
        pair_extent_update_copyto(self, dst_array, src_array, casting)

    @property
    def dst_update_extent(self):
        """
        The locale sub-extent (:obj:`mpi_array.indexing.IndexingExtent`) to be updated.
        """
        return self._dst.region_extent

    @property
    def src_update_extent(self):
        """
        The locale sub-extent (:obj:`mpi_array.indexing.IndexingExtent`) from which the update
        is read.
        """
        return self._src.region_extent

    @property
    def dst_dtype(self):
        """
        A :obj:`numpy.dtype` object indicating the element type
        of the destination array.
        """
        return self._dst._dtype

    @property
    def src_dtype(self):
        """
        A :obj:`numpy.dtype` object indicating the element type
        of the source array.
        """
        return self._src._dtype

    @property
    def dst_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array (:attr:`dst_update_extent`) of the destination
        locale array which receives the update values.
        """
        return self._dst.mpi_data_type

    @property
    def src_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array (:attr:`src_update_extent`) of the source
        locale array from which the update values are read.
        """
        return self._src.mpi_data_type

    @property
    def casting(self):
        """
        A :obj:`str` indicating the casting allowed between different :obj:`numpy.dtype` elements.
        See the :samp:`casting` parameter for the :func:`numpy.copyto` function.
        """
        return self._casting

    @casting.setter
    def casting(self, casting):
        self._casting = casting

    def do_get(self, mpi_win, target_src_rank, origin_dst_buffer):
        """
        Performs calls :meth:`mpi4py.MPI.Win.Get` method of :samp:`mpi_win`
        to perform the RMA data-transfer.

        :type mpi_win: :obj:`mpi4py.MPI.Win`
        :param mpi_win: Window used to retrieve update region for array.
        :type target_src_rank: :obj:`int`
        :param target_src_rank: The rank of the target process in :samp:`mpi_win.group.rank`.
        :type origin_dst_buffer: :obj:`memoryview`
        :param origin_dst_buffer: The destination memory for the update, size of buffer
           should correspond to the size of the :attr:`~ExtentUpdate.dst_extent`.
        """
        mpi_win.Get(
            [origin_dst_buffer, 1, self.dst_data_type],
            target_src_rank,
            [0, 1, self.src_data_type]
        )

    def do_rget(self, mpi_win, target_src_rank, origin_dst_buffer):
        """
        Performs calls :meth:`mpi4py.MPI.Win.Rget` method of :samp:`mpi_win`
        to perform the RMA data-transfer.

        :type mpi_win: :obj:`mpi4py.MPI.Win`
        :param mpi_win: Window used to retrieve update region for array.
        :type target_src_rank: :obj:`int`
        :param target_src_rank: The rank of the target process in :samp:`mpi_win.group.rank`.
        :type origin_dst_buffer: :obj:`memoryview`
        :param origin_dst_buffer: The destination memory for the update, size of buffer
           should correspond to the size of the :attr:`~ExtentUpdate.dst_extent`.
        """
        req = \
            mpi_win.Rget(
                [origin_dst_buffer, 1, self.dst_data_type],
                target_src_rank,
                [0, 1, self.src_data_type]
            )
        return req

    def conclude(self):
        """
        Does nothing, the fetched elements are already in the destination
        (see :meth:`MpiPairExtentUpdateDifferentDtypes.conclude`).
        """
        pass

    def __str__(self):
        """
        Stringify.
        """
        dst_mpi_dtype = None
        if self._dst._dtype is not None:
            dst_mpi_dtype = self._dst._parent_mpi_data_type.Get_name()
        src_mpi_dtype = None
        if self._src._dtype is not None:
            src_mpi_dtype = self._src._parent_mpi_data_type.Get_name()

        return \
            (
                self._str_format
                %
                (
                    self.dst_extent.inter_locale_rank,
                    self.dst_extent.start_h,
                    self.dst_extent.stop_h,
                    self.dst_extent.globale_to_locale_h(self.dst_update_extent.start),
                    self.dst_extent.globale_to_locale_h(self.dst_update_extent.stop),
                    self.dst_update_extent.start,
                    self.dst_update_extent.stop,
                    dst_mpi_dtype,
                    self.src_extent.inter_locale_rank,
                    self.src_extent.start_h,
                    self.src_extent.stop_h,
                    self.src_extent.globale_to_locale_h(self.src_update_extent.start),
                    self.src_extent.globale_to_locale_h(self.src_update_extent.stop),
                    self.src_update_extent.start,
                    self.src_update_extent.stop,
                    src_mpi_dtype
                )
            )


class MpiPairExtentUpdateDifferentDtypes(MpiPairExtentUpdate):

    """
    Over-rides :meth:`MpiPairExtentUpdate.do_get` to buffer-copy and
    subsequent casting when source and destination arrays have different :obj:`numpy.dtype`.
    """

    def __init__(self, dst_extent, src_extent, dst_update_extent, src_update_extent):
        """
        """
        MpiPairExtentUpdate.__init__(
            self,
            dst_extent,
            src_extent,
            dst_update_extent,
            src_update_extent
        )
        self._buffer = None
        self._dst_buffer = None

    def do_get(self, mpi_win, target_src_rank, origin_dst_buffer):
        """
        Performs calls :meth:`mpi4py.MPI.Win.Get` method of :samp:`mpi_win`
        to perform the RMA data-transfer. Uses a locally allocated buffer
        to receive the data and then uses :func:`numpy.copyto` to convert
        the :attr:`~MpiPairExtentUpdate.src_dtype` to the :attr:`~MpiPairExtentUpdate.dst_dtype`.

        :type mpi_win: :obj:`mpi4py.MPI.Win.Get`
        :param mpi_win: Window used to retrieve update region for array.
        :type target_src_rank: :obj:`int`
        :param target_src_rank: The rank of the target process in :samp:`mpi_win.group.rank`.
        :type origin_dst_buffer: :obj:`memoryview`
        :param origin_dst_buffer: The destination memory for the update, size of buffer
           should correspond to the size of the :attr:`~ExtentUpdate.dst_extent`.
        """
        self._buffer = _np.empty(shape=self._src.region_extent.shape, dtype=self._src._dtype)
        self._dst_buffer = origin_dst_buffer

        mpi_win.Get(
            [self._buffer, _np.prod(self._buffer.shape),
             self._src._parent_mpi_data_type],
            target_src_rank,
            [0, 1, self.src_data_type]
        )

    def do_rget(self, mpi_win, target_src_rank, origin_dst_buffer):
        """
        Like :meth:`do_get`, with :meth:`mpi4py.MPI.Win.Rget`: the fetch completes
        with the returned request, then call :meth:`conclude`.

        :rtype: :obj:`mpi4py.MPI.Request`
        :return: The request of the fetch.
        """
        self._buffer = _np.empty(shape=self._src.region_extent.shape, dtype=self._src._dtype)
        self._dst_buffer = origin_dst_buffer

        r = mpi_win.Rget(
            [self._buffer, _np.prod(self._buffer.shape),
             self._src._parent_mpi_data_type],
            target_src_rank,
            [0, 1, self.src_data_type]
        )

        return r

    def conclude(self):
        """
        Copies (and casts) the fetched elements to the destination, after the fetch has
        completed.
        """
        origin_dst_buffer_slice = \
            self._dst.locale_extent.globale_to_locale_slice_h(self._dst.region_extent.to_slice())
        _np.copyto(
            self._dst_buffer[origin_dst_buffer_slice],
            self._buffer,
            casting=self.casting
        )
        self._buffer = None
        self._dst_buffer = None


class HaloSingleExtentUpdate(ExtentUpdate):

    """
    Source and destination indexing info for updating a halo portion.
    """

    def __init__(self, dst_extent, src_extent, update_extent):
        ExtentUpdate.__init__(
            self,
            ExtentAndRegion(dst_extent, update_extent),
            ExtentAndRegion(src_extent, update_extent)
        )

    @property
    def update_extent(self):
        """
        The :obj:`mpi_array.indexing.IndexingExtent` indicating the halo sub-array which is to
        be updated.
        """
        return self._src.region_extent


class MpiHaloSingleExtentUpdate(ExtentUpdate):

    """
    Source and destination indexing info for updating the whole of a halo portion.
    Extends :obj:`ExtentUpdate` with API to create :obj:`mpi4py.MPI.Datatype`
    instances (using :meth:`mpi4py.MPI.Datatype.Create_subarray`) for convenient
    transfer of sub-array data.
    """

    def __init__(self, dst_extent, src_extent, update_extent):
        ExtentUpdate.__init__(
            self,
            MpiExtentAndRegion(dst_extent, update_extent),
            MpiExtentAndRegion(src_extent, update_extent)
        )
        self._str_format = \
            "%8s, %20s, %20s, %20s, %20s, %8s, %20s, %20s, %20s, %20s, %20s, %20s, %16s"
        self._header_str = \
            (
                self._str_format
                %
                (
                    "dst rank",
                    "dst ext  glb start",
                    "dst ext  glb stop ",
                    "dst halo loc start",
                    "dst halo loc stop ",
                    "src rank",
                    "src ext  glb start",
                    "src ext  glb stop ",
                    "src halo loc start",
                    "src halo loc stop ",
                    "    halo glb start",
                    "    halo glb stop ",
                    "MPI datatype",
                )
            )

    def initialise_data_types(self, dtype, order):
        """
        Assigns new instances of `mpi4py.MPI.Datatype` for
        the :attr:`dst_data_type` and :attr:`src_data_type`
        attributes. Only creates new instances when
        the :samp:`{dtype}` and :samp:`{order}` do not match
        existing instances.

        :type dtype: :obj:`numpy.dtype`
        :param dtype: The array element type.
        :type order: :obj:`str`
        :param order: Array memory layout, :samp:`"C"` for C array,
           or :samp:`"F"` for fortran array.
        """
        self._dst.initialise_mpi_data_type(dtype=dtype, order=order)
        self._src.initialise_mpi_data_type(dtype=dtype, order=order)

    @property
    def dst_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array of halo elements which are to
        receive update values.
        """
        return self._dst.mpi_data_type

    @property
    def src_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array of source elements from which
        the halo elements receive their values.
        """
        return self._src.mpi_data_type

    @property
    def update_extent(self):
        """
        The :obj:`mpi_array.indexing.IndexingExtent` indicating the halo sub-array which is to
        be updated.
        """
        return self._src.region_extent

    def __str__(self):
        """
        Stringify.
        """
        mpi_dtype = None
        if self._dst._dtype is not None:
            mpi_dtype = _types.to_datatype(self._dst._dtype).Get_name()
        return \
            (
                self._str_format
                %
                (
                    self.dst_extent.cart_rank,
                    self.dst_extent.start_h,
                    self.dst_extent.stop_h,
                    self.dst_extent.globale_to_locale_h(self.update_extent.start),
                    self.dst_extent.globale_to_locale_h(self.update_extent.stop),
                    self.src_extent.cart_rank,
                    self.src_extent.start_h,
                    self.src_extent.stop_h,
                    self.src_extent.globale_to_locale_h(self.update_extent.start),
                    self.src_extent.globale_to_locale_h(self.update_extent.stop),
                    self.update_extent.start,
                    self.update_extent.stop,
                    mpi_dtype
                )
            )


class HalosUpdate(object):

    """
    Indexing info for updating the halo regions of a single locale
    on MPI rank :samp:`self.dst_rank`.
    """

    #: The "low index" indices.
    LO = HaloIndexingExtent.LO

    #: The "high index" indices.
    HI = HaloIndexingExtent.HI

    def __init__(self, dst_rank, rank_to_extents_map, cart_coord_to_extent=None):
        """
        Construct.

        :type dst_rank: :obj:`int`
        :param dst_rank: The MPI rank (:samp:`cart_comm`) of the MPI
           process which is to receive the halo updates.
        :type rank_to_extents_map: :obj:`dict`
        :param rank_to_extents_map: Dictionary of :samp:`(r, extent)`
           pairs for all ranks :samp:`r` (of :samp:`cart_comm`), where :samp:`extent`
           is a :obj:`mpi_array.distribution.CartLocaleExtent` object indicating the indexing extent
           (tile) on MPI rank :samp:`r.` Only the extent of :samp:`{dst_rank}` is needed
           when :samp:`{cart_coord_to_extent}` is given.
        :type cart_coord_to_extent: callable
        :param cart_coord_to_extent: Returns the extent of the tile with a cartesian
           coordinate (:obj:`tuple`), e.g.
           :meth:`mpi_array.distribution.BlockPartition.get_extent_for_cart_coord`. If
           :samp:`None`, a coordinate to extent map is made from :samp:`{rank_to_extents_map}`.
        """
        self.initialise(dst_rank, rank_to_extents_map, cart_coord_to_extent)

    def create_single_extent_update(self, dst_extent, src_extent, halo_extent):
        """
        Factory method for creating instances of type :obj:`HaloSingleExtentUpdate`.

        :type dst_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param dst_extent: The destination locale extent for halo element update.
        :type src_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param src_extent: The source locale extent for obtaining halo element update.
        :type halo_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param halo_extent: The extent indicating the sub-array of halo elements.

        :rtype: :obj:`HaloSingleExtentUpdate`
        :return: Returns new instance of :obj:`HaloSingleExtentUpdate`.
        """
        return HaloSingleExtentUpdate(dst_extent, src_extent, halo_extent)

    def calc_halo_intersection(self, dst_extent, src_extent, axis, dir):
        """
        Calculates the intersection of :samp:`{dst_extent}` halo slab with
        the update region of :samp:`{src_extent}`.

        :type dst_extent: :obj:`mpi_array.distribution.CartLocaleExtent`
        :param dst_extent: Halo slab indicated by :samp:`{axis}` and :samp:`{dir}`
           taken from this extent.
        :type src_extent: :obj:`mpi_array.distribution.CartLocaleExtent`
        :param src_extent: This extent, minus the halo in the :samp:`{axis}` dimension,
           is intersected with the halo slab.
        :type axis: :obj:`int`
        :param axis: Axis dimension indicating slab.
        :type dir: :attr:`LO` or :attr:`HI`
        :param dir: :attr:`LO` for low-index slab or :attr:`HI` for high-index slab.
        :rtype: :obj:`mpi_array.indexing.IndexingExtent`
        :return: Overlap extent of :samp:{dst_extent} halo-slab and
           the :samp:`{src_extent}` update region.
        """
        return \
            dst_extent.halo_slab_extent(axis, dir).calc_intersection(
                src_extent.no_halo_extent(axis)
            )

    def split_extent_for_max_elements(self, extent, max_elements=None):
        """
        Returns :samp:`[{extent}]`, splitting a halo update into smaller updates
        (of at most :samp:`{max_elements}` elements) is not implemented.
        :meth:`initialise` calls it for each halo update.

        :type extent: :obj:`HaloSingleExtentUpdate`
        :param extent: The halo update.
        :type max_elements: :obj:`int`
        :param max_elements: Ignored.
        :rtype: :obj:`list`
        :return: The list :samp:`[{extent}]`.
        """
        return [extent, ]

    def initialise(self, dst_rank, rank_to_extents_map, cart_coord_to_extent=None):
        """
        Calculates the ranks and regions required to update the
        halo regions of the :samp:`dst_rank` MPI rank.

        :type dst_rank: :obj:`int`
        :param dst_rank: The MPI rank (:samp:`cart_comm`) of the MPI
           process which is to receive the halo updates.
        :type rank_to_extents_map: :obj:`dict`
        :param rank_to_extents_map: Dictionary of :samp:`(r, extent)`
           pairs for all ranks :samp:`r` (of :samp:`cart_comm`), where :samp:`extent`
           is a :obj:`mpi_array.distribution.CartLocaleExtent` object indicating the indexing extent
           (tile) on MPI rank :samp:`r.`
        :type cart_coord_to_extent: callable
        :param cart_coord_to_extent: Returns the extent of the tile with a cartesian
           coordinate, see :obj:`HalosUpdate`.
        """
        self._dst_rank = dst_rank
        self._dst_extent = rank_to_extents_map[dst_rank]
        self._updates = [[[], []] for a in range(self._dst_extent.ndim)]
        if cart_coord_to_extent is None:
            if hasattr(rank_to_extents_map, "keys"):
                ranks = rank_to_extents_map.keys()
            else:
                ranks = range(0, len(rank_to_extents_map))
            cart_coord_to_extents_dict = \
                {
                    tuple(rank_to_extents_map[r].cart_coord): rank_to_extents_map[r]
                    for r in ranks
                }
            cart_coord_to_extent = cart_coord_to_extents_dict.__getitem__
        for dir in [self.LO, self.HI]:
            for a in range(self._dst_extent.ndim):
                if dir == self.LO:
                    i_range = range(-1, -self._dst_extent.cart_coord[a] - 1, -1)
                else:
                    i_range = \
                        range(1, self._dst_extent.cart_shape[a] - self._dst_extent.cart_coord[a], 1)
                for i in i_range:
                    src_cart_coord = _np.array(self._dst_extent.cart_coord, copy=True)
                    src_cart_coord[a] += i
                    src_extent = cart_coord_to_extent(tuple(src_cart_coord))
                    halo_extent = self.calc_halo_intersection(self._dst_extent, src_extent, a, dir)
                    if halo_extent is not None:
                        self._updates[a][dir] += \
                            self.split_extent_for_max_elements(
                                self.create_single_extent_update(
                                    self._dst_extent,
                                    src_extent,
                                    halo_extent
                                )
                        )
                    else:
                        break

    @property
    def updates_per_axis(self):
        """
        A :samp:`ndim` length list of pair elements, each element of the pair
        is a list of :obj:`HaloSingleExtentUpdate` objects.
        """
        return self._updates


class MpiHalosUpdate(HalosUpdate):

    """
    Indexing info for updating the halo regions of a single tile
    on MPI rank :samp:`self.dst_rank`.
    Over-rides the :meth:`create_single_extent_update` to
    return :obj:`MpiHaloSingleExtentUpdate` instances.
    """

    def create_single_extent_update(self, dst_extent, src_extent, halo_extent):
        """
        Factory method for creating instances of type :obj:`MpiHaloSingleExtentUpdate`.

        :type dst_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param dst_extent: The destination locale extent for halo element update.
        :type src_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param src_extent: The source locale extent for obtaining halo element update.
        :type halo_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param halo_extent: The extent indicating the sub-array of halo elements.

        :rtype: :obj:`MpiHaloSingleExtentUpdate`
        :return: Returns new instance of :obj:`MpiHaloSingleExtentUpdate`.
        """
        return MpiHaloSingleExtentUpdate(dst_extent, src_extent, halo_extent)


class UpdatesForRedistribute(object):

    """
    Collection of update extents for re-distribution of array
    elements from one distribution to another.
    """

    def __init__(
        self,
        dst_distrib,
        src_distrib,
        peer_rank_translator=None,
        dst_inter_locale_rank=None
    ):
        """
        Calculates the updates.

        :type dst_distrib: :obj:`mpi_array.distribution.Distribution`
        :param dst_distrib: Distribution of the array which receives the elements.
        :type src_distrib: :obj:`mpi_array.distribution.Distribution`
        :param src_distrib: Distribution of the array from which the elements are copied.
        :type peer_rank_translator: :obj:`mpi_array.globale.RankTranslator`
        :param peer_rank_translator: Translates ranks between the peer communicators
           of the destination and source arrays. If given, the regions which can be copied
           directly from the source locales which share processes with a destination locale
           are calculated first.
        :type dst_inter_locale_rank: :samp:`None` or :obj:`int`
        :param dst_inter_locale_rank: Only calculate the updates of the destination locale
           with this rank, :samp:`None` for the updates of all destination locales.
        :raises ValueError: If elements of the destination locales are not in
           the source distribution.
        """
        object.__init__(self)

        self._dst_distrib = dst_distrib
        self._src_distrib = src_distrib
        self._dst_inter_locale_rank = dst_inter_locale_rank

        self._dst_cpy2_updates = None
        self._dst_rget_updates = None

        self.update_dst_halo = False

        self._dst_translated_peer_ranks = None
        self._dst_peer_ranks = None
        self._src_translated_peer_ranks = None
        self._src_peer_ranks = None
        if peer_rank_translator is not None:
            if dst_distrib.peer_ranks_per_locale.ndim == 2:
                self._dst_peer_ranks = _np.sort(dst_distrib.peer_ranks_per_locale)
                self._dst_translated_peer_ranks = \
                    _np.sort(peer_rank_translator.dst_to_src(self._dst_peer_ranks))
            else:
                self._dst_peer_ranks = _copy.deepcopy(dst_distrib.peer_ranks_per_locale)
                self._dst_translated_peer_ranks = _copy.deepcopy(dst_distrib.peer_ranks_per_locale)
                for r in range(len(self._dst_peer_ranks)):
                    self._dst_peer_ranks[r] = _np.sort(self._dst_peer_ranks[r])
                    self._dst_translated_peer_ranks[r] = \
                        _np.sort(peer_rank_translator.dst_to_src(self._dst_peer_ranks[r]))
            if src_distrib.peer_ranks_per_locale.ndim == 2:
                self._src_peer_ranks = _np.sort(src_distrib.peer_ranks_per_locale)
                self._src_translated_peer_ranks = \
                    _np.sort(peer_rank_translator.src_to_dst(self._src_peer_ranks))
            else:
                self._src_peer_ranks = _copy.deepcopy(src_distrib.peer_ranks_per_locale)
                self._src_translated_peer_ranks = _copy.deepcopy(src_distrib.peer_ranks_per_locale)
                for r in range(len(self._src_peer_ranks)):
                    self._src_peer_ranks[r] = _np.sort(self._src_peer_ranks[r])
                    self._src_translated_peer_ranks[r] = \
                        _np.sort(peer_rank_translator.src_to_dst(self._src_peer_ranks[r]))

        self.initialise()

    @property
    def dst_cpy2_updates(self):
        """
        A :obj:`dict` of :samp:`(dst_inter_locale_rank, updates)` pairs, the updates
        copied directly from the source locales which share processes with the destination
        locale.
        """
        return self._dst_cpy2_updates

    @property
    def dst_rget_updates(self):
        """
        A :obj:`dict` of :samp:`(dst_inter_locale_rank, updates)` pairs, the updates
        fetched from the source locales.
        """
        return self._dst_rget_updates

    @property
    def src_peer_ranks(self):
        """
        The (sorted) peer ranks of each source locale (:samp:`None` without
        a :samp:`peer_rank_translator`).
        """
        return self._src_peer_ranks

    @property
    def dst_translated_peer_ranks(self):
        """
        The (sorted) peer ranks of each destination locale, translated to the ranks of
        the peer communicator of the source (:samp:`None` without
        a :samp:`peer_rank_translator`).
        """
        return self._dst_translated_peer_ranks

    @property
    def src_translated_peer_ranks(self):
        """
        The (sorted) peer ranks of each source locale, translated to the ranks of
        the peer communicator of the destination (:samp:`None` without
        a :samp:`peer_rank_translator`).
        """
        return self._src_translated_peer_ranks

    def create_pair_extent_update(
        self,
        dst_extent,
        src_extent,
        intersection_extent
    ):
        """
        Factory method for creating :obj:`PairExtentUpdate` objects.

        :type dst_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param dst_extent: Destination extent.
        :type src_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param src_extent: Source extent.
        :type intersection_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param intersection_extent: The intersection of :samp:`{src_extent}`
           and :samp:`{dst_extent}` which defines the region of array elements which
           are to be transferred from source to destination.
        :rtype: :obj:`list`
        :return: The :obj:`PairExtentUpdate` defining the source sub-array and
           destination sub-array, in a :obj:`list`.
        """
        peu = \
            PairExtentUpdate(
                self._dst_distrib.get_extent_for_rank(dst_extent.inter_locale_rank),
                self._src_distrib.get_extent_for_rank(src_extent.inter_locale_rank),
                intersection_extent,
                intersection_extent
            )

        return [peu, ]

    def calc_intersection_split(self, dst_extent, src_extent):
        """
        Calculates intersection between :samp:`{dst_extent}` and :samp:`{src_extent}`.
        Any regions of :samp:`{dst_extent}` which **do not** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *left-over* :samp:`type({dst_extent})` elements.
        The regions of :samp:`{dst_extent}` which **do** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *update* :obj:`PairExtentUpdate` elements.
        Returns :obj:`tuple` pair :samp:`(leftovers, updates)`

        :type dst_extent: :obj:`mpi_array.indexing.HaloIndexingExtent`
        :param dst_extent: Extent which is to receive update from intersection
           with :samp:`{src_extent}`.
        :type src_extent: :obj:`mpi_array.indexing.HaloIndexingExtent`
        :param src_extent: Extent which is to provide update for the intersecting
           region of :samp:`{dst_extent}`.
        :rtype: :obj:`tuple`
        :return: Returns :obj:`tuple` pair of :samp:`(leftovers, updates)`.
        """
        return \
            _calc_intersection_split(
                dst_extent,
                src_extent,
                self.create_pair_extent_update,
                self.update_dst_halo
            )

    def calc_intersection_splits(self, dst_extents, src_extent):
        """
        Calls :meth:`calc_intersection_split` for each of the :samp:`{dst_extents}`.

        :type dst_extents: sequence of :obj:`mpi_array.indexing.HaloIndexingExtent`
        :param dst_extents: Extents which are to receive updates.
        :type src_extent: :obj:`mpi_array.indexing.HaloIndexingExtent`
        :param src_extent: Extent which is to provide the updates.
        :rtype: :obj:`tuple`
        :return: The :samp:`(leftovers, updates)` pair of :obj:`list` objects.
        """
        leftovers = []
        updates = []
        for dst_extent in dst_extents:
            dst_leftovers, dst_updates = self.calc_intersection_split(dst_extent, src_extent)
            leftovers += dst_leftovers
            updates += dst_updates
        return leftovers, updates

    def get_cpy2_src_extents(self, dst_inter_locale_rank):
        """
        Returns the extents of the source locales (in rank order) which share processes
        with the destination locale, the elements are copied directly from them.

        :type dst_inter_locale_rank: :obj:`int`
        :param dst_inter_locale_rank: The rank of the destination locale.
        :rtype: :obj:`tuple`
        :return: The source :obj:`mpi_array.distribution.LocaleExtent` objects, empty when
           no :samp:`peer_rank_translator` was given.
        """
        src_extents = ()
        if (self._dst_translated_peer_ranks is not None) and (self._src_peer_ranks is not None):
            dst_translated_peer_ranks = self._dst_translated_peer_ranks[dst_inter_locale_rank]
            if self._src_peer_ranks.ndim == 2:
                is_shared = \
                    _np.any(_np.isin(self._src_peer_ranks, dst_translated_peer_ranks), axis=1)
            else:
                is_shared = \
                    _np.array(
                        tuple(
                            _np.any(_np.isin(src_peer_ranks, dst_translated_peer_ranks))
                            for src_peer_ranks in self._src_peer_ranks
                        ),
                        dtype="bool"
                    )
            src_extents = \
                tuple(
                    self._src_distrib.get_extent_for_rank(src_inter_locale_rank)
                    for src_inter_locale_rank in _np.flatnonzero(is_shared)
                )
        return src_extents

    def get_rget_src_extents(self, dst_extent):
        """
        Returns the extents of the source locales (in rank order) which have elements
        of :samp:`{dst_extent}` (with its halo if :samp:`update_dst_halo`).

        :type dst_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param dst_extent: The extent of the destination locale.
        :rtype: :obj:`tuple`
        :return: The source :obj:`mpi_array.distribution.LocaleExtent` objects.
        """
        if self.update_dst_halo:
            start, stop = dst_extent.start_h, dst_extent.stop_h
        else:
            start, stop = dst_extent.start_n, dst_extent.stop_n
        return \
            tuple(
                self._src_distrib.get_extent_for_rank(src_inter_locale_rank)
                for src_inter_locale_rank in self._src_distrib.find_overlapping_locales(start, stop)
            )

    def initialise_locale_updates(self, dst_inter_locale_rank):
        """
        Calculates the updates of a destination locale: first those copied directly
        from the source locales which share processes with it (:meth:`get_cpy2_src_extents`),
        then those fetched from the source locales which have the remaining elements
        (:meth:`get_rget_src_extents`), in rank order.

        :type dst_inter_locale_rank: :obj:`int`
        :param dst_inter_locale_rank: The rank of the destination locale.
        :rtype: :obj:`list`
        :return: The regions of the destination extent which are not in the source distribution.
        """
        dst_extent = self._dst_distrib.get_extent_for_rank(dst_inter_locale_rank)
        leftovers = [dst_extent, ]
        for src_extent in self.get_cpy2_src_extents(dst_inter_locale_rank):
            leftovers, updates = self.calc_intersection_splits(leftovers, src_extent)
            self._dst_cpy2_updates[dst_inter_locale_rank] += updates
        if len(leftovers) > 0:
            for src_extent in self.get_rget_src_extents(dst_extent):
                leftovers, updates = self.calc_intersection_splits(leftovers, src_extent)
                self._dst_rget_updates[dst_inter_locale_rank] += updates
                if len(leftovers) <= 0:
                    break
        return leftovers

    def check_updates(self):
        """
        Runs consistency checks on the calculated updates, assumes that
        the :samp:`{dst_distrib}` and :samp:`{src_distrib}` distributed
        as a partitioning (no locale extent overlaps except for halo).
        When only the updates of one destination locale were calculated, the updates
        are checked against its extent.

        :raises RuntimeError: If update inconsistency discovered.
        """
        import itertools
        msg = ""

        all_updates = \
            tuple(self._dst_cpy2_updates.values()) + tuple(self._dst_rget_updates.values())
        all_updates = tuple(i for i in itertools.chain(*all_updates))
        total_dst_update_elems = 0
        total_src_update_elems = 0
        for i in range(len(all_updates)):
            u0 = all_updates[i]
            total_dst_update_elems += _np.prod(u0.dst_update_extent.shape)
            total_src_update_elems += _np.prod(u0.src_update_extent.shape)
            for j in range(0, i):
                u1 = all_updates[j]
                isect = u0.dst_update_extent.calc_intersection(u1.dst_update_extent)
                if isect is not None:
                    msg += \
                        (
                            "Got intersecting updates, intersection=%s, updates:\n%s\n%s\n\n"
                            %
                            (isect, u0, u1)
                        )

        if self._dst_inter_locale_rank is None:
            dst_extent = self._dst_distrib.globale_extent
        else:
            dst_extent = self._dst_distrib.get_extent_for_rank(self._dst_inter_locale_rank)
        globale_intersect = dst_extent.calc_intersection(self._src_distrib.globale_extent)
        total_intersect_elems = 0
        if globale_intersect is not None:
            total_intersect_elems = _np.prod(globale_intersect.shape)

        if total_intersect_elems != total_dst_update_elems:
            msg += \
                (
                    "total_intersect_elems=%s != total_dst_update_elems=%s\n"
                    %
                    (total_intersect_elems, total_dst_update_elems)
                )
        if total_intersect_elems != total_src_update_elems:
            msg += \
                (
                    "total_intersect_elems=%s != total_src_update_elems=%s\n"
                    %
                    (total_intersect_elems, total_src_update_elems)
                )

        if (len(msg) > 0):
            raise \
                RuntimeError(
                    "%s.check_updates failed checks:\n%s"
                    %
                    (self.__class__.__name__, msg)
                )

    def initialise(self):
        """
        Calculates the updates of the destination locales (see :meth:`initialise_locale_updates`).

        :raises ValueError: If elements of the destination locales are not in
           the source distribution.
        """
        self._dst_cpy2_updates = _collections.defaultdict(list)
        self._dst_rget_updates = _collections.defaultdict(list)
        if self._dst_inter_locale_rank is None:
            dst_inter_locale_ranks = range(self._dst_distrib.num_locales)
        else:
            dst_inter_locale_ranks = (self._dst_inter_locale_rank,)
        leftovers = []
        for dst_inter_locale_rank in dst_inter_locale_ranks:
            leftovers += self.initialise_locale_updates(dst_inter_locale_rank)
        if len(leftovers) > 0:
            raise_not_in_source_distribution(leftovers, self._src_distrib)


class MpiUpdatesForRedistribute(UpdatesForRedistribute):

    """
    Extends :obj:`UpdatesForRedistribute` by over-riding
    :meth:`~UpdatesForRedistribute.create_pair_extent_update`
    to generate :obj:`MpiPairExtentUpdate` objects (:obj:`MpiPairExtentUpdateDifferentDtypes`
    objects when the element types differ). The updates of a process
    (:meth:`create_process_updates`) have their MPI data types. The updates don't
    refer to arrays, they can be used to copy between any arrays of the distributions,
    element types and orders.
    """

    def __init__(
        self,
        dst_distrib,
        src_distrib,
        dst_dtype,
        src_dtype,
        dst_order="C",
        src_order="C",
        casting="same_kind",
        peer_rank_translator=None,
        dst_inter_locale_rank=None
    ):
        """
        Calculates the updates.

        :type dst_dtype: :obj:`numpy.dtype`
        :param dst_dtype: The element type of the array which receives the elements.
        :type src_dtype: :obj:`numpy.dtype`
        :param src_dtype: The element type of the array from which the elements are copied.
        :type dst_order: :obj:`str`
        :param dst_order: The memory layout of the array which receives the elements.
        :type src_order: :obj:`str`
        :param src_order: The memory layout of the array from which the elements are copied.
        :type casting: :obj:`str`
        :param casting: See :samp:`{casting}` parameter in :func:`numpy.copyto`.

        See :obj:`UpdatesForRedistribute` for the other parameters.
        """
        self._dst_dtype = _np.dtype(dst_dtype)
        self._src_dtype = _np.dtype(src_dtype)
        self._dst_order = dst_order
        self._src_order = src_order
        self._casting = casting
        self._pair_extent_update_type = MpiPairExtentUpdate
        if self._dst_dtype != self._src_dtype:
            self._pair_extent_update_type = MpiPairExtentUpdateDifferentDtypes
        UpdatesForRedistribute.__init__(
            self,
            dst_distrib,
            src_distrib,
            peer_rank_translator=peer_rank_translator,
            dst_inter_locale_rank=dst_inter_locale_rank
        )

    def create_pair_extent_update(
        self,
        dst_extent,
        src_extent,
        intersection_extent
    ):
        """
        Factory method for creating :obj:`MpiPairExtentUpdate` objects.

        :type dst_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param dst_extent: Destination extent.
        :type src_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param src_extent: Source extent.
        :type intersection_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param intersection_extent: The intersection of :samp:`{src_extent}`
           and :samp:`{dst_extent}` which defines the region of array elements which
           are to be transferred from source to destination.
        :rtype: :obj:`list`
        :return: The :obj:`MpiPairExtentUpdate` defining the source and destination sub-arrays
           (without MPI data types).
        """
        update = \
            self._pair_extent_update_type(
                self._dst_distrib.get_extent_for_rank(dst_extent.inter_locale_rank),
                self._src_distrib.get_extent_for_rank(src_extent.inter_locale_rank),
                intersection_extent,
                intersection_extent
            )
        update.casting = self._casting

        return [update, ]

    #: Regions are split into pieces of at least this many bytes (smaller regions aren't
    #: split): many small fetches (or copies) cost more than a few larger ones.
    min_piece_nbytes = 64 * 1024

    def deal_pieces(self, updates, num_processes, process_index):
        """
        Returns the pieces of :samp:`{updates}` of a process, when the pieces are dealt to
        :samp:`{num_processes}` processes in turn: each region is split
        (:func:`split_region`) into as many pieces of at least :attr:`min_piece_nbytes`
        bytes as it has (at most :samp:`{num_processes}`), so the small updates are dealt
        whole to different processes. All processes deal the same pieces in the same order.

        :type updates: sequence of :obj:`MpiPairExtentUpdate`
        :param updates: The updates.
        :type num_processes: :obj:`int`
        :param num_processes: The number of processes which share the updates.
        :type process_index: :obj:`int`
        :param process_index: The index of this process, :samp:`0 <= {process_index}
           < {num_processes}`.
        :rtype: :obj:`list`
        :return: The updates of the pieces of this process.
        """
        itemsize = max(self._dst_dtype.itemsize, self._src_dtype.itemsize)
        pieces = []
        num_dealt = 0
        for update in updates:
            nbytes = int(_np.prod(update.dst_update_extent.shape)) * itemsize
            num_pieces = int(max(1, min(num_processes, nbytes // self.min_piece_nbytes)))
            piece_index = (process_index - num_dealt) % num_processes
            if piece_index < num_pieces:
                if num_pieces == 1:
                    pieces.append(update)
                else:
                    region = split_region(update.dst_update_extent, num_pieces)[piece_index]
                    if region is not None:
                        pieces.append(
                            self.create_pair_extent_update(
                                update.dst_extent,
                                update.src_extent,
                                region
                            )[0]
                        )
            num_dealt += num_pieces
        return pieces

    def create_process_updates(
        self,
        intra_locale_rank,
        intra_locale_size,
        src_inter_locale_rank,
        src_peer_rank
    ):
        """
        Returns the updates of a process of the destination locale
        (:samp:`dst_inter_locale_rank`), its share of the updates of the locale
        (see :meth:`deal_pieces`):

        - the processes of the locale share the fetched regions (with MPI data types),
          they all write to the (shared memory) locale array;
        - the processes of both the destination locale and a source locale share the
          regions copied directly from that source locale (its locale array is the source
          locale array of these processes).

        :type intra_locale_rank: :obj:`int`
        :param intra_locale_rank: The rank of the process in the destination locale.
        :type intra_locale_size: :obj:`int`
        :param intra_locale_size: The number of processes of the destination locale.
        :type src_inter_locale_rank: :obj:`int`
        :param src_inter_locale_rank: The rank of the source locale of the process.
        :type src_peer_rank: :obj:`int`
        :param src_peer_rank: The rank of the process in the peer communicator of the source.
        :rtype: :obj:`ProcessUpdates`
        :return: The updates of the process.
        """
        dst_inter_locale_rank = self._dst_inter_locale_rank
        rget_updates = \
            self.deal_pieces(
                self._dst_rget_updates[dst_inter_locale_rank],
                intra_locale_size,
                intra_locale_rank
            )
        for update in rget_updates:
            update.initialise_data_types(
                dst_dtype=self._dst_dtype,
                src_dtype=self._src_dtype,
                dst_order=self._dst_order,
                src_order=self._src_order
            )

        cpy2_updates = []
        if self._dst_translated_peer_ranks is not None:
            shared_peer_ranks = \
                _np.intersect1d(
                    self._dst_translated_peer_ranks[dst_inter_locale_rank],
                    self._src_peer_ranks[src_inter_locale_rank]
                )
            index = _np.flatnonzero(shared_peer_ranks == src_peer_rank)
            if len(index) > 0:
                cpy2_updates = \
                    self.deal_pieces(
                        [
                            update
                            for update in self._dst_cpy2_updates[dst_inter_locale_rank]
                            if update.src_extent.inter_locale_rank == src_inter_locale_rank
                        ],
                        len(shared_peer_ranks),
                        int(index[0])
                    )

        return ProcessUpdates(cpy2_updates=cpy2_updates, rget_updates=rget_updates)


class UpdatesForGet(object):

    """
    Collection of update extents for fetching an arbitrary sub-extent
    from the globale array.
    """

    def __init__(
        self,
        dst_extent,
        src_distrib,
        update_dst_halo=False
    ):
        """
        """
        object.__init__(self)
        self._dst_extent = dst_extent
        self._src_distrib = src_distrib

        self._dst_extent_queue = None
        self._dst_cpy2_updates = None
        self._dst_rget_updates = None

        self._update_dst_halo = update_dst_halo

        self.initialise()

    def create_pair_extent_update(
        self,
        dst_extent,
        src_extent,
        intersection_extent
    ):
        """
        Factory method for creating :obj:`PairExtentUpdate` objects.

        :type dst_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param dst_extent: Destination extent.
        :type src_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param src_extent: Source extent.
        :type intersection_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param intersection_extent: The intersection of :samp:`{src_extent}`
           and :samp:`{dst_extent}` which defines the region of array elements which
           are to be transferred from source to destination.
        :rtype: :obj:`list`
        :return: The :obj:`PairExtentUpdate` defining the source sub-array and
           destination sub-array, in a :obj:`list`.
        """
        peu = \
            PairExtentUpdate(
                self._dst_extent,
                self._src_distrib.get_extent_for_rank(src_extent.inter_locale_rank),
                intersection_extent,
                intersection_extent
            )

        return [peu, ]

    def calc_intersection_split(self, dst_extent, src_extent):
        """
        Calculates intersection between :samp:`{dst_extent}` and :samp:`{src_extent}`.
        Any regions of :samp:`{dst_extent}` which **do not** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *left-over* :samp:`type({dst_extent})` elements.
        The regions of :samp:`{dst_extent}` which **do** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *update* :obj:`PairExtentUpdate` elements.
        Returns :obj:`tuple` pair :samp:`(leftovers, updates)`

        :type dst_extent: :obj:`mpi_array.indexing.HaloIndexingExtent`
        :param dst_extent: Extent which is to receive update from intersection
           with :samp:`{src_extent}`.
        :type src_extent: :obj:`mpi_array.indexing.HaloIndexingExtent`
        :param src_extent: Extent which is to provide update for the intersecting
           region of :samp:`{dst_extent}`.
        :rtype: :obj:`tuple`
        :return: Returns :obj:`tuple` pair of :samp:`(leftovers, updates)`.
        """
        return \
            _calc_intersection_split(
                dst_extent,
                src_extent,
                self.create_pair_extent_update,
                self._update_dst_halo
            )

    def get_cpy2_src_extents(self, dst_inter_locale_rank):
        """
        Returns the source extents whose elements are copied directly, without RMA: that of
        the source locale with the inter-locale rank of the destination locale.
        """
        src_extents = (self._src_distrib.get_extent_for_rank(dst_inter_locale_rank),)
        return src_extents

    def initialise_cpy2_updates(self):
        """
        Calculates the direct copy updates (see :meth:`get_cpy2_src_extents`), the regions
        which they don't copy are left in the queue of destination extents.
        """
        all_dst_leftovers = []
        for dst_extent_idx in range(len(self._dst_extent_queue)):
            dst_extent = self._dst_extent_queue.pop()
            dst_inter_locale_rank = dst_extent.inter_locale_rank
            src_extents = self.get_cpy2_src_extents(dst_inter_locale_rank)
            dst_extent_leftovers = [dst_extent, ]
            if (src_extents is not None) and (len(src_extents) > 0):
                for src_extent in src_extents:
                    new_dst_extent_leftovers = []
                    for dst_extent in dst_extent_leftovers:
                        dst_leftovers, dst_updates = \
                            self.calc_intersection_split(dst_extent, src_extent)
                        self._dst_cpy2_updates[dst_inter_locale_rank] += dst_updates
                        new_dst_extent_leftovers += dst_leftovers
                    dst_extent_leftovers = new_dst_extent_leftovers
            all_dst_leftovers += dst_extent_leftovers

        self._dst_extent_queue.extend(all_dst_leftovers)

    def initialise_rget_updates(self):
        """
        Calculates the updates fetched from the source locales (in rank order) which
        have the elements not copied by the direct copy updates. Only the source locales
        whose extents intersect the destination extent are considered.
        """
        if self._update_dst_halo:
            start, stop = self._dst_extent.start_h, self._dst_extent.stop_h
        else:
            start, stop = self._dst_extent.start_n, self._dst_extent.stop_n
        for src_rank in self._src_distrib.find_overlapping_locales(start, stop):
            src_extent = self._src_distrib.get_extent_for_rank(src_rank)
            all_dst_leftovers = []
            while len(self._dst_extent_queue) > 0:
                dst_extent = self._dst_extent_queue.pop()
                dst_rank = dst_extent.inter_locale_rank
                dst_leftovers, dst_updates = \
                    self.calc_intersection_split(dst_extent, src_extent)
                self._dst_rget_updates[dst_rank] += dst_updates
                all_dst_leftovers += dst_leftovers
            self._dst_extent_queue.extend(all_dst_leftovers)
            if len(self._dst_extent_queue) <= 0:
                break
        if len(self._dst_extent_queue) > 0:
            raise_not_in_source_distribution(self._dst_extent_queue, self._src_distrib)

    def initialise_updates(self):
        """
        Calculates the direct copy updates, then the fetch updates.
        """
        self.initialise_cpy2_updates()
        self.initialise_rget_updates()

    def initialise(self):
        """
        Calculates the updates of the destination extent (see :meth:`initialise_updates`).
        """
        self._dst_extent_queue = _collections.deque()
        self._dst_extent_queue.extend((self._dst_extent,))
        self._dst_cpy2_updates = _collections.defaultdict(list)
        self._dst_rget_updates = _collections.defaultdict(list)

        self.initialise_updates()


class MpiUpdatesForGet(UpdatesForGet):

    """
    Extends :obj:`UpdatesForGet` by over-riding
    :meth:`~UpdatesForGet.create_pair_extent_update`
    to generate :obj:`MpiPairExtentUpdate` objects.
    """

    def __init__(
        self,
        dst_extent,
        src_distrib,
        dtype,
        order,
        update_dst_halo=False,
    ):
        """
        """
        self.dtype = _np.dtype(dtype)
        self.order = order
        UpdatesForGet.__init__(
            self,
            dst_extent=dst_extent,
            src_distrib=src_distrib,
            update_dst_halo=update_dst_halo
        )

    def create_pair_extent_update(
        self,
        dst_extent,
        src_extent,
        intersection_extent
    ):
        """
        Factory method for creating :obj:`MpiPairExtentUpdate` objects.

        :type dst_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param dst_extent: Destination extent.
        :type src_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param src_extent: Source extent.
        :type intersection_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param intersection_extent: The intersection of :samp:`{src_extent}`
           and :samp:`{dst_extent}` which defines the region of array elements which
           are to be transferred from source to destination.
        :rtype: :obj:`list`
        :return: The :obj:`MpiPairExtentUpdate` defining the source sub-array and
           destination sub-array, in a :obj:`list`.
        """
        peu = \
            MpiPairExtentUpdate(
                self._dst_extent,
                self._src_distrib.get_extent_for_rank(src_extent.inter_locale_rank),
                intersection_extent,
                intersection_extent
            )

        peu_list = [peu, ]

        for peu in peu_list:
            peu.initialise_data_types(
                dst_dtype=self.dtype,
                src_dtype=self.dtype,
                dst_order=self.order,
                src_order=self.order
            )

        return peu_list


class RmaUpdateExecutor(object):
    """
    Performs one-sided fetch of data from remote (source) locale arrays to
    update destination locale array.
    """

    def __init__(
        self,
        inter_win,
        dst_lndarray,
        src_inter_win_rank_attr,
        rank_logger=None,
        casting="same_kind",
        win_rank=0
    ):
        """
        Initialise.

        :type inter_win: :obj:`mpi4py.MPI.Win`
        :param inter_win: The window of the source arrays, :obj:`mpi4py.MPI.WIN_NULL`
           (or :samp:`None`) on processes which don't fetch.
        :type dst_lndarray: :obj:`numpy.ndarray`
        :param dst_lndarray: The destination array.
        :type src_inter_win_rank_attr: :obj:`str`
        :param src_inter_win_rank_attr: The attribute of the source extents which is
           the target rank in :samp:`{inter_win}`.
        :type rank_logger: :obj:`logging.Logger`
        :param rank_logger: Logger for messages.
        :type casting: :obj:`str`
        :param casting: See :samp:`{casting}` parameter in :func:`numpy.copyto`.
        :type win_rank: :obj:`int`
        :param win_rank: The rank of this process in the group of :samp:`{inter_win}`,
           see :meth:`order_updates`.
        """
        object.__init__(self)
        self._dst_lndarray = dst_lndarray
        self._casting = casting
        self._inter_win = inter_win
        self._win_rank = win_rank
        self._rank_logger = rank_logger
        self._src_inter_win_rank_attr = src_inter_win_rank_attr

    def get_src_win_rank(self, src_extent):
        """
        Returns target rank integer (:attr:`inter_win`) for specified :samp:`{src_extent}` extent.

        :type src_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param src_extent: Return target rank for this extent.
        :rtype: :obj:`int`
        :return: Target rank for window :attr:`inter_win`.
        """
        return getattr(src_extent, self._src_inter_win_rank_attr)

    @property
    def inter_win(self):
        """
        The :obj:`mpi4py.MPI.Win` instance used for remote fetch of data.
        """
        return self._inter_win

    @property
    def dst_lndarray(self):
        """
        The destination :obj:`numpy.ndarray` for remote fetches.
        """
        return self._dst_lndarray

    @property
    def rank_logger(self):
        """
        The :obj:`logging.Logger` used for log messages.
        """
        return self._rank_logger

    @rank_logger.setter
    def rank_logger(self, logger):
        """
        """
        self._rank_logger = logger

    def order_updates(self, updates):
        """
        Returns the updates of :samp:`{updates}` grouped by target rank (of :attr:`inter_win`),
        the targets in a random order which depends on the rank of this process
        (the :samp:`win_rank` argument of :obj:`RmaUpdateExecutor`), so that the processes don't
        all fetch from the same targets at the same time.

        :type updates: sequence of :obj:`MpiPairExtentUpdate`
        :param updates: The updates.
        :rtype: :obj:`list`
        :return: The :samp:`(target_rank, target_updates)` pairs, :samp:`target_updates` is
           the :obj:`list` of the updates from the target.
        """
        update_dict = _collections.defaultdict(list)
        for single_update in updates:
            update_dict[int(self.get_src_win_rank(single_update.src_extent))].append(
                single_update
            )
        target_ranks = sorted(update_dict.keys())
        order = _np.random.default_rng(self._win_rank).permutation(len(target_ranks))

        return [(target_ranks[i], update_dict[target_ranks[i]]) for i in order]

    def do_direct_cpy2_update(self, updates, src_lndarray):
        """
        Does direct copy update to :attr:`dst_lndarray` from the
        specified :samp:`{src_lndarray}` array.

        :type updates: sequence of :obj:`PairExtentUpdate`
        :param updates: Sequence of destination and source extents.
        :type src_lndarray: :obj:`numpy.ndarray`
        :param src_lndarray: Elements copied from this array.
        """
        for single_update in updates:
            single_update.copyto(self._dst_lndarray, src_lndarray, casting=self._casting)

    def do_locale_rma_update(self, updates):
        """
        Performs RMA to get elements from remote (source) locales to
        update the (destination) locale extent array. Each target (in the order of
        :meth:`order_updates`) has its own passive target epoch (shared lock), its
        elements are fetched with :meth:`mpi4py.MPI.Win.Get` and the unlock waits for them.
        One epoch for all the targets (:meth:`mpi4py.MPI.Win.Lock_all`, with fetches from
        several targets in flight) intermittently crashed Open MPI 5.0 (one-sided
        communication over the shared memory transport) in stress tests.

        :type updates: sequence of :obj:`MpiPairExtentUpdate`
        :param updates: Sequence of destination and source extents.
        """
        if (
            (self._inter_win is not None)
            and
            (self._inter_win != _mpi.WIN_NULL)
            and
            (len(updates) > 0)
        ):
            for target_rank, target_updates in self.order_updates(updates):
                self._inter_win.Lock(target_rank, _mpi.LOCK_SHARED)
                try:
                    for single_update in target_updates:
                        self.rank_logger.debug(
                            "Getting update from target rank %s:\n%s\n%s",
                            target_rank,
                            single_update._header_str,
                            single_update
                        )
                        single_update.do_get(self._inter_win, target_rank, self._dst_lndarray)
                finally:
                    self._inter_win.Unlock(target_rank)
                for single_update in target_updates:
                    single_update.conclude()


__all__ = [s for s in dir() if not s.startswith('_')]
