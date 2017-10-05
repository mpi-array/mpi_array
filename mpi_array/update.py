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
   MpiPairExtentUpdate - Extends :obj:`PairExtentUpdate` with MPI data type factory.
   MpiPairExtentUpdateDifferentDtypes - Over-rides :meth:`MpiPairExtentUpdate.do_get`.
   HaloSingleExtentUpdate - Describes sub-extent for halo region update.
   MpiHaloSingleExtentUpdate - Extends :obj:`HaloSingleExtentUpdate` with MPI data type factory.
   UpdatesForRedistribute - Calculate sequence of overlapping extents between two distributions.

"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi
import collections as _collections
import copy as _copy
import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from .indexing import HaloIndexingExtent
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
        self._locale_extent = locale_extent
        self._region_extent = region_extent

    @property
    def locale_extent(self):
        return self._locale_extent

    @property
    def region_extent(self):
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
        :type dst_extent_info: :obj:`ExtentAndRegion`
        :param dst_extent_info: Info containing locale extent from which the region update is read.
        """
        object.__init__(self)
        self._dst = dst_extent_info
        self._src = src_extent_info

    @property
    def dst_extent(self):
        """
        The locale :obj:`LocaleExtent` which is to receive sub-array update.
        """
        return self._dst.locale_extent

    @property
    def src_extent(self):
        """
        The locale :obj:`CartLocaleExtent` from which the sub-array update is read.
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
    :param casting: Indicates casting regime, see :func:`numpy.casting`.
    """
    src_slice = peu.src_extent.globale_to_locale_extent_h(peu.src_update_extent).to_slice()
    dst_slice = peu.dst_extent.globale_to_locale_extent_h(peu.dst_update_extent).to_slice()
    _np.copyto(dst_array[dst_slice], src_array[src_slice], casting=casting)


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
        The locale sub-extent (:obj:`IndexingExtent`) to be updated.
        """
        return self._dst.region_extent

    @property
    def src_update_extent(self):
        """
        The locale sub-extent (:obj:`IndexingExtent`) from which the update is read.
        """
        return self._src.region_extent


class MpiExtentAndRegion(ExtentAndRegion):

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
        self._mpi_order = mpi_order

    def create_data_type(self, dtype, order="C"):
        mpi_order = _mpi.ORDER_C
        if order == "F":
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
        dtype = _np.dtype(dtype)
        order = order.lower()
        if (
            (self._dtype is None)
            or
            (self._dtype != dtype)
            or
            (self._order != order)
        ):
            self._mpi_data_type, self._mpi_order, self._parent_mpi_data_type = \
                self.create_data_type(dtype, order)
            self._dtype = dtype
            self._order = order

    @property
    def mpi_data_type(self):
        return self._mpi_data_type


class MpiPairExtentUpdate(ExtentUpdate):

    """
    Source and destination indexing info for updating the whole of a halo portion.
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
        The locale sub-extent (:obj:`IndexingExtent`) to be updated.
        """
        return self._dst.region_extent

    @property
    def src_update_extent(self):
        """
        The locale sub-extent (:obj:`IndexingExtent`) from which the update is read.
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
        defines the sub-array of halo elements which are to
        receive update values.
        """
        return self._dst.mpi_data_type

    @property
    def src_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array of halo elements from which
        receive update values.
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
           should correspond to the size of the :attr:`dst_extent`.
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
           should correspond to the size of the :attr:`dst_extent`.
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
                    self.dst_extent.cart_rank,
                    self.dst_extent.start_h,
                    self.dst_extent.stop_h,
                    self.dst_extent.globale_to_locale_h(self.dst_update_extent.start),
                    self.dst_extent.globale_to_locale_h(self.dst_update_extent.stop),
                    self.dst_update_extent.start,
                    self.dst_update_extent.stop,
                    dst_mpi_dtype,
                    self.src_extent.cart_rank,
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
        the :attr:`src_dtype` to the :attr:`dst_dtype`.

        :type mpi_win: :obj:`mpi4py.MPI.Win.Get`
        :param mpi_win: Window used to retrieve update region for array.
        :type target_src_rank: :obj:`int`
        :param target_src_rank: The rank of the target process in :samp:`mpi_win.group.rank`.
        :type origin_dst_buffer: :obj:`memoryview`
        :param origin_dst_buffer: The destination memory for the update, size of buffer
           should correspond to the size of the :attr:`dst_extent`.
        """
        self._buffer = _np.empty(shape=self._src.region_extent.shape, dtype=self._src._dtype)
        self._dst_buffer = origin_dst_buffer

        mpi_win.Get(
            [self._buffer, _np.product(self._buffer.shape),
             self._src._parent_mpi_data_type],
            target_src_rank,
            [0, 1, self.src_data_type]
        )

    def do_rget(self, mpi_win, target_src_rank, origin_dst_buffer):
        """
        """
        self._buffer = _np.empty(shape=self._src.region_extent.shape, dtype=self._src._dtype)
        self._dst_buffer = origin_dst_buffer

        r = mpi_win.Rget(
            [self._buffer, _np.product(self._buffer.shape),
             self._src._parent_mpi_data_type],
            target_src_rank,
            [0, 1, self.src_data_type]
        )

        return r

    def conclude(self):
        """
        """
        origin_dst_buffer_slice = \
            self._dst.locale_extent.globale_to_locale_extent_h(self._dst.region_extent).to_slice()
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
        The :obj:`IndexingExtent` indicating the halo sub-array which is to be updated.
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
        defines the sub-array of halo elements from which
        receive update values.
        """
        return self._src.mpi_data_type

    @property
    def update_extent(self):
        """
        The :obj:`IndexingExtent` indicating the halo sub-array which is to be updated.
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

    def __init__(self, dst_rank, rank_to_extents_map):
        """
        Construct.

        :type dst_rank: :obj:`int`
        :param dst_rank: The MPI rank (:samp:`cart_comm`) of the MPI
           process which is to receive the halo updates.
        :type rank_to_extents_map: :obj:`dict`
        :param rank_to_extents_map: Dictionary of :samp:`(r, extent)`
           pairs for all ranks :samp:`r` (of :samp:`cart_comm`), where :samp:`extent`
           is a :obj:`CartLocaleExtent` object indicating the indexing extent
           (tile) on MPI rank :samp:`r.`
        """
        self.initialise(dst_rank, rank_to_extents_map)

    def create_single_extent_update(self, dst_extent, src_extent, halo_extent):
        """
        Factory method for creating instances of type :obj:`HaloSingleExtentUpdate`.

        :type dst_extent: :obj:`IndexingExtent`
        :param dst_extent: The destination locale extent for halo element update.
        :type src_extent: :obj:`IndexingExtent`
        :param src_extent: The source locale extent for obtaining halo element update.
        :type halo_extent: :obj:`IndexingExtent`
        :param halo_extent: The extent indicating the sub-array of halo elements.

        :rtype: :obj:`HaloSingleExtentUpdate`
        :return: Returns new instance of :obj:`HaloSingleExtentUpdate`.
        """
        return HaloSingleExtentUpdate(dst_extent, src_extent, halo_extent)

    def calc_halo_intersection(self, dst_extent, src_extent, axis, dir):
        """
        Calculates the intersection of :samp:`{dst_extent}` halo slab with
        the update region of :samp:`{src_extent}`.

        :type dst_extent: :obj:`CartLocaleExtent`
        :param dst_extent: Halo slab indicated by :samp:`{axis}` and :samp:`{dir}`
           taken from this extent.
        :type src_extent: :obj:`CartLocaleExtent`
        :param src_extent: This extent, minus the halo in the :samp:`{axis}` dimension,
           is intersected with the halo slab.
        :type axis: :obj:`int`
        :param axis: Axis dimension indicating slab.
        :type dir: :attr:`LO` or :attr:`HI`
        :param dir: :attr:`LO` for low-index slab or :attr:`HI` for high-index slab.
        :rtype: :obj:`IndexingExtent`
        :return: Overlap extent of :samp:{dst_extent} halo-slab and
           the :samp:`{src_extent}` update region.
        """
        return \
            dst_extent.halo_slab_extent(axis, dir).calc_intersection(
                src_extent.no_halo_extent(axis)
            )

    def split_extent_for_max_elements(self, extent, max_elements=None):
        """
        Partitions the specified extent into smaller extents with number
        of elements no more than :samp:`{max_elements}`.

        :type extent: :obj:`CartLocaleExtent`
        :param extent: The extent to be split.
        :type max_elements: :obj:`int`
        :param max_elements: Each partition of the returned split has no more
           than this many elements.
        :rtype: :obj:`list` of :obj:`CartLocaleExtent`
        :return: List of extents forming a partition of :samp:`{extent}`
           with each extent having no more than :samp:`{max_element}` elements.
        """
        return [extent, ]

    def initialise(self, dst_rank, rank_to_extents_map):
        """
        Calculates the ranks and regions required to update the
        halo regions of the :samp:`dst_rank` MPI rank.

        :type dst_rank: :obj:`int`
        :param dst_rank: The MPI rank (:samp:`cart_comm`) of the MPI
           process which is to receive the halo updates.
        :type rank_to_extents_map: :obj:`dict`
        :param rank_to_extents_map: Dictionary of :samp:`(r, extent)`
           pairs for all ranks :samp:`r` (of :samp:`cart_comm`), where :samp:`extent`
           is a :obj:`CartLocaleExtent` object indicating the indexing extent
           (tile) on MPI rank :samp:`r.`
        """
        self._dst_rank = dst_rank
        self._dst_extent = rank_to_extents_map[dst_rank]
        self._updates = [[[], []]] * self._dst_extent.ndim
        if hasattr(rank_to_extents_map, "keys"):
            ranks = rank_to_extents_map.keys()
        else:
            ranks = range(0, len(rank_to_extents_map))
        cart_coord_to_extents_dict = \
            {
                tuple(rank_to_extents_map[r].cart_coord): rank_to_extents_map[r]
                for r in ranks
            }
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
                    src_extent = cart_coord_to_extents_dict[tuple(src_cart_coord)]
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
        A :attr:`ndim` length list of pair elements, each element of the pair
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

        :type dst_extent: :obj:`IndexingExtent`
        :param dst_extent: The destination locale extent for halo element update.
        :type src_extent: :obj:`IndexingExtent`
        :param src_extent: The source locale extent for obtaining halo element update.
        :type halo_extent: :obj:`IndexingExtent`
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
        peer_rank_translator=None
    ):
        """
        """
        object.__init__(self)

        self._dst_distrib = dst_distrib
        self._src_distrib = src_distrib

        self._dst_extent_queue = None
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
        :param src_extent: The intersection of :samp:`{src_extent}`
           and :samp:`{dst_extent}` which defines the region of array elements which
           are to be transferred from source to destination.
        :rtype: :obj:`PairExtentUpdate`
        :return: Object Defining the source sub-array and destination sub-array.
        """
        peu = \
            PairExtentUpdate(
                self._dst_distrib.locale_extents[dst_extent.inter_locale_rank],
                self._src_distrib.locale_extents[src_extent.inter_locale_rank],
                intersection_extent,
                intersection_extent
            )

        return [peu, ]

    def calc_intersection_split(self, dst_extent, src_extent):
        """
        Calculates intersection between :samp:`{dst_extent}` and `{src_extent}`.
        Any regions of :samp:`{dst_extent}` which **do not** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *left-over* :samp:`type({dst_extent})` elements.
        The regions of :samp:`{dst_extent}` which **do** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *update* :obj:`PairExtentUpdate` elements.
        Returns :obj:`tuple` pair :samp:`(leftovers, updates)`

        :type dst_extent: :obj:`HaloIndexingExtent`
        :param dst_extent: Extent which is to receive update from intersection
           with :samp:`{src_extent}`.
        :type src_extent: :obj:`HaloIndexingExtent`
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

    def get_cpy2_src_extents(self, dst_inter_locale_rank):
        """
        """
        dst_translated_peer_ranks = self._dst_translated_peer_ranks[dst_inter_locale_rank]
        src_locale_extents = self._src_distrib.locale_extents
        src_extent_indices = \
            tuple(
                src_inter_locale_rank
                for src_inter_locale_rank in range(0, len(src_locale_extents))
                if _np.intersect1d(
                    dst_translated_peer_ranks,
                    self._src_peer_ranks[src_inter_locale_rank]
                ).size > 0
            )
        src_extents = tuple(src_locale_extents[e] for e in src_extent_indices)
        return src_extents

    def initialise_cpy2_updates(self):
        """
        """
        if (self._dst_translated_peer_ranks is not None) and (self._src_peer_ranks is not None):
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
        """
        for src_rank in range(len(self._src_distrib.locale_extents)):
            src_extent = self._src_distrib.locale_extents[src_rank]
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
            self._dst_cad.rank_logger.warning(
                "Non-empty leftover queue=%s",
                self._dst_extent_queue
            )

    def check_updates(self):
        """
        Runs consistency checks on the calculated updates, assumes that
        the :attr:`dst_distrib` and :attr:`src_distrib` distributed
        as a partitioning (no locale extent overlaps except for halo).

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
            total_dst_update_elems += _np.product(u0.dst_update_extent.shape)
            total_src_update_elems += _np.product(u0.src_update_extent.shape)
            for j in range(0, i):
                u1 = all_updates[j]
                isect = u0.dst_update_extent.calc_intersection(u1.dst_update_extent)
                if isect is not None:
                    msg += \
                        (
                            "Got intersecting updates, intersection=%s, updates:\n%s\n%s\n\n"
                            (isect, u0, u1)
                        )

        globale_intersect = \
            self._dst_distrib.globale_extent.calc_intersection(self._src_distrib.globale_extent)
        total_intersect_elems = _np.product(globale_intersect.shape)

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

    def initialise_updates(self):
        """
        """
        self.initialise_cpy2_updates()
        self.initialise_rget_updates()

    def initialise(self):
        """
        """
        self._dst_extent_queue = _collections.deque()
        self._dst_extent_queue.extend(self._dst_distrib.locale_extents)
        self._dst_cpy2_updates = _collections.defaultdict(list)
        self._dst_rget_updates = _collections.defaultdict(list)
        self.initialise_updates()


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
        :param src_extent: The intersection of :samp:`{src_extent}`
           and :samp:`{dst_extent}` which defines the region of array elements which
           are to be transferred from source to destination.
        :rtype: :obj:`PairExtentUpdate`
        :return: Object Defining the source sub-array and destination sub-array.
        """
        peu = \
            PairExtentUpdate(
                self._dst_extent,
                self._src_distrib.locale_extents[src_extent.inter_locale_rank],
                intersection_extent,
                intersection_extent
            )

        return [peu, ]

    def calc_intersection_split(self, dst_extent, src_extent):
        """
        Calculates intersection between :samp:`{dst_extent}` and `{src_extent}`.
        Any regions of :samp:`{dst_extent}` which **do not** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *left-over* :samp:`type({dst_extent})` elements.
        The regions of :samp:`{dst_extent}` which **do** intersect with :samp:`{src_extent}`
        are returned as a :obj:`list` of *update* :obj:`PairExtentUpdate` elements.
        Returns :obj:`tuple` pair :samp:`(leftovers, updates)`

        :type dst_extent: :obj:`HaloIndexingExtent`
        :param dst_extent: Extent which is to receive update from intersection
           with :samp:`{src_extent}`.
        :type src_extent: :obj:`HaloIndexingExtent`
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
        """
        src_extents = (self._src_distrib.locale_extents[dst_inter_locale_rank],)
        return src_extents

    def initialise_cpy2_updates(self):
        """
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
        """
        for src_rank in range(len(self._src_distrib.locale_extents)):
            src_extent = self._src_distrib.locale_extents[src_rank]
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
            self._dst_cad.rank_logger.warning(
                "Non-empty leftover queue=%s",
                self._dst_extent_queue
            )

    def initialise_updates(self):
        """
        """
        self.initialise_cpy2_updates()
        self.initialise_rget_updates()

    def initialise(self):
        self._dst_extent_queue = _collections.deque()
        self._dst_extent_queue.extend((self._dst_extent,))
        self._dst_cpy2_updates = _collections.defaultdict(list)
        self._dst_rget_updates = _collections.defaultdict(list)

        self.initialise_updates()


class MpiUpdatesForGet(UpdatesForGet):

    """
    Extends :obj:`UpdatesForGet` by over-riding :meth:`create_pair_extent_update`
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
        Factory method for creating :obj:`PairExtentUpdate` objects.

        :type dst_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param dst_extent: Destination extent.
        :type src_extent: :obj:`mpi_array.distribution.LocaleExtent`
        :param src_extent: Source extent.
        :type intersection_extent: :obj:`mpi_array.indexing.IndexingExtent`
        :param src_extent: The intersection of :samp:`{src_extent}`
           and :samp:`{dst_extent}` which defines the region of array elements which
           are to be transferred from source to destination.
        :rtype: :obj:`PairExtentUpdate`
        :return: Object Defining the source sub-array and destination sub-array.
        """
        peu = \
            MpiPairExtentUpdate(
                self._dst_extent,
                self._src_distrib.locale_extents[src_extent.inter_locale_rank],
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


__all__ = [s for s in dir() if not s.startswith('_')]
