"""
==================================
The :mod:`mpi_array.update` Module
==================================

Helper classes for defining sub-extents to peform RMA array element updates.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   ExtentUpdate - Base class for describing a sub-extent update.
   PairExtentUpdate - Describes sub-extent source and sub-extent destination.
   MpiPairExtentUpdate - Extends :obj:`PairExtentUpdate` with MPI data type factory.
   HaloSingleExtentUpdate - Describes sub-extent for halo region update.
   MpiHaloSingleExtentUpdate - Extends :obj:`HaloSingleExtentUpdate` with MPI data type factory.
   UpdatesForRedistribute - Calculate sequence of overlapping extents between two distributions.

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright

import pkg_resources as _pkg_resources

import mpi4py.MPI as _mpi

from mpi_array.indexing import HaloIndexingExtent
from mpi_array.indexing import calc_intersection_split as _calc_intersection_split

import collections as _collections

import numpy as _np

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


class ExtentUpdate(object):

    """
    Source and destination indexing info for updating a sub-extent region.
    """

    def __init__(self, dst_extent, src_extent):
        """
        Initialise.

        :type dst_extent: :obj:`CartLocaleExtent`
        :param dst_extent: Whole locale extent which receives update.
        :type src_extent: :obj:`CartLocaleExtent`
        :param src_extent: Whole locale extent from which update is read.
        """
        object.__init__(self)
        self._dst_extent = dst_extent  #: whole tile of destination locale
        self._src_extent = src_extent  #: whole tile of source locale

    @property
    def dst_extent(self):
        """
        The locale :obj:`CartLocaleExtent` which is to receive sub-array update.
        """
        return self._dst_extent

    @property
    def src_extent(self):
        """
        The locale :obj:`CartLocaleExtent` from which the sub-array update is read.
        """
        return self._src_extent


class PairExtentUpdate(ExtentUpdate):

    """
    Source and destination indexing info for updating a sub-extent region.
    """

    def __init__(self, dst_extent, src_extent, dst_update_extent, src_update_extent):
        ExtentUpdate.__init__(self, dst_extent, src_extent)
        self._dst_update_extent = dst_update_extent  #: sub-extent of self.dst_extent
        self._src_update_extent = src_update_extent  #: sub-extent of self.src_extent

    @property
    def dst_update_extent(self):
        """
        The locale sub-extent (:obj:`IndexingExtent`) to be updated.
        """
        return self._dst_update_extent

    @property
    def src_update_extent(self):
        """
        The locale sub-extent (:obj:`IndexingExtent`) from which the update is read.
        """
        return self._src_update_extent


class MpiPairExtentUpdate(PairExtentUpdate):

    """
    Source and destination indexing info for updating the whole of a halo portion.
    Extends :obj:`HaloSingleExtentUpdate` with API to create :obj:`mpi4py.MPI.Datatype`
    instances (using :meth:`mpi4py.MPI.Datatype.Create_subarray`) for convenient
    transfer of sub-array data.
    """

    def __init__(self, dst_extent, src_extent, dst_update_extent, src_update_extent):
        PairExtentUpdate.__init__(
            self,
            dst_extent=dst_extent,
            src_extent=src_extent,
            dst_update_extent=dst_update_extent,
            src_update_extent=src_update_extent
        )
        self._dst_order = None
        self._src_order = None
        self._dst_dtype = None
        self._src_dtype = None
        self._dst_data_type = None
        self._src_data_type = None
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
        dst_dtype = _np.dtype(dst_dtype)
        dst_order = dst_order.upper()
        src_dtype = _np.dtype(src_dtype)
        src_order = src_order.upper()
        if (
            (self._dst_dtype is None)
            or
            (self._src_dtype is None)
            or
            (self._dst_dtype != dst_dtype)
            or
            (self._src_dtype != dst_dtype)
            or
            (self._dst_order != dst_order)
            or
            (self._src_order != src_order)
        ):
            self._dst_data_type, self._src_data_type = \
                self.create_data_types(dst_dtype, src_dtype, dst_order, src_order)
            self._dst_dtype = dst_dtype
            self._src_dtype = src_dtype
            self._dst_order = dst_order
            self._src_order = src_order

    def create_data_types(self, dst_dtype, src_dtype, dst_order, src_order):
        """
        Returns pair of new `mpi4py.MPI.Datatype`
        instances :samp:`(dst_data_type, src_data_type)`.

        :type dtype: :obj:`numpy.dtype`
        :param dtype: The array element type.
        :type order: :obj:`str`
        :param order: Array memory layout, :samp:`"C"` for C array,
           or :samp:`"F"` for fortran array.
        :rtype: :obj:`tuple`
        :return: Pair of new `mpi4py.MPI.Datatype`
           instances :samp:`(dst_data_type, src_data_type)`.
        """
        dst_mpi_order = _mpi.ORDER_C
        if dst_order == "F":
            dst_mpi_order = _mpi.ORDER_FORTRAN

        dst_data_type = \
            _mpi._typedict[dst_dtype.char].Create_subarray(
                self._dst_extent.shape_h,
                self._dst_update_extent.shape,
                self._dst_extent.globale_to_locale_h(self._dst_update_extent.start),
                order=dst_mpi_order
            )
        dst_data_type.Commit()

        src_mpi_order = _mpi.ORDER_C
        if src_order == "F":
            src_mpi_order = _mpi.ORDER_FORTRAN

        src_data_type = \
            _mpi._typedict[src_dtype.char].Create_subarray(
                self._src_extent.shape_h,
                self._src_update_extent.shape,
                self._src_extent.globale_to_locale_h(self._src_update_extent.start),
                order=src_mpi_order
            )
        src_data_type.Commit()

        return dst_data_type, src_data_type

    @property
    def dst_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array of halo elements which are to
        receive update values.
        """
        return self._dst_data_type

    @property
    def src_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array of halo elements from which
        receive update values.
        """
        return self._src_data_type

    def __str__(self):
        """
        Stringify.
        """
        dst_mpi_dtype = None
        if self._dst_dtype is not None:
            dst_mpi_dtype = _mpi._typedict[self._dst_dtype.char].Get_name()
        src_mpi_dtype = None
        if self._src_dtype is not None:
            src_mpi_dtype = _mpi._typedict[self._src_dtype.char].Get_name()

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


class HaloSingleExtentUpdate(ExtentUpdate):

    """
    Source and destination indexing info for updating a halo portion.
    """

    def __init__(self, dst_extent, src_extent, update_extent):
        ExtentUpdate.__init__(self, dst_extent, src_extent)
        self._update_extent = update_extent  #: portion from source required for update

    @property
    def update_extent(self):
        """
        The :obj:`IndexingExtent` indicating the halo sub-array which is to be updated.
        """
        return self._update_extent


class MpiHaloSingleExtentUpdate(HaloSingleExtentUpdate):

    """
    Source and destination indexing info for updating the whole of a halo portion.
    Extends :obj:`HaloSingleExtentUpdate` with API to create :obj:`mpi4py.MPI.Datatype`
    instances (using :meth:`mpi4py.MPI.Datatype.Create_subarray`) for convenient
    transfer of sub-array data.
    """

    def __init__(self, dst_extent, src_extent, update_extent):
        HaloSingleExtentUpdate.__init__(self, dst_extent, src_extent, update_extent)
        self._order = None
        self._dtype = None
        self._dst_data_type = None
        self._src_data_type = None
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
        dtype = _np.dtype(dtype)
        order = order.upper()
        if (self._dtype is None) or (self._dtype != dtype) or (self._order != order):
            self._dst_data_type, self._src_data_type = \
                self.create_data_types(dtype, order)
            self._dtype = dtype
            self._order = order

    def create_data_types(self, dtype, order):
        """
        Returns pair of new `mpi4py.MPI.Datatype`
        instances :samp:`(dst_data_type, src_data_type)`.

        :type dtype: :obj:`numpy.dtype`
        :param dtype: The array element type.
        :type order: :obj:`str`
        :param order: Array memory layout, :samp:`"C"` for C array,
           or :samp:`"F"` for fortran array.
        :rtype: :obj:`tuple`
        :return: Pair of new `mpi4py.MPI.Datatype`
           instances :samp:`(dst_data_type, src_data_type)`.
        """
        mpi_order = _mpi.ORDER_C
        if order == "F":
            mpi_order = _mpi.ORDER_FORTRAN

        dst_data_type = \
            _mpi._typedict[dtype.char].Create_subarray(
                self._dst_extent.shape_h,
                self._update_extent.shape,
                self._dst_extent.globale_to_locale_h(self._update_extent.start),
                order=mpi_order
            )
        dst_data_type.Commit()

        src_data_type = \
            _mpi._typedict[dtype.char].Create_subarray(
                self._src_extent.shape_h,
                self._update_extent.shape,
                self._src_extent.globale_to_locale_h(self._update_extent.start),
                order=mpi_order
            )
        src_data_type.Commit()

        return dst_data_type, src_data_type

    @property
    def dst_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array of halo elements which are to
        receive update values.
        """
        return self._dst_data_type

    @property
    def src_data_type(self):
        """
        A :obj:`mpi4py.MPI.Datatype` object created
        using :meth:`mpi4py.MPI.Datatype.Create_subarray` which
        defines the sub-array of halo elements from which
        receive update values.
        """
        return self._src_data_type

    def __str__(self):
        """
        Stringify.
        """
        mpi_dtype = None
        if self._dtype is not None:
            mpi_dtype = _mpi._typedict[self._dtype.char].Get_name()
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

    def __init__(self, dst_rank, rank_to_extents_dict):
        """
        Construct.

        :type dst_rank: :obj:`int`
        :param dst_rank: The MPI rank (:samp:`cart_comm`) of the MPI
           process which is to receive the halo updates.
        :type rank_to_extents_dict: :obj:`dict`
        :param rank_to_extents_dict: Dictionary of :samp:`(r, extent)`
           pairs for all ranks :samp:`r` (of :samp:`cart_comm`), where :samp:`extent`
           is a :obj:`CartLocaleExtent` object indicating the indexing extent
           (tile) on MPI rank :samp:`r.`
        """
        self.initialise(dst_rank, rank_to_extents_dict)

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

    def initialise(self, dst_rank, rank_to_extents_dict):
        """
        Calculates the ranks and regions required to update the
        halo regions of the :samp:`dst_rank` MPI rank.

        :type dst_rank: :obj:`int`
        :param dst_rank: The MPI rank (:samp:`cart_comm`) of the MPI
           process which is to receive the halo updates.
        :type rank_to_extents_dict: :obj:`dict`
        :param rank_to_extents_dict: Dictionary of :samp:`(r, extent)`
           pairs for all ranks :samp:`r` (of :samp:`cart_comm`), where :samp:`extent`
           is a :obj:`CartLocaleExtent` object indicating the indexing extent
           (tile) on MPI rank :samp:`r.`
        """
        self._dst_rank = dst_rank
        self._dst_extent = rank_to_extents_dict[dst_rank]
        self._updates = [[[], []]] * self._dst_extent.ndim
        cart_coord_to_extents_dict = \
            {
                tuple(rank_to_extents_dict[r].cart_coord): rank_to_extents_dict[r]
                for r in rank_to_extents_dict.keys()
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
    elements from one decomposition to another.
    """

    def __init__(self, dst_decomp, src_decomp):
        """
        """
        self._dst_decomp = dst_decomp
        self._src_decomp = src_decomp
        self._inter_comm = None
        self._inter_win = None

        self._updates_dict = None
        self.update_dst_halo = False

        self.initialise()

    def calc_can_use_existing_inter_locale_comm(self):
        can_use_existing_inter_locale_comm = \
            (self._dst_decomp.inter_locale_comm is not None)
        if self._dst_decomp.have_valid_inter_locale_comm:
            if self._src_decomp.have_valid_inter_locale_comm:
                can_use_existing_inter_locale_comm = \
                    (
                        (
                            _mpi.Group.Union(
                                self._dst_decomp.inter_locale_comm.group,
                                self._src_decomp.inter_locale_comm.group
                            ).size
                            ==
                            self._dst_decomp.inter_locale_comm.group.size
                        )
                        and
                        (
                            _mpi.Group.Intersection(
                                self._dst_decomp.inter_locale_comm.group,
                                self._src_decomp.inter_locale_comm.group
                            ).size
                            ==
                            self._dst_decomp.inter_locale_comm.group.size
                        )
                    )
            else:
                can_use_existing_inter_locale_comm = False
        self.rank_logger.debug("BEG: self._dst_decomp.intra_locale_comm.allreduce...")
        can_use_existing_inter_locale_comm = \
            self._dst_decomp.intra_locale_comm.allreduce(
                can_use_existing_inter_locale_comm,
                _mpi.BAND
            )
        self.rank_logger.debug("END: self._dst_decomp.intra_locale_comm.allreduce.")
        self.rank_logger.debug(
            "can_use_existing_inter_locale_comm = %s",
            can_use_existing_inter_locale_comm
        )
        return can_use_existing_inter_locale_comm

    def create_pair_extent_update(
        self,
        dst_extent,
        src_extent,
        intersection_extent
    ):
        peu = \
            MpiPairExtentUpdate(
                self._dst_decomp.get_extent_for_rank(dst_extent.inter_locale_rank),
                self._src_decomp.get_extent_for_rank(src_extent.inter_locale_rank),
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
        are returned as a :obj:`list` of *update* :obj:`MpiPairExtentUpdate` elements.
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

    def calc_direct_mem_copy_updates(self):
        if (self._inter_comm is not None) and (self._inter_comm != _mpi.COMM_NULL):

            dst_rank_to_src_rank = \
                _mpi.Group.Translate_ranks(
                    self._dst_decomp.inter_locale_comm.group,
                    range(0, self._dst_decomp.inter_locale_comm.size),
                    self._src_decomp.inter_locale_comm.group
                )

            for dst_rank in range(self._dst_decomp.inter_locale_comm.size):
                src_rank = dst_rank_to_src_rank[dst_rank]
                dst_extent = self._dst_decomp.get_extent_for_rank(dst_rank)
                src_extent = self._src_decomp.get_extent_for_rank(src_rank)
                dst_leftovers, dst_updates = self.calc_intersection_split(dst_extent, src_extent)
                self._dst_extent_queue.extend(dst_leftovers)
                self._dst_updates[dst_rank] += dst_updates

    def calc_rma_updates(self):
        if (self._inter_comm is not None) and (self._inter_comm != _mpi.COMM_NULL):

            dst_rank_to_src_rank = \
                _mpi.Group.Translate_ranks(
                    self._dst_decomp.inter_locale_comm.group,
                    range(0, self._dst_decomp.inter_locale_comm.size),
                    self._src_decomp.inter_locale_comm.group
                )

            for dst_rank in range(self._dst_decomp.inter_locale_comm.size):
                src_rank = dst_rank_to_src_rank[dst_rank]
                src_extent = self._src_decomp.get_extent_for_rank(src_rank)
                all_dst_leftovers = []
                while len(self._dst_extent_queue) > 0:
                    dst_extent = self._dst_extent_queue.pop()
                    dst_rank = dst_extent.inter_locale_rank
                    dst_leftovers, dst_updates = \
                        self.calc_intersection_split(dst_extent, src_extent)
                    self._dst_updates[dst_rank] += dst_updates
                    all_dst_leftovers += dst_leftovers
                self._dst_extent_queue.extend(all_dst_leftovers)
            if len(self._dst_extent_queue) > 0:
                self._dst_decomp.rank_logger.warning(
                    "Non-empty leftover queue=%s",
                    self._dst_extent_queue
                )

    def initialise(self):
        """
        """
        can_use_existing_inter_locale_comm = self.calc_can_use_existing_inter_locale_comm()

        self._dst_extent_queue = _collections.deque()
        self._dst_updates = _collections.defaultdict(list)
        if can_use_existing_inter_locale_comm:
            self._inter_comm = self._src_decomp.inter_locale_comm
            self._inter_win = self._src_decomp.inter_locale_win
            self.calc_direct_mem_copy_updates()
            self.calc_rma_updates()
            self._dst_decomp.rank_logger.debug(
                "self._dst_updates=%s",
                self._dst_updates
            )

        elif self._dst_decomp.num_locales > 1:
            raise NotImplementedError(
                "Cannot redistribute amongst disparate inter_locale_comm's."
            )

    @property
    def rank_logger(self):
        return self._dst_decomp.rank_logger


__all__ = [s for s in dir() if not s.startswith('_')]
