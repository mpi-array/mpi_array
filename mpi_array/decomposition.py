"""
=========================================
The :mod:`mpi_array.decomposition` Module
=========================================

Sub-division of arrays over nodes and/or MPI processes.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   ExtentUpdate - Base class for describing a sub-extent update.
   PairExtentUpdate - Describes sub-extent source and sub-extent destination.
   MpiPairExtentUpdate - Extends :obj:`PairExtentUpdate` with MPI data type factory.
   HaloSingleExtentUpdate - Describes sub-extent for halo region update.
   MpiHaloSingleExtentUpdate - Extends :obj:`HaloSingleExtentUpdate` with MPI data type factory.
   DecompExtent - Indexing and halo info for a tile in a cartesian decomposition.
   LocaleComms - Shared-memory communicator generation.
   MemAllocTopology - Topology of MPI processes which allocate shared memory.
   CartesianDecomposition - Partition of an array *shape* overs MPI processes and/or nodes.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright

import pkg_resources as _pkg_resources

import sys as _sys
import mpi4py.MPI as _mpi

import array_split as _array_split
import array_split.split  # noqa: F401
from array_split import ARRAY_BOUNDS
from array_split.split import convert_halo_to_array_form, shape_factors as _shape_factors

import mpi_array.logging as _logging
from mpi_array.indexing import IndexingExtent, HaloIndexingExtent
from mpi_array.indexing import calc_intersection_split as _calc_intersection_split

import collections as _collections

import numpy as _np

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


def mpi_version():
    """
    Return the MPI API version.

    :rtype: :obj:`int`
    :return: MPI major version number.
    """
    return _mpi.VERSION


class LocaleComms(object):

    """
    Info on possible shared memory allocation for a specified MPI communicator.
    """

    def __init__(self, comm=None, intra_locale_comm=None):
        """
        Construct.

        :type comm: :obj:`mpi4py.MPI.Comm`
        :param comm: Communicator which is split according to
           shared memory allocation (uses :meth:`mpi4py.MPI.Comm.Split_type`).
        :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param intra_locale_comm: Intra-locale communicator.
           Should be a subset of processes returned
           by :samp:`{comm}.Split_type(mpi4py.MPI.COMM_TYPE_SHARED)`.
           If :samp:`None`, :samp:`{comm}` is *split* into groups
           which can use a MPI window to allocate shared memory
           (i.e. locale is a (NUMA) node).
           Can also specify as :samp:`mpi4py.MPI.COMM_SELF`, in which case the
           locale is a single process.
        """
        if comm is None:
            comm = _mpi.COMM_WORLD
        rank_logger = _logging.get_rank_logger(__name__ + "." + self.__class__.__name__, comm)
        if intra_locale_comm is None:
            if mpi_version() >= 3:
                rank_logger.debug(
                    "BEG: Splitting comm with comm.Split_type(COMM_TYPE_SHARED, ...)"
                )
                intra_locale_comm = comm.Split_type(_mpi.COMM_TYPE_SHARED, key=comm.rank)
                rank_logger.debug(
                    "END: Splitting comm with comm.Split_type(COMM_TYPE_SHARED, ...)"
                )
            else:
                intra_locale_comm = _mpi.COMM_SELF

        self._intra_locale_comm = intra_locale_comm

        # Count the number of self._intra_locale_comm rank-0 processes
        # to work out how many communicators comm was split into.
        is_rank_zero = int(self._intra_locale_comm.rank == 0)

        rank_logger.debug("BEG: comm.allreduce to calculate number of locales...")
        self._num_locales = comm.allreduce(is_rank_zero, _mpi.SUM)
        rank_logger.debug("END: comm.allreduce to calculate number of locales.")

    @property
    def num_locales(self):
        """
        An integer indicating the number of *locales* over which an array is distributed.
        """
        return self._num_locales

    @property
    def intra_locale_comm(self):
        """
        A :obj:`mpi4py.MPI.Comm` object which defines the group of processes
        which can allocate (and access) MPI window shared memory
        (allocated via :meth:`mpi4py.MPI.Win.Allocate_shared` if available).
        """
        return self._intra_locale_comm


class MemAllocTopology(object):

    """
    Defines cartesian communication topology for locales.
    """

    def __init__(
        self,
        ndims=None,
        dims=None,
        rank_comm=None,
        intra_locale_comm=None
    ):
        """
        Initialises cartesian communicator for memory-allocation-nodes.
        Need to specify at least one of the :samp:`{ndims}` or :samp:`{dims}`.
        to indicate the dimension of the cartesian partitioning.

        :type ndims: :obj:`int`
        :param ndims: Dimension of the cartesian partitioning, e.g. 1D, 2D, 3D, etc.
           If :samp:`None`, :samp:`{ndims}=len({dims})`.
        :type dims: sequence of :obj:`int`
        :param dims: The number of partitions along each array axis, zero elements
           are replaced with positive integers such
           that :samp:`numpy.product({dims}) == {rank_comm}.size`.
           If :samp:`None`, :samp:`{dims} = (0,)*{ndims}`.
        :type rank_comm: :obj:`mpi4py.MPI.Comm`
        :param rank_comm: The MPI processes which will have access
           (via a :obj:`mpi4py.MPI.Win` object) to the distributed array.
           If :samp:`None` uses :obj:`mpi4py.MPI.COMM_WORLD`.
        :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param intra_locale_comm: The MPI communicator used to create a window which
            can be used to allocate shared memory
            via :meth:`mpi4py.MPI.Win.Allocate_shared`.
        """
        # No implementation for periodic boundaries yet
        periods = None
        if (ndims is None) and (dims is None):
            raise ValueError("Must specify one of dims or ndims in MemAllocTopology constructor.")
        elif (ndims is not None) and (dims is not None) and (len(dims) != ndims):
            raise ValueError(
                "Length of dims (len(dims)=%s) not equal to ndims=%s." % (len(dims), ndims)
            )
        elif ndims is None:
            ndims = len(dims)

        if dims is None:
            dims = _np.zeros((ndims,), dtype="int")
        if periods is None:
            periods = _np.zeros((ndims,), dtype="bool")
        if rank_comm is None:
            rank_comm = _mpi.COMM_WORLD

        self._rank_comm = rank_comm
        self._locale_comms = LocaleComms(self.rank_comm, intra_locale_comm)
        self._cart_comm = None
        rank_logger = \
            _logging.get_rank_logger(__name__ + "." + self.__class__.__name__, comm=self._rank_comm)

        self._dims = \
            _array_split.split.calculate_num_slices_per_axis(
                dims,
                self.num_locales
            )

        # Create a cartesian grid communicator
        if self.num_locales > 1:
            color = _mpi.UNDEFINED
            if self.intra_locale_comm.rank == 0:
                color = 0
            rank_logger.debug("BEG: self.rank_comm.Split to create self.cart_comm.")
            splt_comm = self.rank_comm.Split(color, self.rank_comm.rank)
            rank_logger.debug("END: self.rank_comm.Split to create self.cart_comm.")
            if splt_comm != _mpi.COMM_NULL:
                rank_logger.debug("BEG: splt_comm.Create to create self.cart_comm.")
                self._cart_comm = splt_comm.Create_cart(self.dims, periods, reorder=True)
                rank_logger.debug("END: splt_comm.Create to create self.cart_comm.")
            else:
                self._cart_comm = _mpi.COMM_NULL

    @property
    def dims(self):
        """
        The number of partitions along each array axis. Defines
        the cartesian topology over which an array is distributed.
        """
        return self._dims

    @property
    def have_valid_cart_comm(self):
        """
        Is :samp:`True` if this rank has :samp:`{self}.cart_comm`
        which is not :samp:`None` and is not :obj:`mpi4py.MPI.COMM_NULL`.
        """
        return \
            (
                (self.cart_comm is not None)
                and
                (self.cart_comm != _mpi.COMM_NULL)
            )

    @property
    def rank_comm(self):
        """
        The group of all MPI processes which have access to array elements.
        """
        return self._rank_comm

    @property
    def cart_comm(self):
        """
        The group of MPI processes (typically one process per memory node)
        which communicate to exchange array data (halo data say) between memory nodes.
        """
        return self._cart_comm

    @property
    def num_locales(self):
        """
        See :attr:`LocaleComms.num_locales`.
        """
        return self._locale_comms.num_locales

    @property
    def intra_locale_comm(self):
        """
        See :attr:`LocaleComms.intra_locale_comm`.
        """
        return self._locale_comms.intra_locale_comm


if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
    # Set docstring for properties.
    MemAllocTopology.num_locales.__doc__ = LocaleComms.num_locales.__doc__
    MemAllocTopology.intra_locale_comm.__doc__ = LocaleComms.intra_locale_comm.__doc__


class DecompExtent(HaloIndexingExtent):

    """
    Indexing extents for single tile of cartesian domain decomposition.
    """

    def __init__(
        self,
        rank,
        cart_rank,
        cart_coord,
        cart_shape,
        array_shape,
        slice,
        halo,
        bounds_policy=ARRAY_BOUNDS
    ):
        """
        Construct.

        :type rank: :obj:`int`
        :param rank: Rank of MPI process in :samp:`rank_comm` communicator.
        :type cart_rank: :obj:`int`
        :param cart_rank: Rank of MPI process in cartesian communicator.
        :type cart_coord: sequence of :obj:`int`
        :param cart_coord: Coordinate index of this tile in the cartesian domain decomposition.
        :type cart_shape: sequence of :obj:`int`
        :param cart_shape: Number of tiles in each axis direction.
        :type slice: sequence of :obj:`slice`
        :param slice: Per-axis start and stop indices (**not including ghost elements**).
        :type halo: :samp:`(len({split}), 2)` shaped array of :obj:`int`
        :param halo: A :samp:`(len(self.start), 2)` shaped array of :obj:`int` indicating the
           per-axis number of outer ghost elements. :samp:`halo[:,0]` is the number
           of elements on the low-index *side* and :samp:`halo[:,1]` is the number of
           elements on the high-index *side*.
        """
        self._rank = rank
        self._cart_rank = cart_rank
        self._cart_coord = _np.array(cart_coord, dtype="int64")
        self._cart_shape = _np.array(cart_shape, dtype=self._cart_coord.dtype)
        self._array_shape = _np.array(array_shape, dtype=self._cart_coord.dtype)
        HaloIndexingExtent.__init__(self, slice, halo=None)
        halo = convert_halo_to_array_form(halo, ndim=len(self._cart_coord))
        if bounds_policy == ARRAY_BOUNDS:
            # Make the halo
            halo = \
                _np.array(
                    (
                        _np.minimum(
                            self.start_n,
                            halo[:, self.LO]
                        ),
                        _np.minimum(
                            self._array_shape - self.stop_n,
                            halo[:, self.HI]
                        ),
                    ),
                    dtype=halo.dtype
                ).T
        self._halo = halo

    @property
    def rank(self):
        """
        MPI rank of the process in the :samp:`rank_comm` communicator.
        """
        return self._rank

    @property
    def cart_rank(self):
        """
        MPI rank of the process in the cartesian decomposition.
        """
        return self._cart_rank

    @property
    def inter_locale_rank(self):
        """
        MPI rank of the process in the :samp:`inter_locale_comm`.
        """
        return self._cart_rank

    @property
    def cart_coord(self):
        """
        Cartesian coordinate of cartesian decomposition.
        """
        return self._cart_coord

    @property
    def cart_shape(self):
        """
        Shape of cartesian decomposition (number of tiles along each axis).
        """
        return self._cart_shape

    def halo_slab_extent(self, axis, dir):
        """
        Returns indexing extent of the halo *slab* for specified axis.

        :type axis: :obj:`int`
        :param axis: Indexing extent of halo slab for this axis.
        :type dir: :attr:`LO` or :attr:`HI`
        :param dir: Indicates low-index halo slab or high-index halo slab.
        :rtype: :obj:`IndexingExtent`
        :return: Indexing extent for halo slab.
        """
        start = self.start_h.copy()
        stop = self.stop_h.copy()
        if dir == self.LO:
            stop[axis] = start[axis] + self.halo[axis, self.LO]
        else:
            start[axis] = stop[axis] - self.halo[axis, self.HI]

        return \
            IndexingExtent(
                start=start,
                stop=stop
            )

    def no_halo_extent(self, axis):
        """
        Returns the indexing extent identical to this extent, except
        has the halo trimmed from the axis specified by :samp:`{axis}`.

        :type axis: :obj:`int` or sequence of :obj:`int`
        :param axis: Axis (or axes) for which halo is trimmed.
        :rtype: :obj:`IndexingExtent`
        :return: Indexing extent with halo trimmed from specified axis (or axes) :samp:`{axis}`.
        """
        start = self.start_h.copy()
        stop = self.stop_h.copy()
        if axis is not None:
            start[axis] += self.halo[axis, self.LO]
            stop[axis] -= self.halo[axis, self.HI]

        return \
            IndexingExtent(
                start=start,
                stop=stop
            )


class ExtentUpdate(object):

    """
    Source and destination indexing info for updating a sub-extent region.
    """

    def __init__(self, dst_extent, src_extent):
        """
        Initialise.

        :type dst_extent: :obj:`DecompExtent`
        :param dst_extent: Whole locale extent which receives update.
        :type src_extent: :obj:`DecompExtent`
        :param src_extent: Whole locale extent from which update is read.
        """
        object.__init__(self)
        self._dst_extent = dst_extent  #: whole tile of destination locale
        self._src_extent = src_extent  #: whole tile of source locale

    @property
    def dst_extent(self):
        """
        The locale :obj:`DecompExtent` which is to receive sub-array update.
        """
        return self._dst_extent

    @property
    def src_extent(self):
        """
        The locale :obj:`DecompExtent` from which the sub-array update is read.
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
        if dst_mpi_order == "F":
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
        if src_mpi_order == "F":
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
           is a :obj:`DecompExtent` object indicating the indexing extent
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

        :type dst_extent: :obj:`DecompExtent`
        :param dst_extent: Halo slab indicated by :samp:`{axis}` and :samp:`{dir}`
           taken from this extent.
        :type src_extent: :obj:`DecompExtent`
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

        :type extent: :obj:`DecompExtent`
        :param extent: The extent to be split.
        :type max_elements: :obj:`int`
        :param max_elements: Each partition of the returned split has no more
           than this many elements.
        :rtype: :obj:`list` of :obj:`DecompExtent`
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
           is a :obj:`DecompExtent` object indicating the indexing extent
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

    def do_locale_update(self):
        """
        """
        pass

    @property
    def rank_logger(self):
        return self._dst_decomp.rank_logger

    def barrier(self):
        pass

    @property
    def update_win(self):
        return None


class CartesianDecomposition(object):

    """
    Partitions an array-shape over MPI memory-nodes.
    """

    #: The "low index" indices.
    LO = HaloIndexingExtent.LO

    #: The "high index" indices.
    HI = HaloIndexingExtent.HI

    def __init__(
        self,
        shape,
        halo=0,
        mem_alloc_topology=None,
        order="C"
    ):
        """
        Create a partitioning of :samp:`{shape}` over memory-nodes.

        :type shape: sequence of :obj:`int`
        :param shape: The shape of the array which is to be partitioned into smaller *sub-shapes*.
        :type halo: :obj:`int`, sequence of :obj:`int` or :samp:`(len({shape}), 2)` shaped array.
        :param halo: Number of *ghost* elements added per axis
           (low and high indices can be different).
        :type mem_alloc_topology: :obj:`MemAllocTopology`
        :param mem_alloc_topology: Object which defines how array
           memory is allocated (distributed) over memory nodes and
           the cartesian topology communicator used to exchange (halo)
           data. If :samp:`None` uses :samp:`MemAllocTopology(dims=numpy.zeros_like({shape}))`.
        """
        self._halo = halo
        self._shape = None
        self._mem_alloc_topology = mem_alloc_topology
        self._shape_decomp = None
        self._rank_logger = None
        self._root_logger = None
        self._cart_win = None
        self._order = order

        self.recalculate(shape, halo)

    def calculate_rank_view_slices(self):
        """
        Splits local array into :samp:`self.intra_locale_comm.size` number
        of tiles. Assigns :attr:`rank_view_slice_n` and :attr:`rank_view_slice_h`
        to :obj:`tuple`-of-:obj:`slice` corresponding to the tile for this MPI rank.
        """
        if self._lndarray_extent.size_n > 0:
            shape_splitter = \
                _array_split.ShapeSplitter(
                    array_shape=self._lndarray_extent.shape_n,
                    axis=_shape_factors(self.intra_locale_comm.size, self.ndim)[::-1],
                    halo=0,
                    array_start=self._lndarray_extent.start_n
                )

            split = shape_splitter.calculate_split()
            rank_extent_n = IndexingExtent(split.flatten()[self.intra_locale_comm.rank])
            rank_extent_h = \
                IndexingExtent(
                    start=_np.maximum(
                        rank_extent_n.start - self._halo[:, self.LO],
                        self._lndarray_extent.start_h
                    ),
                    stop=_np.minimum(
                        rank_extent_n.stop + self._halo[:, self.HI],
                        self._lndarray_extent.stop_h
                    )
                )

            # Convert rank_extent_n and rank_extent_h from global-indices
            # to local-indices
            halo_lo = self._lndarray_extent.halo[:, self.LO]
            rank_extent_n = \
                IndexingExtent(
                    start=rank_extent_n.start - self._lndarray_extent.start_n + halo_lo,
                    stop=rank_extent_n.stop - self._lndarray_extent.start_n + halo_lo,
                )
            rank_extent_h = \
                IndexingExtent(
                    start=rank_extent_h.start - self._lndarray_extent.start_n + halo_lo,
                    stop=rank_extent_h.stop - self._lndarray_extent.start_n + halo_lo,
                )
            rank_h_relative_extent_n = \
                IndexingExtent(
                    start=rank_extent_n.start - rank_extent_h.start,
                    stop=rank_extent_n.start - rank_extent_h.start + rank_extent_n.shape,
                )

            self._rank_view_slice_n = \
                tuple(
                    [slice(rank_extent_n.start[i], rank_extent_n.stop[i]) for i in range(self.ndim)]
                )
            self._rank_view_slice_h = \
                tuple(
                    [slice(rank_extent_h.start[i], rank_extent_h.stop[i]) for i in range(self.ndim)]
                )
            self._rank_view_relative_slice_n = \
                tuple(
                    [
                        slice(rank_h_relative_extent_n.start[i], rank_h_relative_extent_n.stop[i])
                        for i in range(self.ndim)
                    ]
                )
        else:
            self._rank_view_slice_n = tuple([slice(0, 0) for i in range(self.ndim)])
            self._rank_view_slice_h = tuple([slice(0, 0) for i in range(self.ndim)])
            self._rank_view_relative_slice_n = tuple([slice(0, 0) for i in range(self.ndim)])

    def recalculate(self, new_shape, new_halo):
        """
        Recomputes decomposition for :samp:`{new_shape}` and :samp:`{new_halo}`.

        :type new_shape: sequence of :obj:`int`
        :param new_shape: New partition calculated for this shape.
        :type new_halo: :obj:`int`, sequence of :obj:`int` or :samp:`(len{new_shape, 2))` array.
        :param new_halo: New partition calculated for this shape.
        """
        if self._mem_alloc_topology is None:
            self._mem_alloc_topology = MemAllocTopology(ndims=len(new_shape))
        elif (self._shape is not None) and (len(self._shape) != len(new_shape)):
            self._shape = _np.array(new_shape)
            self._mem_alloc_topology = MemAllocTopology(ndims=self._shape.size)
        self._shape = _np.array(new_shape)
        self._halo = new_halo

        shape_splitter = \
            _array_split.ShapeSplitter(
                array_shape=self._shape,
                axis=self._mem_alloc_topology.dims,
                halo=0
            )

        self._halo = convert_halo_to_array_form(halo=self._halo, ndim=len(self._shape))

        self._shape_decomp = shape_splitter.calculate_split()

        self._cart_rank_to_extents_dict = None
        self._halo_updates_dict = None
        self._lndarray_extent = None
        if self.have_valid_cart_comm:
            cart_dims = _np.array(self.cart_comm.dims)
            self._cart_rank_to_extents_dict = dict()
            self._halo_updates_dict = dict()
            for cart_rank in range(0, self.cart_comm.size):
                cart_coords = _np.array(self.cart_comm.Get_coords(cart_rank))
                self._cart_rank_to_extents_dict[cart_rank] = \
                    DecompExtent(
                        rank=self.rank_comm.rank,
                        cart_rank=cart_rank,
                        cart_coord=cart_coords,
                        cart_shape=cart_dims,
                        array_shape=self._shape,
                        slice=self._shape_decomp[tuple(cart_coords)],
                        halo=self._halo,
                        bounds_policy=shape_splitter.tile_bounds_policy
                    )  # noqa: E123
            for cart_rank in range(0, self.cart_comm.size):
                self._halo_updates_dict[cart_rank] = \
                    MpiHalosUpdate(
                        cart_rank,
                        self._cart_rank_to_extents_dict
                )
            self._lndarray_extent = self._cart_rank_to_extents_dict[self.cart_comm.rank]
        elif self.num_locales <= 1:
            slice_tuple = tuple([slice(0, self._shape[i]) for i in range(len(self._shape))])
            self._lndarray_extent = \
                    DecompExtent(
                        rank=0,
                        cart_rank=0,
                        cart_coord=[0, ] * len(self._shape),
                        cart_shape=[1, ] * len(self._shape),
                        array_shape=self._shape,
                        slice=slice_tuple,
                        halo=self._halo,
                        bounds_policy=shape_splitter.tile_bounds_policy
                    )  # noqa: E123
            self._cart_rank_to_extents_dict =\
                {self._lndarray_extent.cart_rank: self._lndarray_extent}

        self._lndarray_extent, self._cart_rank_to_extents_dict = \
            self.intra_locale_comm.bcast((self._lndarray_extent, self._cart_rank_to_extents_dict), 0)

        self._lndarray_view_slice_n = \
            IndexingExtent(
                start=self._lndarray_extent.halo[:, self.LO],
                stop=self._lndarray_extent.halo[:, self.LO] + self._lndarray_extent.shape_n
            ).to_slice()

        self.calculate_rank_view_slices()

    def alloc_local_buffer(self, dtype):
        """
        Allocates a buffer using :meth:`mpi4py.MPI.Win.Allocate_shared` which
        provides storage for the elements of the local (memory-node) multi-dimensional array.

        :rtype: :obj:`tuple`
        :returns: A :obj:`tuple` of :samp:`(buffer, itemsize, shape)`, where :samp:`buffer`
           is the allocated memory for the array, :samp:`itemsize` is :samp:`dtype.itemsize`
           and :samp:`shape` is the shape of the :samp:`numpy.ndarray`.
        """
        self.rank_logger.debug("BEG: alloc_local_buffer")
        num_rank_bytes = 0
        dtype = _np.dtype(dtype)
        if self.intra_locale_comm.rank == 0:
            if (self.num_locales > 1) and (not self.have_valid_cart_comm):
                raise ValueError("Root rank (=0) on intra_locale_comm does not have valid cart_comm.")
            if self.num_locales > 1:
                rank_shape = self._cart_rank_to_extents_dict[self.cart_comm.rank].shape_h
            else:
                rank_shape = self.shape
            num_rank_bytes = int(_np.product(rank_shape) * dtype.itemsize)
        if (mpi_version() >= 3) and (self.intra_locale_comm.size > 1):
            self.rank_logger.debug("BEG: Win.Allocate_shared - allocating %d bytes", num_rank_bytes)
            self._shared_mem_win = \
                _mpi.Win.Allocate_shared(num_rank_bytes, dtype.itemsize, comm=self.intra_locale_comm)
            self.rank_logger.debug("END: Win.Allocate_shared - allocating %d bytes", num_rank_bytes)
            buffer, itemsize = self._shared_mem_win.Shared_query(0)
            self.rank_logger.debug("BEG: Win.Create for self.rank_comm")
            self._rank_win = _mpi.Win.Create(buffer, itemsize, comm=self.rank_comm)
            self.rank_logger.debug("END: Win.Create for self.rank_comm")
        else:
            self.rank_logger.debug("BEG: Win.Allocate - allocating %d bytes", num_rank_bytes)
            self._rank_win = \
                _mpi.Win.Allocate(num_rank_bytes, dtype.itemsize, comm=self.rank_comm)
            self.rank_logger.debug("END: Win.Allocate - allocating %d bytes", num_rank_bytes)
            self._shared_mem_win = self._rank_win
            buffer = self._rank_win.memory
            itemsize = dtype.itemsize

        self._cart_win = None
        lndarray_shape = self.shape
        if self.num_locales > 1:
            self._cart_win = _mpi.WIN_NULL
            if self.have_valid_cart_comm:
                self.rank_logger.debug("BEG: Win.Create for self.cart_comm")
                self._cart_win = _mpi.Win.Create(buffer, itemsize, comm=self.cart_comm)
                self.rank_logger.debug("END: Win.Create for self.cart_comm")
            lndarray_shape = self._lndarray_extent.shape_h

        buffer = _np.array(buffer, dtype='B', copy=False)

        self.rank_logger.debug("END: alloc_local_buffer")
        return buffer, itemsize, lndarray_shape

    def get_updates_for_cart_rank(self, cart_rank):
        return self._halo_updates_dict[cart_rank]

    def __str__(self):
        """
        """
        s = []
        if self.have_valid_cart_comm:
            for cart_rank in range(0, self.cart_comm.size):
                s += \
                    [
                        "{cart_rank = %s, cart_coord = %s, extents=%s}"
                        %
                        (
                            cart_rank,
                            self.cart_comm.Get_coords(cart_rank),
                            self._cart_rank_to_extents_dict[cart_rank],
                        )
                    ]
        return ", ".join(s)

    @property
    def ndim(self):
        """
        Dimension of array.
        """
        return self._shape.size

    @property
    def halo(self):
        """
        Number of *ghost* elements per axis to pad array shape.
        """
        return self._halo

    @halo.setter
    def halo(self, halo):
        if halo is None:
            halo = 0

        self.recalculate(self._shape, halo)

    @property
    def shape(self):
        """
        The shape of the array to be distributed over MPI memory nodes.
        """
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self.recalculate(new_shape, self._halo)

    @property
    def locale_extents(self):
        """
        A :obj:`list` of :obj:`DecompExtent` instances.
        """
        return [
            self._cart_rank_to_extents_dict[cart_rank]
            for cart_rank in self._cart_rank_to_extents_dict.keys()
        ]

    def get_extent_for_rank(self, inter_locale_rank):
        """
        Returns extent associated with the specified rank
        of the :attr:`inter_locale_comm` communicator.
        """
        return self._cart_rank_to_extents_dict[inter_locale_rank]

    @property
    def shape_decomp(self):
        """
        The partition of :samp:`self.shape` over memory nodes.
        """
        return self._shape_decomp

    @property
    def num_locales(self):
        """
        See :attr:`MemAllocTopology.num_locales`.
        """
        return self._mem_alloc_topology.num_locales

    @property
    def intra_locale_comm(self):
        """
        See :attr:`MemAllocTopology.intra_locale_comm`.
        """
        return self._mem_alloc_topology.intra_locale_comm

    @property
    def cart_comm(self):
        """
        See :attr:`MemAllocTopology.cart_comm`.
        """
        return self._mem_alloc_topology.cart_comm

    @property
    def inter_locale_comm(self):
        """
        See :attr:`cart_comm`.
        """
        return self.cart_comm

    @property
    def cart_win(self):
        """
        Window for RMA updates.
        """
        return self._cart_win

    @property
    def inter_locale_win(self):
        """
        See :attr:`cart_win`.
        """
        return self.cart_win

    @property
    def have_valid_cart_comm(self):
        """
        See :attr:`MemAllocTopology.have_valid_cart_comm`.
        """
        return self._mem_alloc_topology.have_valid_cart_comm

    @property
    def have_valid_inter_locale_comm(self):
        """
        See :attr:`have_valid_cart_comm`.
        """
        return self.have_valid_cart_comm

    @property
    def rank_comm(self):
        """
        See :attr:`MemAllocTopology.rank_comm`.
        """
        return self._mem_alloc_topology.rank_comm

    @property
    def rank_view_slice_n(self):
        """
        A :obj:`tuple` of :obj:`slice` indicating the tile (no halo)
        associated with this MPI process (i.e. rank :samp:`self.rank_comm.rank`).
        """
        return self._rank_view_slice_n

    @property
    def rank_view_slice_h(self):
        """
        A :obj:`tuple` of :obj:`slice` indicating the tile (including halo)
        associated with this MPI process (i.e. rank :samp:`self.rank_comm.rank`).
        """
        return self._rank_view_slice_h

    @property
    def rank_view_relative_slice_n(self):
        """
        A :obj:`tuple` of :obj:`slice` which can be used to *slice* (remove)
        the halo from a halo rank view. For example::

           import mpi_array.locale
           lary = mpi_array.locale.zeros((10, 10, 100), dtype="float32")
           _np.all(
               lary.rank_view_h[lary.decomp.rank_view_relative_slice_n]
               ==
               lary.rank_view_n
           )

        """
        return self._rank_view_relative_slice_n

    @property
    def lndarray_extent(self):
        """
        The extent of the locale array.
        """
        return self._lndarray_extent

    @property
    def lndarray_view_slice_n(self):
        """
        Indexing slice which can be used to generate a view of :obj:`mpi_array.locale.lndarray`
        which has the halo removed.
        """
        return self._lndarray_view_slice_n

    @property
    def rank_logger(self):
        """
        A :obj:`logging.Logger` for :attr:`rank_comm` communicator ranks.
        """
        if self._rank_logger is None:
            self._rank_logger = \
                _logging.get_rank_logger(
                    __name__ + "." + self.__class__.__name__,
                    comm=self.rank_comm
                )
        return self._rank_logger

    @property
    def root_logger(self):
        """
        A :obj:`logging.Logger` for rank 0 of the :attr:`rank_comm` communicator.
        """
        if self._root_logger is None:
            self._root_logger = \
                _logging.get_root_logger(
                    __name__ + "." + self.__class__.__name__,
                    comm=self.rank_comm
                )
        return self._root_logger


if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
    # Set docstring for properties.
    CartesianDecomposition.num_locales.__doc__ = \
        MemAllocTopology.num_locales.__doc__
    CartesianDecomposition.intra_locale_comm.__doc__ = MemAllocTopology.intra_locale_comm.__doc__
    CartesianDecomposition.cart_comm.__doc__ = MemAllocTopology.cart_comm.__doc__
    CartesianDecomposition.have_valid_cart_comm.__doc__ = \
        MemAllocTopology.have_valid_cart_comm.__doc__
    CartesianDecomposition.rank_comm.__doc__ = MemAllocTopology.rank_comm.__doc__


__all__ = [s for s in dir() if not s.startswith('_')]
