"""
===================================
The :mod:`mpi_array.globale` Module
===================================

Defines :obj:`gndarray` class and factory functions for
creating multi-dimensional distributed arrays (Partitioned Global Address Space).

Classes
=======

.. autosummary::
   :toctree: generated/

   gndarray - A :obj:`numpy.ndarray` like distributed array.
   PerAxisRmaHaloUpdater - Helper class for performing ghost element updates.
   RmaRedistributeUpdater - Helper class for redistributing elements between distributions.

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
   copyto - Copy elements of one array to another array.


"""

from __future__ import absolute_import

import mpi4py.MPI as _mpi
import numpy as _np
import collections as _collections

from .license import license as _license, copyright as _copyright, version as _version
from . import locale as _locale
from .comms import create_distribution
from .update import UpdatesForRedistribute as _UpdatesForRedistribute
from .update import MpiHalosUpdate as _MpiHalosUpdate
from .update import MpiPairExtentUpdate as _MpiPairExtentUpdate
from .update import MpiPairExtentUpdateDifferentDtypes as _MpiPairExtentUpdateDifferentDtypes
from .indexing import HaloIndexingExtent as _HaloIndexingExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class CommLogger:

    """
    """

    def __init__(self, rank_logger=None, root_logger=None):
        self._rank_logger = None
        self._root_logger = None

    @property
    def rank_logger(self):
        return self._rank_logger

    @rank_logger.setter
    def rank_logger(self, logger):
        self._rank_logger = logger

    @property
    def root_logger(self):
        return self._root_logger

    @root_logger.setter
    def root_logger(self, logger):
        self._root_logger = logger


class PerAxisRmaHaloUpdater(CommLogger):

    """
    Helper class for performing halo data transfer using RMA via MPI windows.
    """
    #: Halo "low index" indices.
    LO = _HaloIndexingExtent.LO

    #: Halo "high index" indices.
    HI = _HaloIndexingExtent.HI

    def __init__(self, locale_extents, dtype, order, inter_locale_win, dst_buffer):
        """
        """
        CommLogger.__init__(self)
        self._locale_extents = locale_extents
        self._dtype = dtype
        self._order = order
        self._inter_locale_win = inter_locale_win
        self._dst_buffer = dst_buffer
        self._halo_updates = None
        self._have_axis_updates = None

    @property
    def locale_extents(self):
        """
        """
        return self._locale_extents

    @property
    def dtype(self):
        return self._dtype

    @property
    def order(self):
        return self._order

    def calc_halo_updates(self):
        """
        """
        halo_updates_dict = dict()
        ndim = self.locale_extents[0].ndim
        have_axis_updates = _np.zeros((ndim, ), dtype="bool")
        for inter_locale_rank in range(len(self.locale_extents)):
            rank_inter_locale_updates = \
                _MpiHalosUpdate(
                    inter_locale_rank,
                    self.locale_extents
                )
            halo_updates_dict[inter_locale_rank] = rank_inter_locale_updates
            have_axis_updates = \
                _np.logical_or(
                    have_axis_updates,
                    _np.array(
                        [
                            (rank_inter_locale_updates.updates_per_axis[a] is not None)
                            and
                            (
                                (len(rank_inter_locale_updates.updates_per_axis[a][self.LO]) > 0)
                                or
                                (len(rank_inter_locale_updates.updates_per_axis[a][self.HI]) > 0)
                            )
                            for a in range(ndim)
                        ],
                        dtype="bool"
                    )
                )

        if _np.any(have_axis_updates):
            for a in range(ndim):
                if not have_axis_updates[a]:
                    for inter_locale_rank in range(len(self.locale_extents)):
                        halo_updates_dict[inter_locale_rank].updates_per_axis[a] = None
        else:
            halo_updates_dict = None

        return halo_updates_dict, have_axis_updates

    @property
    def halo_updates(self):
        if self._halo_updates is None:
            self._halo_updates, self._have_axis_updates = self.calc_halo_updates()

        return self._halo_updates

    def update_halos(self):
        """
        """
        self.do_update_halos(self.halo_updates)

    def do_update_halos(self, halo_updates):
        """
        """
        if halo_updates is not None:
            # Get the halo updates for this rank
            rank_inter_locale_updates = halo_updates[self._inter_locale_win.group.rank]
            # Get the updates separated into per-axis (hyper-slab) updates
            rank_updates_per_axis = rank_inter_locale_updates.updates_per_axis

            # rank_updates_per_axis is None, on *all* inter_locale_win.group ranks,
            # when there are no halos on any axis.
            if rank_updates_per_axis is not None:
                for a in range(len(rank_updates_per_axis)):
                    lo_hi_updates_pair = rank_updates_per_axis[a]

                    # When axis doesn't have a halo then lo_hi_updates_pair
                    # is None on all cart_comm ranks, and we avoid calling the Fence
                    # in this case
                    if lo_hi_updates_pair is not None:
                        axis_inter_locale_rank_updates = \
                            lo_hi_updates_pair[rank_inter_locale_updates.LO] + \
                            lo_hi_updates_pair[rank_inter_locale_updates.HI]
                        self.rank_logger.debug(
                            "BEG: Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)..."
                        )
                        self._inter_locale_win.Fence(
                            _mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)
                        for single_update in axis_inter_locale_rank_updates:
                            single_update.initialise_data_types(self.dtype, self.order)
                            self.rank_logger.debug(
                                "BEG: Getting update:\n%s\n%s",
                                single_update._header_str,
                                single_update
                            )
                            self._inter_locale_win.Get(
                                [self._dst_buffer, 1, single_update.dst_data_type],
                                single_update.src_extent.cart_rank,
                                [0, 1, single_update.src_data_type]
                            )
                            self.rank_logger.debug(
                                "END: Getting update:\n%s\n%s",
                                single_update._header_str,
                                single_update
                            )
                        self._inter_locale_win.Fence(_mpi.MODE_NOSUCCEED)
                        self.rank_logger.debug(
                            "END: Fence(_mpi.MODE_NOSUCCEED)."
                        )


class RmaRedistributeUpdater(_UpdatesForRedistribute):

    """
    Helper class for redistributing array to new distribution.
    Calculates sequence of :obj:`mpi_array.distribution.ExtentUpdate`
    objects which are used to copy elements from
    remote :samp:`{src}` locales to local :samp:`{dst}` locales.
    """

    def __init__(self, dst, src, casting="same_kind"):
        """
        """
        self._dst = dst
        self._src = src
        self._casting = casting
        self._mpi_pair_extent_update_type = _MpiPairExtentUpdate
        if self._dst.dtype != self._src.dtype:
            self._mpi_pair_extent_update_type = _MpiPairExtentUpdateDifferentDtypes

        _UpdatesForRedistribute.__init__(
            self,
            dst.comms_and_distrib.distribution,
            src.comms_and_distrib.distribution
        )
        self._inter_win = self._src.rma_window_buffer.peer_win
        self._dst.rank_logger.debug(
            "self._dst_updates=%s",
            self._dst_updates
        )

    def calc_can_use_existing_src_peer_comm(self):
        can_use_existing_src_peer_comm = self._src.locale_comms.peer_comm is not None
        if self._dst.locale_comms.have_valid_inter_locale_comm:
            if self._src.locale_comms.peer_comm != _mpi.COMM_NULL:
                can_use_existing_src_peer_comm = \
                    (
                        (
                            _mpi.Group.Intersection(
                                self._dst.locale_comms.inter_locale_comm.group,
                                self._src.locale_comms.peer_comm.group
                            ).size
                            ==
                            self._dst.locale_comms.inter_locale_comm.group.size
                        )
                    )
        self._dst.rank_logger.debug(
            "BEG: self._dst_cad.locale_comms.intra_locale_comm.allreduce...")
        can_use_existing_src_peer_comm = \
            self._dst.locale_comms.intra_locale_comm.allreduce(
                can_use_existing_src_peer_comm,
                _mpi.BAND
            )
        self._dst.rank_logger.debug("END: self._dst_cad.locale_comms.intra_locale_comm.allreduce.")
        self._dst.rank_logger.debug(
            "can_use_existing_src_peer_comm = %s",
            can_use_existing_src_peer_comm
        )
        return can_use_existing_src_peer_comm

    def create_pair_extent_update(
        self,
        dst_extent,
        src_extent,
        intersection_extent
    ):
        """
        Factory method which creates sequence of
        of :obj:`mpi_array.distribution.MpiPairExtentUpdate` objects.
        """
        updates = \
            [
                self._mpi_pair_extent_update_type(
                    self._dst.distribution.locale_extents[dst_extent.inter_locale_rank],
                    self._src.distribution.locale_extents[src_extent.inter_locale_rank],
                    intersection_extent,
                    intersection_extent
                ),
            ]
        for update in updates:
            update.initialise_data_types(
                dst_dtype=self._dst.dtype,
                src_dtype=self._src.dtype,
                dst_order=self._dst.lndarray_proxy.md.order,
                src_order=self._src.lndarray_proxy.md.order
            )
            update.casting = self._casting

        return updates

    def do_locale_update(self):
        """
        Performs RMA to get elements from remote locales to
        update the local array.
        """
        if (self._src.locale_comms.num_locales > 1) or (self._dst.locale_comms.num_locales > 1):

            can_use_existing_src_peer_comm = self.calc_can_use_existing_src_peer_comm()
            self._dst.rank_logger.debug(
                "%s.%s: "
                +
                "can_use_existing_src_peer_comm=%s",
                self.__class__.__name__,
                "do_locale_update",
                can_use_existing_src_peer_comm
            )

            if can_use_existing_src_peer_comm:
                if (
                    (self._inter_win is not None)
                    and
                    (self._inter_win != _mpi.WIN_NULL)
                    and
                    self._dst.locale_comms.have_valid_inter_locale_comm
                ):
                    updates = self._dst_updates[self._dst.this_locale.inter_locale_rank]
                    update_dict = _collections.defaultdict(list)
                    for single_update in updates:
                        update_dict[single_update.src_extent.peer_rank].append(single_update)
                    for src_peer_rank in update_dict.keys():
                        self._dst.rank_logger.debug(
                            "BEG: Lock(rank=%s, _mpi.LOCK_SHARED)...", src_peer_rank
                        )
                        self._inter_win.Lock(src_peer_rank, _mpi.LOCK_SHARED)
                        for single_update in update_dict[src_peer_rank]:
                            self._dst.rank_logger.debug(
                                "BEG: Getting update:\n%s\n%s",
                                single_update._header_str,
                                single_update
                            )
                            single_update.do_get(
                                self._inter_win,
                                src_peer_rank,
                                self._dst.lndarray_proxy.lndarray
                            )
                            self._dst.rank_logger.debug(
                                "END: Got update:\n%s\n%s",
                                single_update._header_str,
                                single_update
                            )
                        self._inter_win.Unlock(src_peer_rank)
                        self._dst.rank_logger.debug(
                            "END: Unlock(rank=%s).", src_peer_rank
                        )

            else:
                raise RuntimeError(
                    (
                        "can_use_existing_src_peer_comm=%s: "
                        +
                        "incompatible dst inter_locale_comma and src peer_comm."
                    )
                    %
                    (can_use_existing_src_peer_comm,)
                )
            self._dst.locale_comms.intra_locale_comm.barrier()
        else:
            _np.copyto(
                self._dst.lndarray_proxy.lndarray,
                self._src.lndarray_proxy.lndarray,
                casting=self._casting
            )

    def barrier(self):
        """
        MPI barrier.
        """
        self._dst.locale_comms.rank_logger.debug(
            "BEG: self._src.locale_comms.peer_comm.barrier()..."
        )
        self._src.locale_comms.peer_comm.barrier()
        self._dst.locale_comms.rank_logger.debug(
            "END: self._src.locale_comms.peer_comm.barrier()."
        )


class gndarray(object):

    """
    A Partitioned Global Address Space array with :obj:`numpy.ndarray` API.
    """

    def __new__(
        cls,
        comms_and_distrib,
        rma_window_buffer,
        lndarray_proxy
    ):
        """
        Construct, at least one of :samp:{shape} or :samp:`comms_and_distrib` should
        be specified (i.e. at least one should not be :samp:`None`).

        :type comms_and_distrib: :obj:`mpi_array.distribution.Decomposition`
        :param comms_and_distrib: Array distribution info and used to allocate (possibly)
           shared memory.
        """
        self = object.__new__(cls)
        self._comms_and_distrib = comms_and_distrib
        self._rma_window_buffer = rma_window_buffer
        self._lndarray_proxy = lndarray_proxy
        self._halo_updater = None

        return self

    def __getitem__(self, i):
        """
        """
        self.rank_logger.debug("__getitem__: i=%s", i)
        return None

    def __setitem__(self, i, v):
        """
        """
        self.rank_logger.debug("__setitem__: i=%s, v=%s", i, v)

    def __eq__(self, other):
        """
        """
        ret = empty_like(self, dtype='bool')
        if isinstance(other, gndarray):
            ret.lndarray_proxy.rank_view_n[...] = \
                (self.lndarray_proxy.rank_view_n[...] == other.lndarray_proxy.rank_view_n[...])
        else:
            ret.lndarray_proxy.rank_view_n[...] = \
                (self.lndarray_proxy.rank_view_n[...] == other)

        return ret

    @property
    def this_locale(self):
        return self._comms_and_distrib.this_locale

    @property
    def locale_comms(self):
        return self._comms_and_distrib.locale_comms

    @property
    def distribution(self):
        return self._comms_and_distrib.distribution

    @property
    def comms_and_distrib(self):
        return self._comms_and_distrib

    @property
    def rma_window_buffer(self):
        return self._rma_window_buffer

    @property
    def lndarray_proxy(self):
        return self._lndarray_proxy

    @property
    def shape(self):
        return self._comms_and_distrib.distribution.globale_extent.shape_n

    @property
    def dtype(self):
        return self._lndarray_proxy.dtype

    @property
    def order(self):
        return self._lndarray_proxy.md.order

    @property
    def rank_view_n(self):
        return self._lndarray_proxy.rank_view_n

    @property
    def rank_view_h(self):
        return self._lndarray_proxy.rank_view_h

    @property
    def rank_logger(self):
        """
        """
        return self._comms_and_distrib.locale_comms.rank_logger

    @property
    def root_logger(self):
        """
        """
        return self._comms_and_distrib.locale_comms.root_logger

    def intra_locale_barrier(self):
        """
        """
        self.rank_logger.debug(
            "BEG: self.comms_and_distrib.locale_comms.intra_locale_comm.barrier()..."
        )
        self.comms_and_distrib.locale_comms.intra_locale_comm.barrier()
        self.rank_logger.debug(
            "END: self.comms_and_distrib.locale_comms.intra_locale_comm.barrier()."
        )

    @property
    def halo_updater(self):
        if self._halo_updater is None:
            self._halo_updater = \
                PerAxisRmaHaloUpdater(
                    locale_extents=self.distribution.locale_extents,
                    dtype=self.dtype,
                    order=self.order,
                    inter_locale_win=self.rma_window_buffer.inter_locale_win,
                    dst_buffer=self.lndarray_proxy.lndarray
                )
            self._halo_updater.rank_logger = self.rank_logger
            self._halo_updater.root_logger = self.root_logger

        return self._halo_updater

    def update(self):
        """
        """
        # If running on single locale then there are no halos to update.
        if self.comms_and_distrib.locale_comms.num_locales > 1:
            rank_logger = self.comms_and_distrib.locale_comms.rank_logger
            # Only communicate data between the ranks
            # of self.comms_and_distrib.locale_comms.inter_locale_comm
            if (
                self.comms_and_distrib.locale_comms.have_valid_inter_locale_comm
            ):
                rank_logger.debug(
                    "BEG: update_halos..."
                )
                self.halo_updater.update_halos()
                rank_logger.debug(
                    "END: update_halos."
                )
            self.intra_locale_barrier()

    def calculate_copyfrom_updates(self, src, casting="same_kind"):
        return \
            RmaRedistributeUpdater(
                self,
                src,
                casting
            )

    def copyfrom(self, src, casting="same_kind"):
        """
        Copy the elements of the :samp:`{src}` array to corresponding elements of
        the :samp:`{self}` array.

        :type src: :obj:`gndarray`
        :param src: Global array from which elements are copied.
        :type casting: :obj:`str`
        :param casting: See :samp:`{casting}` parameter in :func:`numpy.copyto`.
        """

        if not isinstance(src, gndarray):
            raise ValueError(
                "Got type(src)=%s, expected %s." % (type(src), gndarray)
            )

        redistribute_updater = self.calculate_copyfrom_updates(src, casting)
        self.rank_logger.debug("BEG: redistribute_updater.barrier()...")
        redistribute_updater.barrier()
        self.rank_logger.debug("END: redistribute_updater.barrier().")
        redistribute_updater.do_locale_update()
        self.rank_logger.debug("BEG: redistribute_updater.barrier()...")
        redistribute_updater.barrier()
        self.rank_logger.debug("END: redistribute_updater.barrier().")

    def all(self, **unused_kwargs):
        return \
            self.locale_comms.peer_comm.allreduce(
                bool(self.lndarray_proxy.rank_view_n.all()),
                op=_mpi.BAND
            )

    def fill(self, value):
        """
        Fill the array (excluding ghost elements) with a scalar value.

        :type value: scalar
        :param value: All non-ghost elements will be assigned this value.
        """
        self.lndarray_proxy.fill(value)
        self.intra_locale_barrier()

    def fill_h(self, value):
        """
        Fill all array elements (including ghost elements) with a scalar value.

        :type value: scalar
        :param value: All elements will be assigned this value.
        """
        self.lndarray_proxy.fill_h(value)
        self.intra_locale_barrier()

    def copy(self):
        ary_out = empty_like(self)
        ary_out.lndarray_proxy.rank_view_partition_h[
            ...] = self.lndarray_proxy.rank_view_partition_h[...]
        self.intra_locale_barrier()

        return ary_out


def empty(
    shape=None,
    dtype="float64",
    comms_and_distrib=None,
    order='C',
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
    :rtype: :obj:`gndarray`
    :return: Newly created array with uninitialised elements.
    """
    if comms_and_distrib is None:
        comms_and_distrib = create_distribution(shape, **kwargs)
    lndarray_proxy, rma_window_buffer = \
        _locale.empty(
            comms_and_distrib=comms_and_distrib,
            dtype=dtype,
            order=order,
            return_rma_window_buffer=True,
            intra_partition_dims=intra_partition_dims
        )
    ary = \
        gndarray(
            comms_and_distrib=comms_and_distrib,
            rma_window_buffer=rma_window_buffer,
            lndarray_proxy=lndarray_proxy
        )

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
    if (isinstance(ary, gndarray)):
        ret_ary = \
            empty(
                dtype=ary.dtype,
                comms_and_distrib=ary.comms_and_distrib,
                order=ary.order,
                intra_partition_dims=ary.lndarray_proxy.intra_partition_dims
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
    :rtype: :obj:`gndarray`
    :return: Newly created array with zero-initialised elements.
    """
    ary = empty(shape, dtype=dtype, comms_and_distrib=comms_and_distrib, order=order, **kwargs)
    ary.fill_h(ary.dtype.type(0))

    return ary


def zeros_like(ary, *args, **kwargs):
    """
    Return a new zero-initialised array with the same shape and type as a given array.

    :type ary: :obj:`gndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`gndarray`
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
    :rtype: :obj:`gndarray`
    :return: Newly created array with one-initialised elements.
    """
    ary = empty(shape, dtype=dtype, comms_and_distrib=comms_and_distrib, order=order, **kwargs)
    ary.fill_h(ary.dtype.type(1))

    return ary


def ones_like(ary, *args, **kwargs):
    """
    Return a new one-initialised array with the same shape and type as a given array.

    :type ary: :obj:`gndarray`
    :param ary: Copy attributes from this array.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Specifies different dtype for the returned array.
    :rtype: :obj:`gndarray`
    :return: Array of one-initialized data with the same shape and type as :samp:`{ary}`.
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.fill_h(ary.dtype.type(1))

    return ary


def copy(ary, **kwargs):
    """
    Return an array copy of the given object.

    :type ary: :obj:`gndarray`
    :param ary: Array to copy.
    :rtype: :obj:`gndarray`
    :return: A copy of :samp:`ary`.
    """
    return ary.copy()


def copyto(dst, src, casting="same_kind", **kwargs):
    """
    Copy the elements of the :samp:`{src}` array to corresponding elements of
    the :samp:`dst` array.

    :type dst: :obj:`gndarray`
    :param dst: Global array which receives elements.
    :type src: :obj:`gndarray`
    :param src: Global array from which elements are copied.
    :type casting: :obj:`str`
    :param casting: See :samp:`{casting}` parameter in :func:`numpy.copyto`.
    """
    if not isinstance(dst, gndarray):
        raise ValueError(
            "Got type(dst)=%s, expected %s." % (type(dst), gndarray)
        )

    dst.copyfrom(src, casting=casting)


__all__ = [s for s in dir() if not s.startswith('_')]
