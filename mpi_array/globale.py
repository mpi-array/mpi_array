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

Functions
=========

.. autosummary::
   :toctree: generated/

   copyto - Copy elements of one array to another array.


"""

from __future__ import absolute_import

import mpi4py.MPI as _mpi
import numpy as _np
from numpy.lib.mixins import NDArrayOperatorsMixin as _NDArrayOperatorsMixin

from .license import license as _license, copyright as _copyright, version as _version
from .update import UpdatesForRedistribute as _UpdatesForRedistribute
from .update import MpiUpdatesForGet as _MpiUpdatesForGet
from .update import MpiHalosUpdate as _MpiHalosUpdate
from .update import MpiPairExtentUpdate as _MpiPairExtentUpdate
from .update import MpiPairExtentUpdateDifferentDtypes as _MpiPairExtentUpdateDifferentDtypes
from .update import RmaUpdateExecutor as _RmaUpdateExecutor
from .locale import win_lndarray as _win_lndarray
from .distribution import LocaleExtent as _LocaleExtent
from .indexing import HaloIndexingExtent as _HaloIndexingExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

_builtin_slice = slice


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
    Helper class for performing halo data transfer using RMA
    via MPI windows (:obj:`mpi4py.MPI.Win` objects).
    """

    #: Halo "low index" indices.
    LO = _HaloIndexingExtent.LO

    #: Halo "high index" indices.
    HI = _HaloIndexingExtent.HI

    def __init__(self, locale_extents, dtype, order, inter_locale_win, dst_buffer):
        """
        Initialise.

        :type locale_extents: sequence of :obj:`mpi_array.distributon.LocaleExtent`
        :param locale_extents: :samp:`locale_extents[r]` is the extent of the array
           elements which reside on rank :samp:`r` of the :samp:`inter_locale_comm`
           communicator.
        :type dtype: :obj:`numpy.dtype`
        :param dtype: Data type of elements in array.
        :type order: :obj:`str`
        :param order: The array order, :samp:`'C'` for C memory layout.
        :type inter_locale_win: :obj:`mpi4py.MPI.Win`
        :param inter_locale_win: The window used to exchange halo element data.
        :type dst_buffer: :obj:`memoryview`
        :param dst_buffer: The buffer into which the halo elements are written.
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
        Sequence of :obj:`mpi_array.distribution.LocaleExtent` objects which
        define the partitioning of the array.
        """
        return self._locale_extents

    @property
    def dtype(self):
        """
        The :obj:`numpy.dtype` of the data to be exchanged in the halo update.
        """
        return self._dtype

    @property
    def order(self):
        """
        Array order :obj:`str`, :samp:`'C'` for C memory layout.
        """
        return self._order

    def calc_halo_updates(self):
        """
        Calculates the per-axis halo-region updates for
        all inter-locale ranks (of the :samp:`inter_locale_comm`).

        :rtype: :obj:`tuple` pair
        :return: A :samp:`(rank_2_updates_dict, bool_sequence)` pair
           where `rank_2_updates` is a :obj:`dict` of :samp:`{inter_locale_rank, halos_update}`,
           where :samp:`inter_locale_rank` is an :obj:`int` indicating the
           rank of the process in :samp:`inter_locale_comm` and :samp:`halos_update`
           is a :obj:`mpi_array.update.MpiHalosUpdate` containing the description
           of regions which are required to be fetched from remote processes.
           The :samp:`bool_sequence` is of length :attr:`ndim` and :samp:`bool_sequence[a] is True`
           indicates that halo updates are required on axis :samp:`a`.
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
    def dst_buffer(self):
        """
        A :obj:`memoryview` which provides the buffer
        into which the halo data is written.
        """
        return self._dst_buffer

    @property
    def halo_updates(self):
        """
        The :samp:`(rank_2_updates_dict, bool_sequence)` pair calculated
        by :meth:`calc_halo_updates`.
        """
        if self._halo_updates is None:
            self._halo_updates, self._have_axis_updates = self.calc_halo_updates()

        return self._halo_updates

    def update_halos(self):
        """
        Performs the data exchange required to update the halo (ghost)
        elements of the array buffer :attr:`dst_buffer`:samp:`.buffer`.
        Can be called :samp:`peer_comm` collectively.
        """
        self.do_update_halos(self.halo_updates)

    def do_update_halos(self, halo_updates):
        """
        Performs the data exchange required to update the halo (ghost)
        elements of the array buffer :attr:`dst_buffer`:samp:`.buffer`.
        Can be called :samp:`peer_comm` collectively.

        :type halo_updates: :obj:`mpi_array.update.MpiHalosUpdate`
        :param halo_updates: A :obj:`dict` of per :samp:`inter_locale_rank`
           halo region updates. See :meth:`calc_halo_updates`.
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
                    # is None on all inter_locale_comm ranks, and we avoid calling the Fence
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
                        self.rank_logger.debug(
                            "END: Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)..."
                        )
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
                        self.rank_logger.debug(
                            "BEG: Fence(_mpi.MODE_NOSUCCEED)."
                        )
                        self._inter_locale_win.Fence(_mpi.MODE_NOSUCCEED)
                        self.rank_logger.debug(
                            "END: Fence(_mpi.MODE_NOSUCCEED)."
                        )


class RankTranslator(object):

    """
    Translate ranks between two `mpi4py.MPI.Group` objects.
    """

    def __init__(self, dst_group, src_group):
        """
        """
        object.__init__(self)
        self._dst_group = dst_group
        self._src_group = src_group

    def dst_to_src(self, ranks):
        """
        Returns :samp:`mpi4py.MPI.Group.Translate_ranks(self.dst_group, ranks, self.src_group)`.
        """
        r = _np.array(ranks, copy=True)
        r.ravel()[...] = _mpi.Group.Translate_ranks(self.dst_group, r.ravel(), self.src_group)
        return r

    def src_to_dst(self, ranks):
        """
        Returns :samp:`mpi4py.MPI.Group.Translate_ranks(self.src_group, ranks, self.dst_group)`.
        """
        r = _np.array(ranks, copy=True)
        r.ravel()[...] = _mpi.Group.Translate_ranks(self.src_group, r.ravel(), self.dst_group)
        return r

    @property
    def dst_group(self):
        """
        A :obj:`mpi4py.MPI.Group`.
        """
        return self._dst_group

    @property
    def src_group(self):
        """
        A :obj:`mpi4py.MPI.Group`.
        """
        return self._src_group


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
        self._max_outstanding_requests = 32 * 32
        self._min_outstanding_requests_per_proc = 2
        self._max_ranks_per_inter_locale_sub_group = 128
        if self._dst.dtype != self._src.dtype:
            self._mpi_pair_extent_update_type = _MpiPairExtentUpdateDifferentDtypes

        _UpdatesForRedistribute.__init__(
            self,
            dst.comms_and_distrib.distribution,
            src.comms_and_distrib.distribution,
            peer_rank_translator=RankTranslator(
                self._dst.locale_comms.peer_comm.group,
                self._src.locale_comms.peer_comm.group
            )
        )
        self._inter_win = self._src.rma_window_buffer.peer_win
        self._max_outstanding_requests_per_proc = \
            _np.max(
                (
                    self._min_outstanding_requests_per_proc,
                    self._max_outstanding_requests // self._max_ranks_per_inter_locale_sub_group
                )
            )
        seed_str = str(2 ** 31)[1:]
        rank_str = str(self._inter_win.group.rank + 1)
        seed_str = rank_str + seed_str[len(rank_str):]
        seed_str = seed_str[0:-len(rank_str)] + rank_str[::-1]
        self._random_state = _np.random.RandomState(seed=int(seed_str))

    def calc_can_use_existing_src_peer_comm(self):
        """
        Returns :samp:`True` if :samp:`self._src.locale_comms.peer_comm`
        can be used to redistribute to the distribution of the :samp:`self._dst` array.

        :rtype: :obj:`bool`
        :return: :samp:`True` if :samp:`self._src.locale_comms.peer_comm` is a super-set
           of the processes of :samp:`self._dst.locale_comms.peer_comm`
        """
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

    def wait_all(self, req_list):
        """
        """
        self._dst.rank_logger.debug(
            "BEG: Waiting for outstanding rget requests, len(req_list)=%s...",
            len(req_list)
        )
        _mpi.Request.Waitall(req_list)
        self._dst.rank_logger.debug(
            "END: Waiting for outstanding rget requests, len(req_list)=%s.",
            len(req_list)
        )

    def do_locale_cpy2_update(self):
        """
        Performs direct copy updates.
        """
        updates = self._dst_cpy2_updates[self._dst.this_locale.inter_locale_rank]
        my_dst_peer_rank = self._dst.locale_comms.peer_comm.rank
        my_src_peer_rank = self._src.locale_comms.peer_comm.rank
        src_lndarray = self._src.lndarray_proxy.lndarray
        dst_lndarray = self._dst.lndarray_proxy.lndarray
        for update in updates:
            src_translated_peer_ranks = \
                self._src_translated_peer_ranks[update.src_extent.inter_locale_rank]
            dst_translated_peer_ranks = \
                self._dst_translated_peer_ranks[update.dst_extent.inter_locale_rank]
            if (
                (
                    (my_src_peer_rank == update.src_extent.peer_rank)
                    and
                    (update.src_extent.peer_rank in dst_translated_peer_ranks)
                )
                or
                (
                    (my_dst_peer_rank == update.dst_extent.peer_rank)
                    and
                    (update.dst_extent.peer_rank in src_translated_peer_ranks)
                )
            ):
                self._dst.rank_logger.debug(
                    "Copying update: mdpr=%s, mspr=%s\nsrc_t_ranks=%s\ndst_t_ranks=%s\n%s\n%s",
                    my_dst_peer_rank,
                    my_src_peer_rank,
                    src_translated_peer_ranks,
                    dst_translated_peer_ranks,
                    update._header_str,
                    update
                )
                update.copyto(dst_lndarray, src_lndarray, casting=self._casting)

    def do_locale_rma_update(self):
        """
        Performs RMA to get elements from remote locales to
        update the locale extent array.
        """
        can_use_existing_src_peer_comm = self.calc_can_use_existing_src_peer_comm()
        self._dst.rank_logger.debug(
            "%s.%s: "
            +
            "can_use_existing_src_peer_comm=%s",
            self.__class__.__name__,
            "do_locale_rma_update",
            can_use_existing_src_peer_comm
        )

        if can_use_existing_src_peer_comm:
            inter_win = _mpi.WIN_NULL
            if self._dst.locale_comms.have_valid_inter_locale_comm:
                inter_win = self._inter_win

            update_executor = \
                _RmaUpdateExecutor(
                    inter_win=inter_win,
                    dst_lndarray=self._dst.lndarray_proxy.lndarray,
                    src_inter_win_rank_attr="peer_rank",
                    rank_logger=self._dst.rank_logger
                )

            # Fetch remote data.
            updates = self._dst_rget_updates[self._dst.this_locale.inter_locale_rank]
            update_executor.do_locale_rma_update(updates)
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

    def do_locale_update(self):
        self.do_locale_rma_update()
        self.do_locale_cpy2_update()

    def do_update(self):
        self.barrier()

        self._dst.locale_comms.rank_logger.debug(
            "%s: BEG: do_locale_cpy2_update()...", self.__class__.__name__
        )
        self.do_locale_cpy2_update()
        self._dst.locale_comms.rank_logger.debug(
            "%s: END: do_locale_cpy2_update().", self.__class__.__name__
        )

        self.barrier()

        self._dst.locale_comms.rank_logger.debug(
            "%s: BEG: do_locale_rma_update()...", self.__class__.__name__
        )
        self.do_locale_rma_update()
        self._dst.locale_comms.rank_logger.debug(
            "%s: END: do_locale_rma_update().", self.__class__.__name__
        )

        self.barrier()

    def barrier(self):
        """
        MPI barrier.
        """
        self._dst.locale_comms.rank_logger.debug(
            "%s: BEG: self._src.locale_comms.peer_comm.barrier()...", self.__class__.__name__
        )
        self._src.locale_comms.peer_comm.barrier()
        self._dst.locale_comms.rank_logger.debug(
            "%s: END: self._src.locale_comms.peer_comm.barrier().", self.__class__.__name__
        )


class gndarray(_NDArrayOperatorsMixin):

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
        self = _NDArrayOperatorsMixin.__new__(cls)
        self._comms_and_distrib = comms_and_distrib
        self._rma_window_buffer = rma_window_buffer
        self._lndarray_proxy = lndarray_proxy
        self._halo_updater = None

        return self

    def free(self):
        """
        Collective (all samp:`peer_comm` processes) free of MPI windows (and locale array memory).
        """
        self._halo_updater = None
        if self._comms_and_distrib is not None:
            self._comms_and_distrib = None
        if self._lndarray_proxy is not None:
            self._lndarray_proxy.free()
            self._lndarray_proxy = None
        if self._rma_window_buffer is not None:
            self._rma_window_buffer.free()
            self._rma_window_buffer = None

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

    def __getitem__(self, i):
        """
        """
        self.rank_logger.debug("__getitem__: i=%s", i)
        return None

    def __setitem__(self, i, v):
        """
        """
        self.rank_logger.debug("__setitem__: i=%s, v=%s", i, v)

    def __array_ufunc__(self, *args, **kwargs):
        """
        """
        from . import globale_ufunc as _globale_ufunc
        return _globale_ufunc.gndarray_array_ufunc(self, *args, **kwargs)

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
    def ndim(self):
        return len(self.shape)

    @property
    def num_locales(self):
        """
        """
        return self._comms_and_distrib.locale_comms.num_locales

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
    def view_n(self):
        return self._lndarray_proxy.view_n

    @property
    def view_h(self):
        return self._lndarray_proxy.view_h

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

    def initialise_windows(self):
        """
        Creates the RMA windows required for inter-locale (and peer) one-sided RMA comms.
        """
        self.rma_window_buffer.initialise_windows()

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

    def inter_locale_barrier(self):
        """
        """
        if self.comms_and_distrib.locale_comms.have_valid_inter_locale_comm:
            self.rank_logger.debug(
                "BEG: self.comms_and_distrib.locale_comms.inter_locale_comm.barrier()..."
            )
            self.comms_and_distrib.locale_comms.inter_locale_comm.barrier()
            self.rank_logger.debug(
                "END: self.comms_and_distrib.locale_comms.inter_locale_comm.barrier()."
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
            self.comms_and_distrib.locale_comms.peer_comm.barrier()
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
        redistribute_updater.do_update()

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

    def copy(self, order='C'):
        from . import globale_creation as _globale_creation

        ary_out = _globale_creation.empty_like(self, order=order)
        ary_out.lndarray_proxy.rank_view_partition_h[...] = \
            self.lndarray_proxy.rank_view_partition_h[...]
        self.intra_locale_barrier()

        return ary_out

    def get_view(self, slice=None, start=None, stop=None, halo=0):
        """
        Returns :samp:`(ary, extent)` pair, where :samp:`ary` is a
        view from the locale extent array corresponding to the
        specified extent arguments. If any of the globale slice
        lies outside the locale extent, then :samp:`ary` is :samp:`None`.
        The :samp:`extent` element is a :obj:`mpi_array.distribution.LocaleExtent`
        instance which corresponds to the specified extent arguments.
        """
        if slice is not None:
            tmp = _np.array(list([s.start, s.stop] for s in slice))
            start = tmp[:, 0]
            stop = tmp[:, 1]

        # Create an extent object equivalent to the argument slice.
        locale_extent = self.lndarray_proxy.locale_extent
        dst_extent =\
            _LocaleExtent(
                peer_rank=locale_extent.peer_rank,
                inter_locale_rank=locale_extent.inter_locale_rank,
                start=start,
                stop=stop,
                slice=slice,
                globale_extent=self.distribution.globale_extent,
                halo=halo,
            )

        locale_ary = None
        if _np.all(
            _np.logical_and(
                dst_extent.start_h >= locale_extent.start_n,
                dst_extent.stop_h <= locale_extent.stop_n
            )
        ):
            # Can return a view of the locale array data
            shape = dst_extent.shape_h
            lstart = locale_extent.globale_to_locale_h(dst_extent.start_h)
            lstop = lstart + shape
            slc = tuple(_builtin_slice(lstart[a], lstop[a]) for a in range(locale_extent.ndim))
            locale_ary = self.lndarray_proxy.lndarray[slc]

        return locale_ary, dst_extent

    def reshape(self, shape):
        """
        Returns an array containing the same data with a new shape equal to :samp:`{shape}`.
        """
        raise NotImplementedError()

    def locale_get(self, slice=None, start=None, stop=None, halo=0):
        """
        Collective over :samp:`{self}.comms.intra_locale_comm` to
        get a portion of the globale array. Returns a view from the
        locale extent of the array if possible, otherwise allocates
        shared memory and performs one-sided RMA to fetch data from
        remote locales.
        """
        locale_ary, dst_extent = self.get_view(slice=slice, start=start, stop=stop, halo=halo)
        if locale_ary is None:
            # Need to fetch remote data

            if not self.rma_window_buffer.inter_locale_win_initialised:
                raise ValueError(
                    "Attempting inter-locale one-sided RMA without having created"
                    +
                    " the inter-locale window, call the initialise_windows method"
                    +
                    " (all *peer* ranks)"
                    +
                    " to create windows before performing one-sided RMA."
                )

            # Allocate (shared) memory for the data to be returned.
            locale_ary = \
                _win_lndarray(
                    shape=dst_extent.shape_h,
                    dtype=self.dtype,
                    comm=self.locale_comms.intra_locale_comm
                )

            if self.locale_comms.have_valid_inter_locale_comm:
                # Calculate the update objects which indicate where to fetch the data.
                update_calculator = \
                    _MpiUpdatesForGet(
                        dst_extent=dst_extent,
                        src_distrib=self.distribution,
                        dtype=self.dtype,
                        order=self.order,
                        update_dst_halo=True
                    )
                update_executor = \
                    _RmaUpdateExecutor(
                        inter_win=self.rma_window_buffer.inter_locale_win,
                        dst_lndarray=locale_ary,
                        src_inter_win_rank_attr="inter_locale_rank",
                        rank_logger=self.rank_logger
                    )
                # Perform the updates, copy locale array data to locale_ary first.
                updates = update_calculator._dst_cpy2_updates[dst_extent.inter_locale_rank]
                update_executor.do_direct_cpy2_update(updates, self.lndarray_proxy.lndarray)

                # Fetch remote data.
                updates = update_calculator._dst_rget_updates[dst_extent.inter_locale_rank]
                update_executor.do_locale_rma_update(updates)

            # All locale processes wait for data fetch to conclude
            self.intra_locale_barrier()

        return locale_ary

    def peer_rank_get(self, slice=None, start=None, stop=None, halo=0):
        """
        Non-collective, one-sided fetch of data to this peer rank process.
        Returns a view from the locale extent of the array if possible,
        otherwise allocates non-shared memory and performs one-sided RMA
        to fetch data from remote locales.
        """
        locale_ary, dst_extent = self.get_view(slice=slice, start=start, stop=stop, halo=halo)
        if locale_ary is None:
            # Need to fetch remote data

            if not self.rma_window_buffer.peer_win_initialised:
                raise ValueError(
                    "Attempting peer one-sided RMA without having created"
                    +
                    " the peer window, call the initialise_windows method (all *peer* ranks)"
                    +
                    " to create windows before performing one-sided RMA."
                )

            # Allocate memory for the data to be returned.
            locale_ary = \
                _win_lndarray(
                    shape=dst_extent.shape_h,
                    dtype=self.dtype,
                    comm=_mpi.COMM_SELF
                )

            update_calculator = \
                _MpiUpdatesForGet(
                    dst_extent=dst_extent,
                    src_distrib=self.distribution,
                    dtype=self.dtype,
                    order=self.order,
                    update_dst_halo=True
                )
            update_executor = \
                _RmaUpdateExecutor(
                    inter_win=self.rma_window_buffer.peer_win,
                    dst_lndarray=locale_ary,
                    src_inter_win_rank_attr="peer_rank",
                    rank_logger=self.rank_logger
                )
            # Perform the updates, copy locale array data to locale_ary first.
            updates = update_calculator._dst_cpy2_updates[dst_extent.inter_locale_rank]
            update_executor.do_direct_cpy2_update(updates, self.lndarray_proxy.lndarray)

            # Fetch remote data.
            updates = update_calculator._dst_rget_updates[dst_extent.inter_locale_rank]
            update_executor.do_locale_rma_update(updates)

        return locale_ary


def free_all(objects):
    """
    Call the :samp:`free` attribute on all arguments.

    :type objects: sequence of :obj:`object`
    :param objects: Call the :samp:`free` attribute for all objects in this
        sequence (if it exists and it is :obj:`callable`).
    """
    for obj in objects:
        if hasattr(obj, "free") and hasattr(obj.free, "__call__"):
            obj.free()


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
