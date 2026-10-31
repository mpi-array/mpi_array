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
   RedistributeUpdatesCache - Cache of the updates of copies between arrays.

Functions
=========

.. autosummary::
   :toctree: generated/

   copyto - Copy elements of one array to another array.


"""

from __future__ import absolute_import

import collections as _collections
import weakref as _weakref
import mpi4py.MPI as _mpi
import numpy as _np
from numpy.lib.mixins import NDArrayOperatorsMixin as _NDArrayOperatorsMixin

from .license import license as _license, copyright as _copyright, version as _version
from .update import MpiUpdatesForRedistribute as _MpiUpdatesForRedistribute
from .update import MpiUpdatesForGet as _MpiUpdatesForGet
from .update import MpiHalosUpdate as _MpiHalosUpdate
from .update import RmaUpdateExecutor as _RmaUpdateExecutor
from .locale import win_lndarray as _win_lndarray
from .comms import agree_on_error as _agree_on_error
from .distribution import LocaleExtent as _LocaleExtent
from .indexing import HaloIndexingExtent as _HaloIndexingExtent

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

_builtin_slice = slice


class CommLogger:

    """
    Has a logger for messages from all processes (:attr:`rank_logger`) and a logger
    for messages from the root process (:attr:`root_logger`).
    """

    def __init__(self, rank_logger=None, root_logger=None):
        self._rank_logger = rank_logger
        self._root_logger = root_logger

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

    def __init__(
        self,
        distribution,
        dtype,
        order,
        inter_locale_win,
        dst_buffer,
        inter_locale_rank
    ):
        """
        Initialise.

        :type distribution: :obj:`mpi_array.distribution.Distribution`
        :param distribution: The distribution of the array.
        :type dtype: :obj:`numpy.dtype`
        :param dtype: Data type of elements in array.
        :type order: :obj:`str`
        :param order: The array order, :samp:`'C'` for C memory layout.
        :type inter_locale_win: :obj:`mpi4py.MPI.Win`
        :param inter_locale_win: The window used to exchange halo element data.
        :type dst_buffer: :obj:`memoryview`
        :param dst_buffer: The buffer into which the halo elements are written.
        :type inter_locale_rank: :obj:`int`
        :param inter_locale_rank: The rank (of the :samp:`inter_locale_comm` communicator)
           of the locale whose halo elements are updated.
        """
        CommLogger.__init__(self)
        self._distribution = distribution
        self._dtype = dtype
        self._order = order
        self._inter_locale_win = inter_locale_win
        self._dst_buffer = dst_buffer
        self._inter_locale_rank = inter_locale_rank
        self._halo_updates = None
        self._have_axis_updates = None

    @property
    def distribution(self):
        """
        The :obj:`mpi_array.distribution.Distribution` of the array.
        """
        return self._distribution

    @property
    def locale_extents(self):
        """
        Sequence of :obj:`mpi_array.distribution.LocaleExtent` objects which
        define the partitioning of the array.
        """
        return self._distribution.locale_extents

    @property
    def inter_locale_rank(self):
        """
        The rank (of the :samp:`inter_locale_comm` communicator) of the locale whose
        halo elements are updated.
        """
        return self._inter_locale_rank

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

    def calc_have_axis_updates(self):
        """
        Returns the axes which have halo updates: those along which a block partition has
        more than one locale and which have a halo. It depends on the distribution only,
        all processes agree on the axes for which they call the (collective) fences without
        calculating the updates of the other locales. Other distributions (e.g. cloned)
        have no halo updates, the halos of their locale extents are empty.

        :rtype: :obj:`numpy.ndarray`
        :return: The :obj:`bool` array, :samp:`have_axis_updates[a]` is true when axis
           :samp:`a` has halo updates.
        """
        ndim = self._distribution.globale_extent.ndim
        have_axis_updates = _np.zeros((ndim,), dtype="bool")
        extent = self._distribution.get_extent_for_rank(self._inter_locale_rank)
        if hasattr(extent, "cart_shape"):
            have_axis_updates = \
                (_np.asarray(extent.cart_shape) > 1) \
                & \
                _np.any(_np.asarray(self._distribution.halo) > 0, axis=1)
        return have_axis_updates

    def calc_halo_updates(self):
        """
        Calculates the per-axis halo-region updates of the locale of :attr:`inter_locale_rank`
        (only, the updates of the other locales are not calculated).

        :rtype: :obj:`tuple` pair
        :return: A :samp:`(halos_update, bool_sequence)` pair where :samp:`halos_update`
           is a :obj:`mpi_array.update.MpiHalosUpdate` containing the description
           of regions which are required to be fetched from remote processes,
           :samp:`None` when no axis has halo updates. The :samp:`bool_sequence` is
           of length :attr:`ndim` and :samp:`bool_sequence[a] is True`
           indicates that halo updates are required on axis :samp:`a`
           (see :meth:`calc_have_axis_updates`), :samp:`halos_update.updates_per_axis[a]`
           is :samp:`None` for the other axes.
        """
        have_axis_updates = self.calc_have_axis_updates()
        halos_update = None
        if _np.any(have_axis_updates):
            distribution = self._distribution
            inter_locale_rank = self._inter_locale_rank
            cart_coord_to_extent = getattr(distribution, "get_extent_for_cart_coord", None)
            if cart_coord_to_extent is not None:
                rank_to_extents_map = \
                    {inter_locale_rank: distribution.get_extent_for_rank(inter_locale_rank)}
            else:
                rank_to_extents_map = distribution.locale_extents
            halos_update = \
                _MpiHalosUpdate(
                    inter_locale_rank,
                    rank_to_extents_map,
                    cart_coord_to_extent=cart_coord_to_extent
                )
            for a in range(len(have_axis_updates)):
                if not have_axis_updates[a]:
                    halos_update.updates_per_axis[a] = None

        return halos_update, have_axis_updates

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
        The :obj:`mpi_array.update.MpiHalosUpdate` (or :samp:`None`) calculated
        by :meth:`calc_halo_updates`.
        """
        if self._have_axis_updates is None:
            self._halo_updates, self._have_axis_updates = self.calc_halo_updates()

        return self._halo_updates

    def update_halos(self):
        """
        Performs the data exchange required to update the halo (ghost)
        elements of the array buffer :attr:`dst_buffer`:samp:`.buffer`.
        Collective over the :samp:`inter_locale_comm` processes.
        """
        self.do_update_halos(self.halo_updates)

    def do_update_halos(self, halo_updates):
        """
        Performs the data exchange required to update the halo (ghost)
        elements of the array buffer :attr:`dst_buffer`:samp:`.buffer`.
        Collective over the :samp:`inter_locale_comm` processes.

        :type halo_updates: :obj:`mpi_array.update.MpiHalosUpdate`
        :param halo_updates: The halo region updates of this locale (:samp:`None`
           on all processes when no axis has halo updates). See :meth:`calc_halo_updates`.
        """
        if halo_updates is not None:
            rank_inter_locale_updates = halo_updates
            # Get the updates separated into per-axis (hyper-slab) updates
            rank_updates_per_axis = rank_inter_locale_updates.updates_per_axis

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


class RedistributeUpdatesCache(object):

    """
    A least recently used cache of the updates calculated for copies between arrays
    (:obj:`RmaRedistributeUpdater`), so that repeated copies between arrays of the same
    distributions (the same objects), element types and orders don't calculate them
    again. The distributions are weakly referenced: the cache doesn't keep them alive,
    the entries of a distribution are removed when it is deleted, and an entry is never
    used for another distribution object.
    """

    def __init__(self, max_size=32):
        """
        Initialise.

        :type max_size: :obj:`int`
        :param max_size: The maximum number of entries, the least recently used entry
           is removed when there are more.
        """
        self._max_size = max_size
        self._entries = _collections.OrderedDict()

    def __len__(self):
        """
        The number of entries.
        """
        return len(self._entries)

    def clear(self):
        """
        Removes all entries.
        """
        self._entries.clear()

    def remove_entry(self, cache_key, ref):
        """
        Removes the entry of :samp:`{cache_key}` if it has the weak reference :samp:`{ref}`
        (called when the referenced distribution is deleted).
        """
        entry = self._entries.get(cache_key, None)
        if (entry is not None) and ((entry[0] is ref) or (entry[1] is ref)):
            del self._entries[cache_key]

    def get(self, dst_distrib, src_distrib, key, calc_updates):
        """
        Returns the updates for copies from arrays of the :samp:`{src_distrib}`
        distribution to arrays of the :samp:`{dst_distrib}` distribution. They are
        calculated (:samp:`{calc_updates}()`) and cached when they are not in the cache.

        :type dst_distrib: :obj:`mpi_array.distribution.Distribution`
        :param dst_distrib: The distribution of the array which receives the elements.
        :type src_distrib: :obj:`mpi_array.distribution.Distribution`
        :param src_distrib: The distribution of the array from which the elements are copied.
        :type key: :obj:`tuple`
        :param key: The other (hashable) parameters of the updates, e.g. the element types.
        :type calc_updates: callable
        :param calc_updates: Calculates the updates.
        :rtype: :obj:`object`
        :return: The updates.
        """
        cache_key = (id(dst_distrib), id(src_distrib)) + tuple(key)
        entry = self._entries.get(cache_key, None)
        if (entry is not None) and (entry[0]() is dst_distrib) and (entry[1]() is src_distrib):
            updates = entry[2]
        else:
            updates = calc_updates()

            def remove_entry(ref, cache=self, cache_key=cache_key):
                cache.remove_entry(cache_key, ref)

            entry = \
                (
                    _weakref.ref(dst_distrib, remove_entry),
                    _weakref.ref(src_distrib, remove_entry),
                    updates
                )
            self._entries[cache_key] = entry
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)
        self._entries.move_to_end(cache_key)
        return updates


#: The updates of copies between arrays.
_redistribute_updates_cache = RedistributeUpdatesCache()


class RmaRedistributeUpdater(object):

    """
    Helper class for redistributing array to new distribution.
    Copies the elements of the :samp:`{src}` array to the :samp:`{dst}` array
    (:meth:`do_update`): the elements of source locales which share processes
    with the destination locale are copied directly, the others are fetched from the
    remote source locales using one-sided RMA. The updates are calculated for the
    destination locale of this process only, and the processes of the locale share
    them (:attr:`updates`).
    """

    def __init__(self, dst, src, casting="same_kind"):
        """
        Initialise.

        :type dst: :obj:`gndarray`
        :param dst: Array which receives the elements.
        :type src: :obj:`gndarray`
        :param src: Array from which the elements are copied.
        :type casting: :obj:`str`
        :param casting: See :samp:`{casting}` parameter in :func:`numpy.copyto`.
        """
        self._dst = dst
        self._src = src
        self._casting = casting
        self._updates = None

    def calc_updates(self):
        """
        Calculates the updates of this process, its share of the updates of its destination
        locale (see :meth:`mpi_array.update.MpiUpdatesForRedistribute.create_process_updates`).

        :rtype: :obj:`mpi_array.update.ProcessUpdates`
        :return: The updates.
        :raises ValueError: If elements of the destination locale are not in the source
           distribution (on the processes of that locale only), or if processes of the
           destination array are not processes of the source array.
        """
        dst_group = self._dst.locale_comms.peer_comm.group
        src_group = self._src.locale_comms.peer_comm.group
        try:
            if _np.any(
                _np.array(
                    _mpi.Group.Translate_ranks(dst_group, range(dst_group.size), src_group)
                )
                ==
                _mpi.UNDEFINED
            ):
                raise ValueError(
                    "Got a destination array with processes which are not processes of the"
                    +
                    " source array."
                )
            locale_updates = \
                _MpiUpdatesForRedistribute(
                    self._dst.distribution,
                    self._src.distribution,
                    dst_dtype=self._dst.dtype,
                    src_dtype=self._src.dtype,
                    dst_order=self._dst.order,
                    src_order=self._src.order,
                    casting=self._casting,
                    peer_rank_translator=RankTranslator(dst_group, src_group),
                    dst_inter_locale_rank=self._dst.this_locale.inter_locale_rank
                )
        finally:
            dst_group.Free()
            src_group.Free()
        intra_locale_comm = self._dst.locale_comms.intra_locale_comm
        return \
            locale_updates.create_process_updates(
                intra_locale_rank=intra_locale_comm.rank,
                intra_locale_size=intra_locale_comm.size,
                src_inter_locale_rank=self._src.this_locale.inter_locale_rank,
                src_peer_rank=self._src.locale_comms.peer_comm.rank
            )

    @property
    def updates(self):
        """
        The :obj:`mpi_array.update.ProcessUpdates` updates of this process, calculated
        (:meth:`calc_updates`) when first used, unless they are in the cache
        (:obj:`RedistributeUpdatesCache`).
        """
        if self._updates is None:
            self._updates = \
                _redistribute_updates_cache.get(
                    self._dst.distribution,
                    self._src.distribution,
                    (
                        self._dst.dtype,
                        self._src.dtype,
                        self._dst.order,
                        self._src.order,
                        self._casting,
                        int(self._dst.this_locale.inter_locale_rank)
                    ),
                    self.calc_updates
                )
        return self._updates

    def do_locale_cpy2_update(self):
        """
        Performs the direct copy updates of this process.
        """
        src_lndarray = self._src.lndarray_proxy.lndarray
        dst_lndarray = self._dst.lndarray_proxy.lndarray
        for update in self.updates.cpy2_updates:
            self._dst.rank_logger.debug("Copying update:\n%s\n%s", update._header_str, update)
            update.copyto(dst_lndarray, src_lndarray, casting=self._casting)

    def do_locale_rma_update(self):
        """
        Performs RMA to get the elements of the updates of this process from remote
        locales (all processes of a locale fetch a share of its elements, into the locale
        array in shared memory). Collective over the :samp:`peer_comm` of the source array
        when its window is created.
        """
        # Created on all processes (collective).
        peer_win = self._src.rma_window_buffer.peer_win
        update_executor = \
            _RmaUpdateExecutor(
                inter_win=peer_win,
                dst_lndarray=self._dst.lndarray_proxy.lndarray,
                src_inter_win_rank_attr="peer_rank",
                rank_logger=self._dst.rank_logger,
                win_rank=self._src.locale_comms.peer_comm.rank
            )
        update_executor.do_locale_rma_update(self.updates.rget_updates)

    def do_update(self):
        """
        Copies the elements, collective over the :samp:`peer_comm` of the source array.
        The first barrier makes sure that the source elements have been written before
        they are read, the second that all processes have finished reading the source
        and writing the destination before any of them returns. The direct copies and
        the fetches read the source and write different regions of the destination,
        they don't need a barrier between them.

        :raises ValueError: (on all processes) If elements of the destination are not
           in the source distribution, or if processes of the destination array are not
           processes of the source array. Each process calculates the updates of its locale
           only, the processes agree on the error in the first barrier.
        """
        error = None
        try:
            self.updates
        except Exception as e:
            error = e
        error = self.barrier(error)
        if error is not None:
            try:
                raise error
            finally:
                # The frame shouldn't reference its own exception (a reference cycle).
                error = None

        self._dst.locale_comms.rank_logger.debug(
            "%s: BEG: do_locale_cpy2_update()...", self.__class__.__name__
        )
        self.do_locale_cpy2_update()
        self._dst.locale_comms.rank_logger.debug(
            "%s: END: do_locale_cpy2_update().", self.__class__.__name__
        )

        self._dst.locale_comms.rank_logger.debug(
            "%s: BEG: do_locale_rma_update()...", self.__class__.__name__
        )
        self.do_locale_rma_update()
        self._dst.locale_comms.rank_logger.debug(
            "%s: END: do_locale_rma_update().", self.__class__.__name__
        )

        self.barrier()

    def barrier(self, error=None):
        """
        Barrier, collective over the :samp:`peer_comm` of the source array, in which
        the processes agree on an error (see :func:`mpi_array.comms.agree_on_error`).

        :type error: :samp:`None` or :obj:`Exception`
        :param error: The error of this process.
        :rtype: :samp:`None` or :obj:`Exception`
        :return: The exception to raise on this process, :samp:`None` if no process
           had an error.
        """
        self._dst.locale_comms.rank_logger.debug(
            "%s: BEG: barrier of self._src.locale_comms.peer_comm...", self.__class__.__name__
        )
        error = _agree_on_error(self._src.locale_comms.peer_comm, error)
        self._dst.locale_comms.rank_logger.debug(
            "%s: END: barrier of self._src.locale_comms.peer_comm.", self.__class__.__name__
        )
        return error


def _check_extent_within(extent, globale_extent):
    """
    Raises :obj:`ValueError` if the elements of :samp:`{extent}` (including
    the halo) are not all within :samp:`{globale_extent}`. Called on all processes
    before a (collective) fetch, so that they all raise.
    """
    if (
        _np.any(extent.start_h < globale_extent.start_n)
        or
        _np.any(extent.stop_h > globale_extent.stop_n)
    ):
        raise ValueError(
            "Got region start=%s, stop=%s which is not within the array, start=%s, stop=%s."
            %
            (
                extent.start_h.tolist(),
                extent.stop_h.tolist(),
                globale_extent.start_n.tolist(),
                globale_extent.stop_n.tolist()
            )
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
        if lndarray_proxy is not None:
            # The views of the locale array keep this array, which frees the memory, alive.
            lndarray_proxy.set_memory_owner(self)

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
        Garbage collection defers the (collective) free of the MPI windows to a point which
        all processes reach (:meth:`mpi_array.comms.RmaWindowBuffer.defer_free`), it can
        happen at different points on different processes. Free explicitly (:meth:`free`
        or a :samp:`with` statement) to release the memory at once.
        """
        rma_window_buffer = getattr(self, "_rma_window_buffer", None)
        if rma_window_buffer is not None:
            rma_window_buffer.defer_free()

    def __enter__(self):
        """
        For use with :samp:`with` contexts.
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        For use with :samp:`with` contexts. Calls :meth:`free` (collective), except
        when an exception is propagating: the other processes may never reach the
        collective free, so it is skipped and the exception propagates without waiting.
        The array is then freed by an explicit :meth:`free`, or after it is garbage
        collected (see :meth:`__del__`).
        """
        if type is None:
            self.free()
        return False

    def __getitem__(self, i):
        """
        Not implemented yet, raises :obj:`NotImplementedError`.
        Use :meth:`locale_get` or :meth:`peer_rank_get` to fetch a region of the array.
        """
        raise NotImplementedError(
            "Indexing a gndarray (index=%r) is not implemented, "
            "use gndarray.locale_get or gndarray.peer_rank_get to fetch a region."
            %
            (i,)
        )

    def __setitem__(self, i, v):
        """
        Not implemented yet, raises :obj:`NotImplementedError`.
        Use :meth:`fill`, :func:`copyto` or a ufunc with the :samp:`out` argument
        to assign elements.
        """
        raise NotImplementedError(
            "Assigning to a gndarray index/slice (index=%r) is not implemented, "
            "use gndarray.fill, mpi_array.globale.copyto or a ufunc with out=."
            %
            (i,)
        )

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
        """
        The globale shape of the array, a :obj:`tuple` of :obj:`int`.
        """
        return tuple(int(s) for s in self._comms_and_distrib.distribution.globale_extent.shape_n)

    @property
    def dtype(self):
        return self._lndarray_proxy.dtype

    @property
    def order(self):
        return self._lndarray_proxy.md.order

    @property
    def view_n(self):
        """
        View of the locale array without the halo. Views, and the arrays (of their type)
        made from them, keep this array alive (its memory is not freed after garbage
        collection while they exist). Views are invalid after an explicit :meth:`free`.
        Views can be released at different points on different processes (e.g.
        :samp:`numpy.ascontiguousarray({view})` makes a view or a copy depending on the layout
        of the tile of a process): the memory is freed when all processes have released
        the array (see :obj:`mpi_array.comms.DeferredFrees`).
        """
        return self._lndarray_proxy.view_n

    @property
    def view_h(self):
        """
        View of the locale array including the halo, see :attr:`view_n`.
        """
        return self._lndarray_proxy.view_h

    @property
    def rank_view_n(self):
        """
        View of the tile of this process without the halo, see :attr:`view_n`.
        """
        return self._lndarray_proxy.rank_view_n

    @property
    def rank_view_h(self):
        """
        View of the tile of this process including the halo, see :attr:`view_n`.
        """
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
                    distribution=self.distribution,
                    dtype=self.dtype,
                    order=self.order,
                    inter_locale_win=self.rma_window_buffer.inter_locale_win,
                    dst_buffer=self.lndarray_proxy.lndarray,
                    inter_locale_rank=self.this_locale.inter_locale_rank
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
            locale_ary = self.lndarray_proxy[slc]

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
        shared memory (a :obj:`mpi_array.locale.win_lndarray`) and performs
        one-sided RMA to fetch data from remote locales. Free the allocated
        array with its :samp:`free` method (collective over the intra-locale
        communicator) when it is no longer needed. Raises :obj:`ValueError` if
        the region is not within the array.

        Either result keeps this array alive (its memory isn't freed after garbage
        collection while the result exists). A fetched result which is garbage collected
        is freed when all the processes of the locale have released it
        (see :obj:`mpi_array.comms.DeferredFrees`).

        Synchronisation is the caller's responsibility (the :obj:`numpy.ufunc`
        implementation in :mod:`mpi_array.globale_ufunc` does this):

        - before the fetch, the remote locales must have finished writing the requested
          elements, e.g. call :meth:`inter_locale_barrier` (all locales) after writing;
        - the remote locales must not overwrite the elements until every locale has
          finished fetching, e.g. call :meth:`inter_locale_barrier` and
          then :meth:`intra_locale_barrier` (all locales) after fetching,
          so that the other processes of each locale wait as well.
        """
        locale_ary, dst_extent = self.get_view(slice=slice, start=start, stop=stop, halo=halo)
        if locale_ary is None:
            # Need to fetch remote data
            _check_extent_within(dst_extent, self.distribution.globale_extent)

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
            locale_ary._fetched_from = self

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
                        rank_logger=self.rank_logger,
                        win_rank=self.locale_comms.inter_locale_comm.rank
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
        otherwise allocates non-shared memory (a :obj:`mpi_array.locale.win_lndarray`)
        and performs one-sided RMA to fetch data from remote locales. Free the
        allocated array with its :samp:`free` method when it is no longer needed.
        Raises :obj:`ValueError` if the region is not within the array. Either
        result keeps this array alive.

        Synchronisation is the caller's responsibility: the processes which write
        the requested elements must have finished before the fetch, and must not
        overwrite them until the fetch has finished, e.g. a :samp:`peer_comm`
        barrier (:samp:`{self}.locale_comms.peer_comm.barrier()`) after writing and
        after fetching.
        """
        locale_ary, dst_extent = self.get_view(slice=slice, start=start, stop=stop, halo=halo)
        if locale_ary is None:
            # Need to fetch remote data
            _check_extent_within(dst_extent, self.distribution.globale_extent)

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
            locale_ary._fetched_from = self

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
                    rank_logger=self.rank_logger,
                    win_rank=self.locale_comms.peer_comm.rank
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
