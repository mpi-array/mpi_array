"""
=================================
The :mod:`mpi_array.comms` Module
=================================

MPI communicators and windows for locales.


Classes
=======

.. autosummary::
   :toctree: generated/


   LocaleCommsInfo - :obj:`LocaleComms` communicator related data members.
   LocaleComms - Intra-locale and inter-locale communicators.
   CartLocaleCommsInfo - :obj:`CartLocaleComms` communicator related data members.
   CartLocaleComms - Intra-locale and cartesian-inter-locale communicators.
   CommsAndDistribution - Tuple with :obj:`LocaleComms` and :obj:`Distribution`.
   ThisLocaleInfo - The inter_locale_comm inter_locale_rank and corresponding peer_comm peer_rank.
   RmaWindowBuffer - Container for locale array buffer and associated RMA windows.
   DeferredFrees - Frees MPI windows at points which all processes reach.
   WindowFree - Frees an MPI window, deferred when its owner is garbage collected.

Factory Functions
=================

.. autosummary::
   :toctree: generated/

   create_locale_comms_info - Creates a :obj:`LocaleCommsInfo` instance.
   get_locale_comms_info - Finds or creates a :obj:`LocaleCommsInfo` instance.
   create_cart_locale_comms_info - Creates a :obj:`CartLocaleCommsInfo` instance.
   get_cart_locale_comms_info - Finds or creates a :obj:`CartLocaleCommsInfo` instance.
   create_locale_comms - Factory function for creating :obj:`LocaleComms` instances.
   create_block_distribution - Factory function for creating :obj:`BlockPartition` instances.
   create_cloned_distribution - Factory function for creating :obj:`ClonedDistribution` instances.
   create_single_locale_distribution - Creating :obj:`SingleLocaleDistribution` instances.
   create_distribution - Factory function for creating :obj:`Distribution` instances.

Functions
=========

.. autosummary::
   :toctree: generated/

   agree_on_error - Processes agree on an error, so that they all raise.
   flush_shared_memory_channels - Processes the messages between the processes of a node.
   get_shared_memory_comm - The communicator of the processes of a node.


Attributes
==========

.. autosummary::
   :toctree: generated/

   LT_PROCESS
   LT_NODE
   DT_BLOCK
   DT_SLAB
   DT_CLONED
   DT_SINGLE_LOCALE


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright, version as _version

import atexit as _atexit
import weakref as _weakref
import sys as _sys
import mpi4py.MPI as _mpi
import numpy as _np
import collections as _collections

import array_split as _array_split

from . import logging as _logging
from .distribution import BlockPartition, ClonedDistribution, SingleLocaleDistribution
from .utils import log_shared_memory_alloc as _log_shared_memory_alloc
from .utils import log_memory_alloc as _log_memory_alloc
from .utils import LruCache as _LruCache

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def mpi_version():
    """
    Return the MPI API version.

    :rtype: :obj:`int`
    :return: MPI major version number.
    """
    return _mpi.VERSION


class CommsCache(object):
    """
    Cache for re-using the MPI communicators (and associated info)
    when constructing :obj:`LocaleComms` (and :obj:`CartLocaleComms`)
    instances.
    """

    def __init__(self):
        """
        Construct cache.
        """
        self._lookup = []
        self._dict = dict()
        self._rank_logger = \
            _logging.get_rank_logger(str(__name__ + "." + self.__class__.__name__))

    @property
    def lookup(self):
        """
        The :obj:`list` of cached items. Each entry is
        a pair :samp:`(comms_tuple, obj)` where the *key* :samp:`comms_tuple` is
        a sequence of :obj:`mpi4py.MPI.Comm` objects and :samp:`obj` is
        the *value* associated with this *key*.
        """
        return self._lookup

    def clear(self):
        """
        Clears the cache.
        """
        self._lookup = []
        self._dict = dict()

    def create_hashable_key(self, key):
        """
        Returns a hashable key which is equal for keys whose elements are all equal
        (see :meth:`create_key_equality_tuple`): :obj:`mpi4py.MPI.Comm` and :samp:`None`
        elements are identified by object identity, others by their values.

        :type key: :obj:`tuple`
        :param key: A look-up key.
        :rtype: :obj:`tuple`
        :return: The hashable key.
        """
        return \
            tuple(
                ("id", id(k))
                if (k is None) or isinstance(k, _mpi.Comm) else
                ("value", tuple(_np.asanyarray(k).ravel().tolist()))
                for k in key
            )

    def create_key_equality_tuple(self, key0, key1):
        """
        Returns a :obj:`tuple` of :obj:`bool` elements indicating
        the per-element equality of the sequence :samp:`{key0}`
        and sequence :samp:`{key1}`. Assumes that :samp:`len({key0}) == len({key1})`.

        :type key0: :obj:`tuple`
        :param key0: A look-up key.
        :type key1: :obj:`tuple`
        :param key1: A look-up key.
        :rtype: :obj:`tuple` of :obj:`bool` elements
        :return: Per element equality of :samp:`{key0}` and :samp:`{key1}` keys.
        """
        return \
            tuple(
                (
                    (
                        (
                            (key0[j] is None)
                            or
                            (key1[j] is None)
                            or
                            isinstance(key0[j], _mpi.Comm)
                            or
                            isinstance(key1[j], _mpi.Comm)
                        )
                        and
                        (key0[j] is key1[j])
                    )
                    or
                    (
                        (
                            (key0[j] is not None)
                            and
                            (key1[j] is not None)
                            and
                            (not isinstance(key0[j], _mpi.Comm))
                            and
                            (not isinstance(key1[j], _mpi.Comm))
                        )
                        and
                        (_np.asanyarray(key0[j]).size == _np.asanyarray(key1[j]).size)
                        and
                        _np.all(_np.asanyarray(key0[j]) == _np.asanyarray(key1[j]))
                    )
                )
                for j in range(len(key1))
            )

    def find(self, key):
        """
        Returns the *value* associated with the *key* :samp:`{key}`.

        :type key: sequence of :obj:`object`
        :param key: Key for look-up, typically a :samp:`tuple`
           of elements of type :obj:`mpi4py.MPI.Comm`, :samp:`None`, integer-scalar
           or integer-sequence.
        :rtype: :obj:`object` or :samp:`None`
        :return: The object associated with the key :samp:`{key}`,
           or :samp:`None` if the key :samp:`{key}` is not found.
        """
        value = None
        entry = self._dict.get(self.create_hashable_key(key), None)
        if entry is not None:
            current_key, value = entry
            if value.rank_logger.isEnabledFor(_logging.DEBUG):
                value.rank_logger.debug(
                    "key:\n%s matched existing key:\n%s, value=\n%s\n%s",
                    key, current_key, value, self.create_key_equality_tuple(key, current_key)
                )
        else:
            self._rank_logger.debug("No match for:\n%s", (key,))
        return value

    def add(self, key, value):
        """
        Adds a :samp:`({key}, {value})` pair to this cache.

        :type key: :obj:`tuple`
        :param key: Key for look-up.
        :type value: :obj:`object`
        :param value: Object associated with the :samp:`{key}` key.
        """
        existing_value = self.find(key)
        if existing_value is not None:
            raise ValueError(
                "The key %s already has an associated value=%s"
                %
                (key, existing_value)
            )
        self._lookup.append((key, value))
        self._dict[self.create_hashable_key(key)] = (key, value)


ThisLocaleInfo = _collections.namedtuple("ThisLocaleInfo", ["inter_locale_rank", "peer_rank"])
ThisLocaleInfo.__doc__ = \
    """
   Pair of communicator rank values :samp:`(inter_locale_rank, peer_rank)` which
   indicates that the rank :samp:`inter_locale_rank` of the :samp:`inter_locale_comm`
   communicator corresponds to the :samp:`peer_rank` rank of the :samp:`peer_comm`
   communicator.
   """
ThisLocaleInfo.inter_locale_rank.__doc__ = \
    "A :obj:`int` indicating the rank of :samp:`inter_locale_comm` communicator."
ThisLocaleInfo.peer_rank.__doc__ = \
    "A :obj:`int` indicating the rank of :samp:`peer_comm` communicator."


class RmaWindowBuffer(object):

    """
    Details of the buffer allocated on a locale.
    """

    def __init__(
        self,
        is_shared,
        shape,
        dtype,
        itemsize,
        intra_locale_win,
        peer_comm=None,
        inter_locale_comm=None,
        peer_win=None,
        inter_locale_win=None,
        root_logger=None,
        rank_logger=None
    ):
        object.__init__(self)
        self._is_shared = is_shared
        self._shape = shape
        self._dtype = dtype
        self._itemsize = itemsize
        self._intra_locale_root_rank = 0
        self._intra_locale_win = intra_locale_win
        self._peer_win = peer_win
        self._inter_locale_win = inter_locale_win
        self._peer_comm = peer_comm
        self._inter_locale_comm = inter_locale_comm
        self._root_logger = root_logger
        self._rank_logger = rank_logger
        self._registration = None
        if peer_comm is not None:
            self._registration = deferred_frees.register(self, peer_comm)
            # All the processes of the peer communicator create the buffer (see register):
            # its node communicator, used when the windows are freed (see free), is created
            # while this is collective over all of them.
            get_shared_memory_comm(peer_comm)

    def defer_free(self):
        """
        Defers the (collective) free of the windows to a point which all the processes
        of :attr:`peer_comm` reach (see :obj:`DeferredFrees`), unless they are freed:
        called when the owner of the buffer is garbage collected, which can happen at
        different points on different processes. No MPI calls, except that the windows
        of a single process are freed at once.
        """
        if getattr(self, "_registration", None) is not None:
            if self._peer_comm.size == 1:
                if can_free_in_del():
                    self.free()
            else:
                deferred_frees.defer(self, self._registration)

    def __del__(self):
        """
        Defers the free of the windows (:meth:`defer_free`).
        """
        self.defer_free()

    @property
    def root_logger(self):
        """
        A :obj:`logging.Logger` for logging messages from the *root* MPI
        process of :attr:`peer_comm`.
        """
        if self._root_logger is None:
            self._root_logger = \
                _logging.get_root_logger(
                    __name__ + "." + self.__class__.__name__,
                    comm=self._peer_comm
                )
        return self._root_logger

    @property
    def rank_logger(self):
        """
        A :obj:`logging.Logger` for logging messages from all rank MPI
        processes of :attr:`peer_comm`.
        """
        if self._rank_logger is None:
            self._rank_logger = \
                _logging.get_rank_logger(
                    __name__ + "." + self.__class__.__name__,
                    comm=self._peer_comm
                )
        return self._rank_logger

    @property
    def is_shared(self):
        """
        A :obj:`bool`, if :samp:`True` then buffer memory was allocated
        using :meth:`mpi4py.MPI.Win.Allocate_shared`, otherwise memory
        was allocated using :meth:`mpi4py.MPI.Win.Allocate`.
        """
        return self._is_shared

    @property
    def shape(self):
        """
        A sequence of :obj:`int` indicating the shape of the locale array.
        """
        return self._shape

    @property
    def dtype(self):
        """
        A :obj:`numpy.dtype` indicating the data type of elements
        stored in the array.
        """
        return self._dtype

    @property
    def itemsize(self):
        """
        An :obj:`int` indicating the number of bytes  per array element
        (same as :attr:`numpy.dtype.itemsize`).
        """
        return self._itemsize

    @property
    def peer_win(self):
        """
        The :obj:`mpi4py.MPI.Win` created from the :samp:`peer_comm` communicator
        which exposes :attr:`buffer` for inter-locale RMA access.
        """
        if self._peer_win is None:
            self._peer_win = self.create_peer_win()
        return self._peer_win

    @property
    def intra_locale_win(self):
        """
        The :obj:`mpi4py.MPI.Win` created from the :samp:`intra_locale_comm` communicator
        which exposes :attr:`buffer` for intra-locale RMA access.
        When :samp:`{intra_locale_win}.group.size > 1` then :attr:`buffer` was
        allocated as shared memory (using :meth:`mpi4py.MPI.Win.Allocate_shared`).
        """
        return self._intra_locale_win

    @property
    def inter_locale_win(self):
        """
        The :obj:`mpi4py.MPI.Win` created from the :samp:`inter_locale_comm` communicator
        which exposes :attr:`buffer` for inter-locale RMA access.
        """
        if self._inter_locale_win is None:
            self._inter_locale_win = self.create_inter_locale_win()
        return self._inter_locale_win

    @property
    def peer_comm(self):
        """
        The *peer* :obj:`mpi4py.MPI.Comm` MPI communicator.
        """
        return self._peer_comm

    @property
    def inter_locale_comm(self):
        """
        The *inter-locale* :obj:`mpi4py.MPI.Win` MPI communicator.
        """
        return self._inter_locale_comm

    @property
    def intra_locale_win_memory(self):
        """
        The memory allocated using one of :meth:`mpi4py.MPI.Win.Allocate`
        or :meth:`mpi4py.MPI.Win.Allocate_shared`. This memory is used to store
        elements of the globale array apportioned to a locale.
        """
        if self.is_shared:
            buffer, itemsize = self.intra_locale_win.Shared_query(0)
        else:
            buffer = get_win_memory(self.intra_locale_win)
        return buffer

    @property
    def buffer(self):
        """
        A :samp:`numpy.ndarray` with memory allocated using one of :meth:`mpi4py.MPI.Win.Allocate`
        or :meth:`mpi4py.MPI.Win.Allocate_shared` as the buffer.

        .. seealso: :attr:`intra_locale_win_memory`
        """
        return _np.frombuffer(self.intra_locale_win_memory, dtype='B')

    def create_inter_locale_win(self):
        """
        Creates a :obj:`mpi4py.MPI.Win` window (over the :attr:`inter_locale_comm`
        communicator) exposing the :attr:`buffer` memory.

        :rtype: :obj:`mpi4py.MPI.Win`
        :return: Newly created window exposing the :attr:`buffer` memory.
           Returns :attr:`mpi4py.MPI.WIN_NULL` on processes which
           have :attr:`inter_locale_comm` equal to :attr:`mpi4py.MPI.COMM_NULL`.
        """
        inter_locale_win = _mpi.WIN_NULL
        if (self._inter_locale_comm != _mpi.COMM_NULL) and (self._inter_locale_comm is not None):
            self.rank_logger.debug("BEG: Win.Create for self.inter_locale_comm")
            inter_locale_win = \
                _mpi.Win.Create(
                    self.intra_locale_win_memory,
                    self.itemsize,
                    comm=self._inter_locale_comm
                )
            self.rank_logger.debug("END: Win.Create for self.inter_locale_comm")
        return inter_locale_win

    def create_peer_win(self):
        """
        Creates a :obj:`mpi4py.MPI.Win` window (over the :attr:`peer_win`
        communicator) exposing the :attr:`buffer` memory.

        :rtype: :obj:`mpi4py.MPI.Win`
        :return: Newly created window exposing the :attr:`buffer` memory.
        """
        if self._inter_locale_comm != _mpi.COMM_NULL:
            peer_buffer = self.intra_locale_win_memory
            peer_buffer_nbytes = len(self.intra_locale_win_memory)
        else:
            peer_buffer_nbytes = 0
            # Zero-length buffer (Win.Create(None) fails with MPI_ERR_WIN
            # on some MPI implementations, e.g. Open MPI 5 with mpi4py 4).
            peer_buffer = bytearray(0)

        self.rank_logger.debug(
            "BEG: Win.Create for self.peer_comm, buffer.nbytes=%s...",
            peer_buffer_nbytes
        )
        peer_win = _mpi.Win.Create(peer_buffer, self.itemsize, comm=self._peer_comm)

        self.rank_logger.debug(
            "END: Win.Create for self.peer_comm, buffer.nbytes=%s...",
            peer_buffer_nbytes
        )
        return peer_win

    @property
    def peer_win_initialised(self):
        """
        Returns :samp:`True` if *peer* RMA window (:attr:`peer_win`) has been
        created. :samp:`False` otherwise.
        """
        return self._peer_win is not None

    @property
    def inter_locale_win_initialised(self):
        """
        Returns :samp:`True` if *inter-locale* RMA window (:attr:`inter_locale_win`)
        has been created. :samp:`False` otherwise.
        """
        return self._inter_locale_win is not None

    @property
    def windows_initialised(self):
        """
        Returns :samp:`True` if both *peer* RMA window (:attr:`peer_win`)
        and *inter-locale* RMA window (:attr:`inter_locale_win`) have been
        created. :samp:`False` otherwise.
        """
        return (self._peer_win is not None) and (self._inter_locale_win is not None)

    def initialise_windows(self):
        """
        Creates both of the :attr:`peer_win`
        and :attr:`inter_locale_win` :obj:`mpi4py.MPI.Win` objects.
        """
        return self.peer_win, self.inter_locale_win

    def free(self):
        """
        Free MPI windows and associated buffer memory (collective). The messages between
        the processes of a node are processed before each window of one-sided communication
        is freed (see :func:`flush_shared_memory_channels`): those between the processes of
        the inter-locale communicator, then those between all the processes of the peer
        communicator, on the node communicator of the peer communicator.
        """
        if self._registration is not None:
            deferred_frees.unregister(self._registration)
            self._registration = None
        if (self._inter_locale_win != _mpi.WIN_NULL) and (self._inter_locale_win is not None):
            flush_shared_memory_channels(self._peer_comm, self._inter_locale_comm)
            self._inter_locale_win.Free()
            self._inter_locale_win = _mpi.WIN_NULL
        if (self._peer_win != _mpi.WIN_NULL) and (self._peer_win is not None):
            flush_shared_memory_channels(self._peer_comm)
            self._peer_win.Free()
            self._peer_win = _mpi.WIN_NULL
        if self._intra_locale_win != _mpi.WIN_NULL:
            self._intra_locale_win.Free()
            self._intra_locale_win = _mpi.WIN_NULL


LocaleCommsInfo = \
    _collections.namedtuple(
        "LocaleCommsInfo",
        [
            "peer_comm",
            "intra_locale_comm",
            "inter_locale_comm",
            "num_locales",
            "peer_ranks_per_locale",
            "rank_logger",
            "root_logger"
        ]
    )
LocaleCommsInfo.__doc__ = \
    """
    Communicators associated with a locale.
    """

_comms_info_cache = CommsCache()


def create_locale_comms_info(
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None
):
    """
    Creates a :obj:`LocaleCommsInfo` associated with
    the specified communicators. Collective over :samp:`{peer_comm}`.
    """
    if peer_comm is None:
        peer_comm = _mpi.COMM_WORLD
    rank_logger = _logging.get_rank_logger(__name__ + "." + "create_locale_comms_info", peer_comm)
    if intra_locale_comm is None:
        rank_logger.debug(
            "BEG: Splitting peer_comm with peer_comm.Split_type(COMM_TYPE_SHARED, ...)"
        )
        intra_locale_comm = peer_comm.Split_type(_mpi.COMM_TYPE_SHARED, key=peer_comm.rank)
        rank_logger.debug(
            "END: Splitting peer_comm with peer_comm.Split_type(COMM_TYPE_SHARED, ...)"
        )

    if inter_locale_comm is None:
        color = _mpi.UNDEFINED
        if intra_locale_comm.rank == 0:
            color = 0
        rank_logger.debug("BEG: peer_comm.Split to create inter_locale_comm.")
        inter_locale_comm = peer_comm.Split(color, peer_comm.rank)
        rank_logger.debug("END: peer_comm.Split to create inter_locale_comm.")

    peer_ranks_per_locale = None
    num_locales = None

    bad_inter_rank_to_intra_rank = tuple()
    if inter_locale_comm != _mpi.COMM_NULL:
        if (
            _mpi.Group.Translate_ranks(
                inter_locale_comm.group,
                (inter_locale_comm.rank,),
                intra_locale_comm.group
            )[0]
            !=
            0
        ):
            # Collect the inconsistent inter-locale-rank to intra-locale-rank
            # translations so that all peer_comm ranks can raise a ValueError
            # (i.e. avoid only raising ValueError on some inter_locale_comm ranks)
            bad_inter_rank_to_intra_rank = \
                (
                    (
                        peer_comm.rank,
                        inter_locale_comm.rank,
                        _mpi.Group.Translate_ranks(
                            inter_locale_comm.group,
                            (inter_locale_comm.rank,),
                            intra_locale_comm.group
                        )[0]
                    ),
                )
        num_locales = inter_locale_comm.size
        peer_ranks_per_locale = \
            _np.array(
                _mpi.Group.Translate_ranks(
                    intra_locale_comm.group,
                    _np.arange(0, intra_locale_comm.size),
                    peer_comm.group
                ),
                dtype="int64"
            ).reshape((intra_locale_comm.size,))

        rank_logger.debug("BEG: inter_locale_comm.allgather for peer-ranks-per-locale...")
        all = \
            inter_locale_comm.allgather(
                (bad_inter_rank_to_intra_rank, peer_ranks_per_locale)
            )
        rank_logger.debug("END: inter_locale_comm.allgather for peer-ranks-per-locale.")

        all_bad = tuple(a[0] for a in all)
        all_prpl = tuple(a[1] for a in all)
        # Assign peer_ranks_per_locale as 1D array of object
        # in case intra_locale.comm.size is not the same on
        # all locales.
        peer_ranks_per_locale = \
            _np.ones((len(all_prpl),), dtype="object")
        for i in range(len(all_prpl)):
            peer_ranks_per_locale[i] = all_prpl[i]

        if (
            _np.all(
                tuple(
                    len(peer_ranks_per_locale[i]) == len(peer_ranks_per_locale[0])
                    for i in range(len(peer_ranks_per_locale))
                )
            )
        ):
            # If all locales have the same intra_locale_comm.size,
            # then just use a 2D numpy.ndarray of 'int64' elements
            # (rather than an 1D array of variable-length 'object' elements
            num_peer_ranks_per_locale = len(peer_ranks_per_locale[0])
            peer_ranks_per_locale = \
                _np.array(peer_ranks_per_locale.tolist(), dtype="int64")
            peer_ranks_per_locale = \
                peer_ranks_per_locale.reshape(
                    (num_locales, num_peer_ranks_per_locale)
                )

        bad_inter_rank_to_intra_rank = sum(all_bad, ())
        del all, all_bad, all_prpl

    rank_logger.debug("BEG: intra_locale_comm.bcast for peer-ranks-per-locale...")
    bad_inter_rank_to_intra_rank, peer_ranks_per_locale, num_locales = \
        intra_locale_comm.bcast(
            (bad_inter_rank_to_intra_rank, peer_ranks_per_locale, num_locales),
            0
        )
    rank_logger.debug("END: intra_locale_comm.bcast for peer-ranks-per-locale...")
    if len(bad_inter_rank_to_intra_rank) > 0:
        raise ValueError(
            "Got invalid inter_locale_comm rank to intra_locale_comm rank translation:\n"
            +
            "\n".join(
                (
                    (
                        "peer rank=%s, inter_locale_comm rank=%s translated to "
                        +
                        "intra_locale_comm rank=%s, should be intra_locale_comm rank=0"
                    )
                    %
                    bad_translation
                    for bad_translation in bad_inter_rank_to_intra_rank
                )
            )
        )

    if num_locales > peer_comm.size:
        raise ValueError(
            "Got number of locales=%s which is greater than peer_comm.size=%s"
            %
            (num_locales, peer_comm.size)
        )

    return \
        LocaleCommsInfo(
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm,
            num_locales=num_locales,
            peer_ranks_per_locale=peer_ranks_per_locale,
            rank_logger=_logging.get_rank_logger(
                __name__ + "." + LocaleComms.__name__,
                comm=peer_comm
            ),
            root_logger=_logging.get_root_logger(
                __name__ + "." + LocaleComms.__name__,
                comm=peer_comm
            )
        )


def get_locale_comms_info(
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None
):
    """
    Finds or creates a :obj:`LocaleCommsInfo` associated with
    the specified communicators. Collective over :samp:`{peer_comm}`.
    """
    key = (peer_comm, intra_locale_comm, inter_locale_comm)
    comms_info = _comms_info_cache.find(key)
    if comms_info is None:
        comms_info = \
            create_locale_comms_info(
                peer_comm=peer_comm,
                intra_locale_comm=intra_locale_comm,
                inter_locale_comm=inter_locale_comm
            )
        _comms_info_cache.add(key, comms_info)
        key = (comms_info.peer_comm, comms_info.intra_locale_comm, comms_info.inter_locale_comm)
        if _comms_info_cache.find(key) is None:
            _comms_info_cache.add(key, comms_info)

    return comms_info


def get_win_memory(win):
    """
    Returns the memory buffer associated with the specified :samp:`{win}` MPI window.

    :type win: :obj:`mpi4py.MPI.Win`
    :param win: Return memory for this window.
    """
    if hasattr(win, "memory"):
        buffer = win.memory
    else:
        buffer = win.tomemory()

    return buffer


#: Set at exit by :func:`_check_mpi4py_abort`, see :func:`can_free_in_del`.
_mpi4py_aborts = False


def _check_mpi4py_abort():
    """
    Called at exit (by an :mod:`atexit` function, before the interpreter shuts down and
    before mpi4py finalizes, or aborts, MPI). Records whether mpi4py is going to abort the
    MPI job: the program was launched with :samp:`python -m mpi4py` and it is exiting
    because of an unhandled exception.
    """
    global _mpi4py_aborts
    unhandled_exception = \
        (getattr(_sys, "last_exc", None) is not None) \
        or \
        (getattr(_sys, "last_value", None) is not None)
    main_spec = getattr(_sys.modules.get("__main__"), "__spec__", None)
    launched_with_mpi4py = \
        (main_spec is not None) and (main_spec.name in ("mpi4py.__main__", "mpi4py.run"))
    _mpi4py_aborts = unhandled_exception and launched_with_mpi4py


def can_free_in_del():
    """
    Returns :samp:`False` if MPI windows must not be freed: MPI is finalized, or mpi4py is
    going to abort the MPI job (launched with :samp:`python -m mpi4py`, and exiting because
    of an unhandled exception). The other processes may never make the collective call,
    and waiting for them would stop the abort.

    In other cases the windows which are still allocated at exit are freed
    (see :obj:`DeferredFrees`), they must not be left to :func:`mpi4py.MPI.Finalize`:
    MPI implementations (e.g. Open MPI 5) can free the windows in a different order on
    different processes, which deadlocks.

    :rtype: :obj:`bool`
    :return: :samp:`True` if windows can be freed.
    """
    return (not _mpi4py_aborts) and (not _mpi.Is_finalized())


#: The node communicators of :func:`get_shared_memory_comm`, by :samp:`id` of the
#: communicator (with a reference to it).
_shared_memory_comms = dict()

#: The ranks, in the node communicator of a communicator, of the processes of a
#: sub-communicator which share the memory of this process (see
#: :func:`flush_shared_memory_channels`), by :samp:`id` of both (with references to them).
_shared_memory_ranks = dict()


def get_shared_memory_comm(comm):
    """
    Returns the communicator of the processes of :samp:`{comm}` which share the memory of
    this process (node), a communicator of the library. It is created
    by :samp:`{comm}.Split_type(COMM_TYPE_SHARED)` when this is first called
    for :samp:`{comm}` (collective over :samp:`{comm}`), and cached.

    :type comm: :obj:`mpi4py.MPI.Comm`
    :param comm: The communicator.
    :rtype: :obj:`mpi4py.MPI.Comm`
    :return: The node communicator.
    """
    key = id(comm)
    entry = _shared_memory_comms.get(key, None)
    if entry is None:
        entry = (comm, comm.Split_type(_mpi.COMM_TYPE_SHARED, key=comm.rank))
        _shared_memory_comms[key] = entry
    return entry[1]


def flush_shared_memory_channels(comm, sub_comm=None):
    """
    Each process sends a message to, and receives one from, every other process
    of :samp:`{comm}` (or of :samp:`{sub_comm}`) which shares its memory (node), on the node
    communicator of :samp:`{comm}` (:func:`get_shared_memory_comm`). The messages which
    these processes sent each other before (e.g. those of one-sided communication) are then
    processed: shared memory transports deliver the messages of a pair of processes in order
    (a barrier doesn't involve every pair). Collective over :samp:`{comm}`, or
    over :samp:`{sub_comm}`, a communicator of some of the processes of :samp:`{comm}`: the
    node communicator of :samp:`{comm}` isn't created then (nothing is exchanged if it
    doesn't exist).

    Called before freeing the windows of one-sided communication. When the shared memory
    transport can't access the memory of other processes (e.g. without permission to
    trace them), Open MPI 5.0 emulates one-sided communication with messages, and messages
    which arrived after their window was freed crashed it (segmentation faults). No node
    communicator is split from a sub-communicator: with Open MPI 5.0.8 on a cluster,
    splitting the inter-locale communicators of arrays (which span several nodes) made
    :func:`mpi4py.MPI.Finalize` abort.

    :type comm: :obj:`mpi4py.MPI.Comm`
    :param comm: The communicator of the processes (the peer communicator of an array).
    :type sub_comm: :samp:`None` or :obj:`mpi4py.MPI.Comm`
    :param sub_comm: The communicator of a window of some of the processes of :samp:`{comm}`
       (e.g. the inter-locale communicator of an array).
    """
    if sub_comm is None:
        node_comm = get_shared_memory_comm(comm)
        ranks = range(node_comm.size)
    else:
        entry = _shared_memory_comms.get(id(comm), None)
        if entry is None:
            return
        node_comm = entry[1]
        key = (id(comm), id(sub_comm))
        entry = _shared_memory_ranks.get(key, None)
        if entry is None:
            sub_group = sub_comm.Get_group()
            node_group = node_comm.Get_group()
            try:
                translated = \
                    _mpi.Group.Translate_ranks(sub_group, list(range(sub_group.size)), node_group)
            finally:
                sub_group.Free()
                node_group.Free()
            entry = (comm, sub_comm, sorted(r for r in translated if r != _mpi.UNDEFINED))
            _shared_memory_ranks[key] = entry
        ranks = entry[2]
    size = len(ranks)
    if size > 1:
        index = ranks.index(node_comm.rank)
        buffers = _np.zeros((size,), dtype="B")
        requests = []
        for i in range(1, size):
            requests.append(
                node_comm.Irecv([buffers[i:i + 1], _mpi.BYTE], ranks[(index - i) % size])
            )
            requests.append(node_comm.Isend([buffers[0:1], _mpi.BYTE], ranks[(index + i) % size]))
        _mpi.Request.Waitall(requests)


class DeferredFrees(object):

    """
    Frees MPI windows at points which all the processes of their communicator reach,
    not when the objects which own them are garbage collected. Freeing a window is
    collective, and garbage collection can happen at different points on different
    processes: while an exception propagates on one process only, for objects in reference
    cycles, or for arrays kept alive by views on some processes only.

    The objects which own windows (an object with a :samp:`free` method which frees them,
    collective over the processes of a communicator) are registered when they are
    created (:meth:`register`). The processes of the communicator create them in the same
    order, so they have the same sequence number on all of them. When an object is garbage
    collected, it is deferred (:meth:`defer`, no MPI calls). When windows are created over
    the communicator (:meth:`note_allocation`), every :attr:`max_allocations` allocations
    or after :attr:`max_bytes` bytes, its processes free the objects which are deferred on
    all of them, in creation order (:meth:`free_agreed`, collective). At exit, the
    registered objects which haven't been freed, deferred or not, are freed in creation
    order (:meth:`free_all`), unless mpi4py is going to abort the MPI job
    (see :func:`can_free_in_del`).
    """

    def __init__(self, max_allocations=32, max_bytes=64 * 2 ** 20):
        """
        Initialise.

        :type max_allocations: :obj:`int`
        :param max_allocations: The processes agree on the frees at least every this many
           allocations over a communicator. An agreement synchronises the processes, the
           deferred windows stay allocated until then.
        :type max_bytes: :obj:`int`
        :param max_bytes: The processes agree on the frees when this many bytes (per process)
           have been allocated over a communicator since they last agreed.
        """
        self.max_allocations = max_allocations
        self.max_bytes = max_bytes
        #: The number of allocations and bytes over each communicator since the last agreement.
        self._allocations = dict()
        #: The communicators of the registered objects, by :samp:`id`.
        self._comms = dict()
        #: The next sequence number for each communicator.
        self._next_seqs = dict()
        #: The deferred objects of each communicator, by sequence number.
        self._deferred = dict()
        #: :samp:`(weakref, comm id, seq)` of the registered objects, in creation order.
        self._registered = _collections.OrderedDict()
        self._next_global_seq = 0

    def register(self, obj, comm):
        """
        Registers :samp:`{obj}`, whose windows are freed collectively over the processes
        of :samp:`{comm}` (all processes of the communicator call this in the same order).

        :type obj: :obj:`object`
        :param obj: The object, its :samp:`free` method frees its windows and
           calls :meth:`unregister`.
        :type comm: :obj:`mpi4py.MPI.Comm`
        :param comm: The communicator whose processes agree on freeing the object.
        :rtype: :obj:`tuple`
        :return: The identifier of the registration (for :meth:`defer` and :meth:`unregister`).
        """
        key = id(comm)
        self._comms[key] = comm
        seq = self._next_seqs.get(key, 0)
        self._next_seqs[key] = seq + 1
        global_seq = self._next_global_seq
        self._next_global_seq += 1
        self._registered[global_seq] = (_weakref.ref(obj), key, seq)
        return (key, seq, global_seq)

    def unregister(self, registration):
        """
        Forgets the object of :samp:`{registration}` (called when its windows are freed).
        """
        key, seq, global_seq = registration
        self._registered.pop(global_seq, None)
        deferred = self._deferred.get(key, None)
        if deferred is not None:
            deferred.pop(seq, None)

    def defer(self, obj, registration):
        """
        Defers freeing the windows of :samp:`{obj}`, which is being garbage collected
        (no MPI calls). It is kept until it is freed.
        """
        key, seq, global_seq = registration
        self._deferred.setdefault(key, dict())[seq] = obj

    def num_deferred(self, comm=None):
        """
        Returns the number of deferred objects (of :samp:`{comm}`, if given).
        """
        if comm is not None:
            return len(self._deferred.get(id(comm), ()))
        return sum(len(deferred) for deferred in self._deferred.values())

    def note_allocation(self, comm, nbytes):
        """
        Called when all the processes of :samp:`{comm}` allocate windows, :samp:`{nbytes}`
        (the same on all of them) bytes per process. Every :attr:`max_allocations`
        allocations, or when :attr:`max_bytes` bytes have been allocated since the processes
        last agreed, they free the objects which all of them have deferred
        (:meth:`free_agreed`, collective). The counts are the same on all processes,
        so they agree at the same allocations: an agreement synchronises the processes.

        :type comm: :obj:`mpi4py.MPI.Comm`
        :param comm: The communicator of the allocation.
        :type nbytes: :obj:`int`
        :param nbytes: The number of bytes, the same on all processes of :samp:`{comm}`.
        """
        if comm.size > 1:
            key = id(comm)
            self._comms[key] = comm
            count, total = self._allocations.get(key, (0, 0))
            count += 1
            total += nbytes
            if (count >= self.max_allocations) or (total >= self.max_bytes):
                self._allocations[key] = (0, 0)
                self.free_agreed(comm)
            else:
                self._allocations[key] = (count, total)

    def free_agreed(self, comm):
        """
        Collective over :samp:`{comm}`. Frees the objects of :samp:`{comm}` which are deferred
        on all of its processes, in creation order.
        """
        deferred = self._deferred.get(id(comm), dict())
        if comm.size > 1:
            agreed_seqs = comm.allreduce(set(deferred.keys()), op=_mpi.BAND)
        else:
            agreed_seqs = set(deferred.keys())
        for seq in sorted(agreed_seqs):
            obj = deferred.get(seq, None)
            if obj is not None:
                obj.free()

    def free_all(self):
        """
        Frees all the registered objects, deferred or not, in creation order. Called at
        exit (on all processes).
        """
        for global_seq in list(self._registered.keys()):
            entry = self._registered.get(global_seq, None)
            if entry is not None:
                ref, key, seq = entry
                obj = self._deferred.get(key, dict()).get(seq, None)
                if obj is None:
                    obj = ref()
                if obj is not None:
                    obj.free()
                else:
                    self._registered.pop(global_seq, None)


#: The windows whose free is deferred (see :obj:`DeferredFrees`).
deferred_frees = DeferredFrees()


def _at_exit():
    """
    An :mod:`atexit` function, it runs before the interpreter shuts down and before
    mpi4py finalizes (or aborts) MPI. Records whether mpi4py is going to abort the job
    (:func:`_check_mpi4py_abort`), otherwise frees the windows which are still allocated,
    in creation order (:meth:`DeferredFrees.free_all`).
    """
    _check_mpi4py_abort()
    if can_free_in_del():
        deferred_frees.free_all()


_atexit.register(_at_exit)


class WindowFree(object):

    """
    Frees an MPI window (collective over the processes of its communicator), registered
    with :obj:`DeferredFrees` so that it isn't freed when its owner is garbage collected.
    """

    def __init__(self, win, comm):
        """
        Initialise.

        :type win: :obj:`mpi4py.MPI.Win`
        :param win: The window.
        :type comm: :obj:`mpi4py.MPI.Comm`
        :param comm: The communicator of the window.
        """
        self._win = win
        self._comm = comm
        self._registration = deferred_frees.register(self, comm)

    def free(self):
        """
        Frees the window (collective), unless it has been freed.
        """
        if self._registration is not None:
            deferred_frees.unregister(self._registration)
            self._registration = None
            if (self._win is not None) and (self._win != _mpi.WIN_NULL):
                self._win.Free()
            self._win = None

    def defer(self):
        """
        Called when the owner of the window is garbage collected: defers the free
        (see :obj:`DeferredFrees`), a window of a single process is freed at once.
        """
        if self._registration is not None:
            if self._comm.size == 1:
                if can_free_in_del():
                    self.free()
            else:
                deferred_frees.defer(self, self._registration)


def agree_on_error(comm, error):
    """
    Collective over :samp:`{comm}` (it synchronises like a barrier). Returns the
    exception to raise on this process: :samp:`{error}` if it is not :samp:`None`, otherwise
    an exception like the error of the lowest rank process which has one (same built-in
    exception type, e.g. :obj:`ValueError` for a sub-class of it, and same message),
    or :samp:`None` if no process has an error. One reduction when no process has an error,
    and a broadcast of the error otherwise.

    :type comm: :obj:`mpi4py.MPI.Comm`
    :param comm: The processes which agree.
    :type error: :samp:`None` or :obj:`Exception`
    :param error: The error of this process.
    :rtype: :samp:`None` or :obj:`Exception`
    :return: The exception to raise.
    """
    error_rank = comm.allreduce(comm.size if error is None else comm.rank, op=_mpi.MIN)
    if error_rank < comm.size:
        error_info = None
        if comm.rank == error_rank:
            error_type = next(t for t in type(error).__mro__ if t.__module__ == "builtins")
            try:
                message = str(error)
            except Exception:
                message = "%s (no message)" % (type(error).__name__,)
            error_info = (error_type, message)
        error_info = comm.bcast(error_info, error_rank)
        if error is None:
            try:
                error = error_info[0](error_info[1])
            except Exception:
                error = RuntimeError(error_info[1])
    return error


class LocaleComms(object):

    """
    MPI communicators for inter and intra locale data exchange. There are three
    communicators:

    :attr:`peer_comm`
       Typically this is :attr:`mpi4py.MPI.COMM_WORLD`. It is the group
       of processes which operate on (perform computations on portions of) a globale array.
    :attr:`intra_locale_comm`
       Can be :attr:`mpi4py.MPI.COMM_SELF`, but is more typically
       the communicator returned
       by :samp:`self.peer_comm.Split_type(mpi4py.MPI.COMM_TYPE_SHARED, key=self.peer_comm.rank)`.
       It is the communicator passed to the :func:`mpi4py.MPI.Win.Allocate_shared` function
       which allocates shared-memory and creates a shared-memory :obj:`mpi4py.MPI.Win` window.
    :attr:`inter_locale_comm`
       Typically this communicator is formed by selecting a single process from each locale.
       This communicator (and associated `mpi4py.MPI.Win` window) is used to exchange
       data between locales.
    """

    def __init__(
        self,
        peer_comm=None,
        intra_locale_comm=None,
        inter_locale_comm=None,
        comms_info=None
    ):
        """
        Construct, this is a collective call over the :samp:`{peer_comm}` communcator.

        :type peer_comm: :obj:`mpi4py.MPI.Comm`
        :param peer_comm: Communicator which is split according to
           shared memory allocation (uses :meth:`mpi4py.MPI.Comm.Split_type`).
           If :samp:`None`, uses :attr:`mpi4py.MPI.COMM_WORLD`.
        :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param intra_locale_comm: Intra-locale communicator.
           Should be a subset of processes returned
           by :samp:`{peer_comm}.Split_type(mpi4py.MPI.COMM_TYPE_SHARED)`.
           If :samp:`None`, :samp:`{peer_comm}` is *split* into groups
           which can use a MPI window to allocate shared memory
           (i.e. in this case locale is a (possibly NUMA) node).
           Can also specify as :samp:`mpi4py.MPI.COMM_SELF`, in which case the
           locale is a single process and regular (non-shared) memory
           is not allocated in :meth:`alloc_locale_buffer`.
        :type inter_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param inter_locale_comm: Inter-locale communicator used to exchange
            data between different locales. If :samp:`None` then one process
            (the :samp:`{intra_locale_comm}.rank == 0`
            process) is selected from each locale to form the :samp:`{inter_locale_comm}`
            communicator group.
        :type comms_info: :obj:`LocaleCommsInfo`
        :param comms_info: Convenience construction without need for lookup
           via :func:`get_locale_comms_info`.
        """
        object.__init__(self)
        if comms_info is None:
            comms_info = \
                get_locale_comms_info(
                    peer_comm=peer_comm,
                    intra_locale_comm=intra_locale_comm,
                    inter_locale_comm=inter_locale_comm
                )
        self._peer_comm = comms_info.peer_comm
        self._intra_locale_comm = comms_info.intra_locale_comm
        self._inter_locale_comm = comms_info.inter_locale_comm
        self._num_locales = comms_info.num_locales
        self._peer_ranks_per_locale = comms_info.peer_ranks_per_locale
        self._rank_logger = comms_info.rank_logger
        self._root_logger = comms_info.root_logger

    def free(self):
        """
        """
        pass

    def alloc_locale_buffer(self, shape, dtype):
        """
        Collective over :attr:`intra_locale_comm`. Allocates a buffer which
        provides storage for the elements of the locale multi-dimensional array,
        using :meth:`mpi4py.MPI.Win.Allocate_shared` (or :meth:`mpi4py.MPI.Win.Allocate`
        when :samp:`{self}.intra_locale_comm.size == 1`). Locales can have different
        numbers of processes.

        :type shape: sequence of :obj:`int`
        :param shape: The shape of the locale array for which a buffer is allocated.
        :type dtype: :obj:`numpy.dtype`
        :param dtype: The array element type.
        :rtype: :obj:`RmaWindowBuffer`
        :returns: A :obj:`collections.namedtuple` containing allocated buffer
           and associated RMA MPI windows.
        """
        is_shared_alloc = False
        self.rank_logger.debug("BEG: alloc_locale_buffer")
        num_rank_bytes = 0
        dtype = _np.dtype(dtype)
        locale_shape = shape
        rank_shape = locale_shape
        if self.intra_locale_comm.rank == 0:
            num_rank_bytes = int(_np.prod(rank_shape) * dtype.itemsize)
        else:
            rank_shape = tuple(_np.zeros_like(rank_shape))
        if self.intra_locale_comm.size > 1:
            is_shared_alloc = True
            _log_shared_memory_alloc(
                self.rank_logger.debug, "BEG: ", num_rank_bytes, rank_shape, dtype
            )
            intra_locale_win = \
                _mpi.Win.Allocate_shared(
                    num_rank_bytes,
                    dtype.itemsize,
                    comm=self.intra_locale_comm
                )
            buffer, itemsize = intra_locale_win.Shared_query(0)
            _log_shared_memory_alloc(
                self.rank_logger.debug, "END: ", num_rank_bytes, rank_shape, dtype, buffer=buffer
            )
        else:
            _log_memory_alloc(
                self.rank_logger.debug, "BEG: ", num_rank_bytes, rank_shape, dtype
            )
            intra_locale_win = \
                _mpi.Win.Allocate(num_rank_bytes, dtype.itemsize, comm=self.intra_locale_comm)
            buffer = get_win_memory(intra_locale_win)

            itemsize = dtype.itemsize
            _log_memory_alloc(
                self.rank_logger.debug, "END: ", num_rank_bytes, rank_shape, dtype, buffer=buffer
            )

        self.rank_logger.debug("END: alloc_local_buffer")
        rma_windows = \
            RmaWindowBuffer(
                is_shared=is_shared_alloc,
                shape=locale_shape,
                dtype=dtype,
                itemsize=itemsize,
                intra_locale_win=intra_locale_win,
                peer_comm=self.peer_comm,
                inter_locale_comm=self.inter_locale_comm,
                root_logger=self.root_logger,
                rank_logger=self.rank_logger
            )
        return rma_windows

    @property
    def num_locales(self):
        """
        An :samp:`int` indicating the number of *locales* over which an array is distributed.
        """
        return self._num_locales

    @property
    def peer_comm(self):
        """
        A :obj:`mpi4py.MPI.Comm` which is super-set of the :attr:`intra_locale_comm`
        and :attr:`inter_locale_comm` communicators.
        """
        return self._peer_comm

    @property
    def intra_locale_comm(self):
        """
        A :obj:`mpi4py.MPI.Comm` object which defines the group of processes
        which can allocate (and access) MPI window shared memory
        (allocated via :meth:`mpi4py.MPI.Win.Allocate_shared` if available).
        """
        return self._intra_locale_comm

    @property
    def inter_locale_comm(self):
        """
        A :obj:`mpi4py.MPI.Comm` communicator defining the group of processes
        which exchange data between locales.
        """
        return self._inter_locale_comm

    @inter_locale_comm.setter
    def inter_locale_comm(self, inter_locale_comm):
        self._inter_locale_comm = inter_locale_comm

    @property
    def have_valid_inter_locale_comm(self):
        """
        Is :samp:`True` if this *peer rank* has :samp:`{self}.inter_locale_comm`
        which is not :samp:`None` and is not :obj:`mpi4py.MPI.COMM_NULL`.
        """
        return \
            (
                (self.inter_locale_comm is not None)
                and
                (self.inter_locale_comm != _mpi.COMM_NULL)
            )

    @property
    def inter_locale_rank_to_peer_rank_map(self):
        """
        Returns sequence, :samp:`m` say, of :obj:`int`
        where :samp:`m[inter_r]` is the *peer rank* of :samp:`self.peer_comm`
        which corresponds to the *inter-locale rank* :samp:`inter_r`
        of :samp:`self.inter_locale_comm`.

        :rtype: :samp:`None` or sequence of :obj:`int`
        :return: Sequence of length :samp:`self.inter_locale_comm.size` on
           ranks for which :samp:`self.have_valid_inter_locale_comm is True`, :samp:`None`
           otherwise.
        """
        m = None
        if self.have_valid_inter_locale_comm:
            m = \
                _mpi.Group.Translate_ranks(
                    self.inter_locale_comm.group,
                    range(0, self.inter_locale_comm.group.size),
                    self.peer_comm.group
                )
        return m

    @property
    def peer_ranks_per_locale(self):
        """
        A :obj:`numpy.ndarray` of shape :samp:`(self.num_locales, num_peer_ranks_per_locale)`
        with :obj:`numpy.int64` elements.
        The :samp:`self.peer_ranks_per_locale[inter_locale_rank]` sequence of elements
        are the :attr:`peer_comm` ranks which make up locale associated
        with :attr:`inter_locale_comm` rank :samp:`inter_locale_rank`.
        """
        return self._peer_ranks_per_locale

    @property
    def this_locale_rank_info(self):
        """
        A :obj:`ThisLocaleInfo` object. Indicates the :samp:`self.inter_locale_comm.rank`
        and `self.peer_comm.rank` on processes for
        which :samp:`self.have_valid_inter_locale_comm is True`.
        Is :attr:`mpi4py.MPI.UNDEFINED` on processes
        where :samp:`self.have_valid_inter_locale_comm is False`.
        """
        if self.have_valid_inter_locale_comm:
            i = ThisLocaleInfo(self.inter_locale_comm.rank, self.peer_comm.rank)
        elif self.inter_locale_comm is None:
            i = ThisLocaleInfo(0, 0)
        else:
            i = _mpi.UNDEFINED
        return i

    @property
    def rank_logger(self):
        """
        A :attr:`peer_comm` :obj:`logging.Logger`.
        """
        return self._rank_logger

    @property
    def root_logger(self):
        """
        A :attr:`peer_comm` :obj:`logging.Logger`.
        """
        return self._root_logger


CartLocaleCommsInfo = \
    _collections.namedtuple(
        "CartLocaleCommsInfo",
        LocaleCommsInfo._fields
        +
        (
            "dims",
            "cart_comm"
        )
    )
CartLocaleCommsInfo.__doc__ = \
    """
    Communicators associated with a cartesian topology locales.
    """


#: The results of :func:`get_cannonical_dims` (least recently used are removed).
_cannonical_dims_cache = _LruCache(max_size=256)


def get_cannonical_dims(num_locales, ndims=None, dims=None, shape=None):
    """
    Returns the block partition per axis for :samp:`{num_locales}` number
    of total number of blocks.

    :type num_locales: :obj:`int`
    :param num_locales: Total number of blocks in the partition.
    :type ndims: :obj:`int`
    :param ndims: The number of dimensions of the block partitioning.
    :type dims: sequence of :obj:`int`
    :param dims: Constraint for the partitioning, positive values indicate
       the number of blocks for the associated axis, zero or negative values
       are chosen (replaced) so that :samp:`num_locales` total number of blocks
       are formed.
    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: The shape of the partitioned array. If given, the chosen numbers of blocks
       are at most the lengths of their axes (so that no block is empty), when such
       a partitioning exists.
    :rtype: :obj:`numpy.ndarray`
    :return: An array of :obj:`int` such that the :samp:`numpy.product` of the
       returned array is equal to :samp:`{num_locales}`.

    Example::

       >>> get_cannonical_dims(16, ndims=2).tolist()
       [4, 4]
       >>> get_cannonical_dims(16, ndims=2, shape=(2, 1000)).tolist()
       [2, 8]
       >>> get_cannonical_dims(4, ndims=2, shape=(1, 1)).tolist()
       [2, 2]
    """
    if (ndims is None) and (dims is None):
        raise ValueError("Must specify one of dims or ndims, got both ndims=None and dims=None.")
    elif (ndims is not None) and (dims is not None) and (len(dims) != ndims):
        raise ValueError(
            "Length of dims (len(dims)=%s) not equal to ndims=%s." % (len(dims), ndims)
        )
    elif ndims is None:
        ndims = len(dims)

    if dims is None:
        dims = _np.zeros((ndims,), dtype="int64")

    cache_key = \
        (
            int(num_locales),
            tuple(int(d) for d in dims),
            None if shape is None else tuple(int(s) for s in shape)
        )
    cached_dims = _cannonical_dims_cache.get(cache_key)
    if cached_dims is not None:
        return cached_dims.copy()

    shape_dims = None
    if (shape is not None) and (len(shape) > 0):
        try:
            shape_dims = \
                _array_split.split.calculate_num_slices_per_axis(
                    dims,
                    num_locales,
                    max_slices_per_axis=_np.maximum(_np.asarray(shape, dtype="int64"), 1)
                )
        except ValueError:
            shape_dims = None
        if (shape_dims is not None) and (_np.prod(shape_dims) != num_locales):
            # The shape is too small for the number of blocks.
            shape_dims = None
    if shape_dims is not None:
        dims = shape_dims
    else:
        dims = \
            _array_split.split.calculate_num_slices_per_axis(
                dims,
                num_locales
            )
    _cannonical_dims_cache[cache_key] = dims.copy()

    return dims


def create_cart_locale_comms_info(
    ndims=None,
    dims=None,
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None,
    cart_comm=None
):
    """
    Creates a :obj:`CartLocaleCommsInfo` associated with
    the specified communicators. Collective over :samp:`{peer_comm}`.
    """
    comms_info = \
        get_locale_comms_info(
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm
        )
    dims = get_cannonical_dims(num_locales=comms_info.num_locales, ndims=ndims, dims=dims)
    periods = [False] * len(dims)
    inter_locale_comm = comms_info.inter_locale_comm
    if (inter_locale_comm != _mpi.COMM_NULL) and (cart_comm is None):
        comms_info.rank_logger.debug("BEG: inter_locale_comm.Create_cart to create cart_comm.")
        cart_comm = \
            inter_locale_comm.Create_cart(
                dims,
                periods,
                reorder=True
            )
        comms_info.rank_logger.debug("END: inter_locale_comm.Create_cart to create cart_comm.")
    elif (inter_locale_comm == _mpi.COMM_NULL) and (cart_comm is None):
        cart_comm = _mpi.COMM_NULL
    if (
        (cart_comm != _mpi.COMM_NULL)
        and
        (cart_comm.group.size != comms_info.num_locales)
    ):
        raise ValueError(
            "Got cart_comm.group.size (=%s) != self._num_locales (=%s)."
            %
            (cart_comm.group.size, comms_info.num_locales)
        )

    return \
        CartLocaleCommsInfo(
            peer_comm=comms_info.peer_comm,
            intra_locale_comm=comms_info.intra_locale_comm,
            inter_locale_comm=comms_info.inter_locale_comm,
            num_locales=comms_info.num_locales,
            peer_ranks_per_locale=comms_info.peer_ranks_per_locale,
            rank_logger=comms_info.rank_logger,
            root_logger=comms_info.root_logger,
            dims=dims,
            cart_comm=cart_comm
        )


def get_cart_locale_comms_info(
    ndims=None,
    dims=None,
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None,
    cart_comm=None
):
    """
    Finds or creates a :obj:`CartLocaleCommsInfo` associated with
    the specified communicators. Collective over :samp:`{peer_comm}`.
    """
    key = (ndims, dims, peer_comm, intra_locale_comm, inter_locale_comm, cart_comm)
    comms_info = _comms_info_cache.find(key)
    if comms_info is None:
        comms_info = \
            create_cart_locale_comms_info(
                ndims,
                dims,
                peer_comm=peer_comm,
                intra_locale_comm=intra_locale_comm,
                inter_locale_comm=inter_locale_comm,
                cart_comm=cart_comm
            )
        _comms_info_cache.add(key, comms_info)
        key = \
            (
                len(comms_info.dims),
                comms_info.dims,
                comms_info.peer_comm,
                comms_info.intra_locale_comm,
                comms_info.inter_locale_comm,
                comms_info.cart_comm
            )
        if _comms_info_cache.find(key) is None:
            _comms_info_cache.add(key, comms_info)
        key = (None,) + key[1:]
        if _comms_info_cache.find(key) is None:
            _comms_info_cache.add(key, comms_info)

    return comms_info


class CartLocaleComms(LocaleComms):

    """
    Defines cartesian communication topology for locales.
    In addition to the :obj:`LocaleComms` communicators, defines:

    :attr:`cart_comm`
       Typically this communicator is created using
       the call :samp:`{inter_locale_comm}.Create_cart(...)`.
       This communicator (and associated `mpi4py.MPI.Win` window) is used to exchange
       data between locales.
       Note that :attr:`inter_locale_comm` property over-rides
       the :attr:`LocaleComms.inter_locale_comm` property to return the :attr:`cart_comm`
       communicator.
    """

    def __init__(
        self,
        ndims=None,
        dims=None,
        peer_comm=None,
        intra_locale_comm=None,
        inter_locale_comm=None,
        cart_comm=None
    ):
        """
        Initialises cartesian communicator for inter-locale data exchange.
        Need to specify at least one of the :samp:`{ndims}` or :samp:`{dims}`.
        to indicate the dimension of the cartesian partitioning.

        :type ndims: :obj:`int`
        :param ndims: Dimension of the cartesian partitioning, e.g. 1D, 2D, 3D, etc.
           If :samp:`None`, :samp:`{ndims}=len({dims})`.
        :type dims: sequence of :obj:`int`
        :param dims: The number of partitions along each array axis, zero elements
           are replaced with positive integers such
           that :samp:`numpy.prod({dims}) == {peer_comm}.size`.
           If :samp:`None`, :samp:`{dims} = (0,)*{ndims}`.
        :type peer_comm: :obj:`mpi4py.MPI.Comm`
        :param peer_comm: The MPI processes which will have access
           (via a :obj:`mpi4py.MPI.Win` object) to the distributed array.
           If :samp:`None` uses :obj:`mpi4py.MPI.COMM_WORLD`.
        :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param intra_locale_comm: The MPI communicator used to create a window which
            can be used to allocate shared memory
            via :meth:`mpi4py.MPI.Win.Allocate_shared`.
        :type inter_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param inter_locale_comm: Inter-locale communicator used to exchange
            data between different locales.
        :type cart_comm: :obj:`mpi4py.MPI.Comm`
        :param cart_comm: Cartesian topology inter-locale communicator used to exchange
            data between different locales.
        """
        comms_info = \
            get_cart_locale_comms_info(
                ndims=ndims,
                dims=dims,
                peer_comm=peer_comm,
                intra_locale_comm=intra_locale_comm,
                inter_locale_comm=inter_locale_comm,
                cart_comm=cart_comm
            )
        LocaleComms.__init__(
            self,
            comms_info=comms_info
        )

        self._dims = comms_info.dims
        self._cart_comm = comms_info.cart_comm
        self._peer_ranks_per_locale = self.calc_cart_peer_ranks_per_locale()

    def calc_cart_peer_ranks_per_locale(self):
        """
        Collective over :attr:`intra_locale_comm`. Returns the peer-ranks-per-locale table
        with rows in :attr:`cart_comm` rank order. The :obj:`LocaleComms` table is in the
        rank order of the inter-locale communicator from which :attr:`cart_comm` was
        created, but the :attr:`cart_comm` ranks can be ordered differently
        (:samp:`Create_cart(reorder=True)` is allowed to permute ranks, and :samp:`cart_comm`
        can be passed in), and block distribution locale extents are indexed by
        cart rank.

        Not done in the (cached) :func:`create_cart_locale_comms_info`, processes of a locale
        can get different cache hit/miss results (e.g. when :samp:`cart_comm` is only
        passed on inter-locale processes), so a collective there could deadlock.

        :rtype: :obj:`numpy.ndarray`
        :return: The table with :samp:`table[cart_rank]` the :attr:`peer_comm` ranks
           of the locale with :attr:`cart_comm` rank :samp:`cart_rank`.
        :raises ValueError: (on all :attr:`intra_locale_comm` processes) if
           the :attr:`cart_comm` group is not a permutation of the inter-locale group.
        """
        error = None
        peer_ranks_per_locale = self._peer_ranks_per_locale
        inter_locale_comm = self._inter_locale_comm
        cart_comm = self._cart_comm
        if (cart_comm is not None) and (cart_comm != _mpi.COMM_NULL):
            if (inter_locale_comm is None) or (inter_locale_comm == _mpi.COMM_NULL):
                error = "Got cart_comm on a process which is not in the inter_locale_comm group."
            else:
                inter_locale_ranks = \
                    _np.array(
                        _mpi.Group.Translate_ranks(
                            cart_comm.group,
                            range(cart_comm.group.size),
                            inter_locale_comm.group
                        ),
                        dtype="int64"
                    )
                if _np.any(inter_locale_ranks == _mpi.UNDEFINED):
                    error = \
                        "Got cart_comm with a group which differs from the inter_locale_comm group."
                else:
                    peer_ranks_per_locale = peer_ranks_per_locale[inter_locale_ranks]
        # intra_locale_comm rank 0 is the inter-locale (cart_comm) process, all processes
        # of the locale use its table, and all raise if any process has an error.
        gathered = self.intra_locale_comm.allgather((error, peer_ranks_per_locale))
        errors = tuple(g[0] for g in gathered if g[0] is not None)
        if len(errors) > 0:
            raise ValueError(errors[0])

        return gathered[0][1]

    @property
    def cart_coord_to_cart_rank_map(self):
        """
        A :obj:`dict` of :obj:`tuple`
        cartesian coordinate (:meth:`mpi4py.MPI.CartComm.Get_coords`) keys
        which map to the associated :attr:`cart_comm` peer_rank.
        """
        d = None
        if self.have_valid_cart_comm:
            d = _np.zeros(self.cart_comm.dims, dtype="int64")
            for cart_rank in range(self.cart_comm.size):
                d[tuple(self.cart_comm.Get_coords(cart_rank))] = cart_rank
        return d

    @property
    def dims(self):
        """
        The number of partitions along each array axis. Defines
        the cartesian topology over which an array is distributed.
        """
        return self._dims

    @property
    def ndim(self):
        """
        An :obj:`int` indicating the dimension of the cartesian topology.
        """
        return self._dims.size

    @property
    def have_valid_cart_comm(self):
        """
        Is :samp:`True` if this peer_rank has :samp:`{self}.cart_comm`
        which is not :samp:`None` and is not :obj:`mpi4py.MPI.COMM_NULL`.
        """
        return \
            (
                (self.cart_comm is not None)
                and
                (self.cart_comm != _mpi.COMM_NULL)
            )

    @property
    def cart_comm(self):
        """
        A :obj:`mpi4py.MPI.CartComm` communicator defining a cartesian topology of
        MPI processes (typically one process per locale) used for inter-locale
        exchange of array data.
        """
        return self._cart_comm

    @property
    def inter_locale_comm(self):
        """
        Overrides :attr:`LocaleComms.inter_locale_comm` to return :attr:`cart_comm`.
        Note that :attr:`peer_ranks_per_locale` is indexed by :attr:`cart_comm` rank.
        """
        return self.cart_comm


def shape_squeeze(shape):
    """
    Returns sequence with the the :samp:`1` elements removed.

    :type shape: sequence of :obj:`int`
    :param shape: A shape.
    :rtype: :obj:`numpy.ndarray`
    :return: The elements of :samp:`{shape}` which are not :samp:`1`.

    Example::

       >>> shape_squeeze((1, 5, 1, 7)).tolist()
       [5, 7]
    """
    shape = _np.asarray(shape, dtype="int64")
    return shape[shape != 1]


def reshape_comms_distribution(comms_distrib, new_globale_shape):
    """
    Returns the communicators and distribution for an array of
    shape :samp:`{new_globale_shape}` which is distributed like the array
    of :samp:`{comms_distrib}`. Only the same shape is implemented (the communicators
    and distribution are shared, a distribution is not modified after it is created),
    for a different shape (including added or removed axes of length one)
    returns :samp:`None`, the caller can then create a new distribution.

    :type comms_distrib: :obj:`CommsAndDistribution`
    :param comms_distrib: The communicators and distribution of an array.
    :type new_globale_shape: sequence of :obj:`int`
    :param new_globale_shape: The globale shape of the new array.
    :rtype: :obj:`CommsAndDistribution` or :samp:`None`
    :return: The :samp:`{comms_distrib}` communicators and distribution,
       or :samp:`None` if :samp:`{new_globale_shape}` differs from the shape
       of :samp:`{comms_distrib}`.
    """
    reshaped_comms_distrib = None
    distrib = comms_distrib.distribution
    globale_shape = tuple(int(s) for s in distrib.globale_extent.shape_n)
    if tuple(int(s) for s in new_globale_shape) == globale_shape:
        reshaped_comms_distrib = \
            CommsAndDistribution(
                comms_distrib.locale_comms,
                distrib,
                comms_distrib.this_locale
            )

    return reshaped_comms_distrib


#: Hyper-block partition distribution type
DT_BLOCK = "block"

#: Hyper-slab partition distribution type
DT_SLAB = "slab"

#: Entire array repeated on each locale.
DT_CLONED = "cloned"

#: Entire array on single locale, no array elements on other locales.
DT_SINGLE_LOCALE = "single_locale"

#: List of value :samp:`distrib_type` values.
_valid_distrib_types = [DT_BLOCK, DT_SLAB, DT_CLONED, DT_SINGLE_LOCALE]

#: Node (NUMA) locale type
LT_NODE = "node"

#: Single process locale type
LT_PROCESS = "process"

#: List of value :samp:`locale_type` values.
_valid_locale_types = [LT_NODE, LT_PROCESS]

CommsAndDistribution = \
    _collections.namedtuple("CommsAndDistribution", ["locale_comms", "distribution", "this_locale"])
CommsAndDistribution.__doc__ = \
    """
   A 3 element tuple :samp:`(locale_comms, distribution, this_locale)`
   describing the apportionment of array elements over MPI processes.
   """
CommsAndDistribution.locale_comms.__doc__ = \
    """
   A :obj:`LocaleComms` object containing communicators for exchanging
   data between locales.
   """
CommsAndDistribution.distribution.__doc__ = \
    """
   A :obj:`mpi_array.distribution.Distribution` object describing the
   apportionment of array elements over locales.
   """
CommsAndDistribution.this_locale.__doc__ = \
    """
   A :obj:`ThisLocaleInfo` with rank pair pertinent for this locale.
   """


def create_locale_comms(
    locale_type=None,
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None
):
    """
    Factory function for creating a :obj:`LocaleComms` object.

    :type locale_type: :obj:`str`
    :param locale_type: One of :attr:`mpi_array.comms.DT_PROCESS`
       or :attr:`mpi_array.comms.DT_NODE`.
    :type peer_comm: :obj:`mpi4py.MPI.Comm`
    :param peer_comm: See :obj:`LocaleComms`.
    :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
    :param intra_locale_comm: See :obj:`LocaleComms`.
    :type inter_locale_comm: :obj:`mpi4py.MPI.Comm`
    :param inter_locale_comm: See :obj:`LocaleComms`.
    :rtype: :obj:`LocaleComms`
    :return: A :obj:`LocaleComms` object.
    """
    if locale_type is None:
        locale_type = LT_NODE
    if locale_type.lower() == LT_PROCESS:
        if (intra_locale_comm is not None) and (intra_locale_comm.size > 1):
            raise ValueError(
                "Got locale_type=%s, but intra_locale_comm.size=%s"
                %
                (locale_type, intra_locale_comm.size)
            )
        elif intra_locale_comm is None:
            intra_locale_comm = _mpi.COMM_SELF
    locale_comms = \
        LocaleComms(
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm
        )
    inter_locale_rank_to_peer_rank = locale_comms.inter_locale_rank_to_peer_rank_map
    this_locale = locale_comms.this_locale_rank_info

    # Broadcast on intra_locale_comm to get peer_rank mapping to all
    # peer_comm ranks
    inter_locale_rank_to_peer_rank, this_locale = \
        locale_comms.intra_locale_comm.bcast(
            (inter_locale_rank_to_peer_rank, this_locale),
            0
        )
    locale_comms.rank_logger.debug(
        "inter_locale_rank_to_peer_rank=%s",
        inter_locale_rank_to_peer_rank
    )

    return locale_comms, inter_locale_rank_to_peer_rank, this_locale


def create_cloned_distribution(
    shape,
    locale_type=None,
    halo=0,
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None
):
    """
    Factory function for creating :obj:`mpi_array.distrbution.ClonedDistribution`
    distribution and associated :obj:`LocaleComms`.

    :rtype: :obj:`CommsAndDistribution`
    :return: A :obj:`CommsAndDistribution` pair.
    """
    locale_comms, inter_locale_rank_to_peer_rank, this_locale =\
        create_locale_comms(
            locale_type=locale_type,
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm
        )

    cloned_distrib = \
        ClonedDistribution(
            globale_extent=shape,
            inter_locale_rank_to_peer_rank=inter_locale_rank_to_peer_rank,
            num_locales=locale_comms.num_locales,
            halo=halo
        )
    cloned_distrib.peer_ranks_per_locale = locale_comms.peer_ranks_per_locale

    return CommsAndDistribution(locale_comms, cloned_distrib, this_locale)


def create_single_locale_distribution(
    shape,
    locale_type=None,
    halo=0,
    inter_locale_rank=0,
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None
):
    """
    Factory function for creating :obj:`mpi_array.distrbution.SingleLocaleDistribution`
    distribution and associated :obj:`LocaleComms`.

    :rtype: :obj:`CommsAndDistribution`
    :return: A :obj:`CommsAndDistribution` pair.
    """
    locale_comms, inter_locale_rank_to_peer_rank, this_locale =\
        create_locale_comms(
            locale_type=locale_type,
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm
        )

    single_locale_distrib = \
        SingleLocaleDistribution(
            globale_extent=shape,
            num_locales=locale_comms.num_locales,
            inter_locale_rank=inter_locale_rank,
            inter_locale_rank_to_peer_rank=inter_locale_rank_to_peer_rank,
            halo=halo
        )
    single_locale_distrib.peer_ranks_per_locale = locale_comms.peer_ranks_per_locale

    return CommsAndDistribution(locale_comms, single_locale_distrib, this_locale)


def create_block_distribution(
    shape,
    locale_type=None,
    dims=None,
    halo=0,
    peer_comm=None,
    intra_locale_comm=None,
    inter_locale_comm=None,
    cart_comm=None
):
    """
    Factory function for creating :obj:`mpi_array.distrbution.BlockPartition`
    distribution and associated :obj:`CartLocaleComms`.


    :type shape: sequence of :obj:`int`
    :param shape: Shape of the globale array.
    :type locale_type: :obj:`str`
    :param locale_type: One of :attr:`mpi_array.comms.DT_PROCESS`
       or :attr:`mpi_array.comms.DT_NODE`. Defines locales.
    :type dims: sequence of :obj:`int`
    :param dims: Defines the partitioning of the globale array axes.
    :type peer_comm: :obj:`mpi4py.MPI.Comm`
    :param peer_comm: See :obj:`LocaleComms`.
    :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
    :param intra_locale_comm: See :obj:`LocaleComms`.
    :type inter_locale_comm: :obj:`mpi4py.MPI.Comm`
    :param inter_locale_comm: See :obj:`LocaleComms`.
    :type cart_comm: :obj:`mpi4py.MPI.Comm`
    :param cart_comm: See :obj:`CartLocaleComms`.
    :rtype: :obj:`CommsAndDistribution`
    :return: A :obj:`CommsAndDistribution` :obj:`collections.namedtuple`.
    """
    if dims is None:
        dims = _np.zeros_like(shape, dtype="int64")
    locale_comms, inter_locale_rank_to_peer_rank, this_locale = \
        create_locale_comms(
            locale_type=locale_type,
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm
        )
    if cart_comm is None:
        # The same on all processes (the shape and the number of locales are).
        dims = get_cannonical_dims(locale_comms.num_locales, dims=dims, shape=shape)
    cart_locale_comms = \
        CartLocaleComms(
            dims=dims,
            peer_comm=locale_comms.peer_comm,
            intra_locale_comm=locale_comms.intra_locale_comm,
            inter_locale_comm=locale_comms.inter_locale_comm,
            cart_comm=cart_comm
        )
    cart_coord_to_cart_rank = cart_locale_comms.cart_coord_to_cart_rank_map
    cart_rank_to_peer_rank = cart_locale_comms.inter_locale_rank_to_peer_rank_map
    this_locale = cart_locale_comms.this_locale_rank_info

    # Broadcast on intra_locale_comm to get peer_rank mapping to all
    # peer_comm ranks
    cart_coord_to_cart_rank, cart_rank_to_peer_rank, this_locale = \
        cart_locale_comms.intra_locale_comm.bcast(
            (cart_coord_to_cart_rank, cart_rank_to_peer_rank, this_locale),
            0
        )
    cart_locale_comms.rank_logger.debug("cart_rank_to_peer_rank=%s", cart_rank_to_peer_rank)

    block_distrib = \
        BlockPartition(
            globale_extent=shape,
            dims=cart_locale_comms.dims,
            cart_coord_to_cart_rank=cart_coord_to_cart_rank,
            inter_locale_rank_to_peer_rank=cart_rank_to_peer_rank,
            halo=halo
        )
    block_distrib.peer_ranks_per_locale = cart_locale_comms.peer_ranks_per_locale

    return CommsAndDistribution(cart_locale_comms, block_distrib, this_locale)


def check_distrib_type(distrib_type):
    """
    Checks :samp:`{distrib_type}` occurs in :samp:`_valid_distrib_types`.

    :type distrib_type: :obj:`str`
    :param distrib_type: String to check.
    :raises ValueError: If :samp:`{distrib_type}` is not a valid *distribution type* specifier.

    """
    if distrib_type.lower() not in _valid_distrib_types:
        raise ValueError(
            "Invalid distrib_type=%s, valid types are: %s."
            %
            (
                distrib_type,
                ", ".join(_valid_distrib_types)
            )
        )


def check_locale_type(locale_type):
    """
    Checks :samp:`{locale_type}` occurs in :samp:`_valid_locale_types`.

    :type locale_type: :obj:`str`
    :param locale_type: String to check.
    :raises ValueError: If :samp:`{locale_type}` is not a valid *locale type* specifier.
    """
    if locale_type.lower() not in _valid_locale_types:
        raise ValueError(
            "Invalid locale_type=%s, valid types are: %s."
            %
            (
                locale_type,
                ", ".join(_valid_locale_types)
            )
        )


def create_distribution(shape, distrib_type=None, locale_type=LT_NODE, **kwargs):
    """
    Factory function for creating :obj:`mpi_array.distribution.Distribution`
    and associated :obj:`LocaleComms`.

    :type shape: sequence of :obj:`int`
    :param shape: Shape of the globale array.
    :type distrib_type: :obj:`str`
    :param distrib_type: One
       of :attr:`mpi_array.comms.DT_BLOCK` or :attr:`mpi_array.comms.DT_SLAB`
       or :attr:`mpi_array.comms.DT_CLONED` or :attr:`mpi_array.comms.DT_SINGLE_LOCALE`.
       Defines how the globale array is dstributed over locales.
       If :samp:`None` defaults to :attr:`mpi_array.comms.DT_BLOCK`
       if :samp:`numpy.prod(shape) > 0` otherwise :attr:`mpi_array.comms.DT_CLONED`.
    :type locale_type: :obj:`str`
    :param locale_type: One of :attr:`mpi_array.comms.DT_PROCESS`
       or :attr:`mpi_array.comms.DT_NODE`. Defines locales.
    :type dims: sequence of :obj:`int`
    :param dims: Only relevant when :samp:`{distrib_type} == DT_BLOCK`.
       Defines the partitioning of the globale array axes.
    :type axis: :obj:`int`
    :param axis: Only relevant when :samp:`{distrib_type} == DT_SLAB`.
       Indicates the single axis of the globale array partitioned into slabs.
    :type peer_comm: :obj:`mpi4py.MPI.Comm`
    :param peer_comm: See :obj:`LocaleComms`.
    :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
    :param intra_locale_comm: See :obj:`LocaleComms`.
    :type inter_locale_comm: :obj:`mpi4py.MPI.Comm`
    :param inter_locale_comm: See :obj:`LocaleComms`.
    :type cart_comm: :obj:`mpi4py.MPI.Comm`
    :param cart_comm: Only relevant when :samp:`{distrib_type} == DT_BLOCK`
        or :samp:`{distrib_type} == DT_SLAB`. See :obj:`CartLocaleComms`.
    :rtype: :obj:`CommsAndDistribution`
    :return: A :obj:`CommsAndDistribution` :obj:`collections.namedtuple`.

    See also:

       :func:`create_block_distribution`
       :func:`create_cloned_distribution`
       :func:`create_single_locale_distribution`
    """
    if distrib_type is None:
        if _np.prod(shape) > 0:
            distrib_type = DT_BLOCK
        else:
            distrib_type = DT_CLONED

    check_distrib_type(distrib_type)
    check_locale_type(locale_type)

    if distrib_type.lower() == DT_BLOCK:
        comms_and_distrib = create_block_distribution(shape, locale_type, **kwargs)
    elif distrib_type.lower() == DT_SLAB:
        if "axis" in kwargs.keys():
            axis = kwargs["axis"]
            del kwargs["axis"]
        else:
            axis = 0
        dims = _np.ones_like(shape, dtype="int64")
        dims[axis] = 0
        comms_and_distrib = create_block_distribution(shape, locale_type, dims=dims, **kwargs)
    elif distrib_type.lower() == DT_CLONED:
        comms_and_distrib = create_cloned_distribution(shape, locale_type, **kwargs)
    elif distrib_type.lower() == DT_SINGLE_LOCALE:
        comms_and_distrib = create_single_locale_distribution(shape, locale_type, **kwargs)

    return comms_and_distrib


__all__ = [s for s in dir() if not s.startswith('_')]
