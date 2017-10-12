"""
=================================
The :mod:`mpi_array.comms` Module
=================================

MPI communicators and windows for locales.


Classes
=======

.. autosummary::
   :toctree: generated/


   LocaleComms - Intra-locale and inter-locale communicators.
   CartLocaleComms - Intra-locale and cartesian-inter-locale communicators.
   CommsAndDistribution - Tuple with :obj:`LocaleComms` and :obj:`Distribution`.
   ThisLocaleInfo - The inter_locale_comm inter_locale_rank and corresponding peer_comm peer_rank.
   RmaWindowBuffer - Container for locale array buffer and associated RMA windows.

Factory Functions
=================

.. autosummary::
   :toctree: generated/

   create_locale_comms - Factory function for creating :obj:`LocaleComms` instances.
   create_block_distribution - Factory function for creating :obj:`BlockPartition` instances.
   create_cloned_distribution - Factory function for creating :obj:`ClonedDistribution` instances.
   create_single_locale_distribution - Creating :obj:`SingleLocaleDistribution` instances.
   create_distribution - Factory function for creating :obj:`Distribution` instances.


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

import sys as _sys
import copy as _copy
import mpi4py.MPI as _mpi
import numpy as _np
import collections as _collections

import array_split as _array_split

from . import logging as _logging
from .distribution import BlockPartition, ClonedDistribution, SingleLocaleDistribution
from .utils import log_shared_memory_alloc as _log_shared_memory_alloc
from .utils import log_memory_alloc as _log_memory_alloc

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


ThisLocaleInfo = _collections.namedtuple("ThisLocaleInfo", ["inter_locale_rank", "peer_rank"])
if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
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


RmaWindowBufferTuple = \
    _collections.namedtuple(
        "RmaWindowBuffer",
        [
            "is_shared",
            "shape",
            "dtype",
            "itemsize",
            "peer_win",
            "intra_locale_win",
            "inter_locale_win"
        ]
    )


class RmaWindowBuffer(RmaWindowBufferTuple):

    __slots__ = ()

    @property
    def buffer(self):
        """
        The memory allocated using one of :meth:`mpi4py.MPI.Win.Allocate`
        or :meth:`mpi4py.MPI.Win.Allocate_shared`. This memory is used to store
        elements of the globale array apportioned to a locale.
        """
        if self.is_shared:
            buffer, itemsize = self.intra_locale_win.Shared_query(0)
        else:
            buffer = self.intra_locale_win.memory
        buffer = _np.array(buffer, dtype='B', copy=False)

        return buffer

    def free(self):
        """
        Free MPI windows and associated buffer memory.
        """
        if self.inter_locale_win != _mpi.WIN_NULL:
            self.inter_locale_win.Free()
        if self.peer_win != _mpi.WIN_NULL:
            self.peer_win.Free()
        if self.intra_locale_win != _mpi.WIN_NULL:
            self.intra_locale_win.Free()


if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
    RmaWindowBuffer.__doc__ = \
        """
       Details of the buffer allocated on a locale.
       """
    RmaWindowBuffer.is_shared.__doc__ = \
        """
        A :obj:`bool`, if :samp:`True` then buffer memory was allocated
        using :meth:`mpi4py.MPI.Win.Allocate_shared`, otherwise memory
        was allocated using :meth:`mpi4py.MPI.Win.Allocate`.
        """
    RmaWindowBuffer.shape.__doc__ = \
        "A sequence of :obj:`int` indicating the shape of the locale array."
    RmaWindowBuffer.dtype.__doc__ = \
        """
        A :obj:`numpy.dtype` indicating the data type of elements
        stored in the array.
        """
    RmaWindowBuffer.itemsize.__doc__ = \
        """
        An :obj:`int` indicating the number of bytes  per array element
        (same as :attr:`numpy.dtype.itemsize`).
        """
    RmaWindowBuffer.peer_win.__doc__ = \
        """
        The :obj:`mpi4py.MPI.Win` created from the :samp:`peer_comm` communicator
        which exposes :attr:`buffer` for inter-locale RMA access.
        """
    RmaWindowBuffer.intra_locale_win.__doc__ = \
        """
        The :obj:`mpi4py.MPI.Win` created from the :samp:`intra_locale_comm` communicator
        which exposes :attr:`buffer` for intra-locale RMA access.
        When :samp:`{intra_locale_win}.group.size > 1` then :attr:`buffer` was
        allocated as shared memory (using :meth:`mpi4py.MPI.Win.Allocate_shared`).
        """
    RmaWindowBuffer.inter_locale_win.__doc__ = \
        """
        The :obj:`mpi4py.MPI.Win` created from the :samp:`inter_locale_comm` communicator
        which exposes :attr:`buffer` for inter-locale RMA access.
        """


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

    def __init__(self, peer_comm=None, intra_locale_comm=None, inter_locale_comm=None):
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
        """
        if peer_comm is None:
            peer_comm = _mpi.COMM_WORLD
        self._peer_comm = peer_comm
        rank_logger = _logging.get_rank_logger(__name__ + "." + self.__class__.__name__, peer_comm)
        if intra_locale_comm is None:
            if mpi_version() >= 3:
                rank_logger.debug(
                    "BEG: Splitting peer_comm with peer_comm.Split_type(COMM_TYPE_SHARED, ...)"
                )
                intra_locale_comm = peer_comm.Split_type(_mpi.COMM_TYPE_SHARED, key=peer_comm.rank)
                rank_logger.debug(
                    "END: Splitting peer_comm with peer_comm.Split_type(COMM_TYPE_SHARED, ...)"
                )
            else:
                intra_locale_comm = _mpi.COMM_SELF

        self._intra_locale_comm = intra_locale_comm

        self._inter_locale_comm = None

        if inter_locale_comm is None:
            color = _mpi.UNDEFINED
            if self.intra_locale_comm.rank == 0:
                color = 0
            rank_logger.debug("BEG: self.peer_comm.Split to create self.inter_locale_comm.")
            inter_locale_comm = self._peer_comm.Split(color, self._peer_comm.rank)
            rank_logger.debug("END: self.peer_comm.Split to create self.inter_locale_comm.")

        self._peer_ranks_per_locale = None

        self._num_locales = None
        bad_inter_rank_to_intra_rank = tuple()
        if inter_locale_comm != _mpi.COMM_NULL:
            if (
                _mpi.Group.Translate_ranks(
                    inter_locale_comm.group,
                    (inter_locale_comm.rank,),
                    self._intra_locale_comm.group
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
                            self._peer_comm.rank,
                            inter_locale_comm.rank,
                            _mpi.Group.Translate_ranks(
                                inter_locale_comm.group,
                                (inter_locale_comm.rank,),
                                self._intra_locale_comm.group
                            )[0]
                        ),
                    )
            self._num_locales = inter_locale_comm.size
            self._peer_ranks_per_locale = \
                _np.array(
                    _mpi.Group.Translate_ranks(
                        self._intra_locale_comm.group,
                        _np.arange(0, self._intra_locale_comm.size),
                        self._peer_comm.group
                    ),
                    dtype="int64"
                ).reshape((self._intra_locale_comm.size,))

            rank_logger.debug("BEG: inter_locale_comm.allgather for peer-ranks-per-locale...")
            all = \
                inter_locale_comm.allgather(
                    (bad_inter_rank_to_intra_rank, self._peer_ranks_per_locale)
                )
            rank_logger.debug("END: inter_locale_comm.allgather for peer-ranks-per-locale.")

            all_bad = tuple(a[0] for a in all)
            all_prpl = tuple(a[1] for a in all)
            # Assign self._peer_ranks_per_locale as 1D array of object
            # in case self.intra_locale.comm.size is not the same on
            # all locales.
            self._peer_ranks_per_locale = \
                _np.ones((len(all_prpl),), dtype="object")
            for i in range(len(all_prpl)):
                self._peer_ranks_per_locale[i] = all_prpl[i]

            if (
                _np.all(
                    tuple(
                        len(self._peer_ranks_per_locale[i]) == len(self._peer_ranks_per_locale[0])
                        for i in range(len(self._peer_ranks_per_locale))
                    )
                )
            ):
                # If all locales have the same intra_locale_comm.size,
                # then just use a 2D numpy.ndarray of 'int64' elements
                # (rather than an 1D array of variable-length 'object' elements
                num_peer_ranks_per_locale = len(self._peer_ranks_per_locale[0])
                self._peer_ranks_per_locale = \
                    _np.array(self._peer_ranks_per_locale.tolist(), dtype="int64")
                self._peer_ranks_per_locale = \
                    self._peer_ranks_per_locale.reshape(
                        (self._num_locales, num_peer_ranks_per_locale)
                    )

            bad_inter_rank_to_intra_rank = sum(all_bad, ())
            del all, all_bad, all_prpl

        rank_logger.debug("BEG: intra_locale_comm.bcast for peer-ranks-per-locale...")
        bad_inter_rank_to_intra_rank, self._peer_ranks_per_locale, self._num_locales = \
            self._intra_locale_comm.bcast(
                (bad_inter_rank_to_intra_rank, self._peer_ranks_per_locale, self._num_locales),
                0
            )
        rank_logger.debug("END: intra_locale_comm.bcast for peer-ranks-per-locale...")
        if len(bad_inter_rank_to_intra_rank) > 0:
            raise ValueError(
                "Got invalid inter_locale_comm rank to intra_locale_comm rank translation:\n"
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

        if self._num_locales > self._peer_comm.size:
            raise ValueError(
                "Got number of locales=%s which is greated than peer_comm.size=%s"
                %
                (self._num_locales, self._peer_comm.size)
            )

        self._inter_locale_comm = inter_locale_comm

        self._rank_logger = \
            _logging.get_rank_logger(
                __name__ + "." + self.__class__.__name__,
                comm=self._peer_comm
            )

        self._root_logger = \
            _logging.get_root_logger(
                __name__ + "." + self.__class__.__name__,
                comm=self._peer_comm
            )

    def alloc_locale_buffer(self, shape, dtype):
        """
        Allocates a buffer using :meth:`mpi4py.MPI.Win.Allocate_shared` which
        provides storage for the elements of the locale multi-dimensional array.

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
            num_rank_bytes = int(_np.product(rank_shape) * dtype.itemsize)
        else:
            rank_shape = tuple(_np.zeros_like(rank_shape))
        if (mpi_version() >= 3) and (self.intra_locale_comm.size > 1):
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
            self.rank_logger.debug(
                "BEG: Win.Allocate - allocating buffer of %d bytes for shape=%s, dtype=%s",
                num_rank_bytes,
                rank_shape,
                dtype
            )
            _log_memory_alloc(
                self.rank_logger.debug, "BEG: ", num_rank_bytes, rank_shape, dtype
            )
            intra_locale_win = \
                _mpi.Win.Allocate(num_rank_bytes, dtype.itemsize, comm=self.peer_comm)
            buffer = intra_locale_win.memory
            itemsize = dtype.itemsize
            _log_memory_alloc(
                self.rank_logger.debug, "END: ", num_rank_bytes, rank_shape, dtype, buffer=buffer
            )

        if num_rank_bytes > 0:
            peer_buffer = buffer
        else:
            peer_buffer = None
        if peer_buffer is None:
            peer_buffer_nbytes = None
        elif (hasattr(peer_buffer, 'nbytes')):
            peer_buffer_nbytes = peer_buffer.nbytes
        else:
            peer_buffer_nbytes = \
                _np.product(peer_buffer.shape) * peer_buffer.itemsize

        self.rank_logger.debug(
            "BEG: Win.Create for self.peer_comm, buffer.nbytes=%s...",
            peer_buffer_nbytes
        )
        peer_win = _mpi.Win.Create(peer_buffer, itemsize, comm=self.peer_comm)
        if peer_win.memory is None:
            peer_win_memory_nbytes = None
        elif (hasattr(peer_win.memory, 'nbytes')):
            peer_win_memory_nbytes = peer_win.memory.nbytes
        else:
            peer_win_memory_nbytes = \
                _np.product(peer_win.memory.shape) * peer_win.memory.itemsize
        self.rank_logger.debug(
            "END: Win.Create for self.peer_comm, peer_win.memory.nbytes=%s...",
            peer_win_memory_nbytes
        )

        inter_locale_win = None
        inter_locale_win = _mpi.WIN_NULL
        if self.have_valid_inter_locale_comm:
            self.rank_logger.debug("BEG: Win.Create for self.inter_locale_comm")
            inter_locale_win = \
                _mpi.Win.Create(
                    buffer,
                    itemsize,
                    comm=self.inter_locale_comm
                )
            self.rank_logger.debug("END: Win.Create for self.inter_locale_comm")

        buffer = _np.array(buffer, dtype='B', copy=False)

        self.rank_logger.debug("END: alloc_local_buffer")
        return \
            RmaWindowBuffer(
                is_shared=is_shared_alloc,
                shape=locale_shape,
                dtype=dtype,
                itemsize=itemsize,
                peer_win=peer_win,
                intra_locale_win=intra_locale_win,
                inter_locale_win=inter_locale_win
            )

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


class CartLocaleComms(LocaleComms):

    """
    Defines cartesian communication topology for locales.
    In addition to the :obj:`LocaleComms` communicators, defines:

    :attr:`cart_comm`
       Typically this communicator is created using
       the call :samp:`{inter_locale_comm}.Create_cart(...)`.
       This communicator (and associated `mpi4py.MPI.Win` window) is used to exchange
       data between locales.
       In construction, this :obj:`mpi4py.MPI.CartComm` communicator
       replaces the :attr:`LocaleComms.inter_locale_comm` communicator.
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
           that :samp:`numpy.product({dims}) == {peer_comm}.size`.
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
        LocaleComms.__init__(
            self,
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm
        )

        # No implementation for periodic boundaries yet
        periods = None
        if (ndims is None) and (dims is None):
            raise ValueError("Must specify one of dims or ndims in CartLocaleComms constructor.")
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

        self._cart_comm = cart_comm
        rank_logger = \
            _logging.get_rank_logger(__name__ + "." + self.__class__.__name__, comm=self.peer_comm)

        self._dims = \
            _array_split.split.calculate_num_slices_per_axis(
                dims,
                self.num_locales
            )

        # Create a cartesian grid communicator
        # NB: use of self._inter_locale_comm (not self.inter_locale_comm)
        # important here because self.inter_locale_comm is over-ridden to
        # return self._cart_comm.
        inter_locale_comm = self._inter_locale_comm
        if (inter_locale_comm != _mpi.COMM_NULL) and (cart_comm is None):
            rank_logger.debug("BEG: inter_locale_comm.Create to create cart_comm.")
            cart_comm = \
                inter_locale_comm.Create_cart(
                    self.dims,
                    periods,
                    reorder=True
                )
            rank_logger.debug("END: inter_locale_comm.Create to create cart_comm.")
        elif (inter_locale_comm == _mpi.COMM_NULL) and (cart_comm is None):
            cart_comm = _mpi.COMM_NULL
        if (
            (cart_comm != _mpi.COMM_NULL)
            and
            (cart_comm.group.size != self._num_locales)
        ):
            raise ValueError(
                "Got cart_comm.group.size (=%s) != self._num_locales (=%s)."
                %
                (self._cart_comm.group.size, self._num_locales)
            )

        self._cart_comm = cart_comm

    @property
    def cart_coord_to_cart_rank_map(self):
        """
        A :obj:`dict` of :obj:`tuple`
        cartesian coordinate (:meth:`mpi4py.MPI.CartComm.Get_coords`) keys
        which map to the associated :attr:`cart_comm` peer_rank.
        """
        d = dict()
        if self.have_valid_cart_comm:
            for cart_rank in range(self.cart_comm.size):
                d[tuple(self.cart_comm.Get_coords(cart_rank))] = cart_rank
        elif self.cart_comm is None:
            d = None
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
        """
        return self.cart_comm


def shape_squeeze(shape):
    """
    Returns sequence with the the :samp:`1` elements removed.
    """
    return _np.asarray(shape)[_np.where(shape == 1)]


def reshape_comms_distribution(comms_distrib, new_globale_shape):
    """
    """
    reshaped_comms_distrib = None

    locale_comms = comms_distrib.locale_comms
    distrib = comms_distrib.distribution
    this_locale = comms_distrib.this_locale
    globale_shape = _np.array(distrib.globale_extent.shape)

    new_globale_shape = _np.array(new_globale_shape)

    if (
        (new_globale_shape.size == globale_shape.size)
        and
        _np.all(new_globale_shape == globale_shape)
    ):
        reshaped_comms_distrib = \
            CommsAndDistribution(locale_comms, _copy.deepcopy(distrib), this_locale)
    else:
        new_squeeze_shape = shape_squeeze(new_globale_shape)
        org_squeeze_shape = shape_squeeze(globale_shape)
        if len(new_squeeze_shape) == len(org_squeeze_shape):
            if _np.all(_np.sort(org_squeeze_shape) == _np.sort(org_squeeze_shape)):
                _np.all(new_squeeze_shape == org_squeeze_shape)
                if new_globale_shape.size >= globale_shape.size:
                    idx = \
                        [
                            _np.nonzero(new_globale_shape == globale_shape[i])[0][0]
                            for i in range(len(globale_shape))
                        ]
                    reshaped_comms_distrib = \
                        CommsAndDistribution(
                            locale_comms.expand_dims(
                                ndim=new_globale_shape.size,
                                orig_axis_new_pos=idx
                            ),
                            distrib.expand_dims(ndim=new_globale_shape.size, orig_axis_new_pos=idx),
                            this_locale
                        )
                else:
                    idx = [_np.nonzero(new_globale_shape[0] == globale_shape)[0][0], ]
                    for i in range(1, new_globale_shape.size):
                        idx.append(
                            _np.nonzero(new_globale_shape[i] == globale_shape[idx[-1]:])[0][0]
                        )
                    idx = _np.setdiff1d(idx, _np.arange(0, new_globale_shape.size))
                    reshaped_comms_distrib = \
                        CommsAndDistribution(
                            locale_comms.squeeze(axis=idx),
                            distrib.squeeze(axis=idx),
                            this_locale
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
if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
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
       if :samp:`numpy.product(shape) > 0` otherwise :attr:`mpi_array.comms.DT_CLONED`.
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
        if _np.product(shape) > 0:
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
