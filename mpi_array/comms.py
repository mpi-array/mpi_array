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
   CommsAndDistribution - Pair consisting of :obj:`LocaleComms` and :obj:`Distribution`.
   ThisLocaleInfo - Info on inter_locale_comm peer_rank and corresponding peer_comm peer_rank.
   RmaWindowBuffer - Container for array buffer and associated RMA windows.

Factory Functions
=================

.. autosummary::
   :toctree: generated/

   create_distribution - Factory function for creating :obj:`Distribution` instances.
   create_block_distribution - Factory function for creating :obj:`BlockPartition` instances.

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright, version as _version

import mpi4py.MPI as _mpi
import numpy as _np
import collections as _collections

import array_split as _array_split

from . import logging as _logging
from .distribution import BlockPartition

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


RmaWindowBuffer = \
    _collections.namedtuple(
        "RmaWindowBuffer",
        [
            "buffer",
            "shape",
            "dtype",
            "itemsize",
            "peer_win",
            "intra_locale_win",
            "inter_locale_win"
        ]
    )


class LocaleComms(object):

    """
    Info on possible shared memory allocation for a specified MPI communicator.
    """

    def __init__(self, peer_comm=None, intra_locale_comm=None, inter_locale_comm=None):
        """
        Construct.

        :type peer_comm: :obj:`mpi4py.MPI.Comm`
        :param peer_comm: Communicator which is split according to
           shared memory allocation (uses :meth:`mpi4py.MPI.Comm.Split_type`).
        :type intra_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param intra_locale_comm: Intra-locale communicator.
           Should be a subset of processes returned
           by :samp:`{peer_comm}.Split_type(mpi4py.MPI.COMM_TYPE_SHARED)`.
           If :samp:`None`, :samp:`{peer_comm}` is *split* into groups
           which can use a MPI window to allocate shared memory
           (i.e. locale is a (NUMA) node).
           Can also specify as :samp:`mpi4py.MPI.COMM_SELF`, in which case the
           locale is a single process.
        :type inter_locale_comm: :obj:`mpi4py.MPI.Comm`
        :param inter_locale_comm: Inter-locale communicator used to exchange
            data between different locales.
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

        # Count the number of self._intra_locale_comm peer_rank-0 processes
        # to work out how many communicators peer_comm was split into.
        is_rank_zero = int(self._intra_locale_comm.rank == 0)

        rank_logger.debug("BEG: peer_comm.allreduce to calculate number of locales...")
        self._num_locales = peer_comm.allreduce(is_rank_zero, _mpi.SUM)
        rank_logger.debug("END: peer_comm.allreduce to calculate number of locales.")

        self._inter_locale_comm = None

        if (self._num_locales > 1):
            if inter_locale_comm is None:
                color = _mpi.UNDEFINED
                if self.intra_locale_comm.rank == 0:
                    color = 0
                rank_logger.debug("BEG: self.peer_comm.Split to create self.inter_locale_comm.")
                inter_locale_comm = self._peer_comm.Split(color, self._peer_comm.rank)
                rank_logger.debug("END: self.peer_comm.Split to create self.inter_locale_comm.")
            self._inter_locale_comm = inter_locale_comm
        elif (inter_locale_comm is not None) and (inter_locale_comm != _mpi.COMM_NULL):
            raise ValueError(
                "Got valid inter_local_comm=%s when self.num_locales <= 1"
                %
                (inter_locale_comm, )
            )

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

        :rtype: :obj:`RmaWindowBuffer`
        :returns: A :obj:`collections.namedtuple` containing allocated buffer
           and associated RMA MPI windows.
        """
        self.rank_logger.debug("BEG: alloc_locale_buffer")
        num_rank_bytes = 0
        dtype = _np.dtype(dtype)
        rank_shape = shape
        if self.intra_locale_comm.rank == 0:
            num_rank_bytes = int(_np.product(rank_shape) * dtype.itemsize)
        if (mpi_version() >= 3) and (self.intra_locale_comm.size > 1):
            self.rank_logger.debug("BEG: Win.Allocate_shared - allocating %d bytes", num_rank_bytes)
            intra_locale_win = \
                _mpi.Win.Allocate_shared(num_rank_bytes, dtype.itemsize,
                                         comm=self.intra_locale_comm)
            self.rank_logger.debug("END: Win.Allocate_shared - allocating %d bytes", num_rank_bytes)
            buffer, itemsize = intra_locale_win.Shared_query(0)
            self.rank_logger.debug("BEG: Win.Create for self.peer_comm")
            peer_win = _mpi.Win.Create(buffer, itemsize, comm=self.peer_comm)
            self.rank_logger.debug("END: Win.Create for self.peer_comm")
        else:
            self.rank_logger.debug("BEG: Win.Allocate - allocating %d bytes", num_rank_bytes)
            peer_win = \
                _mpi.Win.Allocate(num_rank_bytes, dtype.itemsize, comm=self.peer_comm)
            self.rank_logger.debug("END: Win.Allocate - allocating %d bytes", num_rank_bytes)
            intra_locale_win = peer_win
            buffer = peer_win.memory
            itemsize = dtype.itemsize

        inter_locale_win = None
        if self.num_locales > 1:
            inter_locale_win = _mpi.WIN_NULL
            if self.have_valid_inter_locale_comm:
                self.rank_logger.debug("BEG: Win.Create for self.inter_locale_comm")
                inter_locale_win = _mpi.Win.Create(buffer, itemsize, comm=self.inter_locale_comm)
                self.rank_logger.debug("END: Win.Create for self.inter_locale_comm")

        buffer = _np.array(buffer, dtype='B', copy=False)

        self.rank_logger.debug("END: alloc_local_buffer")
        return \
            RmaWindowBuffer(
                buffer=buffer,
                shape=rank_shape,
                dtype=dtype,
                itemsize=itemsize,
                peer_win=peer_win,
                intra_locale_win=intra_locale_win,
                inter_locale_win=inter_locale_win
            )

    @property
    def num_locales(self):
        """
        An integer indicating the number of *locales* over which an array is distributed.
        """
        return self._num_locales

    @property
    def peer_comm(self):
        """
        MPI communicator which is super-set of :attr:`intra_locale_comm`
        and :attr:`inter_locale_comm`.
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
        Is :samp:`True` if this peer_rank has :samp:`{self}.inter_locale_comm`
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
        where :samp:`m[inter_r]` is the peer_rank of :samp:`self.peer_comm`
        corresponding to peer_rank :samp:`inter_r` of :samp:`self.inter_locale_comm`.

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
    def this_locale_rank_info(self):
        """
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
        # important here because self.inter_locale_comm us over-ridden to
        # return self._cart_comm.
        inter_locale_comm = self._inter_locale_comm
        if self.num_locales > 1:
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
            elif cart_comm != _mpi.COMM_NULL:
                raise ValueError(
                    (
                        "Got object cart_comm=%s when expecting cart_comm to match "
                        +
                        "self._inter_locale_comm=%s"
                    )
                    %
                    (cart_comm, inter_locale_comm)
                )
            self._cart_comm = cart_comm
        elif (cart_comm is not None) and (cart_comm != _mpi.COMM_NULL):
            raise ValueError(
                "Got object cart_comm=%s when self.num_locales <= 1, cart_comm should be None"
                %
                (cart_comm, )
            )

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
        Dimension (:obj:`int`) of the cartesian topology.
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


#: Hyper-block partition distribution type
DT_BLOCK = "block"

#: Hyper-slab partition distribution type
DT_SLAB = "slab"

#: List of value :samp:`distrib_type` values.
_valid_distrib_types = [DT_BLOCK, DT_SLAB]

#: Node (NUMA) locale type
LT_NODE = "node"

#: Single process locale type
LT_PROCESS = "process"

#: List of value :samp:`locale_type` values.
_valid_locale_types = [LT_NODE, LT_PROCESS]

CommsAndDistribution = \
    _collections.namedtuple("CommsAndDistribution", ["locale_comms", "distribution", "this_locale"])


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
    Factory function for creating :obj:`BlockPartition` distribution instance.

    :rtype: :obj:`CommsAndDistribution`
    :return: A :obj:`CommsAndDistribution` pair.
    """
    if dims is None:
        dims = _np.zeros_like(shape, dtype="int64")
    if locale_type.lower() == LT_PROCESS:
        if (intra_locale_comm is not None) and (intra_locale_comm.size > 1):
            raise ValueError(
                "Got locale_type=%s, but intra_locale_comm.size=%s"
                %
                (locale_type, intra_locale_comm.size)
            )
        intra_locale_comm = _mpi.COMM_SELF
    cart_locale_comms = \
        CartLocaleComms(
            dims=dims,
            peer_comm=peer_comm,
            intra_locale_comm=intra_locale_comm,
            inter_locale_comm=inter_locale_comm,
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

    block_distrib = \
        BlockPartition(
            globale_extent=shape,
            dims=cart_locale_comms.dims,
            cart_coord_to_cart_rank=cart_coord_to_cart_rank,
            inter_locale_rank_to_peer_rank=cart_rank_to_peer_rank,
            halo=halo
        )
    return CommsAndDistribution(cart_locale_comms, block_distrib, this_locale)


def check_distrib_type(distrib_type):
    """
    Checks :samp:`{distrib_type}` occurs in :samp:`_valid_distrib_types`.
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


def create_distribution(shape, distrib_type=DT_BLOCK, locale_type=LT_NODE, **kwargs):
    """
    Factory function for creating :obj:`mpi_array.distribution.Distribution` instance.

    :rtype: :obj:`CommsAndDistribution`
    :return: A :obj:`CommsAndDistribution` pair.
    """
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

    return comms_and_distrib


__all__ = [s for s in dir() if not s.startswith('_')]
