"""
=========================================
The :mod:`mpi_array.decomposition` Module
=========================================

Sub-division of arrays over nodes and/or MPI processes.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   SharedMemInfo - Shared-memory communicator generation.
   MemNodeTopology - Topology of MPI processes which allocate shared memory.
   Decomposition - Partition of an array *shape* overs MPI processes and/or nodes.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources
import sys as _sys
import mpi4py.MPI as _mpi
import array_split as _array_split
import array_split.split  # noqa: F401
import numpy as _np

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


class SharedMemInfo(object):
    """
    Info on possible shared memory allocation for a specified MPI communicator.
    """

    def __init__(self, comm=None, shared_mem_comm=None):
        """
        Construct.

        :type comm: :obj:`mpi4py.MPI.Comm`
        :param comm: Communicator used to split according to
           shared memory allocation (uses :meth:`mpi4py.MPI.Comm.Split_type`).
        :type shared_mem_comm: :obj:`mpi4py.MPI.Comm`
        :param shared_mem_comm: Shared memory communicator, can explicitly
           specify (should be a subset of processes returned
           by :samp:`{comm}.Split_type(_mpi.COMM_TYPE_SHARED)`.
           If :samp:`None`, :samp:`{comm}` is *split* into groups
           which can use a MPI window to allocate shared memory.
        """
        if comm is None:
            comm = _mpi.COMM_WORLD
        if shared_mem_comm is None:
            if _mpi.VERSION >= 3:
                shared_mem_comm = comm.Split_type(_mpi.COMM_TYPE_SHARED, key=comm.rank)
            else:
                shared_mem_comm = comm.Split(comm.rank, key=comm.rank)

        self._shared_mem_comm = shared_mem_comm

        # Count the number of self._shared_mem_comm rank-0 processes
        # to work out how many communicators comm was split into.
        is_rank_zero = 0
        if self._shared_mem_comm.rank == 0:
            is_rank_zero = 1
        self._num_shared_mem_nodes = comm.allreduce(is_rank_zero, _mpi.SUM)

    @property
    def num_shared_mem_nodes(self):
        """
        An integer indicating the number of *memory nodes* over which an array is distributed.
        """
        return self._num_shared_mem_nodes

    @property
    def shared_mem_comm(self):
        """
        A :obj:`mpi4py.MPI.Comm` object which defines the group of processes
        which can allocate (and access) MPI window shared memory
        (via  :meth:`mpi4py.MPI.Win.Allocate_shared`).
        """
        return self._shared_mem_comm


class MemNodeTopology(object):
    """
    Defines cartesian communication topology for MPI processes.
    """

    def __init__(
        self,
        ndims=None,
        dims=None,
        rank_comm=None,
        shared_mem_comm=None
    ):
        """
        Initialises cartesian communicator mem-nodes.
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
        :param rank_comm: The MPI processes over which an array is to be distributed.
           If :samp:`None` uses :obj:`mpi4py.MPI.COMM_WORLD`.
        :type shared_mem_comm: :obj:`mpi4py.MPI.Comm`
        :param shared_mem_comm: The MPI communicator used to create a window which
            can be used to allocate shared memory
            via :meth:`mpi4py.MPI.Win.Allocate_shared`.
        """
        # No implementation for periodic boundaries
        periods = None
        if (ndims is None) and (dims is None):
            raise ValueError("Must specify one of dims or ndims in MemNodeTopology constructor.")
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
        self._shared_mem_info = SharedMemInfo(self.rank_comm, shared_mem_comm)
        self._cart_comm = None

        self._dims = \
            _array_split.split.calculate_num_slices_per_axis(
                dims,
                self.num_shared_mem_nodes
            )

        # Create a cartesian grid communicator
        if self.num_shared_mem_nodes > 1:
            color = _mpi.UNDEFINED
            if self.shared_mem_comm.rank == 0:
                color = 0
            splt_comm = self.rank_comm.Split(color, self.rank_comm.rank)
            if splt_comm != _mpi.COMM_NULL:
                self._cart_comm = splt_comm.Create_cart(self.dims, periods, reorder=True)
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
    def num_shared_mem_nodes(self):
        """
        See :attr:`SharedMemInfo.num_shared_mem_nodes`.
        """
        return self._shared_mem_info.num_shared_mem_nodes

    @property
    def shared_mem_comm(self):
        """
        See :attr:`SharedMemInfo.shared_mem_comm`.
        """
        return self._shared_mem_info.shared_mem_comm


if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
    # Set docstring for properties.
    MemNodeTopology.num_shared_mem_nodes.__doc__ = SharedMemInfo.num_shared_mem_nodes.__doc__
    MemNodeTopology.shared_mem_comm.__doc__ = SharedMemInfo.shared_mem_comm.__doc__


class Decomposition(object):
    """
    Partitions an array-shape over MPI memory-nodes.
    """

    def __init__(
        self,
        shape,
        halo=0,
        mem_node_topology=None,
    ):
        """
        Create a partitioning of :samp:`{shape}` over memory-nodes.

        :type shape: sequence of :obj:`int`
        :param shape: The shape of the array which is to be partitioned into smaller *sub-shapes*.
        :type halo: :obj:`int`, sequence of :obj:`int` or :samp:`(len({shape}), 2)` shaped array.
        :param halo: Number of *ghost* elements added per axis
           (low and high indices can be different).
        :type mem_node_topology: :obj:`MemNodeTopology`
        :param mem_node_topology: Object which defines how array
           memory is allocated (distributed) over memory nodes and
           the cartesian topology communicator used to exchange (halo)
           data. If :samp:`None` uses :samp:`MemNodeTopology(dims=numpy.zeros_like({shape}))`.
        """
        self._halo = halo
        self._shape = None
        self._mem_node_topology = mem_node_topology
        self._shape_decomp = None

        self.recalculate(shape, halo)

    def recalculate(self, new_shape, new_halo):
        """
        Recomputes decomposition for :samp:`{new_shape}` and :samp:`{new_halo}`.

        :type new_shape: sequence of :obj:`int`
        :param new_shape: New partition calculated for this shape.
        :type new_halo: :obj:`int`, sequence of :obj:`int` or :samp:`(len{new_shape, 2))` array.
        :param new_halo: New partition calculated for this shape.
        """
        if self._mem_node_topology is None:
            self._mem_node_topology = MemNodeTopology(ndims=len(new_shape))
        elif (self._shape is not None) and (len(self._shape) != len(new_shape)):
            self._shape = _np.array(new_shape)
            self._mem_node_topology = MemNodeTopology(ndims=self._shape.size)
        self._shape = _np.array(new_shape)
        self._halo = new_halo

        shape_splitter = \
            _array_split.ShapeSplitter(
                array_shape=self._shape,
                axis=self._mem_node_topology.dims,
                halo=self._halo
            )

        self._halo = shape_splitter.halo

        self._shape_decomp = shape_splitter.calculate_split()

        if self.have_valid_cart_comm:
            self._cart_rank_to_extents_dict = dict()
            for cart_rank in range(0, self.cart_comm.size):
                self._cart_rank_to_extents_dict[cart_rank] = \
                    self._shape_decomp[tuple(self.cart_comm.Get_coords(cart_rank))]
                self._cart_rank_to_extents_dict[cart_rank] = \
                    _np.array(
                        [
                            [s.start for s in self._cart_rank_to_extents_dict[cart_rank]],
                            [s.stop for s in self._cart_rank_to_extents_dict[cart_rank]],
                        ]
                )

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
    def halo(self):
        """
        Number of *ghost* elements per axis padding array shape.
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
    def shape_decomp(self):
        """
        The partition of :samp:`self.shape` over memory nodes.
        """
        return self._shape_decomp

    @property
    def num_shared_mem_nodes(self):
        """
        See :attr:`MemNodeTopology.num_shared_mem_nodes`.
        """
        return self._mem_node_topology.num_shared_mem_nodes

    @property
    def shared_mem_comm(self):
        """
        See :attr:`MemNodeTopology.shared_mem_comm`.
        """
        return self._mem_node_topology.shared_mem_comm

    @property
    def cart_comm(self):
        """
        See :attr:`MemNodeTopology.cart_comm`.
        """
        return self._mem_node_topology.cart_comm

    @property
    def have_valid_cart_comm(self):
        """
        See :attr:`MemNodeTopology.have_valid_cart_comm`.
        """
        return self._mem_node_topology.have_valid_cart_comm

    @property
    def rank_comm(self):
        """
        See :attr:`MemNodeTopology.rank_comm`.
        """
        return self._mem_node_topology.rank_comm


if (_sys.version_info[0] >= 3) and (_sys.version_info[1] >= 5):
    # Set docstring for properties.
    Decomposition.num_shared_mem_nodes.__doc__ = MemNodeTopology.num_shared_mem_nodes.__doc__
    Decomposition.shared_mem_comm.__doc__ = MemNodeTopology.shared_mem_comm.__doc__
    Decomposition.cart_comm.__doc__ = MemNodeTopology.cart_comm.__doc__
    Decomposition.have_valid_cart_comm.__doc__ = MemNodeTopology.have_valid_cart_comm.__doc__
    Decomposition.rank_comm.__doc__ = MemNodeTopology.rank_comm.__doc__

__all__ = [s for s in dir() if not s.startswith('_')]
