"""
==================================
The :mod:`mpi_array.global` Module
==================================

Defines :obj:`gndarray` class and factory functions for
creating multi-dimensional distributed arrays (Partitioned Global Address Space).

Classes
=======

.. autosummary::
   :toctree: generated/

   gndarray - A :obj:`numpy.ndarray` like distributed array.
   GndarrayRedistributeUpdater - Helper class for redistributing elements between distributions.

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

import pkg_resources as _pkg_resources
import mpi4py.MPI as _mpi
import numpy as _np

from .license import license as _license, copyright as _copyright
from . import locale as _locale
from .distribution import create_distribution
from .update import UpdatesForRedistribute as _UpdatesForRedistribute

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()

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

    def __init__(self, dtype, order, inter_locale_win, dst_buffer):
        """
        """
        CommLogger.__init__(self)
        self._dtype = dtype
        self._order = order
        self._inter_locale_win = inter_locale_win
        self._dst_buffer = dst_buffer
    
    @property
    def dtype(self):
        return self._dtype

    @property
    def order(self):
        return self._order

    def update_halos(self, halo_updates):
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

        
class GndarrayRedistributeUpdater(_UpdatesForRedistribute):

    """
    Helper class for redistributing array to new distribution.
    Calculates sequence of :obj:`mpi_array.distribution.ExtentUpdate`
    objects which are used to copy elements from
    remote locales to local locale.
    """

    def __init__(self, dst, src):
        """
        """
        self._dst = dst
        self._src = src
        _UpdatesForRedistribute.__init__(self, dst.comms_and_distrib, src.comms_and_distrib)

    def create_pair_extent_update(
        self,
        dst_extent,
        src_extent,
        intersection_extent
    ):
        """
        Factory method for which creates sequence of
        of :obj:`mpi_array.distribution.MpiPairExtentUpdate` objects.
        """
        updates = \
            _UpdatesForRedistribute.create_pair_extent_update(
                self,
                dst_extent,
                src_extent,
                intersection_extent
            )
        for update in updates:
            update.initialise_data_types(
                dst_dtype=self._dst.dtype,
                src_dtype=self._src.dtype,
                dst_order=self._dst.lndarray.md.order,
                src_order=self._src.lndarray.md.order
            )
        return updates

    def do_locale_update(self):
        """
        Performs RMA to get elements from remote locales to
        update the local array.
        """
        if (self._inter_win is not None) and (self._inter_win != _mpi.WIN_NULL):
            updates = self._dst_updates[self._dst_decomp.lndarray_extent.inter_locale_rank]
            self._dst_decomp.rank_logger.debug(
                "BEG: Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)..."
            )
            self._inter_win.Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)
            for single_update in updates:
                self._dst_decomp.rank_logger.debug(
                    "BEG: Getting update:\n%s\n%s",
                    single_update._header_str,
                    single_update
                )
                self._inter_win.Get(
                    [self._dst.lndarray.slndarray, 1, single_update.dst_data_type],
                    single_update.src_extent.inter_locale_rank,
                    [0, 1, single_update.src_data_type]
                )
                self._dst_decomp.rank_logger.debug(
                    "END: Getting update:\n%s\n%s",
                    single_update._header_str,
                    single_update
                )
            self._inter_win.Fence(_mpi.MODE_NOSUCCEED)
            self._dst_decomp.rank_logger.debug(
                "END: Fence(_mpi.MODE_NOSUCCEED)."
            )
        if self._dst_decomp.num_locales > 1:
            self._dst_decomp.intra_locale_comm.barrier()
        else:
            _np.copyto(self._dst.lndarray.slndarray, self._src.lndarray.slndarray)

    def barrier(self):
        """
        MPI barrier.
        """
        self._dst_decomp.rank_logger.debug(
            "BEG: self._src.comms_and_distrib.intra_locale_comm.barrier()..."
        )
        self._src.comms_and_distrib.intra_locale_comm.barrier()
        self._dst_decomp.rank_logger.debug(
            "END: self._src.comms_and_distrib.intra_locale_comm.barrier()."
        )


class gndarray(object):

    """
    A Partitioned Global Address Space array with :obj:`numpy.ndarray` API.
    """

    def __new__(
        cls,
        comms_and_distrib,
        rma_window_buffer,
        lndarray
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
        self._lndarray = lndarray
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
            ret.lndarray.rank_view_n[...] = \
                (self.lndarray.rank_view_n[...] == other.lndarray.rank_view_n[...])
        else:
            ret.lndarray.rank_view_n[...] = \
                (self.lndarray.rank_view_n[...] == other)

        return ret

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
    def lndarray(self):
        return self._lndarray

    @property
    def shape(self):
        return self._comms_and_distrib.distribution.globale_extent.shape_n

    @property
    def dtype(self):
        return self._lndarray.dtype

    @property
    def order(self):
        return self._lndarray.md.order

    @property
    def rank_view_n(self):
        return self._lndarray.rank_view_n

    @property
    def rank_view_h(self):
        return self._lndarray.rank_view_h

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

    @property
    def halo_updater(self):
        if self._halo_updater is None:
            self._halo_updater = \
                PerAxisRmaHaloUpdater(
                    dtype=self.dtype,
                    order=self.order,
                    inter_locale_win=self.rma_window_buffer.inter_locale_win,
                    dst_buffer=self.lndarray.slndarray
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
#                and
#                hasattr(self.comms_and_distrib.distribution, "halo_updates")
            ):
                rank_logger.debug(
                    "BEG: update_halos..."
                )
                halo_updates = self.comms_and_distrib.distribution.halo_updates
                self.halo_updater.update_halos(halo_updates)
                rank_logger.debug(
                    "END: update_halos."
                )

            # All ranks on locale wait for halo update to complete
            rank_logger.debug(
                "BEG: self.comms_and_distrib.locale_comms.intra_locale_comm.barrier()..."
            )
            self.comms_and_distrib.locale_comms.intra_locale_comm.barrier()
            rank_logger.debug(
                "END: self.comms_and_distrib.locale_comms.intra_locale_comm.barrier()."
            )

    def calculate_copyfrom_updates(self, src):
        return \
            GndarrayRedistributeUpdater(
                self,
                src
            )

    def copyfrom(self, src):
        """
        Copy the elements of the :samp:`{src}` array to corresponding elements of
        the :samp:`{self}` array.

        :type src: :obj:`gndarray`
        :type src: Global array from which elements are copied.
        """

        if not isinstance(src, gndarray):
            raise ValueError(
                "Got type(src)=%s, expected %s." % (type(src), gndarray)
            )

        redistribute_updater = self.calculate_copyfrom_updates(src)
        self.comms_and_distrib.rank_logger.debug("BEG: redistribute_updater.barrier()...")
        redistribute_updater.barrier()
        self.comms_and_distrib.rank_logger.debug("END: redistribute_updater.barrier().")
        redistribute_updater.do_locale_update()
        self.comms_and_distrib.rank_logger.debug("BEG: redistribute_updater.barrier()...")
        redistribute_updater.barrier()
        self.comms_and_distrib.rank_logger.debug("END: redistribute_updater.barrier().")

    def all(self, **unused_kwargs):
        return \
            self.locale_comms.rank_comm.allreduce(
                bool(self.lndarray.rank_view_n.all()),
                op=_mpi.BAND
            )

    def fill(self, value):
        """
        Fill the array (excluding ghost elements) with a scalar value.

        :type value: scalar
        :param value: All non-ghost elements will be assigned this value.
        """
        self.lndarray.fill(value)

        self.locale_comms.rank_logger.debug("BEG: self.locale_comms.intra_locale_comm...")
        self.locale_comms.intra_locale_comm.barrier()
        self.locale_comms.rank_logger.debug("END: self.locale_comms.intra_locale_comm.")

    def fill_h(self, value):
        """
        Fill all array elements (including ghost elements) with a scalar value.

        :type value: scalar
        :param value: All elements will be assigned this value.
        """
        self.lndarray.fill_h(value)

        self.locale_comms.rank_logger.debug("BEG: self.locale_comms.intra_locale_comm...")
        self.locale_comms.intra_locale_comm.barrier()
        self.locale_comms.rank_logger.debug("END: self.locale_comms.intra_locale_comm.")

    def copy(self):
        ary_out = empty_like(self)
        ary_out.lndarray.rank_view_partition_h[...] = self.lndarray.rank_view_partition_h[...]

        ary_out.locale_comms.rank_logger.debug("BEG: ary_out.locale_comms.intra_locale_comm...")
        ary_out.locale_comms.intra_locale_comm.barrier()
        ary_out.locale_comms.rank_logger.debug("END: ary_out.locale_comms.intra_locale_comm.")

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
    lndarray, rma_window_buffer = \
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
            lndarray=lndarray
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
                intra_partition_dims=ary.lndarray.intra_partition_dims
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


def copyto(dst, src, **kwargs):
    """
    Copy the elements of the :samp:`{src}` array to corresponding elements of
    the :samp:`dst` array.

    :type dst: :obj:`gndarray`
    :type dst: Global array which receives elements.
    :type src: :obj:`gndarray`
    :type src: Global array from which elements are copied.
    """
    if not isinstance(dst, gndarray):
        raise ValueError(
            "Got type(dst)=%s, expected %s." % (type(dst), gndarray)
        )

    dst.copyfrom(src)


__all__ = [s for s in dir() if not s.startswith('_')]
