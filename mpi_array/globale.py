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
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources
import mpi4py.MPI as _mpi
import numpy as _np
from mpi_array.locale import lndarray as _lndarray
from mpi_array.distribution import BlockPartition as _BlockPartition
from mpi_array.update import UpdatesForRedistribute as _UpdatesForRedistribute

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


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
        _UpdatesForRedistribute.__init__(self, dst.decomp, src.decomp)

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
            "BEG: self._src.decomp.intra_locale_comm.barrier()..."
        )
        self._src.decomp.intra_locale_comm.barrier()
        self._dst_decomp.rank_logger.debug(
            "END: self._src.decomp.intra_locale_comm.barrier()."
        )


class gndarray(object):

    """
    A Partitioned Global Address Space array with :obj:`numpy.ndarray` API.
    """

    def __new__(
        cls,
        shape=None,
        dtype=_np.dtype("float64"),
        order=None,
        decomp=None
    ):
        """
        Construct, at least one of :samp:{shape} or :samp:`decomp` should
        be specified (i.e. at least one should not be :samp:`None`).

        :type shape: :samp:`None` or sequence of :obj:`int`
        :param shape: **Global** shape of the array. If :samp:`None`
           global array shape is taken as :samp:`{decomp}.shape`.
        :type dtype: :obj:`numpy.dtype`
        :param dtype: Data type for elements of the array.
        :type strides: :samp:`None` or sequence of :obj:`int`
        :param strides: Strides of data in memory.
        :type order: {:samp:`C`, :samp:`F`} or :samp:`None`
        :param order: Row-major (C-style) or column-major (Fortran-style) order.
        :type decomp: :obj:`mpi_array.distribution.Decomposition`
        :param decomp: Array distribution info and used to allocate (possibly)
           shared memory.
        """
        if (shape is not None) and (decomp is None):
            decomp = _BlockPartition(shape)
        elif (shape is not None) and (decomp is not None) and (_np.any(decomp.shape != shape)):
            raise ValueError(
                "Got inconsistent shape argument=%s and decomp.shape=%s, should be the same."
                %
                (shape, decomp.shape)
            )
        elif (shape is None) and (decomp is None):
            raise ValueError(
                (
                    "Got None for both shape argument and decomp argument,"
                    +
                    " at least one should be provided."
                )
            )
        self = object.__new__(cls)
        self._lndarray = _lndarray(shape=shape, dtype=dtype, order=order, decomp=decomp)
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

    def all(self, **unused_kwargs):
        return \
            self.decomp.rank_comm.allreduce(
                bool(self.lndarray.rank_view_n.all()),
                op=_mpi.BAND
            )

    @property
    def decomp(self):
        return self._lndarray.decomp

    @property
    def lndarray(self):
        return self._lndarray

    @property
    def shape(self):
        return self._lndarray.decomp.shape

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
        return self._lndarray.decomp.rank_logger

    @property
    def root_logger(self):
        """
        """
        return self._lndarray.decomp.root_logger

    def update(self):
        """
        """
        # If running on single locale then there are no halos to update.
        if self.decomp.num_locales > 1:
            # Only do comms between the ranks of self.decomp.cart_comm
            self.decomp.rank_logger.debug("BEG: self.decomp.intra_locale_comm.barrier()...")
            self.decomp.intra_locale_comm.barrier()
            self.decomp.rank_logger.debug("END: self.decomp.intra_locale_comm.barrier().")
            if self.decomp.have_valid_cart_comm:
                self.decomp.rank_logger.debug("BEG: self.decomp.cart_mem_comm.barrier()...")
                self.decomp.cart_comm.barrier()
                self.decomp.rank_logger.debug("END: self.decomp.cart_mem_comm.barrier().")

                cart_rank_updates = \
                    self.decomp.get_updates_for_cart_rank(self.decomp.cart_comm.rank)
                per_axis_cart_rank_updates = cart_rank_updates.updates_per_axis

                # per_axis_cart_rank_updates is None on all cart_comm ranks only
                # if there are no halos on any axis.
                if per_axis_cart_rank_updates is not None:
                    for a in range(self.decomp.ndim):
                        lo_hi_updates_pair = per_axis_cart_rank_updates[a]

                        # When axis a doesn't have a halo, then lo_hi_updates_pair
                        # is None on all cart_comm ranks, and we avoid calling the Fence
                        # in this case
                        if lo_hi_updates_pair is not None:
                            axis_cart_rank_updates = \
                                lo_hi_updates_pair[self.decomp.LO] + \
                                lo_hi_updates_pair[self.decomp.HI]
                            self.decomp.rank_logger.debug(
                                "BEG: Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)..."
                            )
                            self.decomp.cart_win.Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)
                            for single_update in axis_cart_rank_updates:
                                single_update.initialise_data_types(self.dtype, self.order)
                                self.decomp.rank_logger.debug(
                                    "BEG: Getting update:\n%s\n%s",
                                    single_update._header_str,
                                    single_update
                                )
                                self.decomp.cart_win.Get(
                                    [self.lndarray.slndarray, 1, single_update.dst_data_type],
                                    single_update.src_extent.cart_rank,
                                    [0, 1, single_update.src_data_type]
                                )
                                self.decomp.rank_logger.debug(
                                    "END: Getting update:\n%s\n%s",
                                    single_update._header_str,
                                    single_update
                                )
                            self.decomp.cart_win.Fence(_mpi.MODE_NOSUCCEED)
                            self.decomp.rank_logger.debug(
                                "END: Fence(_mpi.MODE_NOSUCCEED)."
                            )

            # All ranks on locale wait for halo update to complete
            self.decomp.rank_logger.debug("BEG: self.decomp.intra_locale_comm.barrier()...")
            self.decomp.intra_locale_comm.barrier()
            self.decomp.rank_logger.debug("END: self.decomp.intra_locale_comm.barrier().")

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
        self.decomp.rank_logger.debug("BEG: redistribute_updater.barrier()...")
        redistribute_updater.barrier()
        self.decomp.rank_logger.debug("END: redistribute_updater.barrier().")
        redistribute_updater.do_locale_update()
        self.decomp.rank_logger.debug("BEG: redistribute_updater.barrier()...")
        redistribute_updater.barrier()
        self.decomp.rank_logger.debug("END: redistribute_updater.barrier().")


def empty(shape=None, dtype="float64", decomp=None, order='C'):
    """
    Creates array of uninitialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`gndarray`
    :return: Newly created array with uninitialised elements.
    """
    ary = gndarray(shape=shape, dtype=dtype, decomp=decomp, order=order)

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
        ret_ary = empty(dtype=ary.dtype, decomp=ary.decomp)
    else:
        ret_ary = _np.empty_like(ary, dtype=dtype)

    return ret_ary


def zeros(shape=None, dtype="float64", decomp=None, order='C'):
    """
    Creates array of zero-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`gndarray`
    :return: Newly created array with zero-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view_n[...] = _np.zeros(1, dtype)

    ary.decomp.rank_logger.debug("BEG: ary.decomp.intra_locale_comm...")
    ary.decomp.intra_locale_comm.barrier()
    ary.decomp.rank_logger.debug("END: ary.decomp.intra_locale_comm.")

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
    ary.rank_view_n[...] = _np.zeros(1, ary.dtype)

    ary.decomp.rank_logger.debug("BEG: ary.decomp.intra_locale_comm...")
    ary.decomp.intra_locale_comm.barrier()
    ary.decomp.rank_logger.debug("END: ary.decomp.intra_locale_comm.")

    return ary


def ones(shape=None, dtype="float64", decomp=None, order='C'):
    """
    Creates array of one-initialised elements.

    :type shape: :samp:`None` or sequence of :obj:`int`
    :param shape: **Global** shape to be distributed amongst
       memory nodes.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: Data type of array elements.
    :type decomp: :obj:`numpy.dtype`
    :param decomp: Data type of array elements.
    :rtype: :obj:`gndarray`
    :return: Newly created array with one-initialised elements.
    """
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view_n[...] = _np.ones(1, dtype)

    ary.decomp.rank_logger.debug("BEG: ary.decomp.intra_locale_comm...")
    ary.decomp.intra_locale_comm.barrier()
    ary.decomp.rank_logger.debug("END: ary.decomp.intra_locale_comm.")

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
    ary.rank_view_n[...] = _np.ones(1, ary.dtype)

    ary.decomp.rank_logger.debug("BEG: ary.decomp.intra_locale_comm...")
    ary.decomp.intra_locale_comm.barrier()
    ary.decomp.rank_logger.debug("END: ary.decomp.intra_locale_comm.")

    return ary


def copy(ary, **kwargs):
    """
    Return an array copy of the given object.

    :type ary: :obj:`gndarray`
    :param ary: Array to copy.
    :rtype: :obj:`gndarray`
    :return: A copy of :samp:`ary`.
    """
    ary_out = empty_like(ary)
    ary_out.rank_view_n[...] = ary.rank_view_n[...]

    ary_out.decomp.rank_logger.debug("BEG: ary_out.decomp.intra_locale_comm...")
    ary_out.decomp.intra_locale_comm.barrier()
    ary_out.decomp.rank_logger.debug("END: ary_out.decomp.intra_locale_comm.")

    return ary_out


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
