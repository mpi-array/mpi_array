"""
========================================
The :mod:`mpi_array.distribution` Module
========================================

Apportionment of arrays over locales.

Classes
=======

.. autosummary::
   :toctree: generated/

   GlobaleExtent - Indexing and halo info for globale array.
   HaloSubExtent - Indexing sub-extent of globale extent.
   LocaleExtent - Indexing and halo info for locale array region.
   CartLocaleExtent - Indexing and halo info for a tile in a cartesian distribution.
   Distribution - Apportionment of extents amongst locales.
   ClonedDistribution - Entire array occurs in each locale.
   SingleLocaleDistribution - Entire array occurs on a single locale.
   BlockPartition - Block partition distribution of array extents amongst locales.

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright, version as _version

import mpi4py.MPI as _mpi
import numpy as _np
import copy as _copy
import collections as _collections

import array_split as _array_split
import array_split.split  # noqa: F401
from array_split.split import convert_halo_to_array_form as _convert_halo_to_array_form

from .indexing import IndexingExtent, HaloIndexingExtent


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class GlobaleExtent(HaloIndexingExtent):

    """
    Indexing extent for an entire globale array.
    """

    pass


class ScalarGlobaleExtent(GlobaleExtent):

    """
    Indexing extent for a scalar.
    """

    def __init__(self):
        GlobaleExtent.__init__(self, start=(), stop=(), halo=0)


class HaloSubExtent(HaloIndexingExtent):

    """
    Indexing extent for single region of a larger globale extent.
    Simply over-rides construction to trim the halo to the
    the :samp:`{globale_extent}` (with halo) bounds.
    """

    def __init__(
        self,
        globale_extent=None,
        slice=None,
        halo=0,
        start=None,
        stop=None,
        struct=None
    ):
        """
        Construct. Takes care of trimming the halo of this extent so
        that this extent does not stray outside the halo region of
        the :samp:`{globale_extent}`

        :type globale_extent: :obj:`GlobaleExtent`
        :param globale_extent: The indexing extent of the entire array.
        :type slice: sequence of :obj:`slice`
        :param slice: Per-axis start and stop indices (**not including ghost elements**).
        :type halo: :samp:`(len({split}), 2)` shaped array of :obj:`int`
        :param halo: Desired halo, a :samp:`(len(self.start), 2)` shaped array of :obj:`int`
           indicating the per-axis number of outer ghost elements. :samp:`halo[:,0]` is the
           number of ghost elements on the low-index *side* and :samp:`halo[:,1]` is the number
           of ghost elements on the high-index *side*. **Note**: that the halo will be truncated
           so that this halo extent does not extend beyond the halo :samp:`{globale_extent}`.
        :type start: sequence of :obj:`slice`
        :param start: Per-axis start indices (**not including ghost elements**).
        :type stop: sequence of :obj:`slice`
        :param stop: Per-axis stop indices (**not including ghost elements**).
        """
        struct_is_none = (struct is None)
        if (not struct_is_none) or (globale_extent is None):
            HaloIndexingExtent.__init__(
                self,
                slice=slice,
                start=start,
                stop=stop,
                halo=halo,
                struct=struct
            )
        elif struct_is_none:
            HaloIndexingExtent.__init__(self, slice=slice, start=start, stop=stop, halo=None)
            halo = _convert_halo_to_array_form(halo, ndim=self.ndim)

            # Axes with size=0 always get zero halo
            halo[_np.where(self.stop_n <= self.start_n)] = 0
            if globale_extent is not None:
                # Calculate the locale halo, truncate if it strays outside
                # the globale_extent halo region.
                halo = \
                    _np.maximum(
                        0,
                        _np.minimum(
                            _np.asarray(
                                [
                                    self.start_n - globale_extent.start_h,
                                    globale_extent.stop_h - self.stop_n
                                ],
                                dtype=halo.dtype
                            ).T,
                            halo
                        )
                    )
            self.halo = halo


class LocaleExtent(HaloSubExtent):

    """
    Indexing extent for single region of array residing on a locale.
    Extends :obj:`HaloSubExtent` by storing additional :attr:`{peer_rank}`
    and :attr:`inter_locale_rank` rank integers indicating the process
    responsible for exchanging the data to/from this extent.
    """

    PEER_RANK = 3
    PEER_RANK_STR = "peer_rank"
    INTER_LOCALE_RANK = 4
    INTER_LOCALE_RANK_STR = "inter_locale_rank"

    struct_dtype_dict = _collections.defaultdict(lambda: None)

    @staticmethod
    def create_struct_dtype_from_ndim(cls, ndim):
        """
        Creates a :obj:`numpy.dtype` structure for holding start and stop indices.

        :rtype: :obj:`numpy.dtype`
        :return: :obj:`numpy.dtype` with :samp:`"start"` and :samp:`"stop"` multi-index
           fields of dimension :samp:`{ndim}`.
        """
        return \
            _np.dtype(
                [
                    (cls.START_STR, _np.int64, (ndim,)),
                    (cls.STOP_STR, _np.int64, (ndim,)),
                    (cls.HALO_STR, _np.int64, (ndim, 2)),
                    (cls.PEER_RANK_STR, _np.int64),
                    (cls.INTER_LOCALE_RANK_STR, _np.int64)
                ]
            )

    def __init__(
        self,
        peer_rank=None,
        inter_locale_rank=None,
        globale_extent=None,
        slice=None,
        halo=0,
        start=None,
        stop=None,
        struct=None
    ):
        """
        Construct.

        :type peer_rank: :obj:`int`
        :param peer_rank: Rank of MPI process in :samp:`peer_comm` communicator which
           corresponds to :samp:`{inter_locale_rank}` rank of :samp:`{inter_locale_comm}`.
        :type inter_locale_rank: :obj:`int`
        :param inter_locale_rank: Rank of MPI process in :samp:`inter_locale_comm` communicator.
        :type globale_extent: :obj:`GlobaleExtent`
        :param globale_extent: The indexing extent of the entire array.
        :type slice: sequence of :obj:`slice`
        :param slice: Per-axis start and stop indices (**not including ghost elements**).
        :type halo: :samp:`(len({split}), 2)` shaped array of :obj:`int`
        :param halo: Desired halo, a :samp:`(len(self.start), 2)` shaped array of :obj:`int`
           indicating the per-axis number of outer ghost elements. :samp:`halo[:,0]` is the
           number of ghost elements on the low-index *side* and :samp:`halo[:,1]` is the number
           of ghost elements on the high-index *side*. **Note**: that the halo will be truncated
           so that this halo extent does not extend beyond the halo :samp:`{globale_extent}`.
        :type start: sequence of :obj:`slice`
        :param start: Per-axis start indices (**not including ghost elements**).
        :type stop: sequence of :obj:`slice`
        :param stop: Per-axis stop indices (**not including ghost elements**).
        """
        struct_is_none = (struct is None)
        HaloSubExtent.__init__(
            self,
            globale_extent=globale_extent,
            slice=slice,
            start=start,
            stop=stop,
            halo=halo,
            struct=struct
        )
        if struct_is_none:
            self._struct[self.PEER_RANK] = peer_rank
            self._struct[self.INTER_LOCALE_RANK] = inter_locale_rank

    def __eq__(self, other):
        """
        Equality
        """
        return \
            (
                HaloSubExtent.__eq__(self, other)
                and
                (self.peer_rank == other.peer_rank)
                and
                (self.inter_locale_rank == other.inter_locale_rank)
            )

    @property
    def peer_rank(self):
        """
        An :obj:`int` indicating the rank of the process in the :samp:`peer_comm` communicator
        which corresponds to the :attr:`inter_locale_rank` in
        the :samp:`inter_locale_comm` communicator.
        """
        return self._struct[self.PEER_RANK]

    @property
    def inter_locale_rank(self):
        """
        An :obj:`int` indicating the rank of the process in the :samp:`inter_locale_comm`
        responsible for exchanging data to/from this extent.
        """
        return self._struct[self.INTER_LOCALE_RANK]

    def halo_slab_extent(self, axis, dir):
        """
        Returns indexing extent of the halo *slab* for specified axis.

        :type axis: :obj:`int`
        :param axis: Indexing extent of halo slab for this axis.
        :type dir: :attr:`LO` or :attr:`HI`
        :param dir: Indicates low-index halo slab or high-index halo slab.
        :rtype: :obj:`IndexingExtent`
        :return: Indexing extent for halo slab.

        .. todo::

           Provide an example code here.

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

        .. todo::

           Provide an example code here.
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

    def to_tuple(self):
        """
        Convert this instance to a :obj:`tuple` which can be passed to constructor
        (or used as a :obj:`dict` key).

        :rtype: :obj:`tuple`
        :return: The :obj:`tuple` representation of this object.
        """
        return \
            (
                self.peer_rank,
                self.inter_locale_rank,
                None,
                None,
                tuple(tuple(row) for row in self.halo.tolist()),
                tuple(self.start_n),
                tuple(self.stop_n),
                None  # struct arg
            )

    def __repr__(self):
        """
        Stringize.
        """
        return \
            (
                (
                    "%s("
                    "start=%s, stop=%s, halo=%s, peer_rank=%s, inter_locale_rank=%s"
                    +
                    ", globale_extent=None"
                    +
                    ", struct=None"
                    ")"
                )
                %
                (
                    self.__class__.__name__,
                    repr(self.start_n.tolist()),
                    repr(self.stop_n.tolist()),
                    repr(self.halo.tolist()),
                    repr(self.peer_rank),
                    repr(self.inter_locale_rank),
                )
            )

    def __str__(self):
        """
        """
        return self.__repr__()


class ScalarLocaleExtent(LocaleExtent):

    """
    Indexing extent for a scalar.
    """

    def __init__(
        self,
        peer_rank,
        inter_locale_rank,
        struct=None
    ):
        LocaleExtent.__init__(
            self,
            peer_rank=peer_rank,
            inter_locale_rank=inter_locale_rank,
            globale_extent=ScalarGlobaleExtent(),
            start=(),
            stop=(),
            struct=struct
        )


class CartLocaleExtent(LocaleExtent):

    """
    Indexing extents for single tile of cartesian domain distribution.
    """

    CART_COORD = 5
    CART_COORD_STR = "cart_coord"
    CART_SHAPE = 6
    CART_SHAPE_STR = "cart_shape"

    struct_dtype_dict = _collections.defaultdict(lambda: None)

    @staticmethod
    def create_struct_dtype_from_ndim(cls, ndim):
        """
        Creates a :obj:`numpy.dtype` structure for holding start and stop indices.

        :rtype: :obj:`numpy.dtype`
        :return: :obj:`numpy.dtype` with :samp:`"start"` and :samp:`"stop"` multi-index
           fields of dimension :samp:`{ndim}`.
        """
        return \
            _np.dtype(
                [
                    (cls.START_STR, _np.int64, (ndim,)),
                    (cls.STOP_STR, _np.int64, (ndim,)),
                    (cls.HALO_STR, _np.int64, (ndim, 2)),
                    (cls.PEER_RANK_STR, _np.int64),
                    (cls.INTER_LOCALE_RANK_STR, _np.int64),
                    (cls.CART_COORD_STR, _np.int64, (ndim,)),
                    (cls.CART_SHAPE_STR, _np.int64, (ndim,))
                ]
            )

    def __init__(
        self,
        peer_rank=None,
        inter_locale_rank=None,
        cart_coord=None,
        cart_shape=None,
        globale_extent=None,
        slice=None,
        halo=None,
        start=None,
        stop=None,
        struct=None
    ):
        """
        Construct.

        :type peer_rank: :obj:`int`
        :param peer_rank: Rank of MPI process in :samp:`peer_comm` communicator which
           corresponds to the :samp:`{inter_locale_rank}` peer_rank in the :samp:`cart_comm`
           cartesian communicator.
        :type inter_locale_rank: :obj:`int`
        :param inter_locale_rank: Rank of MPI process in :samp:`cart_comm` cartesian communicator
           which corresponds to the :samp:`{peer_comm}` peer_rank in the :samp:`peer_comm`
           communicator.
        :type cart_coord: sequence of :obj:`int`
        :param cart_coord: Coordinate index (:meth:`mpi4py.MPI.CartComm.Get_coordinate`) of
           this :obj:`LocaleExtent` in the cartesian domain distribution.
        :type cart_shape: sequence of :obj:`int`
        :param cart_shape: Number of :obj:`LocaleExtent` regions in each axis direction
           of the cartesian distribution.
        :type globale_extent: :obj:`GlobaleExtent`
        :param globale_extent: The indexing extent of the entire array.
        :type slice: sequence of :obj:`slice`
        :param slice: Per-axis start and stop indices (**not including ghost elements**).
        :type halo: :samp:`(len({split}), 2)` shaped array of :obj:`int`
        :param halo: Desired halo, a :samp:`(len(self.start), 2)` shaped array of :obj:`int`
           indicating the per-axis number of outer ghost elements. :samp:`halo[:,0]` is the
           number of ghost elements on the low-index *side* and :samp:`halo[:,1]` is the number
           of ghost elements on the high-index *side*. **Note**: that the halo will be truncated
           so that this halo extent does not extend beyond the halo :samp:`{globale_extent}`.
        :type start: sequence of :obj:`slice`
        :param start: Per-axis start indices (**not including ghost elements**).
        :type stop: sequence of :obj:`slice`
        :param stop: Per-axis stop indices (**not including ghost elements**).
        """
        struct_is_none = (struct is None)
        LocaleExtent.__init__(
            self,
            peer_rank=peer_rank,
            inter_locale_rank=inter_locale_rank,
            globale_extent=globale_extent,
            slice=slice,
            halo=halo,
            start=start,
            stop=stop,
            struct=struct
        )
        if struct_is_none:
            self._struct[self.CART_COORD] = cart_coord
            self._struct[self.CART_SHAPE] = cart_shape

    def __eq__(self, other):
        """
        Equality.
        """
        return \
            (
                LocaleExtent.__eq__(self, other)
                and
                _np.all(self.cart_coord == other.cart_coord)
                and
                _np.all(self.cart_shape == other.cart_shape)
            )

    @property
    def cart_rank(self):
        """
        An :obj:`int` indicating the rank of the process in the :samp:`cart_comm`
        cartesian communicator which corresponds to the :attr:`{peer_rank}` rank
        in the :samp:`peer_comm` communicator.
        """
        return self.inter_locale_rank

    @property
    def cart_coord(self):
        """
        A :obj:`tuple` of :obj:`int` indicating the coordinate
        index (:meth:`mpi4py.MPI.CartComm.Get_coordinate`) of
        this :obj:`LocaleExtent` in the cartesian domain distribution.
        """
        return self._struct[self.CART_COORD]

    @property
    def cart_shape(self):
        """
        A :obj:`tuple` of :obj:`int` indicating the number of :obj:`LocaleExtent`
        regions in each axis direction of the cartesian distribution.
        """
        return self._struct[self.CART_SHAPE]

    def to_tuple(self):
        """
        Convert this instance to a :obj:`tuple` which can be passed to constructor
        (or used as a :obj:`dict` key).

        :rtype: :obj:`tuple`
        :return: The :obj:`tuple` representation of this object.
        """
        return \
            (
                self.peer_rank,
                self.inter_locale_rank,
                tuple(self.cart_coord),
                tuple(self.cart_shape),
                None,
                None,
                tuple(tuple(row) for row in self.halo.tolist()),
                tuple(self.start_n),
                tuple(self.stop_n),
                None  # struct arg
            )

    def __repr__(self):
        """
        Stringize.
        """
        return \
            (
                (
                    "%s("
                    +
                    "start=%s, stop=%s, halo=%s, peer_rank=%s, "
                    +
                    "inter_locale_rank=%s, "
                    +
                    "cart_coord=%s, cart_shape=%s"
                    +
                    ", globale_extent=None"
                    +
                    ", struct=None"
                    +
                    ")"
                )
                %
                (
                    self.__class__.__name__,
                    repr(self.start_n.tolist()),
                    repr(self.stop_n.tolist()),
                    repr(self.halo.tolist()),
                    repr(self.peer_rank),
                    repr(self.inter_locale_rank),
                    repr(tuple(self.cart_coord)),
                    repr(tuple(self.cart_shape)),
                )
            )

    def __str__(self):
        """
        """
        return self.__repr__()


class Distribution(object):

    """
    Describes the apportionment of array extents amongst locales.
    """

    #: The "low index" indices.
    LO = HaloIndexingExtent.LO

    #: The "high index" indices.
    HI = HaloIndexingExtent.HI

    def __init__(
        self,
        globale_extent,
        locale_extents,
        halo=0,
        globale_extent_type=GlobaleExtent,
        locale_extent_type=LocaleExtent,
        inter_locale_rank_to_peer_rank=None
    ):
        """
        Construct.

        :type globale_extent: :obj:`object`
        :param globale_extent: Can be specified as a *sequence-of-int* shape,
            *sequence-of-slice* slice or a :obj:`mpi_array.indexing.IndexingExtent`.
            Defines the globale extent of the array.
        :type locale_extents: sequence of :obj:`object`
        :param locale_extents: Can be specified as a
            sequence of *sequence-of-slice* slices or
            a sequence of :obj:`mpi_array.indexing.IndexingExtent`.
            Defines the distribution of the globale array over locales.
            The element :samp:`locale_extents[r]` defines the extent
            which is to be allocated by :samp:`inter_locale_comm` rank :samp:`r`.
        :type halo: :obj:`int`, sequence of :obj:`int` or :samp:`(ndim, 2)` shaped array
        :param halo: Locale array halo (ghost elements).
        :type globale_extent_type: :obj:`object`
        :param globale_extent_type: The class/type for :attr:`globale_extent`.
        :type locale_extent_type: :obj:`object`
        :param locale_extent_type: The class/type for elements of :attr:`locale_extents`.
        :type inter_locale_rank_to_peer_rank: sequence of :obj:`int` or :obj:`dict`
        :param inter_locale_rank_to_peer_rank: A :obj:`dict`
           of :samp:`(inter_locale_rank, peer_rank)` pairs. If a sequence,
           then :samp:`{inter_locale_rank_to_peer_rank}[inter_locale_rank] = peer_rank`.

        .. todo::

           Provide example here.

        """
        self._peer_ranks_per_locale = None
        self._locale_extent_type = locale_extent_type
        self._globale_extent_type = globale_extent_type
        self._inter_locale_rank_to_peer_rank = inter_locale_rank_to_peer_rank

        self._globale_extent = self.create_globale_extent(globale_extent, halo=0)
        ndim = self._globale_extent.ndim
        self._halo = \
            _convert_halo_to_array_form(halo=_copy.deepcopy(halo), ndim=ndim)
        num_locales = len(locale_extents)

        self._struct_locale_extents = \
            self.create_struct_locale_extents(num_locales=num_locales, ndim=ndim)

        self.initialise_struct_locale_extents(self._struct_locale_extents, locale_extents)

        self._locale_extents = self.create_locale_extents(self._struct_locale_extents)

    def create_struct_locale_extents(self, num_locales, ndim):
        """
        """
        return \
            _np.zeros(
                (num_locales,),
                dtype=self._locale_extent_type.get_struct_dtype_from_ndim(
                    self._locale_extent_type,
                    ndim
                )
            )

    def initialise_struct_locale_extents(self, struct_locale_extents, locale_extents_descr):
        """
        """
        num_locales = len(locale_extents_descr)
        START_STR = self._locale_extent_type.START_STR
        STOP_STR = self._locale_extent_type.STOP_STR
        HALO_STR = self._locale_extent_type.HALO_STR
        PEER_RANK_STR = self._locale_extent_type.PEER_RANK_STR
        INTER_LOCALE_RANK_STR = self._locale_extent_type.INTER_LOCALE_RANK_STR
        ndim = 0
        if num_locales > 0:
            ndim = len(struct_locale_extents[START_STR][0])
        if (num_locales > 0) and (ndim > 0):
            initialise_halos = True
            struct_locale_extents[INTER_LOCALE_RANK_STR] = _np.arange(0, num_locales)
            if self._inter_locale_rank_to_peer_rank is not None:
                if isinstance(self._inter_locale_rank_to_peer_rank, _np.ndarray):
                    struct_locale_extents[PEER_RANK_STR] = \
                        self._inter_locale_rank_to_peer_rank[
                            struct_locale_extents[INTER_LOCALE_RANK_STR]
                    ]
                else:
                    struct_locale_extents[PEER_RANK_STR] = \
                        tuple(
                            self._inter_locale_rank_to_peer_rank[inter_locale_rank]
                            for inter_locale_rank in range(num_locales)
                    )
            else:
                struct_locale_extents[PEER_RANK_STR] = _mpi.UNDEFINED

            if (
                isinstance(locale_extents_descr, _np.ndarray)
                and
                (START_STR in locale_extents_descr.dtype.names)
                and
                (STOP_STR in locale_extents_descr.dtype.names)
            ):
                struct_locale_extents[START_STR] = locale_extents_descr[START_STR]
                struct_locale_extents[STOP_STR] = locale_extents_descr[STOP_STR]
                if HALO_STR in locale_extents_descr.dtype.names:
                    struct_locale_extents[HALO_STR] = locale_extents_descr[HALO_STR]
                    initialise_halos = False
            elif (
                (
                    hasattr(locale_extents_descr[0], "__iter__")
                    or
                    hasattr(locale_extents_descr[0], "__getitem__")
                )
                and
                _np.all([isinstance(e, slice) for e in locale_extents_descr[0]])
            ):
                # assume they are all slices
                start_stop = \
                    _np.array(
                        tuple(
                            _np.array(
                                tuple(
                                    (slc.start, slc.stop)
                                    for slc in le_descr
                                )
                            ).T
                            for le_descr in locale_extents_descr
                        )
                    )
                struct_locale_extents[START_STR] = start_stop[:, 0]
                struct_locale_extents[STOP_STR] = start_stop[:, 1]
            elif (
                (
                    hasattr(locale_extents_descr[0], "start")
                    or
                    hasattr(locale_extents_descr[0], "stop")
                )
            ):
                start_stop = \
                    _np.array(
                        tuple(
                            (le_descr.start, le_descr.stop)
                            for le_descr in locale_extents_descr
                        )
                    )
                struct_locale_extents[START_STR] = start_stop[:, 0]
                struct_locale_extents[STOP_STR] = start_stop[:, 1]
            elif (
                (
                    hasattr(locale_extents_descr[0], "__iter__")
                    or
                    hasattr(locale_extents_descr[0], "__getitem__")
                )
                and
                _np.all(
                    [
                        (hasattr(e, "__int__") or hasattr(e, "__long__"))
                        for e in iter(locale_extents_descr[0])
                    ]
                )
            ):
                struct_locale_extents[START_STR] = 0
                struct_locale_extents[STOP_STR] = locale_extents_descr
            else:
                raise ValueError(
                    "Could not construct dtype=%s instances from locale_extents_descr=%s."
                    %
                    (struct_locale_extents.dtype, locale_extents_descr,)
                )
            if initialise_halos:
                self.initialise_struct_locale_extents_halos(
                    struct_locale_extents,
                    self._globale_extent,
                    self._halo
                )

    def initialise_struct_locale_extents_halos(self, struct_locale_extents, globale_extent, halo):
        """
        Trim locale extent halos so they don't extend beyond the :samp:`{globale_extent}` halo.
        """
        START_N_STR = self._locale_extent_type.START_N_STR
        STOP_N_STR = self._locale_extent_type.STOP_N_STR
        HALO_STR = self._locale_extent_type.HALO_STR
        ge_start_h = globale_extent.start_h
        ge_stop_h = globale_extent.stop_h

        msk = struct_locale_extents[STOP_N_STR] > struct_locale_extents[START_N_STR]
        struct_locale_extents[HALO_STR][:, :, self.LO] = \
            _np.maximum(
                0,
                _np.minimum(
                    _np.where(msk, struct_locale_extents[START_N_STR] - ge_start_h, 0),
                    halo[:, self.LO]
                )
        )
        struct_locale_extents[HALO_STR][:, :, self.HI] = \
            _np.maximum(
                0,
                _np.minimum(
                    _np.where(msk, ge_stop_h - struct_locale_extents[STOP_N_STR], 0),
                    halo[:, self.HI]
                )
        )

    def create_locale_extents(self, struct_locale_extents):
        """
        """
        return \
            _np.array(
                tuple(
                    self._locale_extent_type(struct=struct_locale_extents[i])
                    for i in range(len(struct_locale_extents))
                ),
                dtype="object"
            )

    def get_peer_rank(self, inter_locale_rank):
        """
        Returns the :samp:`peer_rank` rank (of :samp:`peer_comm`) which is
        is the equivalent process of the  :samp:`{inter_locale_rank}` rank
        of the :samp:`inter_locale_comm` communicator.

        :type inter_locale_rank: :obj:`int`
        :param inter_locale_rank: A rank of :samp:`inter_locale_comm`.
        :rtype: :obj:`int`
        :return: The equivalent rank from :samp:`peer_comm`.
        """
        rank = _mpi.UNDEFINED
        if self._inter_locale_rank_to_peer_rank is not None:
            rank = self._inter_locale_rank_to_peer_rank[inter_locale_rank]
        return rank

    def create_globale_extent(self, globale_extent, halo=0):
        """
        Factory function for creating :obj:`GlobaleExtent` object.

        :type globale_extent: :obj:`object`
        :param globale_extent: Can be specified as a *sequence-of-int* shape,
            *sequence-of-slice* slice or a :obj:`mpi_array.indexing.IndexingExtent`.
            Defines the globale extent of the array.
        :type halo: :obj:`int`
        :param halo: Globale array halo (border), currently ignored.
        :rtype: :obj:`GlobaleExtent`
        :return: A :samp:`self._globale_extent_type` instance.

        .. todo::

           Handle :samp:`{globale_extent}` for non-zero :samp:`{halo}`.
        """

        # Don't support globale halo/border yet.
        halo = 0
        if isinstance(globale_extent, self._globale_extent_type):
            globale_extent = _copy.deepcopy(globale_extent)
            globale_extent.halo = halo
        elif (
            hasattr(globale_extent, "__iter__")
            and
            _np.all([isinstance(e, slice) for e in iter(globale_extent)])
        ):
            globale_extent = self._globale_extent_type(slice=globale_extent, halo=halo)
        elif hasattr(globale_extent, "start") and hasattr(globale_extent, "stop"):
            globale_extent = \
                self._globale_extent_type(
                    start=globale_extent.start,
                    stop=globale_extent.stop,
                    halo=halo
                )
        elif (
            (hasattr(globale_extent, "__iter__") or hasattr(globale_extent, "__getitem__"))
            and
            _np.all(
                [
                    (hasattr(e, "__int__") or hasattr(e, "__long__"))
                    for e in iter(globale_extent)
                ]
            )
        ):
            stop = _np.array(globale_extent)
            globale_extent = \
                self._globale_extent_type(start=_np.zeros_like(stop), stop=stop, halo=halo)
        else:
            raise ValueError(
                "Could not construct %s instance from globale_extent=%s."
                %
                (self._globale_extent_type.__name__, globale_extent,)
            )

        return globale_extent

    def create_locale_extent(
            self,
            inter_locale_rank,
            locale_extent,
            globale_extent,
            halo=0,
            **kwargs
    ):
        """
        Factory function for creating :obj:`LocaleExtent` object.
        The :samp:`**kwargs` are passed through to
        the :samp:`self._locale_extent_type` constructor.

        :type inter_locale_rank: :obj:`int`
        :param inter_locale_rank: Rank of :samp:`inter_locale_comm` which is
            responsible for exchanging data to/from the array region defined
            by the returned locale extent.
        :type locale_extent: :obj:`object`
        :param locale_extent: Can be specified as a *sequence-of-int* shape,
            *sequence-of-slice* slice or a :obj:`mpi_array.indexing.IndexingExtent`.
            Defines the locale extent of the array.
        :type globale_extent: :obj:`GlobaleExtent`
        :param globale_extent: The globale array extent.
        :type halo: :obj:`int`, sequence of :obj:`int`,...
        :param halo: Locale array halo (ghost elements).
        :rtype: :obj:`LocaleExtent`
        :return: A :samp:`self._locale_extent_type` instance.
        """
        peer_rank = self.get_peer_rank(inter_locale_rank)
        if hasattr(locale_extent, "start") and hasattr(locale_extent, "stop"):
            locale_extent = \
                self._locale_extent_type(
                    peer_rank=peer_rank,
                    inter_locale_rank=inter_locale_rank,
                    globale_extent=globale_extent,
                    start=locale_extent.start,
                    stop=locale_extent.stop,
                    halo=halo,
                    **kwargs
                )
        elif (
            (hasattr(locale_extent, "__iter__") or hasattr(locale_extent, "__getitem__"))
            and
            _np.all([isinstance(e, slice) for e in locale_extent])
        ):
            locale_extent = \
                self._locale_extent_type(
                    peer_rank=peer_rank,
                    inter_locale_rank=inter_locale_rank,
                    globale_extent=globale_extent,
                    slice=locale_extent,
                    halo=halo,
                    **kwargs
                )
        elif (
            (hasattr(locale_extent, "__iter__") or hasattr(locale_extent, "__getitem__"))
            and
            _np.all(
                [
                    (hasattr(e, "__int__") or hasattr(e, "__long__"))
                    for e in iter(locale_extent)
                ]
            )
        ):
            stop = _np.array(locale_extent)
            locale_extent = \
                self._locale_extent_type(
                    peer_rank=peer_rank,
                    inter_locale_rank=inter_locale_rank,
                    globale_extent=globale_extent,
                    start=_np.zeros_like(stop),
                    stop=stop,
                    halo=halo,
                    **kwargs
                )
        else:
            raise ValueError(
                "Could not construct %s instance from locale_extent=%s."
                %
                (self._locale_extent_type.__name__, locale_extent,)
            )

        return locale_extent

    def get_extent_for_rank(self, inter_locale_rank):
        """
        Returns extent associated with the specified rank
        of the :attr:`inter_locale_comm` communicator.

        :type inter_locale_rank: :obj:`int`
        :param inter_locale_rank: Return the locale extent
           associated with this rank.
        :rtype: :obj:`LocaleExtent`
        :return: The locale extent for the specified :samp:`{inter_locale_rank}` rank.
        """
        return self._locale_extents[inter_locale_rank]

    @property
    def halo(self):
        """
        A :samp:`(ndim, 2)` shaped array of :obj:`int` indicating the
        halo (ghost elements) for extents. This may differ from the :attr:`LocaleExtent.halo`
        value due to the locale extent halos getting trimmed to lie within the globale extent.
        """
        return self._halo

    @property
    def globale_extent(self):
        """
        A :obj:`GlobaleExtent` specifying the globale array indexing extent.
        """
        return self._globale_extent

    @property
    def locale_extents(self):
        """
        Sequence of :obj:`LocaleExtent` objects where :samp:`locale_extents[r]`
        is the extent assigned to locale with :samp:`inter_locale_comm` rank :samp:`r`.
        """
        return self._locale_extents

    @property
    def num_locales(self):
        """
        An :obj:`int` specifying the number of locales in this distribution.
        """
        return len(self._locale_extents)

    @property
    def peer_ranks_per_locale(self):
        """
        A :obj:`numpy.ndarray` of length :attr:`num_locales`. Each element
        of the array is a sequence of :obj:`int` such that :samp:`self.peer_ranks_per_locale[r]`
        are the ranks of :samp:`peer_comm` which belong to the locale associated
        with :samp:`self.locale_extents[r]`.
        """
        return self._peer_ranks_per_locale

    @peer_ranks_per_locale.setter
    def peer_ranks_per_locale(self, prpl):
        self._peer_ranks_per_locale = prpl


class ClonedDistribution(Distribution):

    """
    Distribution where entire globale extent elements occur on every locale.
    """

    def __init__(self, globale_extent, num_locales, halo=0, inter_locale_rank_to_peer_rank=None):
        """
        Initialise.

        :type globale_extent: :obj:`object`
        :param globale_extent: Can be specified as a *sequence-of-int* shape,
            *sequence-of-slice* slice or a :obj:`mpi_array.indexing.IndexingExtent`.
            Defines the globale extent of the array.
        :type num_locales: :obj:`int`
        :param num_locales: Number of locales over which the globale array is cloned.
        :type halo: :obj:`int`, sequence of :obj:`int` or :samp:`(ndim, 2)` shaped array
        :param halo: Locale array halo (ghost elements).
        :type inter_locale_rank_to_peer_rank: sequence of :obj:`int` or :obj:`dict`
        :param inter_locale_rank_to_peer_rank: A :obj:`dict`
           of :samp:`(inter_locale_rank, peer_rank)` pairs. If a sequence,
           then :samp:`{inter_locale_rank_to_peer_rank}[inter_locale_rank] = peer_rank`.

        .. todo::

           Provide example here.
        """
        Distribution.__init__(
            self,
            globale_extent=globale_extent,
            locale_extents=[_copy.deepcopy(globale_extent) for i in range(num_locales)],
            halo=halo,
            inter_locale_rank_to_peer_rank=inter_locale_rank_to_peer_rank
        )


class SingleLocaleDistribution(Distribution):

    """
    Distribution where entire globale extent elements occur on just a single locale.
    """

    def __init__(
        self,
        globale_extent,
        num_locales,
        inter_locale_rank=0,
        halo=0,
        inter_locale_rank_to_peer_rank=None
    ):
        """
        Initialise.

        :type globale_extent: :obj:`object`
        :param globale_extent: Can be specified as a *sequence-of-int* shape,
            *sequence-of-slice* slice or a :obj:`mpi_array.indexing.IndexingExtent`.
            Defines the globale extent of the array.
        :type num_locales: :obj:`int`
        :param num_locales: Number of locales. One non-empty extent on the first
           (i.e. :samp:`inter_locale_rank == 0` rank) locale, empty extents on
           remaining locales.
        :type halo: :obj:`int`, sequence of :obj:`int` or :samp:`(ndim, 2)` shaped array
        :param halo: Locale array halo (ghost elements).
        :type inter_locale_rank_to_peer_rank: sequence of :obj:`int` or :obj:`dict`
        :param inter_locale_rank_to_peer_rank: A :obj:`dict`
           of :samp:`(inter_locale_rank, peer_rank)` pairs. If a sequence,
           then :samp:`{inter_locale_rank_to_peer_rank}[inter_locale_rank] = peer_rank`.
        """
        self._halo = halo
        self._globale_extent_type = GlobaleExtent
        globale_extent = self.create_globale_extent(globale_extent)
        sidx = _np.array(globale_extent.start_n)
        locale_extents = [HaloIndexingExtent(start=sidx, stop=sidx) for i in range(num_locales)]
        locale_extent = locale_extents[inter_locale_rank]
        locale_extent.start_n = globale_extent.start_n
        locale_extent.stop_n = globale_extent.stop_n
        Distribution.__init__(
            self,
            globale_extent=globale_extent,
            locale_extents=locale_extents,
            halo=halo,
            inter_locale_rank_to_peer_rank=inter_locale_rank_to_peer_rank,
            globale_extent_type=self._globale_extent_type
        )


def convert_slice_split_to_struct(splt, globale_extent, halo):
    """
    """
    LO = HaloIndexingExtent.LO
    HI = HaloIndexingExtent.HI

    START_N_STR = HaloIndexingExtent.START_N_STR
    STOP_N_STR = HaloIndexingExtent.STOP_N_STR
    HALO_STR = HaloIndexingExtent.HALO_STR

    ndim = len(splt.flatten()[0])
    dt = \
        HaloIndexingExtent.get_struct_dtype_from_ndim(
            HaloIndexingExtent,
            ndim=ndim
        )
    splt_ary = _np.zeros(splt.shape, dtype=dt)
    for i in range(splt.size):
        midx = _np.unravel_index(i, splt.shape)
        splt_ary[START_N_STR][midx] = tuple(slc.start for slc in splt[midx])
        splt_ary[STOP_N_STR][midx] = tuple(slc.stop for slc in splt[midx])

    ge_start_h = globale_extent.start_h
    ge_stop_h = globale_extent.stop_h

    halo = _convert_halo_to_array_form(halo, ndim)
    msk = splt_ary[STOP_N_STR] > splt_ary[START_N_STR]
    slice_lo = (slice(None),) * ndim + (slice(None), LO)
    splt_ary[HALO_STR][slice_lo] = \
        _np.maximum(
            0,
            _np.minimum(
                _np.where(msk, splt_ary[START_N_STR] - ge_start_h, 0),
                halo[:, LO]
            )
    )
    slice_hi = (slice(None),) * ndim + (slice(None), HI)
    splt_ary[HALO_STR][slice_hi] = \
        _np.maximum(
            0,
            _np.minimum(
                _np.where(msk, ge_stop_h - splt_ary[STOP_N_STR], 0),
                halo[:, HI]
            )
    )
    return splt_ary


class BlockPartition(Distribution):

    """
    Block partition of an array (shape) over locales.
    """

    _split_cache = _collections.defaultdict(lambda: None)

    def __init__(
        self,
        globale_extent,
        dims,
        cart_coord_to_cart_rank,
        halo=0,
        order="C",
        inter_locale_rank_to_peer_rank=None
    ):
        """
        Creates a block-partitioning of :samp:`{shape}` over locales.

        :type globale_extent: :obj:`GlobaleExtent`
        :param globale_extent: The globale extent to be partitioned.
        :type dims: sequence of :obj:`int`
        :param dims: The number of partitions along each
            dimension, :samp:`len({dims}) == len({globale_extent}.shape_n)`
            and :samp:`num_locales = numpy.product({dims})`.
        :type halo: :obj:`int`, sequence of :obj:`int` or :samp:`(len({shape}), 2)` shaped array.
        :param halo: Number of *ghost* elements added per axis
           (low-index number of ghost elements may differ to the
           number of high-index ghost elements).
        :type cart_coord_to_cart_rank: :obj:`dict`
        :param cart_coord_to_cart_rank: Mapping between cartesian
           communicator coordinate (:meth:`mpi4py.MPI.CartComm.Get_coords`)
           and cartesian communicator rank.
        """
        self._globale_extent_type = GlobaleExtent
        globale_extent = self.create_globale_extent(globale_extent, halo)
        self._dims = dims
        ndim = len(self._dims)
        halo = _convert_halo_to_array_form(halo, ndim)

        key = (globale_extent.to_tuple(), tuple(self._dims), tuple(tuple(row) for row in halo))
        splt = self._split_cache[key]
        if splt is None:
            shape_splitter = \
                _array_split.ShapeSplitter(
                    array_shape=globale_extent.shape_n,
                    array_start=globale_extent.start_n,
                    axis=self._dims,
                    halo=0
                )
            splt = \
                convert_slice_split_to_struct(
                    shape_splitter.calculate_split(),
                    globale_extent,
                    halo
                )
            self._split_cache[key] = splt

        locale_extents = _np.empty(splt.size, dtype=splt.dtype)
        for i in range(locale_extents.size):
            cart_coord = tuple(_np.unravel_index(i, splt.shape))
            locale_extents[cart_coord_to_cart_rank[cart_coord]] = splt[cart_coord]

        self._cart_coord_to_cart_rank = cart_coord_to_cart_rank
        self._cart_rank_to_cart_coord_map = \
            {cart_coord_to_cart_rank[c]: c for c in cart_coord_to_cart_rank.keys()}
        self._cart_rank_to_cart_coord_map = \
            _np.array(
                tuple(
                    self._cart_rank_to_cart_coord_map[cart_rank]
                    for cart_rank in range(len(self._cart_rank_to_cart_coord_map.keys()))
                ),
                dtype="int64"
            )
        Distribution.__init__(
            self,
            globale_extent=globale_extent,
            locale_extents=locale_extents,
            inter_locale_rank_to_peer_rank=inter_locale_rank_to_peer_rank,
            halo=halo,
            locale_extent_type=CartLocaleExtent,
            globale_extent_type=self._globale_extent_type
        )

    def initialise_struct_locale_extents(self, struct_locale_extents, locale_extents_descr):
        """
        """
        Distribution.initialise_struct_locale_extents(
            self,
            struct_locale_extents,
            locale_extents_descr
        )
        num_locales = len(locale_extents_descr)
        CART_COORD_STR = self._locale_extent_type.CART_COORD_STR
        CART_SHAPE_STR = self._locale_extent_type.CART_SHAPE_STR
        if num_locales > 0:
            struct_locale_extents[CART_COORD_STR] = \
                self._cart_rank_to_cart_coord_map[_np.arange(num_locales)]
            struct_locale_extents[CART_SHAPE_STR] = self._dims

    def create_locale_extent(
            self,
            inter_locale_rank,
            locale_extent,
            globale_extent,
            halo=0,
            **kwargs
    ):
        """
        Over-rides :meth:`Distrbution.create_locale_extent`.

        :rtype: :obj:`CartLocaleExtent`
        :returns: A :obj:`CartLocaleExtent` instance.
        """
        return \
            Distribution.create_locale_extent(
                self,
                inter_locale_rank,
                locale_extent,
                globale_extent,
                halo,
                cart_coord=self._cart_rank_to_cart_coord_map[inter_locale_rank],
                cart_shape=self._dims,
                **kwargs
            )

    def __str__(self):
        """
        Stringify.
        """
        s = [str(le) for le in self.locale_extents]
        return ", ".join(s)


__all__ = [s for s in dir() if not s.startswith('_')]
