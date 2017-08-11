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

import array_split as _array_split
import array_split.split  # noqa: F401
from array_split.split import convert_halo_to_array_form as _convert_halo_to_array_form

from . import logging as _logging
from .indexing import IndexingExtent, HaloIndexingExtent


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class GlobaleExtent(HaloIndexingExtent):

    """
    Indexing extent for an entire array.
    """

    pass


class HaloSubExtent(HaloIndexingExtent):

    """
    Indexing extent for single region of a larger globale extent.
    """

    def __init__(
        self,
        globale_extent,
        slice=None,
        halo=0,
        start=None,
        stop=None
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
        HaloIndexingExtent.__init__(self, slice=slice, start=start, stop=stop, halo=None)
        halo = _convert_halo_to_array_form(halo, ndim=self.ndim)
        # Calculate the locale halo, truncate if it strays outside
        # the globale_extent halo region.
        halo = \
            _np.maximum(
                _np.array((0,), dtype=halo.dtype),
                _np.array(
                    (
                        _np.minimum(
                            self.start_n - globale_extent.start_h,
                            halo[:, self.LO]
                        ),
                        _np.minimum(
                            globale_extent.stop_h - self.stop_n,
                            halo[:, self.HI]
                        ),
                    ),
                    dtype=halo.dtype
                ).T
            )
        self._halo = halo


class LocaleExtent(HaloSubExtent):

    """
    Indexing extent for single region of array residing on a locale.
    """

    def __init__(
        self,
        peer_rank,
        inter_locale_rank,
        globale_extent,
        slice=None,
        halo=0,
        start=None,
        stop=None
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
        self._peer_rank = peer_rank
        self._inter_locale_rank = inter_locale_rank
        HaloSubExtent.__init__(
            self,
            globale_extent=globale_extent,
            slice=slice,
            start=start,
            stop=stop,
            halo=halo
        )

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
        return self._peer_rank

    @property
    def inter_locale_rank(self):
        """
        An :obj:`int` indicating the rank of the process in the :samp:`inter_locale_comm`
        responsible for exchanging data to/from this extent.
        """
        return self._inter_locale_rank

    def halo_slab_extent(self, axis, dir):
        """
        Returns indexing extent of the halo *slab* for specified axis.

        :type axis: :obj:`int`
        :param axis: Indexing extent of halo slab for this axis.
        :type dir: :attr:`LO` or :attr:`HI`
        :param dir: Indicates low-index halo slab or high-index halo slab.
        :rtype: :obj:`IndexingExtent`
        :return: Indexing extent for halo slab.
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

    def __repr__(self):
        """
        Stringize.
        """
        return \
            (
                (
                    "LocaleExtent(start=%s, stop=%s, halo=%s, peer_rank=%s, inter_locale_rank=%s)"
                )
                %
                (
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


class CartLocaleExtent(LocaleExtent):

    """
    Indexing extents for single tile of cartesian domain distribution.
    """

    def __init__(
        self,
        peer_rank,
        inter_locale_rank,
        cart_coord,
        cart_shape,
        globale_extent,
        slice=None,
        halo=None,
        start=None,
        stop=None
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
        LocaleExtent.__init__(
            self,
            peer_rank=peer_rank,
            inter_locale_rank=inter_locale_rank,
            globale_extent=globale_extent,
            slice=slice,
            halo=halo,
            start=start,
            stop=stop
        )
        self._cart_coord = _np.array(cart_coord, dtype="int64")
        self._cart_shape = _np.array(cart_shape, dtype=self._cart_coord.dtype)

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
        return self._cart_coord

    @property
    def cart_shape(self):
        """
        A :obj:`tuple` of :obj:`int` indicating the number of :obj:`LocaleExtent`
        regions in each axis direction of the cartesian distribution.
        """
        return self._cart_shape

    def __repr__(self):
        """
        Stringize.
        """
        return \
            (
                (
                    "CartLocaleExtent(start=%s, stop=%s, halo=%s, peer_rank=%s, "
                    +
                    "inter_locale_rank=%s, "
                    +
                    "cart_coord=%s, cart_shape=%s)"
                )
                %
                (
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
        Initialise.
        """
        self._locale_extent_type = locale_extent_type
        self._globale_extent_type = globale_extent_type
        self._inter_locale_rank_to_peer_rank = inter_locale_rank_to_peer_rank

        self._globale_extent = self.create_globale_extent(globale_extent, halo=0)
        self._halo = \
            _convert_halo_to_array_form(halo=_copy.deepcopy(halo), ndim=self._globale_extent.ndim)
        self._locale_extents = _copy.copy(locale_extents)
        for i in range(len(locale_extents)):
            self._locale_extents[i] = \
                self.create_locale_extent(i, locale_extents[i], self._globale_extent, halo)

    def get_peer_rank(self, inter_locale_rank):
        """
        Returns the :samp:`peer_rank` rank (of :samp:`peer_comm`) which is
        is the equivalent process of the  :samp:`{inter_locale_rank}` rank
        of the :samp:`inter_locale_comm` communicator.
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

        :todo: Handle globale_extent halo.
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
                (self._globale_extent.__class__.__name__, globale_extent,)
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
                (self._locale_extent_type.__class__.__name__, locale_extent,)
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


class ClonedDistribution(Distribution):

    """
    Distribution where entire globale extent elements occur on every locale.
    """

    def __init__(self, globale_extent, num_locales, halo=0, inter_locale_rank_to_peer_rank=None):
        """
        Initialise.
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
        """
        self._halo = halo
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
            inter_locale_rank_to_peer_rank=inter_locale_rank_to_peer_rank
        )


class BlockPartition(Distribution):

    """
    Block partition of an array (shape) over locales.
    """

    #: The "low index" indices.
    LO = HaloIndexingExtent.LO

    #: The "high index" indices.
    HI = HaloIndexingExtent.HI

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
        Create a partitioning of :samp:`{shape}` over locales.

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
        self._num_locales = _np.product(dims)
        self._dims = dims
        self._rank_logger = None
        self._root_logger = None
        self._order = order
        self._halo_updates_dict = None

        if self._num_locales > 1:
            shape_splitter = \
                _array_split.ShapeSplitter(
                    array_shape=globale_extent.shape_n,
                    array_start=globale_extent.start_n,
                    axis=self._dims,
                    halo=0
                )
            splt = shape_splitter.calculate_split()

            locale_extents = _np.empty(splt.size, dtype="object")
            for i in range(locale_extents.size):
                cart_coord = tuple(_np.unravel_index(i, splt.shape))
                locale_extents[cart_coord_to_cart_rank[cart_coord]] = splt[cart_coord]
        else:
            locale_extents = [globale_extent, ]
            if cart_coord_to_cart_rank is None:
                cart_coord_to_cart_rank = {tuple(_np.zeros_like(globale_extent.shape_n)): 0}

        self._cart_coord_to_cart_rank = cart_coord_to_cart_rank
        self._cart_rank_to_cart_coord_map = \
            {cart_coord_to_cart_rank[c]: c for c in cart_coord_to_cart_rank.keys()}
        Distribution.__init__(
            self,
            globale_extent=globale_extent,
            locale_extents=locale_extents,
            inter_locale_rank_to_peer_rank=inter_locale_rank_to_peer_rank,
            halo=halo,
            locale_extent_type=CartLocaleExtent,
            globale_extent_type=self._globale_extent_type
        )

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

    @property
    def rank_logger(self):
        """
        A :obj:`logging.Logger` for :attr:`peer_comm` communicator ranks.
        """
        if self._rank_logger is None:
            self._rank_logger = \
                _logging.get_rank_logger(
                    __name__ + "." + self.__class__.__name__,
                    comm=self.peer_comm
                )
        return self._rank_logger

    @property
    def root_logger(self):
        """
        A :obj:`logging.Logger` for *peer rank* :samp:`0` of the :attr:`peer_comm` communicator.
        """
        if self._root_logger is None:
            self._root_logger = \
                _logging.get_root_logger(
                    __name__ + "." + self.__class__.__name__,
                    comm=self.peer_comm
                )
        return self._root_logger


__all__ = [s for s in dir() if not s.startswith('_')]
