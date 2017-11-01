"""
====================================
The :mod:`mpi_array.indexing` Module
====================================

Various calculations for array indexing and array indexing extents.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   IndexingExtent - Index range for a tile of a decomposition.
   HaloIndexingExtent - Index range, with ghost elements, for a tile of a decomposition.
   calc_intersection_split - decompose an extent based on intersection with another extent.

"""
from __future__ import absolute_import

import numpy as _np
import copy as _copy
import collections as _collections
from array_split.split import convert_halo_to_array_form

from .license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

START = 0
STOP = 1
HALO = 2


class IndexingExtent(object):

    """
    Indexing bounds for a single tile of domain decomposition.
    """

    struct_dtype_dict = _collections.defaultdict(lambda: None)

    def create_struct_dtype(self, ndim):
        """
        Creates a :obj:`numpy.dtype` structure for holding start and stop indices.

        :rtype: :obj:`numpy.dtype`
        :return: :obj:`numpy.dtype` with :samp:`"start"` and :samp:`"stop"` multi-index
           fields of dimension :samp:`{ndim}`.
        """
        return _np.dtype([("start", _np.int64, (ndim,)), ("stop", _np.int64, (ndim,))])

    def get_struct_dtype(self, ndim):
        """
        """
        dtype = self.struct_dtype_dict[ndim]
        if dtype is None:
            dtype = self.create_struct_dtype(ndim)
            self.struct_dtype_dict[ndim] = dtype
        return dtype

    def __init__(self, slice=None, start=None, stop=None, struct=None):
        """
        Construct, must specify either :samp:`{slice}` or
        both of :samp:`{start}` and :samp:`{stop}`.

        :type slice: sequence of :obj:`slice`
        :param slice: Per axis start and stop indices defining the extent.
        :type start: sequence of :obj:`int`
        :param start: Per axis *start* indices defining the start of extent.
        :type stop: sequence of :obj:`int`
        :param stop: Per axis *stop* indices defining the extent.

        """
        object.__init__(self)
        if struct is None:
            if slice is not None:
                struct = self.create_struct_instance(ndim=len(slice))
                struct[START] = _np.array([s.start for s in slice], dtype="int64")
                struct[STOP] = _np.array([s.stop for s in slice], dtype=struct[START].dtype)
            elif (start is None) and (stop is not None):
                struct = self.create_struct_instance(ndim=len(stop))
                struct[STOP] = _np.array(stop, dtype="int64")
                struct[START] = _np.zeros_like(struct[STOP])
            elif (start is not None) and (stop is not None):
                struct = self.create_struct_instance(ndim=len(start))
                struct[START] = _np.array(start, dtype="int64")
                struct[STOP] = _np.array(stop, dtype="int64")

        self._struct = struct

    def __eq__(self, other):
        """
        Return :samp:`True` for identical :attr:`start` and :attr:`stop`.
        """
        return \
            (
                (self.ndim == other.ndim)
                and
                _np.all(self.start == other.start)
                and
                _np.all(self.stop == other.stop)
            )

    def create_struct_instance(self, ndim):
        """
        Creates a struct instance with :obj:`numpy.dtype` :samp:`self.struct_dtype_dict[ndim]`.

        :rtype: struct
        :return: A struct.
        """
        return _np.zeros((1,), dtype=self.get_struct_dtype(ndim))[0]

    @property
    def start(self):
        """
        Sequence of :obj:`int` indicating the per-axis start indices of this extent
        (including halo).
        """
        return self._struct[START]

    @start.setter
    def start(self, start):
        if len(start) != len(self._struct[START]):
            self._struct = self.create_struct_dtype(ndim=len(start))
        self._struct[START][...] = start

    @property
    def stop(self):
        """
        Sequence of :obj:`int` indicating the per-axis stop indices of this extent
        (including halo).
        """
        return self._struct[STOP]

    @stop.setter
    def stop(self, stop):
        if len(stop) != len(self._struct[STOP]):
            self._struct = self.create_struct_dtype(ndim=len(stop))
        self._struct[STOP][...] = stop

    @property
    def shape(self):
        """
        Sequence of :obj:`int` indicating the shape of this extent
        (including halo).
        """
        return self._struct[STOP] - self._struct[START]

    @property
    def ndim(self):
        """
        Dimension of indexing.
        """
        return self._struct[START].size

    def calc_intersection(self, other):
        """
        Returns the indexing extent which is the intersection of
        this extent with the :samp:`{other}` extent.

        :type other: :obj:`IndexingExtent`
        :param other: Perform intersection calculation using this extent.
        :rtype: :obj:`IndexingExtent`
        :return: :samp:`None` if the extents do not intersect, otherwise
           returns the extent of overlapping indices.
        """
        intersection_extent = \
            IndexingExtent(
                start=_np.maximum(self.start, other.start),
                stop=_np.minimum(self.stop, other.stop)
            )
        if _np.any(intersection_extent.start >= intersection_extent.stop):
            intersection_extent = None

        return intersection_extent

    def split(self, a, index):
        """
        Split this extent into two extents by cutting along
        axis :samp:`{a}` at index :samp:`{index}`.

        :type a: :obj:`int`
        :param a: Cut along this axis.
        :type index: :obj:`int`
        :param index: Location of cut.
        :rtype: :obj:`tuple`
        :return: A :samp:`(lo, hi)` pair.
        """
        if index <= self.start[a]:
            lo, hi = None, self
        elif index >= self.stop[a]:
            lo, hi = self, None
        else:
            b = self.start.copy()
            e = self.stop.copy()
            e[a] = index
            lo = IndexingExtent(start=b, stop=e)
            b[a] = index
            hi = IndexingExtent(start=b, stop=self.stop.copy())

        return lo, hi

    def calc_intersection_split(self, other):
        """
        Returns :samp:`(leftovers, intersection)` pair, where :samp:`intersection`
        is the :obj:`IndexingExtent` object (possibly :samp:`None`) indicating
        the intersection of this (:samp:`{self}`) extent with the :samp:`other` extent
        and :samp:`leftovers` is a list of :obj:`IndexingExtent` objects
        indicating regions of :samp:`self` which do not intersect with
        the :samp:`other` extent.

        :type other: :obj:`IndexingExtent`
        :param other: Perform intersection calculation using this extent.
        :rtype: :obj:`tuple`
        :return: :samp:`(leftovers, intersection)` pair.
        """
        intersection = self.calc_intersection(other)
        leftovers = []
        if intersection is not None:
            q = _collections.deque()
            q.append(self)
            for a in range(self.ndim):
                o = q.pop()
                lo, hi = o.split(a, intersection.start[a])
                if lo is not None:
                    leftovers.append(lo)
                if hi is not None:
                    lo, hi = hi.split(a, intersection.stop[a])
                    if lo is not None:
                        q.append(lo)
                    if hi is not None:
                        leftovers.append(hi)
        else:
            leftovers.append(other)

        return leftovers, intersection

    def to_slice(self):
        """
        Returns ":obj:`tuple` of :obj:`slice`" equivalent of this indexing extent.

        :rtype: :obj:`tuple` of :obj:`slice` elements
        :return: Tuple of slice equivalent to this indexing extent.
        """
        return \
            tuple(
                [
                    slice(self._struct[START][i], self._struct[STOP][i])
                    for i in range(len(self._struct[START]))
                ]
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
                None,  # slice arg
                tuple(self.start),
                tuple(self.stop),
                None  # struct arg
            )

    def __repr__(self):
        """
        Stringize.
        """
        return "IndexingExtent(start=%s, stop=%s)" % (tuple(self.start), tuple(self.stop))

    def __str__(self):
        """
        """
        return self.__repr__()


class HaloIndexingExtent(IndexingExtent):

    """
    Indexing bounds with ghost (halo) elements, for a single tile of domain decomposition.

    Example::

        >>> from mpi_array.indexing import HaloIndexingExtent
        >>>
        >>> hie = HaloIndexingExtent(start=(10,), stop=(20,), halo=((2,4),))
        >>> print("hie.start_n = %s" % (hie.start_n,)) # start without halo
        hie.start_n = [10]
        >>> print("hie.start_h = %s" % (hie.start_h,)) # start with halo
        hie.start_h = [8]
        >>> print("hie.stop_n  = %s" % (hie.stop_n,))  # stop without halo
        hie.stop_n  = [20]
        >>> print("hie.stop_h  = %s" % (hie.stop_h,))  # stop with halo
        hie.stop_h  = [24]
    """

    #: The "low index" indices.
    LO = 0

    #: The "high index" indices.
    HI = 1

    struct_dtype_dict = _collections.defaultdict(lambda: None)

    def create_struct_dtype(self, ndim):
        """
        Creates a :obj:`numpy.dtype` structure for holding start and stop indices.

        :rtype: :obj:`numpy.dtype`
        :return: :obj:`numpy.dtype` with :samp:`"start"` and :samp:`"stop"` multi-index
           fields of dimension :samp:`{ndim}`.
        """
        return \
            _np.dtype(
                [
                    ("start", _np.int64, (ndim,)),
                    ("stop", _np.int64, (ndim,)),
                    ("halo", _np.int64, (ndim, 2))
                ]
            )

    def __init__(self, slice=None, start=None, stop=None, halo=None, struct=None):
        """
        Construct.

        :type slice: sequence of :obj:`slice`
        :param slice: Per axis start and stop indices defining the extent (**not including ghost
           elements**).
        :type start: sequence of :obj:`int`
        :param start: Per axis *start* indices defining the start of extent  (**not including ghost
           elements**).
        :type stop: sequence of :obj:`int`
        :param stop: Per axis *stop* indices defining the extent  (**not including ghost
           elements**).
        :type halo: :samp:`(len({slice}), 2)` shaped array of :obj:`int`
        :param halo: A :samp:`(len(self.start), 2)` shaped array of :obj:`int` indicating the
           per-axis number of outer ghost elements. :samp:`halo[:,0]` is the number
           of elements on the low-index *side* and :samp:`halo[:,1]` is the number of
           elements on the high-index *side*.

        """
        IndexingExtent.__init__(self, slice, start, stop, struct)
        if halo is None:
            halo = _np.zeros((self._struct[START].shape[0], 2), dtype=self._struct[START].dtype)
        else:
            halo = convert_halo_to_array_form(halo, self._struct[START].size)
        self._struct[HALO][...] = halo

    @property
    def halo(self):
        """
        A :samp:`(len(self.start), 2)` shaped array of :obj:`int` indicating the
        per-axis number of outer ghost elements. :samp:`halo[:,0]` is the number
        of elements on the low-index *side* and :samp:`halo[:,1]` is the number of
        elements on the high-index *side*.
        """
        return self._struct[HALO]

    @halo.setter
    def halo(self, halo):
        self._struct[HALO] = convert_halo_to_array_form(halo, self.ndim)

    @property
    def start_h(self):
        """
        The start index of the tile with "halo" elements.
        """
        return self.start_n - self.halo[:, self.LO]

    @property
    def start_n(self):
        """
        The start index of the tile without "halo" elements ("no halo").
        """
        return self.start

    @start_n.setter
    def start_n(self, start_n):
        self.start[...] = start_n

    @property
    def stop_h(self):
        """
        The stop index of the tile with "halo" elements.
        """
        return self.stop + self.halo[:, self.HI]

    @property
    def stop_n(self):
        """
        The stop index of the tile without "halo" elements ("no halo").
        """
        return self.stop

    @stop_n.setter
    def stop_n(self, stop_n):
        self.stop[...] = stop_n

    @property
    def shape_h(self):
        """
        The shape of the tile with "halo" elements.
        """
        return self.stop_n + self.halo[:, self.HI] - self.start_n + self.halo[:, self.LO]

    @property
    def shape_n(self):
        """
        The shape of the tile without "halo" elements ("no halo").
        """
        return self.stop_n - self.start_n

    @property
    def start(self):
        """
        Same as :attr:`start_n`.
        """
        return self._struct[START]

    @property
    def stop(self):
        """
        Same as :attr:`stop_n`.
        """
        return self._struct[STOP]

    @property
    def shape(self):
        """
        Same as :attr:`shape_n`.
        """
        return self._struct[STOP] - self._struct[START]

    @property
    def size_n(self):
        """
        Integer indicating the number of elements in this extent without halo ("no halo")
        """
        return _np.product(self.shape_n)

    @property
    def size_h(self):
        """
        Integer indicating the number of elements in this extent including halo.
        """
        return _np.product(self.shape_h)

    def __eq__(self, other):
        """
        Equality.
        """
        return IndexingExtent.__eq__(self, other) and _np.all(self.halo == other.halo)

    def to_slice_n(self):
        """
        Returns ":obj:`tuple` of :obj:`slice`" equivalent of this
        indexing extent without halo ("no halo").

        :rtype: :obj:`tuple` of :obj:`slice` elements
        :return: Tuple of slice equivalent to this no-halo indexing extent.
        """
        b = self.start_n
        e = self.stop_n
        return tuple([slice(b[i], e[i]) for i in range(len(b))])

    def to_slice_h(self):
        """
        Returns ":obj:`tuple` of :obj:`slice`" equivalent of this
        indexing extent including halo.

        :rtype: :obj:`tuple` of :obj:`slice` elements
        :return: Tuple of slice equivalent to this indexing extent including halo.
        """
        b = self.start_h
        e = self.stop_h
        return tuple([slice(b[i], e[i]) for i in range(len(b))])

    def globale_to_locale_h(self, gidx):
        """
        Convert globale array index to locale array index.

        :type gidx: sequence of :obj:`int`
        :param gidx: Globale index.

        :rtype: :obj:`numpy.ndarray`
        :return: Locale index.
        """
        return -self.start_h + gidx

    def locale_to_globale_h(self, lidx):
        """
        Convert locale array index to globale array index.

        :type lidx: sequence of :obj:`int`
        :param lidx: Locale index.

        :rtype: :obj:`numpy.ndarray`
        :return: Globale index.
        """
        return self.start_h + lidx

    def globale_to_locale_n(self, gidx):
        """
        Convert globale array index to locale array index.

        :type gidx: sequence of :obj:`int`
        :param gidx: Globale index.

        :rtype: :obj:`numpy.ndarray`
        :return: Locale index.
        """
        return -self.start_n + gidx

    def locale_to_globale_n(self, lidx):
        """
        Convert locale array index to globale array index.

        :type lidx: sequence of :obj:`int`
        :param lidx: Locale index.

        :rtype: :obj:`numpy.ndarray`
        :return: Globale index.
        """
        return self.start_n + lidx

    def globale_to_locale_extent_h(self, gext):
        """
        Return :samp:`gext` converted to locale index.
        """
        ext = _copy.deepcopy(gext)
        if isinstance(gext, HaloIndexingExtent):
            ext.start_n = self.globale_to_locale_h(gext.start_n)
            ext.stop_n = self.globale_to_locale_h(gext.stop_n)
        else:
            ext.start = self.globale_to_locale_h(gext.start)
            ext.stop = self.globale_to_locale_h(gext.stop)

        return ext

    def locale_to_globale_extent_h(self, lext):
        """
        Return :samp:`lext` converted to globale index.
        """
        ext = _copy.deepcopy(lext)
        if isinstance(lext, HaloIndexingExtent):
            ext.start_n = self.locale_to_globale_h(lext.start_n)
            ext.stop_n = self.locale_to_globale_h(lext.stop_n)
        else:
            ext.start = self.locale_to_globale_h(lext.start)
            ext.stop = self.locale_to_globale_h(lext.stop)
        return ext

    def globale_to_locale_slice_h(self, gslice):
        """
        Return :samp:`gslice` converted to locale slice.
        """
        b = self.start_h
        slc = \
            tuple(
                slice(
                    gslice[i].start - b[i],
                    gslice[i].stop - b[i],
                )
                for i in range(len(gslice))
            )

        return slc

    def locale_to_globale_slice_h(self, lslice):
        """
        Return :samp:`lslice` converted to globale slice.
        """
        b = self.start_h
        slc = \
            tuple(
                slice(
                    lslice[i].start + b[i],
                    lslice[i].stop + b[i],
                )
                for i in range(len(lslice))
            )

        return slc

    def globale_to_locale_slice_n(self, gslice):
        """
        Return :samp:`gslice` converted to locale slice.
        """
        b = self.start_n
        slc = \
            tuple(
                slice(
                    gslice[i].start - b[i],
                    gslice[i].stop - b[i],
                )
                for i in range(len(gslice))
            )

        return slc

    def locale_to_globale_slice_n(self, lslice):
        """
        Return :samp:`lslice` converted to globale slice.
        """
        b = self.start_n
        slc = \
            tuple(
                slice(
                    lslice[i].start + b[i],
                    lslice[i].stop + b[i],
                )
                for i in range(len(lslice))
            )

        return slc

    def to_slice(self):
        """
        Same as :meth:`to_slice_n`.
        """
        return self.to_slice_n()

    def to_tuple(self):
        """
        Convert this instance to a :obj:`tuple` which can be passed to constructor
        (or used as a :obj:`dict` key).

        :rtype: :obj:`tuple`
        :return: The :obj:`tuple` representation of this object.
        """
        return \
            (
                None,  # slice arg
                tuple(self.start_n),
                tuple(self.stop_n),
                tuple(tuple(row) for row in self.halo.tolist()),
                None,  # struct arg
            )

    def __repr__(self):
        """
        Stringize.
        """
        return \
            (
                "HaloIndexingExtent(start=%s, stop=%s, halo=%s)"
                %
                (self.start_n.tolist(), self.stop_n.tolist(), self.halo.tolist())
            )

    def __str__(self):
        """
        """
        return self.__repr__()


def calc_intersection_split(
    dst_extent,
    src_extent,
    update_factory,
    update_dst_halo
):
    """
    Calculates intersection between :samp:`{dst_extent}` and `{src_extent}`.
    Any regions of :samp:`{dst_extent}` which **do not** intersect with :samp:`{src_extent}`
    are returned as a :obj:`list` of *left-over* :samp:`type({dst_extent})` elements.
    The regions of :samp:`{dst_extent}` which **do** intersect with :samp:`{src_extent}`
    are returned as a :obj:`list` of *update* elements. The *update* elements
    are created with a call to the factory object :samp:`update_factory`::

       update_factory(dst_extent, src_extent, intersection)

    Returns :obj:`tuple` pair :samp:`(leftovers, updates)`.

    :type dst_extent: :obj:`HaloIndexingExtent`
    :param dst_extent: Extent which is to receive update from intersection
       with :samp:`{src_extent}`.
    :type src_extent: :obj:`HaloIndexingExtent`
    :param src_extent: Extent which is to provide update for the intersecting
       region of :samp:`{dst_extent}`.
    :type update_factory: callable :obj:`object`
    :param update_factory: Object called to create instances
       of :obj:`mpi_array.decomposition.PairUpdateExtent`.
    :type update_dst_halo: :obj:`bool`
    :param update_dst_halo: If true, then the halo of :samp:`{dst_extent}` is
       include when calculating the intersection with :samp:`{src_extent}`.
    :rtype: :obj:`tuple`
    :return: Returns :obj:`tuple` pair of :samp:`(leftovers, updates)`.
    """

    leftovers = []
    updates = []

    if update_dst_halo:
        dst_ie = IndexingExtent(start=dst_extent.start_h, stop=dst_extent.stop_h)
        halo = 0
    else:
        dst_ie = IndexingExtent(start=dst_extent.start_n, stop=dst_extent.stop_n)
        halo = dst_extent.halo
    src_ie = IndexingExtent(start=src_extent.start_n, stop=src_extent.stop_n)

    ie_leftovers, intersection = dst_ie.calc_intersection_split(src_ie)

    for ie_leftover in ie_leftovers:
        de = _copy.deepcopy(dst_extent)
        de.start_n = ie_leftover.start
        de.stop_n = ie_leftover.stop
        de.halo = halo
        leftovers.append(de)

    if intersection is not None:
        updates += update_factory(dst_extent, src_extent, intersection)
    else:
        leftovers = [dst_extent, ]

    return leftovers, updates


__all__ = [s for s in dir() if not s.startswith('_')]
