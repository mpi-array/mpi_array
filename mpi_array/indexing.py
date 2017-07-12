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


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources
import numpy as _np

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


class IndexingExtent(object):

    """
    Indexing bounds for a single tile of domain decomposition.
    """

    def __init__(self, split=None, start=None, stop=None):
        """
        Construct.

        :type split: sequence of :obj:`slice`
        :param split: Per axis start and stop indices defining the extent.
        :type start: sequence of :obj:`int`
        :param start: Per axis *start* indices defining the start of extent.
        :type stop: sequence of :obj:`int`
        :param stop: Per axis *stop* indices defining the extent.

        """
        object.__init__(self)
        if split is not None:
            self._beg = _np.array([s.start for s in split], dtype="int64")
            self._end = _np.array([s.stop for s in split], dtype=self._beg.dtype)
        elif (start is not None) and (stop is not None):
            self._beg = _np.array(start, dtype="int64")
            self._end = _np.array(stop, dtype="int64")

    def __eq__(self, other):
        """
        Return :samp:`True` for identical :attr:`start` and :attr:`stop`.
        """
        return _np.all(self._beg == other._beg) and _np.all(self._end == other._end)

    @property
    def start(self):
        """
        Sequence of :obj:`int` indicating the per-axis start indices of this extent
        (including halo).
        """
        return self._beg

    @property
    def stop(self):
        """
        Sequence of :obj:`int` indicating the per-axis stop indices of this extent
        (including halo).
        """
        return self._end

    @property
    def shape(self):
        """
        Sequence of :obj:`int` indicating the shape of this extent
        (including halo).
        """
        return self._end - self._beg

    @property
    def ndim(self):
        """
        Dimension of indexing.
        """
        return len(self._beg)

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
                start=_np.maximum(self._beg, other._beg),
                stop=_np.minimum(self._end, other._end)
            )
        if _np.any(intersection_extent._beg >= intersection_extent._end):
            intersection_extent = None

        return intersection_extent

    def to_slice(self):
        """
        Returns ":obj:`tuple` of :obj:`slice`" equivalent of this indexing extent.

        :rtype: :obj:`tuple` of :obj:`slice` elements
        :return: Tuple of slice equivalent to this indexing extent.
        """
        return tuple([slice(self._beg[i], self._end[i]) for i in range(len(self._beg))])

    def __repr__(self):
        """
        Stringize.
        """
        return "IndexingExtent(start=%s, stop=%s)" % (tuple(self._beg), tuple(self._end))

    def __str__(self):
        """
        """
        return self.__repr__()


class HaloIndexingExtent(IndexingExtent):

    """
    Indexing bounds with ghost (halo) elements, for a single tile of domain decomposition.
    """

    #: The "low index" indices.
    LO = 0

    #: The "high index" indices.
    HI = 1

    def __init__(self, split=None, start=None, stop=None, halo=None):
        """
        Construct.

        :type split: sequence of :obj:`slice`
        :param split: Per axis start and stop indices defining the extent (**not including ghost
           elements**).
        :type halo: :samp:`(len({split}), 2)` shaped array of :obj:`int`
        :param halo: A :samp:`(len(self.start), 2)` shaped array of :obj:`int` indicating the
           per-axis number of outer ghost elements. :samp:`halo[:,0]` is the number
           of elements on the low-index *side* and :samp:`halo[:,1]` is the number of
           elements on the high-index *side*.

        """
        IndexingExtent.__init__(self, split, start, stop)
        if halo is None:
            halo = _np.zeros((self._beg.shape[0], 2), dtype=self._beg.dtype)
        self._halo = halo

    @property
    def halo(self):
        """
        A :samp:`(len(self.start), 2)` shaped array of :obj:`int` indicating the
        per-axis number of outer ghost elements. :samp:`halo[:,0]` is the number
        of elements on the low-index *side* and :samp:`halo[:,1]` is the number of
        elements on the high-index *side*.
        """
        return self._halo

    @property
    def start_h(self):
        """
        The start index of the tile with "halo" elements.
        """
        return self._beg - self._halo[:, self.LO]

    @property
    def start_n(self):
        """
        The start index of the tile without "halo" elements ("no halo").
        """
        return self._beg

    @property
    def stop_h(self):
        """
        The stop index of the tile with "halo" elements.
        """
        return self._end + self._halo[:, self.HI]

    @property
    def stop_n(self):
        """
        The stop index of the tile without "halo" elements ("no halo").
        """
        return self._end

    @property
    def shape_h(self):
        """
        The shape of the tile with "halo" elements.
        """
        return self._end + self._halo[:, self.HI] - self._beg + self._halo[:, self.LO]

    @property
    def shape_n(self):
        """
        The shape of the tile without "halo" elements ("no halo").
        """
        return self._end - self._beg

    @property
    def start(self):
        """
        Same as :attr:`start_n`.
        """
        return self.start_n

    @property
    def stop(self):
        """
        Same as :attr:`stop_n`.
        """
        return self.stop_n

    @property
    def shape(self):
        """
        Same as :attr:`shape_n`.
        """
        return self.shape_n

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

    def to_slice_n(self):
        """
        Returns ":obj:`tuple` of :obj:`slice`" equivalent of this
        indexing extent without halo ("no halo").

        :rtype: :obj:`tuple` of :obj:`slice` elements
        :return: Tuple of slice equivalent to this no-halo indexing extent.
        """
        return tuple([slice(self._beg[i], self._end[i]) for i in range(len(self._beg))])

    def to_slice_h(self):
        """
        Returns ":obj:`tuple` of :obj:`slice`" equivalent of this
        indexing extent including halo.

        :rtype: :obj:`tuple` of :obj:`slice` elements
        :return: Tuple of slice equivalent to this indexing extent including halo.
        """
        return tuple(
            [
                slice(
                    self._beg[i] - self.halo[i, self.LO],
                    self._end[i] + self.halo[i, self.HI]
                ) for i in range(len(self._beg))
            ]
        )

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

    def to_slice(self):
        """
        Same as :meth:`to_slice_n`.
        """
        return self.to_slice_n()

    def __repr__(self):
        """
        Stringize.
        """
        return \
            (
                "HaloIndexingExtent(start=%s, stop=%s, halo=%s)"
                %
                (tuple(self._beg), tuple(self._end), tuple(self._halo))
            )

    def __str__(self):
        """
        """
        return self.__repr__()


__all__ = [s for s in dir() if not s.startswith('_')]
