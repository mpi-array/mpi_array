"""
=================================
The :mod:`mpi_array.types` Module
=================================

Convert :obj:`numpy.dtype` to :obj:`mpi4py.MPI.Datatype`.

Functions
=========

.. autosummary::
   :toctree: generated/

   to_datatype - Convert :obj:`numpy.dtype` to :obj:`mpi4py.MPI.Datatype`.

Utilities
=========

.. autosummary::
   :toctree: generated/

   create_lookup - Creates :obj:`dict` of (:obj:`numpy.dtype`, :obj:`mpi4py.MPI.Datatype`) pairs.
   is_usable_datatype - Whether a basic MPI data type can be used for a :obj:`numpy.dtype`.
   create_datatype - Creates a :obj:`mpi4py.MPI.Datatype` from a given :obj:`numpy.dtype`.
   find_or_create_datatype - Returns a :obj:`mpi4py.MPI.Datatype` given a :obj:`numpy.dtype`.

"""
from __future__ import absolute_import

import mpi4py.MPI as _mpi
import numpy as _np
import collections as _collections

from .license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

_lookup = None


def is_usable_datatype(datatype, dtyp):
    """
    Returns :samp:`True` if the basic MPI data type :samp:`{datatype}` can be used for elements
    of :obj:`numpy.dtype` :samp:`{dtyp}`: it is valid, it has the size of an element, and so
    does each element of a data type derived from it. MPI implementations differ in their
    support for some types, e.g. :obj:`numpy.float16`: Open MPI 4.1 has no
    valid :samp:`MPI_SHORT_FLOAT`, the types derived from the 16 bit float type of MPICH 4.2
    have size zero.

    :type datatype: :obj:`mpi4py.MPI.Datatype`
    :param datatype: The MPI data type.
    :type dtyp: :obj:`numpy.dtype`
    :param dtyp: The element type.
    :rtype: :obj:`bool`
    :return: Whether :samp:`{datatype}` can be used for :samp:`{dtyp}` elements.
    """
    try:
        if (datatype == _mpi.DATATYPE_NULL) or (datatype.Get_size() != dtyp.itemsize):
            return False
        derived = datatype.Create_contiguous(2)
        try:
            return derived.Get_size() == 2 * dtyp.itemsize
        finally:
            derived.Free()
    except _mpi.Exception:
        return False


def create_lookup():
    """
    Creates a :obj:`collections.defaultdict` of (:obj:`numpy.dtype`, :obj:`mpi4py.MPI.Datatype`)
    key-value pairs. Populated with basic types from :samp:`mpi4py.MPI._typedict`, those which
    can be used (see :func:`is_usable_datatype`), the others get byte types
    (see :func:`create_datatype`).

    :rtype: :obj:`collections.defaultdict`
    :return: A :obj:`collections.defaultdict` with default element being :samp:`None`.
    """
    lu = _collections.defaultdict(lambda: None)

    if hasattr(_mpi, "_typedict"):
        for c in _mpi._typedict.keys():
            try:
                dtyp = _np.dtype(c)
            except TypeError:
                continue
            if is_usable_datatype(_mpi._typedict[c], dtyp):
                lu[dtyp] = _mpi._typedict[c]

    return lu


def create_datatype(dtyp):
    """
    Creates a :obj:`mpi4py.MPI.Datatype` from a given :obj:`numpy.dtype`.

    :type dtyp: :obj:`numpy.dtype`
    :param dtyp: This gets converted to a :obj:`mpi4py.MPI.Datatype`.
    :rtype: :obj:`mpi4py.MPI.Datatype`
    :returns: A :obj:`mpi4py.MPI.Datatype` for the given :samp:`{dtyp}`.
    """
    dtyp = _np.dtype(dtyp)
    datatype = None
    if dtyp.names is not None:
        datatypes = \
            _np.array(
                tuple(
                    find_or_create_datatype(dtyp.fields[name][0]) for name in dtyp.names
                ),
                dtype="object"
            )
        blocklengths = _np.ones_like(datatypes, dtype="int64")
        displacements = _np.array(tuple(dtyp.fields[name][1] for name in dtyp.names))

        if dtyp.isalignedstruct:
            struct_datatype = _mpi.Datatype.Create_struct(blocklengths, displacements, datatypes)
        else:
            blocklengths[...] = tuple(dt.extent for dt in datatypes)
            struct_datatype = _mpi.BYTE.Create_hindexed(blocklengths, displacements)
        # The MPI extent (and lower bound) of the struct depends on the field offsets
        # and sizes, but consecutive array elements are dtyp.itemsize bytes apart
        # (e.g. padding after the last field, or an offset first field).
        datatype = struct_datatype.Create_resized(0, dtyp.itemsize)
        struct_datatype.Free()
    elif dtyp.subdtype is not None:
        datatype = \
            find_or_create_datatype(dtyp.subdtype[0]).Create_contiguous(
                _np.prod(dtyp.subdtype[1]))
    else:
        datatype = _mpi.BYTE.Create_contiguous(dtyp.itemsize)

    datatype.Commit()
    datatype.Set_name(repr(dtyp))

    return datatype


def find_or_create_datatype(dtyp):
    """
    Converts a :obj:`numpy.dtype` to a :obj:`mpi4py.MPI.Datatype`.
    Uses a lookup dictionary to find the MPI data-type
    which matches the :samp:`{dtype}` :obj:`numpy.dtype`. If
    no match is found, a new instance of a :obj:`mpi4py.MPI.Datatype`
    is created (and added to the lookup dictionary).

    :type dtyp: :obj:`numpy.dtype`
    :param dtyp: This gets converted to a :obj:`mpi4py.MPI.Datatype`.
    :rtype: :obj:`mpi4py.MPI.Datatype`
    :returns: A :obj:`mpi4py.MPI.Datatype` for the given :samp:`{dtyp}`.
    """
    global _lookup

    if _lookup is None:
        _lookup = create_lookup()

    datatype = _lookup[dtyp]
    if datatype is None:
        datatype = create_datatype(dtyp)
        _lookup[dtyp] = datatype

    return datatype


def to_datatype(dtyp):
    """
    Converts a :obj:`numpy.dtype` to a :obj:`mpi4py.MPI.Datatype`.

    :type dtyp: :obj:`numpy.dtype`
    :param dtyp: This gets converted to a :obj:`mpi4py.MPI.Datatype`.
    :rtype: :obj:`mpi4py.MPI.Datatype`
    :returns: A :obj:`mpi4py.MPI.Datatype` for the given :samp:`{dtyp}`.
    """

    return find_or_create_datatype(_np.dtype(dtyp))


__all__ = [s for s in dir() if not s.startswith('_')]
