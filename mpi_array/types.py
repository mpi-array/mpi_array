"""
=================================
The :mod:`mpi_array.types` Module
=================================

Convert :obj:`numpy.dtype` to :obj:`mpi4pi.MPI.Datatype`.

Functions
=========

.. autosummary::
   :toctree: generated/

   to_datatype - Convert :obj:`numpy.dtype` to :obj:`mpi4pi.MPI.Datatype`.

Utilities
=========

.. autosummary::
   :toctree: generated/

   create_lookup - Creates :obj:`dict` of (:obj:`numpy.dtype`, :obj:`mpi4py.MPI.Datatype`) pairs.
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


def create_lookup():
    """
    Creates a :obj:`collections.defaultdict` of (:obj:`numpy.dtype`, :obj:`mpi4py.MPI.Datatype`)
    key-value pairs. Populated with basic types from :obj:`mpi4py.MPI._typedict`.

    :rtype: :obj:`collections.defaultdict`
    :return: A :obj:`collections.defaultdict` with default element being :samp:`None`.
    """
    lu = _collections.defaultdict(lambda: None)

    if hasattr(_mpi, "_typedict"):
        for c in _mpi._typedict.keys():
            try:
                lu[_np.dtype(c)] = _mpi._typedict[c]
            except TypeError:
                pass

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
            datatype = _mpi.Datatype.Create_struct(blocklengths, displacements, datatypes)
        else:
            blocklengths[...] = tuple(dt.extent for dt in datatypes)
            datatype = _mpi.BYTE.Create_hindexed(blocklengths, displacements)
    elif dtyp.subdtype is not None:
        datatype = \
            find_or_create_datatype(dtyp.subdtype[0]).Create_contiguous(
                _np.product(dtyp.subdtype[1]))
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
