"""
=================================
The :mod:`mpi_array.local` Module
=================================

Defines :obj:`lndarray` class and factory functions for
creating multi-dimensional arrays where memory is allocated
using :meth:`mpi4py.MPI.Win.Allocate_shared` or :meth:`mpi4py.MPI.Win.Allocate`.

Classes
=======

..
   Special template for mpi_array.local.lndarray to avoid numpydoc
   documentation style sphinx warnings/errors from numpy.ndarray inheritance.

.. autosummary::
   :toctree: generated/
   :template: autosummary/lndarray_class.rst

   lndarray - Sub-class of :obj:`numpy.ndarray` which uses MPI allocated memory.

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

Utilities
=========

.. autosummary::
   :toctree: generated/

   NdarrayMetaData - Strides, offset and order info.

"""

from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources
import numpy as _np

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


class NdarrayMetaData(object):
    """
    Encapsulates, strides, offset and order argument of :meth:`lndarray.__new__`.
    """

    def __init__(self, strides, offset, order):
        object.__init__(self)
        self.strides = strides
        self.offset = offset
        self.order = order


class lndarray(_np.ndarray):
    """
    Sub-class of :obj:`np.ndarray` for use with the :obj:`mpi4py` parallel processing.
    Allocates a shared memory buffer using :func:`mpi4py.MPI.Win.Allocate_shared`.
    (if available, otherwise uses :func:`mpi4py.MPI.Win.Allocate`).
    """

    def __new__(
        cls,
        shape,
        dtype=_np.dtype("float64"),
        buffer=None,
        strides=None,
        offset=0,
        order=None,
        decomp=None
    ):
        """
        """
        if buffer is None:
            buffer, itemsize, shape = decomp.alloc_local_buffer(shape, dtype)
        else:
            if not isinstance(buffer, memoryview):
                raise ValueError(
                    "Got buffer type=%s which is not an instance of %s"
                    %
                    (
                        type(buffer),
                        memoryview
                    )
                )

            if decomp is None:
                raise ValueError("Got None value for decomp with non-None buffer = %s" % (buffer,))

        self = \
            _np.ndarray.__new__(
                cls,
                shape,
                dtype,
                buffer,
                offset,
                strides,
                order
            )
        self.md = NdarrayMetaData(strides=strides, offset=offset, order=order)
        self.decomp = decomp

        return self

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.md = getattr(obj, 'md', None)
        self.decomp = getattr(obj, 'decomp', None)

    def rank_view(self):
        return self[self.decomp.rank_view_slice]

    def __reduce_ex__(self, protocol):
        """
        Pickle *reference* to shared memory.
        """
        raise NotImplementedError("Cannot pickle objects of type %s" % type(self))
        # return ndarray, (self.shape, self.dtype, self.mp_Array,
        #                 self.strides, self.offset, self.order)

    def __reduce__(self):
        return self.__reduce_ex__(self, 0)


def empty(shape=None, dtype="float64", decomp=None):

    ary = lndarray(shape=shape, dtype=dtype, decomp=decomp)

    return ary


def empty_like(ary):
    """
    """
    if (isinstance(ary, lndarray)):
        ret_ary = empty(dtype=ary.dtype, decomp=ary.decomp)
    else:
        ret_ary = _np.empty_like(ary)

    return ret_ary


def zeros(shape, dtype="float64", decomp=None):
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view()[...] = _np.zeros(1, dtype)

    return ary


def zeros_like(ary, *args, **kwargs):
    """
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.rank_view()[...] = _np.zeros(1, ary.dtype)

    return ary


def ones(shape, dtype="float64", decomp=None):
    ary = empty(shape, dtype=dtype, decomp=decomp)
    ary.rank_view()[...] = _np.ones(1, dtype)

    return ary


def ones_like(ary, *args, **kwargs):
    """
    """
    ary = empty_like(ary, *args, **kwargs)
    ary.rank_view()[...] = _np.ones(1, ary.dtype)

    return ary


def copy(ary_in):
    ary = empty(dtype=ary_in.dtype, decomp=ary_in.decomp)
    ary[...] = ary_in[...]

    return ary
