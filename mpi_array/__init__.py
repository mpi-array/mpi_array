"""
============================
The :mod:`mpi_array` Package
============================

Python package for multi-dimensional distributed arrays
(
`Partitioned Global Address Space <https://en.wikipedia.org/wiki/Partitioned_global_address_space>`_
)
with :obj:`numpy.ndarray`-like API. The classes and functions below are attributes of the
package (e.g. :samp:`mpi_array.zeros` is :func:`mpi_array.globale_creation.zeros`).

Classes
=======

.. autosummary::

   ~globale.gndarray - A :obj:`numpy.ndarray` like distributed array.

Array Creation
==============

.. autosummary::

   ~globale_creation.empty - Create uninitialised array.
   ~globale_creation.empty_like - Create uninitialised array like another array.
   ~globale_creation.zeros - Create zero-initialised array.
   ~globale_creation.zeros_like - Create zero-initialised array like another array.
   ~globale_creation.ones - Create one-initialised array.
   ~globale_creation.ones_like - Create one-initialised array like another array.
   ~globale_creation.full - Create *fill value* initialised array.
   ~globale_creation.full_like - Create *fill value* initialised array like another.
   ~globale_creation.array - Create array from an existing array-like object.
   ~globale_creation.asarray - Convert to an array.
   ~globale_creation.asanyarray - Convert to an array, sub-classes pass through.
   ~globale_creation.copy - Create a replica of an array.
   ~globale_creation.eye - Not implemented.
   ~globale_creation.identity - Not implemented.

Functions
=========

.. autosummary::

   ~globale.copyto - Copy elements of one array to another array.
   ~globale.free_all - Free the arrays of a sequence.

Ufuncs
======

The :obj:`numpy.ufunc` objects of NumPy are attributes of the package too
(e.g. :samp:`mpi_array.sqrt` is :obj:`numpy.sqrt`), they apply to
:obj:`mpi_array.globale.gndarray` arguments (see :mod:`mpi_array.globale_ufunc`), as
does the operator of an :obj:`mpi_array.globale.gndarray` (e.g. :samp:`a + 1`).
:samp:`mpi_array.dtype` is :obj:`numpy.dtype`.

"""
import sys as _sys
import numpy as _np
from . import init as _init  # noqa: E402,F401
from . import rtd as _rtd  # noqa: E402,F401
from .license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

_this_module = _sys.modules[__name__]
from .globale import gndarray  # noqa: E402,F401
from .globale import free_all  # noqa: E402,F401
from .globale import copyto  # noqa: E402,F401
from . import globale_ufunc as _ufunc  # noqa: E402,F401

from .globale_creation import empty, empty_like, eye, identity  # noqa: E402,F401
from .globale_creation import ones, ones_like, zeros, zeros_like, full, full_like  # noqa: E402,F401
from .globale_creation import array, asarray, asanyarray, copy  # noqa: E402,F401

_ufunc.set_numpy_ufuncs_as_module_attr(_this_module, _ufunc)

dtype = _np.dtype

__all__ = \
    [
        "gndarray", "free_all", "copyto", "dtype",
        "empty", "empty_like", "eye", "identity",
        "ones", "ones_like", "zeros", "zeros_like", "full", "full_like",
        "array", "asarray", "asanyarray", "copy",
    ] \
    + \
    sorted(_name for _name in dir(_ufunc) if isinstance(getattr(_ufunc, _name), _np.ufunc))
