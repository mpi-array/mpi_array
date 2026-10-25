"""
============================
The :mod:`mpi_array` Package
============================

Python package for multi-dimensional distributed arrays
(
`Partitioned Global Address Space <https://en.wikipedia.org/wiki/Partitioned_global_address_space>`_
)
with :obj:`numpy.ndarray`-like API.


Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

Attributes
==========


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
