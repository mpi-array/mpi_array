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
from __future__ import absolute_import  # noqa: E402,F401

import sys as _sys
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
from . import globale_ufunc as _ufunc  # noqa: E402,F401

from . import globale_creation as _creation  # noqa: E402,F401
for s in _creation.__all__:
    setattr(_this_module, s, getattr(_creation, s))

_ufunc.set_numpy_ufuncs_as_module_attr(_this_module, _ufunc)

__all__ = [s for s in dir() if not s.startswith('_')]
