"""
======================================
The :mod:`mpi_array.rtd` Module
======================================

.. currentmodule:: mpi_array.rtd

Sets up :samp:`mock` modules for readthedocs.org sphinx builds.
See `this FAQ <http://read-the-docs.readthedocs.io/en/latest/faq.html
#i-get-import-errors-on-libraries-that-depend-on-c-modules>`_.

Functions
=========

.. autosummary::
   :toctree: generated/

   initialise_mock_modules - Replaces/initialise mock modules in :obj:`sys.modules`.

Attributes
==========

.. autodata:: MOCK_MODULES

"""
from __future__ import absolute_import
import pkg_resources as _pkg_resources
import os as _os

#: List of module names
MOCK_MODULES = ['mpi4py', "mpi4py.MPI"]


def initialise_mock_modules(module_name_list):
    """
    Updates system modules (:func:`sys.modules.update`) with
    :samp:`unittest.mock.MagicMock` objects.

    :type module_name_list: sequence of :obj:`str`
    :param module_name_list: List of module names to be replaced/initialised
       with :samp:`MagicMock` instances in :obj:`sys.modules`.

    """
    import sys as _sys
    try:
        from unittest.mock import MagicMock as _MagicMock
    except (ImportError,):
        from mock import Mock as _MagicMock

    class Mock(_MagicMock):

        @classmethod
        def __getattr__(cls, name):
            return _MagicMock()

    _sys.modules.update((mod_name, Mock()) for mod_name in module_name_list)


if "READTHEDOCS" in _os.environ.keys():
    initialise_mock_modules(MOCK_MODULES)

from .license import license as _license, copyright as _copyright  # noqa: E402,F401

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


__all__ = [s for s in dir() if not s.startswith('_')]
