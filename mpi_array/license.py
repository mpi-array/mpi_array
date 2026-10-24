from __future__ import absolute_import
from importlib.resources import files as _files


def _read_package_text(file_name):
    """
    Returns the text of the :mod:`mpi_array` package data file :samp:`{file_name}`.
    """
    return _files("mpi_array").joinpath(file_name).read_text(encoding="utf-8")


__copyright__ = _read_package_text("copyright.txt")
__license__ = (
    __copyright__
    +
    "\n\n"
    +
    _read_package_text("license.txt")
)
__author__ = "Shane J. Latham"
__version__ = _read_package_text("version.txt").strip()


def version():
    """
    Returns :mod:`mpi_array` version string.

    :rtype: :obj:`str`
    :return: Version string.
    """
    return __version__


def license():
    """
    Returns the :mod:`mpi_array` license string.

    :rtype: :obj:`str`
    :return: License string.
    """
    return __license__


def copyright():
    """
    Returns the :mod:`mpi_array` copyright string.

    :rtype: :obj:`str`
    :return: Copyright string.
    """
    return __copyright__


__doc__ = \
    """
=====================================
The :mod:`mpi_array.license` Module
=====================================

License and copyright info.

License
=======

%s

Copyright
=========

%s

Functions
=========

.. autosummary::
   :toctree: generated/

   version - Returns :mod:`mpi_array` version string.
   license - Returns :mod:`mpi_array` license string.
   copyright - Returns :mod:`mpi_array` copyright string.


""" % (license(), copyright())

__all__ = [s for s in dir() if not s.startswith('_')]
