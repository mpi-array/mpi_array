from __future__ import absolute_import
import pkg_resources as _pkg_resources

__copyright__ = _pkg_resources.resource_string("mpi_array", "copyright.txt").decode()
__license__ = (
    __copyright__
    +
    "\n\n"
    +
    _pkg_resources.resource_string("mpi_array", "license.txt").decode()
)
__author__ = "Shane J. Latham"
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode().strip()


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
