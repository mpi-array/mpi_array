"""
========================================
The :mod:`mpi_array.globale_test` Module
========================================

Module defining :mod:`mpi_array.globale` unit-tests.
Execute as::

   python -m mpi_array.globale_test


Classes
=======

.. autosummary::
   :toctree: generated/

   GndarrayTest - Tests for :obj:`mpi_array.globale.gndarray`.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array
from mpi_array.decomposition import CartesianDecomposition
import mpi_array.globale

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class GndarrayTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.globale.gndarray`.
    """

    def test_construct_arg_checking(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.__new__`.
        """
        self.assertRaises(
            ValueError,
            mpi_array.globale.gndarray,
            shape=None,
            decomp=None,
            dtype="int64"
        )

        self.assertRaises(
            ValueError,
            mpi_array.globale.gndarray,
            shape=(100, 100, 100),
            decomp=CartesianDecomposition(shape=(100, 99, 100)),
            dtype="int64"
        )

    def test_get_item(self):
        """
        """
        gary = mpi_array.globale.zeros((20, 20, 20), dtype="int8")
        gary[1, 2, 8] = 22
        gary[1:10, 2:4, 4:8]
        gary[...] = 19
        gary[:] = 101


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
