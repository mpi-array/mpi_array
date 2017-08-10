"""
=================================================
The :mod:`mpi_array.globale_creation_test` Module
=================================================

Module for testing creation/factory functions which
generate instances of :mod:`mpi_array.globale.gndarray`.
Execute as::

   python -m mpi_array.globale_creation_test

and with parallelism::

   mpirun -n  2 python -m mpi_array.globale_creation_test
   mpirun -n  4 python -m mpi_array.globale_creation_test
   mpirun -n 27 python -m mpi_array.globale_creation_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   GndarrayCreationTest - Tests for :func:`mpi_array.globale.gndarray` creation functions.
"""
from __future__ import absolute_import

import numpy as _np
import mpi4py.MPI as _mpi

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging  # noqa: E402,F401
from .globale import gndarray as _gndarray
from .globale_creation import asarray as _asarray
from . import comms as _comms
from . import locale as _locale

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class GndarrayCreationTest(_unittest.TestCase):
    """
    :obj:`unittest.TestCase` for :func:`mpi_array.globale.gndarray` instance generation.
    """

    def test_asarray_with_scalar(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asarray`.
        """

        sary0 = _asarray(5.0)
        self.assertTrue(sary0.__class__ is _gndarray)
        self.assertEqual(_np.dtype("float64"), sary0)
        self.assertEqual(0, sary0.ndim)
        self.assertSequenceEqual((), sary0.shape)
        self.assertTrue(sary0.locale_comms.peer_comm is _mpi.COMM_WORLD)

        sary1 = _asarray(sary0)
        self.assertTrue(sary1 is sary0)

    def test_asarray_with_tuple(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asarray`.
        """

        tary0 = _asarray(_np.linspace(100.0, 200.0, 101).tolist())
        tary0.rank_logger.debug("tary0.num_locales = %s" % (tary0.num_locales,))
        self.assertTrue(tary0.__class__ is _gndarray)
        self.assertEqual(_np.dtype("float64"), tary0)
        self.assertTrue(tary0.locale_comms.peer_comm is _mpi.COMM_WORLD)

        tary1 = _asarray(tary0)
        self.assertTrue(tary1 is tary0)

    def test_asarray_with_subclass(self):
        """
        :obj:`unittest.TestCase` for :func:`mpi_array.globale_creation.asarray`.
        """

        class GndarraySubclass(_gndarray):
            pass

        candd = _comms.create_distribution(shape=(8, 32, 32, 32))
        lndarray_proxy, rma_window_buffer = \
            _locale.empty(
                comms_and_distrib=candd,
                dtype="int8",
                order='C',
                return_rma_window_buffer=True
            )

        ary_subclass = GndarraySubclass(candd, rma_window_buffer, lndarray_proxy)
        self.assertTrue(ary_subclass.__class__ is not _gndarray)
        self.assertTrue(isinstance(ary_subclass, _gndarray))
        asary0 = _asarray(ary_subclass)
        self.assertTrue(asary0.__class__ is _gndarray)


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
