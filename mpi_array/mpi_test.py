"""
====================================
The :mod:`mpi_array.mpi_test` Module
====================================

Module defining :mod:`mpi4py` unit-tests.
Execute as::

   python -m mpi_array.mpi_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   MpiTest - Tests for :obj:`mpi4py` one-sided comms.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array as _mpi_array
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401

import mpi4py.MPI as _mpi
import numpy as _np

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class MpiTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi4py` one-sided comms.
    """

    def setUp(self):
        self.rank_logger = _logging.get_rank_logger(name=self.id(), comm=_mpi.COMM_WORLD)

    def create_window(self, np_ndarray, comm=_mpi.COMM_WORLD):
        win = _mpi.Win.Create(np_ndarray, np_ndarray.itemsize, comm=comm)
        return win

    def test_numpy_array_get_builtin_datatype(self):
        """
        Test for :meth:`mpi_array.MPI.Win.Get` with built-in .
        """
        comm = _mpi.COMM_WORLD
        comm.barrier()
        src_ary = _np.zeros((1000,), dtype="int32")
        dst_ary = _np.zeros_like(src_ary)

        win = self.create_window(src_ary, comm)
        my_rank = comm.rank
        ne_rank = (my_rank + 1) % comm.size
        src_ary[...] = my_rank
        win.Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)
        win.Get(
            [dst_ary, dst_ary.size, _mpi._typedict[dst_ary.dtype.char]],
            ne_rank,
            [0, src_ary.size, _mpi._typedict[src_ary.dtype.char]]
        )
        win.Fence(_mpi.MODE_NOSUCCEED)
        self.assertTrue(_np.all(dst_ary == ne_rank))

    def test_numpy_array_get_contiguous_datatype(self):
        """
        Test for :meth:`mpi_array.MPI.Win.Get`.
        """
        comm = _mpi.COMM_WORLD
        comm.barrier()
        ary = _np.zeros((1000,), dtype="int32")
        dst_ary = ary.copy()

        win = self.create_window(ary, comm)
        my_rank = comm.rank
        ne_rank = (my_rank + 1) % comm.size
        ary[...] = my_rank

        ary_datatype = _mpi._typedict[ary.dtype.char]
        ctg_datatype = ary_datatype.Create_contiguous(ary.size // 2)
        ctg_datatype.Commit()

        win.Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)
        win.Get(
            [dst_ary, 1, ctg_datatype],
            ne_rank,
            [0, 1, ctg_datatype]
        )
        win.Get(
            [dst_ary[ary.size // 2:], 1, ctg_datatype],
            ne_rank,
            [ary.size // 2, 1, ctg_datatype]
        )
        win.Fence(_mpi.MODE_NOSUCCEED)
        self.assertTrue(_np.all(dst_ary == ne_rank))

    def test_numpy_array_get_subarray_datatype(self):
        """
        Test for :meth:`mpi_array.MPI.Win.Get`.
        """
        comm = _mpi.COMM_WORLD
        comm.barrier()
        ary = _np.zeros((1000,), dtype="int32")
        dst_ary = ary.copy()

        win = self.create_window(ary, comm)
        my_rank = comm.rank
        ne_rank = (my_rank + 1) % comm.size
        ary[...] = my_rank

        ary_datatype = _mpi._typedict[ary.dtype.char]
        suba_lo_datatype = ary_datatype.Create_subarray((ary.size,), (ary.size // 2,), (0,))
        suba_lo_datatype.Commit()
        suba_hi_datatype = ary_datatype.Create_subarray(
            (ary.size,), (ary.size // 2,), (ary.size // 2,))
        suba_hi_datatype.Commit()

        win.Fence(_mpi.MODE_NOPUT | _mpi.MODE_NOPRECEDE)
        win.Get(
            [dst_ary, 1, suba_lo_datatype],
            ne_rank,
            [0, 1, suba_lo_datatype]
        )
        win.Get(
            [dst_ary, 1, suba_hi_datatype],
            ne_rank,
            [0, 1, suba_hi_datatype]
        )
        win.Fence(_mpi.MODE_NOSUCCEED)
        self.assertTrue(_np.all(dst_ary == ne_rank))


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
