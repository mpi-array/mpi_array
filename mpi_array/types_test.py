"""
=========================================
The :mod:`mpi_array.indexing_test` Module
=========================================

Module defining :mod:`mpi_array.indexing` unit-tests.
Execute as::

   python -m mpi_array.indexing_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   TypesTest - Tests for :func:`mpi_array.types.to_datatype`.


"""
from __future__ import absolute_import

import numpy as _np  # noqa: E402,F401
import mpi4py.MPI as _mpi

from .license import license as _license, copyright as _copyright, version as _version
from . import types as _types
from . import logging as _logging
from . import unittest as _unittest

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version


class TypesTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.types.to_datatype`.
    """

    def setUp(self):
        """
        Set up, assign :obj:`logging.Logger` object.
        """
        self.logger = _logging.get_rank_logger(self.id())

    def test_basic(self):
        """
        Test for :obj:`mpi_array.types.to_datatype` converting basic :obj:`numpy.dtype`
        to MPI data types.
        """

        type_pairs = [
            [_mpi.BOOL, "bool"],
            [_mpi.UNSIGNED_CHAR, "uint8"],
            [_mpi.SIGNED_CHAR, "int8"],
            [_mpi.UNSIGNED_SHORT, "uint16"],
            [_mpi.SIGNED_SHORT, "int16"],
            [_mpi.UNSIGNED_INT, "uint32"],
            [_mpi.SIGNED_INT, "int32"],
            [_mpi.UNSIGNED_LONG_LONG, "uint64"],
            [_mpi.AINT, "int64"],
            [_mpi.FLOAT, "float32"],
            [_mpi.DOUBLE, "float64"],
        ]
        for (mpi_dt, np_dt) in type_pairs:
            mpi_dt_from_np_dt = _types.to_datatype(np_dt)
            self.assertEqual(
                mpi_dt,
                mpi_dt_from_np_dt,
                "str(%s) != str(%s)" % (mpi_dt.Get_name(), mpi_dt_from_np_dt.Get_name())
            )

    def test_contiguous(self):
        """
        Test for :obj:`mpi_array.types.to_datatype` converting contiguous :obj:`numpy.dtype`
        to MPI data types.
        """
        mpi_dt_from_np_dt = _types.to_datatype("float16")
        self.assertEqual(2, mpi_dt_from_np_dt.Get_extent()[1])
        mpi_dt_from_np_dt = _types.to_datatype(("float16", (4,)))
        self.assertEqual(8, mpi_dt_from_np_dt.Get_extent()[1])
        mpi_dt_from_np_dt = _types.to_datatype(("int32", (4,)))
        self.assertEqual(16, mpi_dt_from_np_dt.Get_extent()[1])
        mpi_dt_from_np_dt = _types.to_datatype(("int8", (3,)))
        self.assertEqual(3, mpi_dt_from_np_dt.Get_extent()[1])

    def test_struct(self):
        """
        Test for :obj:`mpi_array.types.to_datatype` converting structure :obj:`numpy.dtype`
        to MPI data types.
        """
        mpi_dt_from_np_dt = _types.to_datatype([("m0", "float16"), ])
        self.assertEqual(2, mpi_dt_from_np_dt.Get_extent()[1])
        mpi_dt_from_np_dt = _types.to_datatype([("m0", "float16"), ("m1", "float16")])
        self.assertEqual(4, mpi_dt_from_np_dt.Get_extent()[1])

        mpi_dt_from_np_dt = _types.to_datatype([("b0", "int8"), ("b1", "int8")])
        self.assertEqual(2, mpi_dt_from_np_dt.Get_extent()[1])

        mpi_dt_from_np_dt = _types.to_datatype([("b0", "int8"), ("b1", "int8"), ("b2", "int8")])
        self.assertEqual(3, mpi_dt_from_np_dt.Get_extent()[1])

        mpi_dt_from_np_dt = \
            _types.to_datatype("float64")
        self.assertEqual(8, mpi_dt_from_np_dt.Get_extent()[1])

        mpi_dt_from_np_dt = \
            _types.to_datatype([("m0", "float16"), ])
        self.assertEqual(2, mpi_dt_from_np_dt.Get_extent()[1])

        mpi_dt_from_np_dt = \
            _types.to_datatype([("m0", "int32"), ])
        self.assertEqual(4, mpi_dt_from_np_dt.Get_extent()[1])

        mpi_dt_from_np_dt = \
            _types.to_datatype([("m0", "float64"), ("m1", "float64"), ("m2", "float64")])
        self.assertEqual(24, mpi_dt_from_np_dt.Get_extent()[1])

        mpi_dt_from_np_dt = \
            _types.to_datatype(
                [("m0", "float64"), ("m1", "int16"), ("m2", "float16"), ("m3", "int32")]
            )
        self.assertEqual(0, mpi_dt_from_np_dt.Get_extent()[0])
        self.assertEqual(16, mpi_dt_from_np_dt.Get_extent()[1])

    def test_struct_bcast(self):
        """
        Tests MPI communications with
        `structured arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_.
        """
        comm = _mpi.COMM_WORLD
        for align in [True, False]:
            np_dt = _np.dtype([("m0", "int8"), ("m1", "float64"), ("m2", "int16"), ], align=align)
            mpi_dt = _types.to_datatype(np_dt)

            ary = _np.empty((101,), dtype=np_dt)
            self.logger.info("ary.dtype.isalignedstruct = %s" % (ary.dtype.isalignedstruct,))
            self.logger.info("ary.nbytes          = %s" % (ary.nbytes,))
            self.logger.info("ary.itemsize        = %s" % (ary.itemsize,))
            self.logger.info("mpi_dt.Get_extent() = %s" % (mpi_dt.Get_extent(),))
            ary["m0"] = _np.arange(0, ary.shape[0])
            ary["m1"] = _np.arange(ary.shape[0], 2 * ary.shape[0])
            ary["m2"] = _np.arange(2 * ary.shape[0], 3 * ary.shape[0])
            root_rank = 0
            bcast_ary = _np.empty_like(ary)
            if comm.rank == root_rank:
                bcast_ary = ary.copy()
            comm.Bcast([bcast_ary, mpi_dt], root_rank)

            all_bcast_ary = _np.empty((comm.size, bcast_ary.shape[0]), dtype=bcast_ary.dtype)
            comm.Allgather([bcast_ary, mpi_dt], [all_bcast_ary, mpi_dt])

            self.assertSequenceEqual(ary.tolist(), bcast_ary.tolist())
            for i in range(all_bcast_ary.shape[0]):
                self.assertSequenceEqual(ary.tolist(), all_bcast_ary[i].tolist())


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
