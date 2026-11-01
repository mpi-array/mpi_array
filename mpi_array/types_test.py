"""
=========================================
The :mod:`mpi_array.types_test` Module
=========================================

Module defining :mod:`mpi_array.types` unit-tests.
Execute as::

   python -m mpi_array.types_test


Classes
=======

.. autosummary::
   :toctree: generated/

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

        # mpi4py >= 4 maps the fixed width integer types to MPI_INT8_T, MPI_UINT8_T, etc.
        type_tuples = [
            ["bool", _mpi.BOOL, _mpi.C_BOOL, ],
            ["uint8", _mpi.UNSIGNED_CHAR, _mpi.UINT8_T, ],
            ["int8", _mpi.SIGNED_CHAR, _mpi.INT8_T, ],
            ["uint16", _mpi.UNSIGNED_SHORT, _mpi.UINT16_T, ],
            ["int16", _mpi.SIGNED_SHORT, _mpi.INT16_T, ],
            ["uint32", _mpi.UNSIGNED_INT, _mpi.UNSIGNED_LONG, _mpi.UINT32_T, ],
            ["int32", _mpi.SIGNED_INT, _mpi.INT, _mpi.LONG, _mpi.INT32_T, ],
            ["uint64", _mpi.UNSIGNED_LONG_LONG, _mpi.UNSIGNED_LONG, _mpi.UINT64_T, ],
            [
                "int64",
                _mpi.AINT, _mpi.LONG, _mpi.LONG_INT, _mpi.SIGNED_LONG_LONG, _mpi.SIGNED_LONG,
                _mpi.INT64_T,
            ],
            ["float32", _mpi.FLOAT, ],
            ["float64", _mpi.DOUBLE, _mpi.LONG_DOUBLE, ],
        ]
        for tup in type_tuples:
            np_dt = tup[0]
            mpi_dt_from_np_dt = _types.to_datatype(np_dt)
            matches =\
                _np.any(
                    tuple(
                        mpi_dt_from_np_dt == mpi_dt for mpi_dt in tup[1:]
                    )
                )
            self.assertTrue(
                matches,
                "%s converted to %s is not equal to any of %s"
                %
                (
                    np_dt,
                    mpi_dt_from_np_dt.Get_name(),
                    tuple(mpi_dt.Get_name() for mpi_dt in tup[1:])
                )
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

    def test_derived_types_of_basic_types(self):
        """
        The data types derived from those of the basic :obj:`numpy.dtype` have the size
        and extent of their elements. MPI implementations differ in their support of some
        basic types: the derived types of the 16 bit float of MPICH 4.2 had size zero,
        and Open MPI 4.1 has no valid :samp:`MPI_SHORT_FLOAT`
        (see :func:`mpi_array.types.is_usable_datatype`).
        """
        for np_dt in _np.typecodes["AllInteger"] + _np.typecodes["AllFloat"] + "?":
            itemsize = _np.dtype(np_dt).itemsize
            mpi_dt = _types.to_datatype(np_dt)
            self.assertEqual(itemsize, mpi_dt.Get_size(), np_dt)
            self.assertEqual((0, itemsize), mpi_dt.Get_extent(), np_dt)
            mpi_dt = _types.to_datatype((np_dt, (3,)))
            self.assertEqual(3 * itemsize, mpi_dt.Get_size(), np_dt)
            self.assertEqual((0, 3 * itemsize), mpi_dt.Get_extent(), np_dt)

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

    #: Structured dtypes whose fields do not span the whole :samp:`itemsize`,
    #: or whose first field does not start at byte 0.
    padded_dtypes = \
        (
            _np.dtype([("m0", "int8"), ("m1", "float64")], align=True),
            _np.dtype([("m0", "float64"), ("m1", "int8")], align=True),
            _np.dtype(
                {
                    "names": ["m0", "m1"],
                    "formats": ["int8", "float64"],
                    "offsets": [0, 1],
                    "itemsize": 16
                }
            ),
            _np.dtype(
                {
                    "names": ["m0", "m1"],
                    "formats": ["int32", "int16"],
                    "offsets": [4, 8],
                    "itemsize": 12
                }
            ),
            _np.dtype([("m0", "int8"), ("m1", [("n0", "int16"), ("n1", "int8")])], align=True),
            _np.dtype(
                [
                    ("m0", "int8"),
                    (
                        "m1",
                        {"names": ["n0"], "formats": ["int16"], "offsets": [2], "itemsize": 6}
                    )
                ]
            ),
        )

    def test_struct_extent(self):
        """
        The MPI datatype extent must equal :attr:`numpy.dtype.itemsize` (with lower bound 0)
        for structured dtypes, otherwise multi-element transfers use the wrong stride.
        Padded structs used to get extent 9 for itemsize 16.
        """
        dtypes = \
            (
                _np.dtype([("m0", "int8"), ("m1", "float64")]),
                _np.dtype([("m0", "float64"), ("m1", "int16"), ("m2", "float16"), ("m3", "int32")]),
            ) + self.padded_dtypes
        for dtyp in dtypes:
            mpi_dt = _types.to_datatype(dtyp)
            self.assertSequenceEqual((0, dtyp.itemsize), tuple(mpi_dt.Get_extent()), repr(dtyp))

    def test_struct_get(self):
        """
        One-sided :meth:`mpi4py.MPI.Win.Get` of multiple structured elements with padding
        (the last element(s) used to be lost or misplaced).
        """
        comm = _mpi.COMM_WORLD
        num_elems = 7
        for dtyp in self.padded_dtypes:
            mpi_dt = _types.to_datatype(dtyp)

            def fill(ary, rank):
                for name in ary.dtype.names:
                    if ary.dtype[name].names is None:
                        ary[name] = 10 * rank + _np.arange(1, ary.shape[0] + 1)
                    else:
                        fill(ary[name], rank + 1)

            src = _np.zeros((num_elems,), dtype=dtyp)
            fill(src, comm.rank)
            dst = _np.zeros_like(src)
            target_rank = (comm.rank + 1) % comm.size
            win = _mpi.Win.Create(src, dtyp.itemsize, comm=comm)
            try:
                win.Fence()
                win.Get([dst, num_elems, mpi_dt], target_rank, [0, num_elems, mpi_dt])
                win.Fence()
            finally:
                win.Free()
            expected = _np.zeros_like(src)
            fill(expected, target_rank)
            self.assertSequenceEqual(expected.tolist(), dst.tolist(), repr(dtyp))

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
