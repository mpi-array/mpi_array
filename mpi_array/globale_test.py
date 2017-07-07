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
from mpi_array.decomposition import CartesianDecomposition, MemAllocTopology
import mpi_array.globale
import mpi4py.MPI as _mpi
import numpy as _np

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

    def test_attr(self):
        """
        Test various attributes of :obj:`mpi_array.globale.gndarray`.
        """
        gary = mpi_array.globale.zeros((21, 35, 50), dtype="int8")
        if gary.decomp.shared_mem_comm.rank == 0:
            gary.lndarray.view_h[...] = 0
        gary.decomp.shared_mem_comm.barrier()
        rank_val = gary.decomp.rank_comm.rank
        gary.rank_view_n[...] = rank_val

        self.assertEqual(gary.dtype, _np.dtype("int8"))
        self.assertSequenceEqual(list(gary.shape), list((21, 35, 50)))
        self.assertTrue(gary.decomp is not None)
        self.assertTrue(gary.lndarray is not None)
        self.assertTrue(isinstance(gary.lndarray, mpi_array.locale.lndarray))
        self.assertTrue(gary.decomp is gary.lndarray.decomp)
        self.assertEqual("C", gary.order)
        self.assertTrue(gary.rank_logger is not None)
        self.assertTrue(isinstance(gary.rank_logger, _logging.Logger))
        self.assertTrue(gary.root_logger is not None)
        self.assertTrue(isinstance(gary.root_logger, _logging.Logger))
        self.assertTrue(_np.all(gary.rank_view_n == rank_val))
        if _np.any(gary.rank_view_h.shape > gary.rank_view_n.shape):
            self.assertTrue(
                _np.all(_np.where(gary.rank_view_h == rank_val, 0, gary.rank_view_h) == 0)
            )

    def test_get_item_and_set_item(self):
        """
        Test the :meth:`mpi_array.globale.gndarray.__getitem__`
        and :meth:`mpi_array.globale.gndarray.__setitem__` methods.
        """
        gary = mpi_array.globale.zeros((20, 20, 20), dtype="int8")
        gary[1, 2, 8] = 22
        gary[1:10, 2:4, 4:8]
        gary[...] = 19
        gary[:] = 101

    def test_update_1d(self):
        """
        Test for :meth:`mpi_array.globale.gndarray.update`, 1D decomposition.
        """

        halo = 4
        for lshape in ((100, 200), (1000,), ):
            gshape = (_mpi.COMM_WORLD.size * lshape[0],) + lshape[1:]
            ndims = len(lshape)
            mat = \
                MemAllocTopology(
                    ndims=ndims,
                    dims=(0,) + (1,) * (ndims - 1),
                    rank_comm=_mpi.COMM_WORLD,
                    shared_mem_comm=_mpi.COMM_SELF
                )
            non_shared_decomp = \
                CartesianDecomposition(shape=gshape, mem_alloc_topology=mat, halo=halo)

            lshape = (10000,) + lshape[1:]
            mat = \
                MemAllocTopology(
                    ndims=ndims,
                    dims=(0,) + (1,) * (ndims - 1),
                    rank_comm=_mpi.COMM_WORLD
                )
            gshape = (mat.num_shared_mem_nodes * lshape[0],) + lshape[1:]
            halo = 4
            shared_decomp = CartesianDecomposition(shape=gshape, mem_alloc_topology=mat, halo=halo)

            decomp_list = [non_shared_decomp, shared_decomp]

            for decomp in decomp_list:
                gary = mpi_array.globale.empty(decomp=decomp, dtype="int32")
                self.assertEqual(_np.dtype("int32"), gary.dtype)

                if gary.decomp.have_valid_cart_comm:
                    cart_rank_val = gary.decomp.cart_comm.rank + 1
                    gary.lndarray.view_h[...] = 0
                    gary.lndarray.view_n[...] = cart_rank_val

                    if gary.decomp.cart_comm.size > 1:
                        if gary.decomp.cart_comm.rank == 0:
                            self.assertTrue(_np.all(gary.lndarray[-halo:] == 0))
                        elif gary.decomp.cart_comm.rank == (gary.decomp.cart_comm.size - 1):
                            self.assertTrue(_np.all(gary.lndarray[0:halo] == 0))
                        else:
                            self.assertTrue(_np.all(gary.lndarray[0:halo] == 0))
                            self.assertTrue(_np.all(gary.lndarray[-halo:] == 0))

                gary.update()

                if gary.decomp.have_valid_cart_comm:

                    self.assertTrue(_np.all(gary.lndarray.view_n[...] == cart_rank_val))

                    if gary.decomp.cart_comm.size > 1:
                        if gary.decomp.cart_comm.rank == 0:
                            self.assertTrue(_np.all(gary.lndarray[-halo:] == (cart_rank_val + 1)))
                        elif gary.decomp.cart_comm.rank == (gary.decomp.cart_comm.size - 1):
                            self.assertTrue(
                                _np.all(gary.lndarray[0:halo] == (cart_rank_val - 1))
                            )
                        else:
                            self.assertTrue(
                                _np.all(
                                    gary.lndarray[0:halo]
                                    ==
                                    (cart_rank_val - 1)
                                )
                            )
                            self.assertTrue(
                                _np.all(
                                    gary.lndarray[-halo:]
                                    ==
                                    (cart_rank_val + 1)
                                )
                            )
                gary.decomp.shared_mem_comm.barrier()


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
