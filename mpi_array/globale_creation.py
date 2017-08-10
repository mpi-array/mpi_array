"""
============================================
The :mod:`mpi_array.globale_creation` Module
============================================

Defines :obj:`mpi_array.globale.gndarray` creation functions.

Functions
=========

.. autosummary::
   :toctree: generated/

   asarray - Returns :obj:`mpi_array.globale.gndarray` equivalent of input.


"""

from __future__ import absolute_import

import numpy as _np

from .license import license as _license, copyright as _copyright, version as _version
from .globale import gndarray as _gndarray, empty as _empty
from . import logging as _logging  # noqa: E402,F401
from . import comms as _comms

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def asarray(a, dtype=None, order=None, **kwargs):
    """
    :rtype: :obj:`mpi_array.globale.gndarray`
    """
    if hasattr(a, "__class__") and (a.__class__ is _gndarray):
        ret_ary = a
    elif isinstance(a, _gndarray):
        ret_ary =\
            _gndarray(
                comms_and_distrib=a.comms_and_distrib,
                rma_window_buffer=a.rma_window_buffer,
                lndarray_proxy=a.lndarray_proxy
            )
    else:
        if "distrib_type" not in kwargs.keys() or kwargs["distrib_type"] is None:
            kwargs["distrib_type"] = _comms.DT_CLONED
        np_ary = _np.asanyarray(a, dtype, order)
        ret_ary = \
            _empty(
                np_ary.shape,
                dtype=np_ary.dtype,
                **kwargs
            )
        if (ret_ary.ndim == 0) and (ret_ary.locale_comms.have_valid_inter_locale_comm):
            ret_ary.lndarray_proxy.lndarray[...] = np_ary
        else:
            locale_rank_view_slice_n = ret_ary.lndarray_proxy.rank_view_slice_n
            globale_rank_view_slice_n = \
                ret_ary.lndarray_proxy.locale_extent.locale_to_globale_slice_h(
                    locale_rank_view_slice_n
                )

            ret_ary.lndarray_proxy.lndarray[locale_rank_view_slice_n] =\
                np_ary[globale_rank_view_slice_n]

        ret_ary.intra_locale_barrier()

    return ret_ary


__all__ = [s for s in dir() if not s.startswith('_')]
