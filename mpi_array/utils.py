"""
=================================
The :mod:`mpi_array.utils` Module
=================================

Various utilities.

Functions
=========

.. autosummary::
   :toctree: generated/

   get_shared_mem_usage_percent_string - Returns current consumed percentage of system shared-mem.
   log_shared_memory_alloc - Generates logging message with amount of shared memory allocated.
   log_memory_alloc - Generates logging message with amount of memory allocated.

"""
from __future__ import absolute_import

import psutil as _psutil

from .license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def get_shared_mem_usage_percent_string(shm_file_name="/dev/shm"):
    """
    Returns a string indicating the current percentage of available
    shared memory which is allocated.

    :type shm_file_name: :obj:`str`
    :param shm_file_name: Absolute path of shared-memory file.
    """
    usage_percent = "unknown"
    try:
        usage_percent_float = _psutil.disk_usage(shm_file_name).percent
        usage_percent = "%5.2f%%" % usage_percent_float
    except Exception:
        pass
    return usage_percent


def log_shared_memory_alloc(logger, pfx, num_rank_bytes, rank_shape, dtype):
    """
    Generates logging message which indicates amount of shared-memory allocated
    using call to :meth:`mpi4py.MPI.Win.Allocate_shared`.
    """
    sfx = "."
    if pfx.find("BEG") >= 0:
        sfx = sfx + ".."
    logger(
        "%sWin.Allocate_shared - allocating buffer of %12d bytes for shape=%s, "
        +
        "dtype=%s, shared-mem-usage=%s%s",
        pfx,
        num_rank_bytes,
        rank_shape,
        dtype,
        get_shared_mem_usage_percent_string(),
        sfx
    )


def log_memory_alloc(logger, pfx, num_rank_bytes, rank_shape, dtype):
    """
    Generates logging message which indicates amount of memory allocated
    using call to :meth:`mpi4py.MPI.Win.Allocate`.
    """
    sfx = "."
    if pfx.find("BEG") >= 0:
        sfx = sfx + ".."
    logger(
        "%sWin.Allocate - allocating buffer of %12d bytes for shape=%s, "
        +
        "dtype=%s%s",
        pfx,
        num_rank_bytes,
        rank_shape,
        dtype,
        sfx
    )


__all__ = [s for s in dir() if not s.startswith('_')]
