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

Classes
=======

.. autosummary::
   :toctree: generated/

   LruCache - A least recently used cache.
   SharedMemUsagePercentString - Shared memory use string, calculated when formatted.

"""
from __future__ import absolute_import

import collections as _collections
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


class SharedMemUsagePercentString(object):

    """
    Its string is :func:`get_shared_mem_usage_percent_string`, only calculated when
    a logging message which has it is formatted.
    """

    def __str__(self):
        return get_shared_mem_usage_percent_string()


def log_shared_memory_alloc(logger, pfx, num_rank_bytes, rank_shape, dtype, buffer=None):
    """
    Generates logging message which indicates amount of shared-memory allocated
    using call to :meth:`mpi4py.MPI.Win.Allocate_shared`.
    """
    sfx = "."
    if pfx.find("BEG") >= 0:
        sfx = sfx + ".."
    logger_args = \
        (pfx, num_rank_bytes, rank_shape, dtype, SharedMemUsagePercentString())
    buffer_fmt = ""
    if buffer is not None:
        buffer_fmt = ", buffer=%s"
        logger_args = logger_args + (buffer, )
    logger_args = logger_args + (sfx,)
    logger(
        "%sWin.Allocate_shared - allocating buffer of %12d bytes for shape=%s, "
        +
        "dtype=%s, shared-mem-usage=%s" + buffer_fmt + "%s",
        *logger_args
    )


def log_memory_alloc(logger, pfx, num_rank_bytes, rank_shape, dtype, buffer=None):
    """
    Generates logging message which indicates amount of memory allocated
    using call to :meth:`mpi4py.MPI.Win.Allocate`.
    """
    sfx = "."
    if pfx.find("BEG") >= 0:
        sfx = sfx + ".."
    logger_args = \
        (pfx, num_rank_bytes, rank_shape, dtype)
    buffer_fmt = ""
    if buffer is not None:
        buffer_fmt = ", buffer=%s"
        logger_args = logger_args + (buffer, )
    logger_args = logger_args + (sfx,)
    logger(
        "%sWin.Allocate - allocating buffer of %12d bytes for shape=%s, "
        +
        "dtype=%s" + buffer_fmt + "%s",
        *logger_args
    )


class LruCache(object):

    """
    A least recently used cache: a mapping of at most :attr:`~LruCache.max_size` entries,
    the least
    recently used entry is removed when there are more. For values which are the same
    on every process (the entries which are removed can then differ between processes).
    """

    def __init__(self, max_size=256):
        """
        Initialise.

        :type max_size: :obj:`int`
        :param max_size: The maximum number of entries.
        """
        self._max_size = max_size
        self._entries = _collections.OrderedDict()

    @property
    def max_size(self):
        """
        The maximum number of entries.
        """
        return self._max_size

    def __len__(self):
        """
        The number of entries.
        """
        return len(self._entries)

    def __contains__(self, key):
        """
        Whether :samp:`{key}` has an entry.
        """
        return key in self._entries

    def get(self, key, default=None):
        """
        Returns the value of :samp:`{key}` (and marks it as the most recently used), or
        :samp:`{default}` if it has no entry.
        """
        value = self._entries.get(key, default)
        if key in self._entries:
            self._entries.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        """
        Sets the value of :samp:`{key}`, removing the least recently used entries
        when there are more than :attr:`max_size`.
        """
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def clear(self):
        """
        Removes all entries.
        """
        self._entries.clear()


__all__ = [s for s in dir() if not s.startswith('_')]
