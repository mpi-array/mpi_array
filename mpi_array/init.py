"""
Initialisation which needs to occur prior to :samp:`MPI_Init`.

Parts of this source borrows from
the `airspeed velocity (asv) <http://asv.readthedocs.io/en/latest>`_
file `benchmark.py <https://github.com/spacetelescope/asv/blob/master/asv/benchmark.py>`_.

See the `LICENSE <https://github.com/spacetelescope/asv/blob/master/LICENSE.rst>`_.

"""
# Licensed under a 3-clause BSD style license - see LICENSE.rst

__license__ = "https://github.com/spacetelescope/asv/blob/master/LICENSE.rst"


def create_linux_process_time():
    """
    The best timer we can use is time.process_time, but it is not
    available in the Python stdlib until Python 3.3.  This is a ctypes
    backport (Linux only) for Python versions which don't have it.
    """
    import ctypes
    from ctypes.util import find_library
    import errno

    CLOCK_PROCESS_CPUTIME_ID = 2  # time.h

    clockid_t = ctypes.c_int
    time_t = ctypes.c_long

    class timespec(ctypes.Structure):
        _fields_ = [
            ('tv_sec', time_t),         # seconds
            ('tv_nsec', ctypes.c_long)  # nanoseconds
        ]
    _clock_gettime = ctypes.CDLL(
        find_library('rt'), use_errno=True).clock_gettime
    _clock_gettime.argtypes = [clockid_t, ctypes.POINTER(timespec)]

    def process_time():
        tp = timespec()
        if _clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ctypes.byref(tp)) < 0:
            err = ctypes.get_errno()
            msg = errno.errorcode[err]
            if err == errno.EINVAL:
                msg += (
                    "The clk_id (4) specified is not supported on this system")
            raise OSError(err, msg)
        return tp.tv_sec + tp.tv_nsec * 1e-9

    return process_time


def create_darwin_process_time():
    """
    The best timer we can use is time.process_time, but it is not
    available in the Python stdlib until Python 3.3.  This is a ctypes
    backport (Darwin only) for Python versions which don't have it.
    """
    import ctypes
    from ctypes.util import find_library
    import errno

    RUSAGE_SELF = 0  # sys/resources.h

    time_t = ctypes.c_long
    suseconds_t = ctypes.c_int32

    class timeval(ctypes.Structure):
        _fields_ = [
            ('tv_sec', time_t),
            ('tv_usec', suseconds_t)
        ]

    class rusage(ctypes.Structure):
        _fields_ = [
            ('ru_utime', timeval),
            ('ru_stime', timeval),
            ('ru_maxrss', ctypes.c_long),
            ('ru_ixrss', ctypes.c_long),
            ('ru_idrss', ctypes.c_long),
            ('ru_isrss', ctypes.c_long),
            ('ru_minflt', ctypes.c_long),
            ('ru_majflt', ctypes.c_long),
            ('ru_nswap', ctypes.c_long),
            ('ru_inblock', ctypes.c_long),
            ('ru_oublock', ctypes.c_long),
            ('ru_msgsnd', ctypes.c_long),
            ('ru_msgrcv', ctypes.c_long),
            ('ru_nsignals', ctypes.c_long),
            ('ru_nvcsw', ctypes.c_long),
            ('ru_nivcsw', ctypes.c_long)
        ]

    _getrusage = ctypes.CDLL(find_library('c'), use_errno=True).getrusage
    _getrusage.argtypes = [ctypes.c_int, ctypes.POINTER(rusage)]

    def process_time():
        ru = rusage()
        if _getrusage(RUSAGE_SELF, ctypes.byref(ru)) < 0:
            err = ctypes.get_errno()
            msg = errno.errorcode[err]
            if err == errno.EINVAL:
                msg += (
                    "The clk_id (0) specified is not supported on this system")
            raise OSError(err, msg)
        return float(ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 +
                     ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6)

    return process_time


_process_time = None


def initialise_process_time_timer():
    """
    The best timer we can use is time.process_time, but it is not
    available in the Python stdlib until Python 3.3.  This is a ctypes
    backport for Pythons that don't have it.

    :rtype: :obj:`function`
    :return: The :func:`time.process_time` function, if available, otherwise, if possible,
       an equivalent created using :mod:`ctypes`, otherwise :func:`timeit.default_timer`.
    """
    global _process_time

    if _process_time is None:
        try:
            from time import process_time
        except ImportError:  # Python <3.3
            import sys
            if sys.platform.startswith("linux"):
                process_time = create_linux_process_time()
            elif sys.platform == 'darwin':
                process_time = create_darwin_process_time()
            else:
                # Fallback to default timer
                import timeit
                process_time = timeit.default_timer
        _process_time = process_time

    return _process_time


def get_process_time_timer():
    """
    The best timer we can use is time.process_time, but it is not
    available in the Python stdlib until Python 3.3.  This is a ctypes
    backport for Pythons that don't have it.

    :rtype: :obj:`function`
    :return: The :func:`time.process_time` function, if available, otherwise, if possible,
       an equivalent created using :mod:`ctypes`, otherwise :func:`timeit.default_timer`.
    """
    global _process_time

    if _process_time is None:
        initialise_process_time_timer()
    return _process_time


initialise_process_time_timer()

__all__ = [s for s in dir() if not s.startswith('_')]
