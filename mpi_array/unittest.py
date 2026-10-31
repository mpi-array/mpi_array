"""
======================================
The :mod:`mpi_array.unittest` Module
======================================

Some simple wrappers of python built-in :mod:`unittest` module
for :mod:`mpi_array` unit-tests.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   TestCase - Extends :obj:`unittest.TestCase` with :obj:`TestCase.assertArraySplitEqual`.

.. autosummary::
   :toctree: generated/

   TestProgram - Over-ride to use :obj:`logging.Logger` output.
   TextTestRunner - Over-ride to use :obj:`logging.Logger` output.
   TextTestResult - Over-ride to use :obj:`logging.Logger` output.
   LoggerDecorator - A stream like object which writes to a :obj:`logging.Logger`.
   main - Convenience command-line test-case *search and run* function.
   handle_arg - Replaces an argument unless it was given.

Hang detection
==============

A deadlock in a multi-process run would otherwise hang forever. Each test is
therefore run under a :func:`faulthandler.dump_traceback_later` timer: if a single
test takes longer than the timeout, every rank dumps the stacks of all its threads
to :obj:`sys.stderr` and exits with a non-zero status (which makes :samp:`mpirun`
abort the job). The per-test timeout (in seconds) is read from
the :samp:`MPI_ARRAY_TEST_TIMEOUT` environment variable, a value of :samp:`0`
disables the timer. If the :samp:`MPI_ARRAY_TEST_HANG_DIR` environment variable
names a directory, each rank writes its stacks to the
file :samp:`hang_stacks_rank{NNNN}.txt` in that directory instead of :obj:`sys.stderr`
(:samp:`mpirun` may stop forwarding the other ranks' output once the first rank exits).

.. autosummary::
   :toctree: generated/

   get_test_timeout - Returns the per-test hang-detection timeout.
   arm_hang_detection - (Re)starts the hang-detection timer.
   cancel_hang_detection - Stops the hang-detection timer.
   open_hang_dump_file - Opens the per-rank file for hang-detection stacks.
   close_hang_dump_file - Closes the per-rank file for hang-detection stacks.

Attributes
==========

.. autosummary::
   :toctree: generated/

   TEST_TIMEOUT_ENV_VAR
   DEFAULT_TEST_TIMEOUT
   TEST_HANG_DIR_ENV_VAR
   NO_TESTS_EXIT_STATUS

"""
from __future__ import absolute_import

import faulthandler as _faulthandler
import os as _os
import signal as _signal
import sys as _sys
import unittest as _builtin_unittest
import mpi_array.logging
import numpy as _np
import mpi4py.MPI as _mpi
import time as _time
import warnings as _warnings

#: Name of the environment variable which sets the per-test hang-detection timeout (seconds).
TEST_TIMEOUT_ENV_VAR = "MPI_ARRAY_TEST_TIMEOUT"

#: Per-test hang-detection timeout (seconds) used when :samp:`MPI_ARRAY_TEST_TIMEOUT` is not set.
DEFAULT_TEST_TIMEOUT = 300.0

#: Name of the environment variable naming a directory for per-rank hang-detection stack files.
TEST_HANG_DIR_ENV_VAR = "MPI_ARRAY_TEST_HANG_DIR"

#: Exit status when no tests were run (same as :func:`unittest.main` in Python 3.12+).
NO_TESTS_EXIT_STATUS = 5

#: File object to which hang-detection stacks are written (:samp:`None` means :obj:`sys.stderr`).
_hang_dump_file = None


def get_test_timeout():
    """
    Returns the per-test hang-detection timeout.

    :rtype: :obj:`float`
    :return: Timeout in seconds, read from the :samp:`MPI_ARRAY_TEST_TIMEOUT` environment
       variable (default :attr:`DEFAULT_TEST_TIMEOUT`). Zero (or negative) means no timeout.
    :raises ValueError: If the environment variable is not a number.
    """
    value = _os.environ.get(TEST_TIMEOUT_ENV_VAR, "").strip()
    if len(value) == 0:
        return DEFAULT_TEST_TIMEOUT
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            "Environment variable %s=%r is not a number of seconds."
            %
            (TEST_TIMEOUT_ENV_VAR, value)
        )


def _hang_dump_target():
    """
    Returns the file object to which hang-detection stacks are written.
    """
    return _hang_dump_file if _hang_dump_file is not None else _sys.stderr


def open_hang_dump_file():
    """
    If the :samp:`MPI_ARRAY_TEST_HANG_DIR` environment variable names a directory,
    opens the file :samp:`hang_stacks_rank{NNNN}.txt` (:samp:`{NNNN}` is
    the :obj:`mpi4py.MPI.COMM_WORLD` rank) in that directory, and the hang detection
    writes stacks to this file instead of :obj:`sys.stderr`.

    :rtype: :samp:`None` or :obj:`str`
    :return: The file name, :samp:`None` if the environment variable is not set.
    """
    global _hang_dump_file
    close_hang_dump_file()
    hang_dir = _os.environ.get(TEST_HANG_DIR_ENV_VAR, "").strip()
    if len(hang_dir) == 0:
        return None
    _os.makedirs(hang_dir, exist_ok=True)
    file_name = _os.path.join(hang_dir, "hang_stacks_rank%04d.txt" % _mpi.COMM_WORLD.rank)
    _hang_dump_file = open(file_name, "w")
    return file_name


def close_hang_dump_file():
    """
    Closes the file opened by :func:`open_hang_dump_file` (removing it if it is empty).
    """
    global _hang_dump_file
    if _hang_dump_file is not None:
        file_name = _hang_dump_file.name
        _hang_dump_file.close()
        _hang_dump_file = None
        if _os.path.exists(file_name) and (_os.path.getsize(file_name) == 0):
            _os.remove(file_name)


def arm_hang_detection(timeout=None):
    """
    (Re)starts the hang-detection timer. If the timer is not re-armed (or cancelled)
    within :samp:`{timeout}` seconds, the stacks of all threads are written
    to :obj:`sys.stderr` (or the file opened by :func:`open_hang_dump_file`)
    and the process exits with status 1.

    :type timeout: :obj:`float`
    :param timeout: Timeout in seconds, if :samp:`None` uses :func:`get_test_timeout`.
       Zero (or negative) cancels the timer.
    """
    if timeout is None:
        timeout = get_test_timeout()
    _faulthandler.cancel_dump_traceback_later()
    if timeout > 0:
        _faulthandler.dump_traceback_later(timeout, exit=True, file=_hang_dump_target())


def cancel_hang_detection():
    """
    Stops the hang-detection timer started by :func:`arm_hang_detection`.
    """
    _faulthandler.cancel_dump_traceback_later()


def _fix_docstring_for_sphinx(docstr):
    lines = docstr.split("\n")
    for i in range(len(lines)):
        if lines[i].find(" " * 8) == 0:
            lines[i] = lines[i][8:]
    return "\n".join(lines)


class TestCase(_builtin_unittest.TestCase):

    """
    Extends :obj:`unittest.TestCase` with the :meth:`~TestCase.assertArraySplitEqual`.
    """

    def assertArraySplitEqual(self, splt1, splt2):
        """
        Compares :obj:`list` of :obj:`numpy.ndarray` results returned by
        :func:`numpy.array_split` and :func:`array_split.split.array_split` functions.

        :type splt1: :obj:`list` of :obj:`numpy.ndarray`
        :param splt1: First object in equality comparison.
        :type splt2: :obj:`list` of :obj:`numpy.ndarray`
        :param splt2: Second object in equality comparison.
        :raises AssertionError: If any element of :samp:`{splt1}` is not equal to
            the corresponding element of :samp:`splt2`.
        """
        self.assertEqual(len(splt1), len(splt2))
        for i in range(len(splt1)):
            self.assertTrue(
                (
                    _np.all(_np.array(splt1[i]) == _np.array(splt2[i]))
                    or
                    ((_np.array(splt1[i]).size == 0) and (_np.array(splt2[i]).size == 0))
                ),
                msg=(
                    "element %d of split is not equal %s != %s"
                    %
                    (i, _np.array(splt1[i]), _np.array(splt2[i]))
                )
            )

    #
    # Method over-rides below are just to avoid sphinx warnings.
    # They return the base-class result so that the context-manager forms
    # (e.g. ``with self.assertRaisesRegex(...):``) work.
    #
    def assertListEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertListEqual`.
        """
        return _builtin_unittest.TestCase.assertListEqual(self, *args, **kwargs)

    def assertRaisesRegex(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertRaisesRegex`.
        """
        return _builtin_unittest.TestCase.assertRaisesRegex(self, *args, **kwargs)

    def assertSequenceEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertSequenceEqual`.
        """
        return _builtin_unittest.TestCase.assertSequenceEqual(self, *args, **kwargs)

    def assertSetEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertSetEqual`.
        """
        return _builtin_unittest.TestCase.assertSetEqual(self, *args, **kwargs)

    def assertTupleEqual(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertTupleEqual`.
        """
        return _builtin_unittest.TestCase.assertTupleEqual(self, *args, **kwargs)

    def assertWarnsRegex(self, *args, **kwargs):
        """
        See :obj:`unittest.TestCase.assertWarnsRegex`.
        """
        return _builtin_unittest.TestCase.assertWarnsRegex(self, *args, **kwargs)


class LoggerDecorator:

    """
    Decorator for :obj:`logging.Logger` to provide :meth:`~LoggerDecorator.write`,
    :meth:`~LoggerDecorator.writeln` and :meth:`~LoggerDecorator.flush` methods.
    """

    def __init__(self, logger):
        """
        Initialise.

        :type logger: :obj:`logging.Logger`
        :param logger: The logger to which the messages are written.
        """
        self.logger = logger

    def write(self, v=""):
        """
        Writes :samp:`{v}` as an info message.
        """
        self.logger.info(v)

    def writeln(self, v=""):
        """
        Writes :samp:`{v}` as an info message.
        """
        self.logger.info(v)

    def write_error(self, v):
        """
        Writes :samp:`{v}` as an error message.
        """
        self.logger.error(v)

    def flush(self):
        """
        Does nothing.
        """
        pass


class TextTestResult(_builtin_unittest.TextTestResult):

    """
    Over-ride to use :obj:`logging.Logger` output. Each test is run
    under the hang-detection timer (see :func:`arm_hang_detection`).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialise, the arguments are those of :obj:`unittest.TextTestResult`.
        """
        _builtin_unittest.TextTestResult.__init__(self, *args, **kwargs)

    def startTest(self, test):
        arm_hang_detection()
        _builtin_unittest.result.TestResult.startTest(self, test)
        if self.showAll:
            self.stream.write(self.getDescription(test) + "...")
            self.stream.flush()

    def stopTestRun(self):
        cancel_hang_detection()
        _builtin_unittest.TextTestResult.stopTestRun(self)

    def addSuccess(self, test):
        _builtin_unittest.result.TestResult.addSuccess(self, test)
        if self.showAll:
            self.stream.write(self.getDescription(test) + "..." + "ok")
        elif self.dots:
            self.stream.write('.')
            self.stream.flush()

    def addError(self, test, err):
        _builtin_unittest.result.TestResult.addError(self, test, err)
        if self.showAll:
            self.stream.write_error("ERROR")
        elif self.dots:
            self.stream.write_error('E')
            self.stream.flush()

    def addFailure(self, test, err):
        _builtin_unittest.result.TestResult.addFailure(self, test, err)
        if self.showAll:
            self.stream.write_error("FAIL")
        elif self.dots:
            self.stream.write_error('F')
            self.stream.flush()

    def addSkip(self, test, reason):
        _builtin_unittest.result.TestResult.addSkip(self, test, reason)
        if self.showAll:
            self.stream.write("skipped {0!r}".format(reason))
        elif self.dots:
            self.stream.write("s")
            self.stream.flush()

    def addExpectedFailure(self, test, err):
        _builtin_unittest.result.TestResult.addExpectedFailure(self, test, err)
        if self.showAll:
            self.stream.write("expected failure")
        elif self.dots:
            self.stream.write("x")
            self.stream.flush()

    def addUnexpectedSuccess(self, test):
        _builtin_unittest.result.TestResult.addUnexpectedSuccess(self, test)
        if self.showAll:
            self.stream.write("unexpected success")
        elif self.dots:
            self.stream.write("u")
            self.stream.flush()

    def printErrors(self):
        if self.dots or self.showAll:
            self.stream.write()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.write_error(self.separator1)
            self.stream.write_error("%s: %s" % (flavour, self.getDescription(test)))
            self.stream.write_error(self.separator2)
            self.stream.write_error("%s" % err)


def handle_arg(arg_index, arg_key, arg_value, args, kwargs):
    """
    Replace an argument with a specified value if it does not
    appear in :samp:`{args}` or :samp:`{kwargs}` argument list.

    :type arg_index: :obj:`int`
    :param arg_index: Index of argument in :samp:`{args}`
    :type arg_key: :obj:`str`
    :param arg_index: String key of argument in :samp:`{kwargs}`
    :type arg_value: :obj:`object`
    :param arg_value: Value for argument if it does not appear in argument
       lists or has :samp:`None` value in argument lists.
    :type args: :obj:`list`
    :param args: List of arguments.
    :type kwargs: :obj:`dict`
    :param kwargs: Dictionary of key-word arguments.
    """
    a = None
    if len(args) > arg_index:
        a = args[arg_index]
    if arg_key in kwargs.keys():
        a = kwargs[arg_key]

    if a is None:
        a = arg_value

    if len(args) > arg_index:
        args[arg_index] = a
    if arg_key in kwargs.keys():
        kwargs[arg_key] = a
    if (len(args) <= arg_index) and (arg_key not in kwargs.keys()):
        kwargs[arg_key] = a

    return a


class TextTestRunner(_builtin_unittest.TextTestRunner):

    """
    A test runner class that displays results in textual form.

    Extends :obj:`unittest.TextTestRunner` with logging output
    instead of :obj:`sys.stderr` output.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialise, the arguments are those of :obj:`unittest.TextTestRunner`. The default
        result class is :obj:`TextTestResult`, and the default stream is
        a :obj:`LoggerDecorator`.
        """
        handle_arg(5, "resultclass", TextTestResult, args, kwargs)
        verbosity = handle_arg(2, "verbosity", 0, args, kwargs)

        logger_name = __name__ + ".TextTestRunner"
        logger = mpi_array.logging.get_rank_logger(logger_name)
        log_level = mpi_array.logging.WARN
        if verbosity <= 1:
            if _mpi.COMM_WORLD.rank == 0:
                log_level = mpi_array.logging.INFO
        else:
            log_level = mpi_array.logging.INFO
        mpi_array.logging.initialise_loggers([logger_name, ], log_level)
        stream = LoggerDecorator(logger)

        handle_arg(0, "stream", stream, args, kwargs)

        _builtin_unittest.TextTestRunner.__init__(self, *args, **kwargs)

        # Remove _WritelnDecorator decoration for LoggerDecorator
        if hasattr(self.stream, "stream") and isinstance(self.stream.stream, LoggerDecorator):
            self.stream = self.stream.stream

        if not hasattr(self, "warnings"):
            self.warnings = None

    def run(self, test):
        """
        Run the given test case or test suite.
        """
        result = self._makeResult()
        _builtin_unittest.registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        with _warnings.catch_warnings():
            if self.warnings:
                # if self.warnings is set, use it to filter all the warnings
                _warnings.simplefilter(self.warnings)
                # if the filter is 'default' or 'always', special-case the
                # warnings from the deprecated unittest methods to show them
                # no more than once per module, because they can be fairly
                # noisy.  The -Wd and -Wa flags can be used to bypass this
                # only when self.warnings is None.
                if self.warnings in ['default', 'always']:
                    _warnings.filterwarnings(
                        'module',
                        category=DeprecationWarning,
                        message=r'Please use assert\w+ instead.'
                    )
            startTime = _time.time()
            startTestRun = getattr(result, 'startTestRun', None)
            if startTestRun is not None:
                startTestRun()
            try:
                test(result)
            finally:
                stopTestRun = getattr(result, 'stopTestRun', None)
                if stopTestRun is not None:
                    stopTestRun()
            stopTime = _time.time()
        timeTaken = stopTime - startTime
        result.printErrors()
        if hasattr(result, 'separator2'):
            self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs (COMM_WORLD.size=%3d)" %
                            (run, run != 1 and "s" or "", timeTaken, _mpi.COMM_WORLD.size))
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(len, (result.expectedFailures,
                                result.unexpectedSuccesses,
                                result.skipped))
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped = results

        infos = []
        if not result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = len(result.failures), len(result.errors)
            if failed:
                infos.append("failures=%d" % failed)
            if errored:
                infos.append("errors=%d" % errored)
        else:
            self.stream.write("OK")
        if skipped:
            infos.append("skipped=%d" % skipped)
        if expectedFails:
            infos.append("expected failures=%d" % expectedFails)
        if unexpectedSuccesses:
            infos.append("unexpected successes=%d" % unexpectedSuccesses)
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        return result


class TestProgram(_builtin_unittest.TestProgram):

    """
    A command-line program that runs a set of tests, extends :samp:`unittest.TestProgram`
    by using logging rather than standard stream.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialise, the arguments are those of :func:`unittest.main`. The default test
        runner is :obj:`TextTestRunner`.
        """
        handle_arg(3, "testRunner", TextTestRunner, args, kwargs)
        _builtin_unittest.TestProgram.__init__(self, *args, **kwargs)


def main(
    module_name,
    log_level=mpi_array.logging.DEBUG,
    init_logger_names=None,
    verbosity=None,
    failfast=None
):
    """
    Like :func:`unittest.main`, initialises :obj:`logging.Logger` objects
    and instantiates a :obj:`TestProgram` to discover and run :obj:`TestCase` objects.
    Loads a set of tests from module and runs them;
    this is primarily for making test modules conveniently executable.
    The simplest use for this function is to include the following line at
    the end of a test module::

       mpi_array.unittest.main(__name__)

    If :samp:`__name__ == "__main__"`, then *discoverable* :obj:`unittest.TestCase`
    test cases are executed.
    Logging level can be explicitly set for a group of modules using::

       import logging

       mpi_array.unittest.main(
           __name__,
           logging.DEBUG,
           [__name__, "module_name_0", "module_name_1", "package.module_name_2"]
       )


    :type module_name: :obj:`str`
    :param module_name: If :samp:`{module_name} == "__main__"` then unit-tests
       are *discovered* and run.
    :type log_level: :obj:`int`
    :param log_level: The default logging level for all
       :obj:`logging.Logger` objects.
    :type init_logger_names: sequence of :obj:`str`
    :param init_logger_names: List of logger names to initialise
       (using :func:`mpi_array.logging.initialise_loggers`). If :samp:`None`,
       then the list defaults to :samp:`[{module_name}, "mpi_array"]`. If list
       is empty no loggers are initialised.

    Each test runs under the hang-detection timer (see :func:`arm_hang_detection`).
    After the tests have run, the ranks of :obj:`mpi4py.MPI.COMM_WORLD` agree on
    the exit status, so every rank exits with :samp:`0` if all ranks passed, with :samp:`1`
    if any rank had a failure or error, and with :attr:`NO_TESTS_EXIT_STATUS` if no
    tests were found.
    """
    if module_name == "__main__":
        if (init_logger_names is None):
            init_logger_names = [module_name, "mpi_array"]

        if (len(init_logger_names) > 0):
            mpi_array.logging.initialise_loggers(
                init_logger_names, log_level=log_level)

        kwargs = dict()
        if failfast is not None:
            kwargs["failfast"] = failfast
        if verbosity is not None:
            kwargs["verbosity"] = verbosity

        timeout = get_test_timeout()
        if timeout > 0:
            open_hang_dump_file()
        dump_on_sigterm = (timeout > 0) and hasattr(_faulthandler, "register")
        if dump_on_sigterm:
            # When mpirun kills the remaining ranks (because the timer fired on another
            # rank first) these ranks also dump their stacks.
            _faulthandler.register(
                _signal.SIGTERM,
                file=_hang_dump_target(),
                all_threads=True,
                chain=True
            )
        arm_hang_detection(timeout)
        try:
            program = TestProgram(exit=False, **kwargs)
            result = program.result
            if (result.testsRun == 0) and (len(result.skipped) == 0):
                exit_status = NO_TESTS_EXIT_STATUS
            elif result.wasSuccessful():
                exit_status = 0
            else:
                exit_status = 1
            arm_hang_detection(timeout)
            exit_status = _mpi.COMM_WORLD.allreduce(exit_status, op=_mpi.MAX)
        finally:
            cancel_hang_detection()
            if dump_on_sigterm:
                _faulthandler.unregister(_signal.SIGTERM)
            close_hang_dump_file()

        _sys.exit(exit_status)


__all__ = [s for s in dir() if not s.startswith('_')]
