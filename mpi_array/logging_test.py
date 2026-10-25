"""
=========================================
The :mod:`mpi_array.logging_test` Module
=========================================

Module defining :mod:`mpi_array.logging` unit-tests.
Execute as::

   python -m mpi_array.logging_test


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: autosummary/inherits_TestCase_class.rst

   LoggingTest - Tests for :mod:`mpi_array.logging` functions.

"""
import io as _io
import os as _os
import re as _re
import tempfile as _tempfile
import logging as _builtin_logging
import mpi4py.MPI as _mpi

from .license import license as _license, copyright as _copyright, version as _version
from . import unittest as _unittest
from . import logging as _logging

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class LoggingTest(_unittest.TestCase):

    """
    Tests for :mod:`mpi_array.logging` functions. The loggers used are not
    descendants of the :samp:`"mpi_array"` logger, so that they only have
    the handlers which the tests add.
    """

    def setUp(self):
        """
        Sets :attr:`logger_name`, a name for loggers which is unique to the test.
        """
        self.logger_name = "mpi_array_logging_test." + self._testMethodName
        self.handlers = []

    def tearDown(self):
        """
        Removes the handlers added with :meth:`add_handler`.
        """
        for logger, handler in self.handlers:
            logger.removeHandler(handler)
            handler.close()

    def add_handler(self, logger, handler):
        """
        Adds :samp:`{handler}` to :samp:`{logger}`, it is removed in :meth:`tearDown`.
        """
        logger.addHandler(handler)
        self.handlers.append((logger, handler))
        return handler

    def test_get_handlers(self):
        """
        :func:`mpi_array.logging.get_handlers` returns the handlers of a logger and
        of the ancestors it propagates to, without changing them. It used to extend
        the :samp:`handlers` list of the logger, which grew on every call.
        """
        parent = _builtin_logging.getLogger(self.logger_name)
        child = _builtin_logging.getLogger(self.logger_name + ".child.grandchild")
        parent_handler = self.add_handler(parent, _builtin_logging.StreamHandler(_io.StringIO()))
        child_handler = self.add_handler(child, _builtin_logging.StreamHandler(_io.StringIO()))
        for i in range(3):
            handlers = _logging.get_handlers(child)
            self.assertEqual(2, len(handlers) - len(_logging.get_handlers(parent.parent)))
            self.assertTrue(handlers[0] is child_handler)
            self.assertTrue(handlers[1] is parent_handler)
            self.assertEqual([child_handler], child.handlers)
            self.assertEqual([parent_handler], parent.handlers)
            _logging.get_rank_logger(self.logger_name + ".child")
            self.assertEqual([parent_handler], parent.handlers)

        child.propagate = False
        self.assertEqual([child_handler], _logging.get_handlers(child))
        self.assertEqual([_builtin_logging.StreamHandler], _logging.get_handler_classes(child))

    def test_multi_line_formatter(self):
        """
        :obj:`mpi_array.logging.MultiLineFormatter` formats each line of a message,
        and adds an exception traceback (and stack information) once. They used to be
        added again to every line of the message.
        """
        stream = _io.StringIO()
        handler = _builtin_logging.StreamHandler(stream)
        handler.setFormatter(_logging.MultiLineFormatter("prefix|%(message)s"))
        logger = _builtin_logging.getLogger(self.logger_name)
        logger.propagate = False
        self.add_handler(logger, handler)
        try:
            raise RuntimeError("Raised for the test.")
        except RuntimeError:
            logger.exception("line %s\nline %s\nline %s", 1, 2, 3)
        lines = stream.getvalue().splitlines()
        self.assertEqual(["prefix|line 1", "prefix|line 2", "prefix|line 3"], lines[0:3])
        self.assertEqual("prefix|Traceback (most recent call last):", lines[3])
        self.assertEqual("prefix|RuntimeError: Raised for the test.", lines[-1])
        self.assertEqual(1, stream.getvalue().count("Traceback"))
        self.assertTrue(all(line.startswith("prefix|") for line in lines))

        stream.truncate(0)
        stream.seek(0)
        logger.warning("line 1\nline 2", stack_info=True)
        lines = stream.getvalue().splitlines()
        self.assertEqual(
            ["prefix|line 1", "prefix|line 2", "prefix|Stack (most recent call last):"],
            lines[0:3]
        )
        self.assertEqual(1, stream.getvalue().count("Stack (most recent call last)"))

    def rank_prefix_regex(self, comm=None, rank_string="rank"):
        """
        Returns the regular expression for the prefix which
        :func:`mpi_array.logging.get_rank_logger` adds to the lines of a message.
        """
        if comm is None:
            comm = _mpi.COMM_WORLD
        rank = ("%%0%dd" % (len(str(_mpi.COMM_WORLD.size - 1)),)) % comm.rank
        return r"\d\d:\d\d:\d\d\|%s\|%s%s\|" % (_re.escape(comm.Get_name()), rank_string, rank)

    def test_rank_logger_uses_configured_handlers(self):
        """
        The messages of a :func:`mpi_array.logging.get_rank_logger` logger, with the
        time, communicator name and rank prefix on each line, go to the handlers of
        the ancestor loggers. New handlers of the same classes used to be created with
        no arguments, so a :obj:`logging.FileHandler` raised :obj:`TypeError` (and array
        creation failed), and a :obj:`logging.StreamHandler` wrote to standard error
        instead of its stream.
        """
        parent = _builtin_logging.getLogger(self.logger_name)
        parent.setLevel(_builtin_logging.INFO)
        stream = _io.StringIO()
        self.add_handler(parent, _builtin_logging.StreamHandler(stream))
        with _tempfile.TemporaryDirectory() as dir_name:
            file_name = _os.path.join(dir_name, "rank_logger.log")
            file_handler = self.add_handler(parent, _builtin_logging.FileHandler(file_name))
            for i in range(2):
                rank_logger = _logging.get_rank_logger(self.logger_name + ".child")
            rank_logger.info("line %s\nline %s", 1, 2)
            rank_logger.debug("not logged at level INFO")
            try:
                raise RuntimeError("Raised for the test.")
            except RuntimeError:
                rank_logger.exception("line 3")
            file_handler.flush()
            with open(file_name) as f:
                file_text = f.read()

        prefix = self.rank_prefix_regex()
        for text in (stream.getvalue(), file_text):
            lines = text.splitlines()
            self.assertRegex(lines[0], "^" + prefix + "line 1$")
            self.assertRegex(lines[1], "^" + prefix + "line 2$")
            self.assertRegex(lines[2], "^" + prefix + "line 3$")
            self.assertRegex(lines[3], "^" + prefix + "Traceback")
            self.assertRegex(lines[-1], "^" + prefix + "RuntimeError: Raised for the test.$")
            self.assertTrue(all(_re.match(prefix, line) for line in lines))
            self.assertEqual(1, len([line for line in lines if line.endswith("|line 1")]))
            self.assertEqual(1, text.count("Traceback"))

    def test_rank_logger_ranks(self):
        """
        Only the processes in the :samp:`ranks` argument of
        :func:`mpi_array.logging.get_rank_logger` (rank 0 for
        :func:`mpi_array.logging.get_root_logger`) log messages.
        """
        parent = _builtin_logging.getLogger(self.logger_name)
        stream = _io.StringIO()
        self.add_handler(parent, _builtin_logging.StreamHandler(stream))
        comm = _mpi.COMM_WORLD
        _logging.get_rank_logger(self.logger_name + ".child", ranks=[comm.size - 1]).warning("A")
        _logging.get_root_logger(self.logger_name + ".child").warning("B")
        lines = stream.getvalue().splitlines()
        expected = []
        if comm.rank == (comm.size - 1):
            expected.append("^" + self.rank_prefix_regex() + "A$")
        if comm.rank == 0:
            expected.append("^" + self.rank_prefix_regex(rank_string="root") + "B$")
        self.assertEqual(len(expected), len(lines))
        for line, regex in zip(lines, expected):
            self.assertRegex(line, regex)

    def test_initialise_loggers_does_not_duplicate_handlers(self):
        """
        :func:`mpi_array.logging.initialise_loggers` doesn't add a handler to a logger
        which (or an ancestor of which) already has a handler of that class, a message
        would then be output more than once.
        """
        parent = _builtin_logging.getLogger(self.logger_name)
        child = _builtin_logging.getLogger(self.logger_name + ".child")
        try:
            for i in range(2):
                _logging.initialise_loggers([self.logger_name], log_level=_builtin_logging.INFO)
            self.assertEqual(1, len(parent.handlers))
            self.assertTrue(isinstance(parent.handlers[0], _logging.SplitStreamHandler))
            _logging.initialise_loggers([child.name], log_level=_builtin_logging.DEBUG)
            self.assertEqual(0, len(child.handlers))
            self.assertEqual(_builtin_logging.DEBUG, child.level)
        finally:
            for logger in (parent, child):
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
