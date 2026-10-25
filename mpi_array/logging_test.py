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
import logging as _builtin_logging

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


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
