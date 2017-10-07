"""
=================================
The :mod:`mpi_array.tests` Module
=================================

Module for running all :mod:`mpi_array` unit-tests, including :mod:`unittest` test-cases
and :mod:`doctest` tests for module doc-strings and sphinx (RST) documentation.
Execute as::

   python -m mpi_array.tests

"""
from __future__ import absolute_import

import sys as _sys
import re as _re
import unittest as _unittest
import doctest as _doctest  # noqa: E402,F401
import os.path as _os_path
import mpi_array as _mpi_array

from .license import license as _license, copyright as _copyright, version as _version
from .unittest import main as _unittest_main
from . import logging as _logging

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

_doctest_OuputChecker = _doctest.OutputChecker


class MultiPlatformAnd23Checker(_doctest_OuputChecker):

    """
    Overrides the :meth:`doctest.OutputChecker.check_output` method
    to remove the :samp:`'L'` from integer literals
    """

    def check_output(self, want, got, optionflags):
        """
        For python-2 replaces "124L" with "124". For python 2 and 3,
        replaces :samp:`", dtype=int64)"` with :samp:`")"`.

        See :meth:`doctest.OutputChecker.check_output`.

        """
        if _sys.version_info[0] <= 2:
            got = _re.sub("([0-9]+)L", "\\1", got)

        got = _re.sub(", dtype=int64\\)", ")", got)

        return _doctest_OuputChecker.check_output(self, want, got, optionflags)


_doctest.OutputChecker = MultiPlatformAnd23Checker


class DocTestTestSuite(_unittest.TestSuite):

    def __init__(self):
        readme_file_name = \
            _os_path.realpath(
                _os_path.join(_os_path.dirname(__file__), "..", "README.rst")
            )
        suite = _unittest.TestSuite()
        if _os_path.exists(readme_file_name):
            suite.addTests(
                _doctest.DocFileSuite(
                    readme_file_name,
                    module_relative=False,
                    optionflags=_doctest.NORMALIZE_WHITESPACE
                )
            )
        suite.addTests(
            _doctest.DocTestSuite(
                _mpi_array.globale_ufunc,
                optionflags=_doctest.NORMALIZE_WHITESPACE
            )
        )

        _unittest.TestSuite.__init__(self, suite)


def load_tests(loader, tests, pattern):
    suite = \
        loader.loadTestsFromNames(
            [
                "mpi_array.mpi_test",
                "mpi_array.types_test",
                "mpi_array.comms_test",
                "mpi_array.indexing_test",
                "mpi_array.distribution_test",
                "mpi_array.update_test",
                "mpi_array.locale_test",
                "mpi_array.globale_test",
                "mpi_array.globale_creation_test",
                "mpi_array.globale_ufunc_test",
            ]
        )
    suite.addTests(DocTestTestSuite())

    import mpi_array.indexing as _indexing
    suite.addTests(_doctest.DocTestSuite(_indexing))

    return suite


__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest_main(__name__, log_level=_logging.WARNING, verbosity=2)
