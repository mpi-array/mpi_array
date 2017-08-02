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

import unittest as _unittest
import doctest as _doctest  # noqa: E402,F401
import os.path

from .license import license as _license, copyright as _copyright, version as _version
from .unittest import main as _unittest_main
from . import logging as _logging

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class DocTestTestSuite(_unittest.TestSuite):

    def __init__(self):
        readme_file_name = \
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "README.rst")
            )
        suite = _unittest.TestSuite()
        if os.path.exists(readme_file_name):
            suite.addTests(
                _doctest.DocFileSuite(
                    readme_file_name,
                    module_relative=False,
                    optionflags=_doctest.NORMALIZE_WHITESPACE
                )
            )

        _unittest.TestSuite.__init__(self, suite)


def load_tests(loader, tests, pattern):
    suite = \
        loader.loadTestsFromNames(
            [
                "mpi_array.mpi_test",
                "mpi_array.indexing_test",
                "mpi_array.distribution_test",
                "mpi_array.update_test",
                "mpi_array.locale_test",
                "mpi_array.globale_test",
            ]
        )
    suite.addTests(DocTestTestSuite())

    import mpi_array.indexing as _indexing
    suite.addTests(_doctest.DocTestSuite(_indexing))

    return suite


__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest_main(__name__, log_level=_logging.WARNING, verbosity=0)
