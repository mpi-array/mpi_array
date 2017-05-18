"""
===================================
The :mod:`mpi_array.tests` Module
===================================

Module for running all :mod:`mpi_array` unit-tests, including :mod:`unittest` test-cases
and :mod:`doctest` tests for module doc-strings and sphinx (RST) documentation.
Execute as::

   python -m mpi_array.tests

.. currentmodule:: mpi_array.tests

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources
import unittest as _unittest
import doctest as _doctest  # noqa: E402,F401

import os.path

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()


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
                "mpi_array.array_test",
                "mpi_array.decomposition_test",
            ]
        )
    suite.addTests(DocTestTestSuite())
    return suite


__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest.main()
