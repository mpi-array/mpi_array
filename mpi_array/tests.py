"""
=================================
The :mod:`mpi_array.tests` Module
=================================

Module for running all :mod:`mpi_array` unit-tests, including :mod:`unittest` test-cases
and :mod:`doctest` tests of the docstring examples (see :obj:`DocTestTestSuite`).
Execute as::

   python -m mpi_array.tests

"""
from __future__ import absolute_import

import importlib as _importlib
import re as _re
import unittest as _unittest
import doctest as _doctest  # noqa: E402,F401
import os.path as _os_path

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
    to remove the :samp:`", dtype=int64)"` from array representations
    (the default integer type differs between platforms).
    """

    def check_output(self, want, got, optionflags):
        """
        Replaces :samp:`", dtype=int64)"` with :samp:`")"`.

        See :meth:`doctest.OutputChecker.check_output`.

        """
        got = _re.sub(", dtype=int64\\)", ")", got)

        return _doctest_OuputChecker.check_output(self, want, got, optionflags)


_doctest.OutputChecker = MultiPlatformAnd23Checker


#: The modules whose docstring examples are tested.
DOCTEST_MODULE_NAMES = \
    (
        "mpi_array.comms",
        "mpi_array.globale_ufunc",
        "mpi_array.indexing",
        "mpi_array.locale",
        "mpi_array.update",
    )


class DocTestTestSuite(_unittest.TestSuite):

    """
    The :mod:`doctest` tests of the README (when it is next to the package) and of
    the docstrings of the :attr:`DOCTEST_MODULE_NAMES` modules.
    """

    def __init__(self):
        """
        Initialise with the tests.
        """
        readme_file_name = \
            _os_path.realpath(
                _os_path.join(_os_path.dirname(__file__), "..", "README.md")
            )
        suite = _unittest.TestSuite()
        optionflags = _doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS

        if _os_path.exists(readme_file_name):
            suite.addTests(
                _doctest.DocFileSuite(
                    readme_file_name,
                    module_relative=False,
                    optionflags=optionflags
                )
            )
        for module_name in DOCTEST_MODULE_NAMES:
            suite.addTests(
                _doctest.DocTestSuite(
                    _importlib.import_module(module_name),
                    optionflags=optionflags
                )
            )

        _unittest.TestSuite.__init__(self, suite)


def load_tests(loader, tests, pattern):
    suite = \
        loader.loadTestsFromNames(
            [
                "mpi_array.mpi_test",
                "mpi_array.logging_test",
                "mpi_array.types_test",
                "mpi_array.comms_test",
                "mpi_array.indexing_test",
                "mpi_array.distribution_test",
                "mpi_array.update_test",
                "mpi_array.locale_test",
                "mpi_array.globale_test",
                "mpi_array.globale_creation_test",
                "mpi_array.globale_ufunc_test",
                "mpi_array.benchmarks.utils.wlm_test",
            ]
        )
    suite.addTests(DocTestTestSuite())

    return suite


__all__ = [s for s in dir() if not s.startswith('_')]

if __name__ == "__main__":
    _unittest_main(__name__, log_level=_logging.WARNING, verbosity=0)
