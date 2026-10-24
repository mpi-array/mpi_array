"""
Benchmark utilities.

Parts of this source borrows from
the `airspeed velocity (asv) <http://asv.readthedocs.io/en/latest>`_
file `benchmark.py <https://github.com/spacetelescope/asv/blob/master/asv/benchmark.py>`_.

See the `LICENSE <https://github.com/spacetelescope/asv/blob/master/LICENSE.rst>`_.

"""
from __future__ import absolute_import
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ...init import get_process_time_timer  # noqa: F401

__license__ = "https://github.com/spacetelescope/asv/blob/master/LICENSE.rst"


def try_import_for_setup(module_name):
    """
    Returns the imported module named :samp:`{module_name}`.
    Attempts to import the module named :samp:`{module_name}` and
    translates any raised :obj:`ImportError` into a :obj:`NotImplementedError`.
    This is useful in benchmark :samp:`setup`, so that a failed import results in
    the benchmark getting *skipped*.

    :type module_name: :obj:`str`
    :param module_name: Attempt to import this module.
    :rtype: :obj:`module`
    :returns: The imported module named :samp:`{module_name}`.
    :raises NotImplementedError: if there is an :obj:`ImportError`.
    """
    from importlib import import_module
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise NotImplementedError("Error during '%s' module import: %s." % (module_name, str(e)))

    return module


class SpecificImporter(object):
    """
    Module importer that only allows loading a given module from the
    given path.

    Using this enables importing the asv benchmark suite without
    adding its parent directory to sys.path. The parent directory can
    in principle contain anything, including some version of the
    project module (common situation if asv.conf.json is on project
    repository top level).
    """

    def __init__(self, name, root):
        self._name = name
        self._root = root

    def find_spec(self, fullname, path=None, target=None):
        """
        Returns the module spec for module :samp:`{fullname}` when it is the
        module named in the constructor (searched for only in the :samp:`root` directory),
        otherwise :samp:`None`.
        """
        if fullname == self._name:
            from importlib.machinery import PathFinder
            return PathFinder.find_spec(fullname, [self._root])
        return None


def update_sys_path(root):
    """
    Inserts a :obj:`SpecificImporter` instance into the :attr:`sys.meta_path`.

    :type root: :obj:`str`
    :param root: The :obj`SpecificImported` path inserted at the head of :attr:`sys.meta_path`.
    """
    import os
    import sys
    sys.meta_path.insert(
        0,
        SpecificImporter(
            os.path.basename(root),
            os.path.dirname(root)
        )
    )


__all__ = [s for s in dir() if not s.startswith('_')]
