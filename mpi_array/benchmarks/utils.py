"""
Benchmark utilities.

Parts of this source borrows from
the `airspeed velocity (asv) <http://asv.readthedocs.io/en/latest>`_
file `benchmark.py <https://github.com/spacetelescope/asv/blob/master/asv/benchmark.py>`_.

See the `LICENSE <https://github.com/spacetelescope/asv/blob/master/LICENSE.rst>`_.

"""
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys as _sys
from ..init import get_process_time_timer  # noqa: F401

__license__ = "https://github.com/spacetelescope/asv/blob/master/LICENSE.rst"

if _sys.version_info[0] <= 2:
    # Work around lack of pickling of instance methods in python-2
    from copy_reg import pickle as _pickle
    from types import MethodType as _MethodType

    def _pickle_method(method):
        func_name = method.im_func.__name__
        obj = method.im_self
        cls = method.im_class
        if func_name.startswith('__') and not func_name.endswith('__'):
            cls_name = cls.__name__.lstrip('_')
            if cls_name:
                func_name = '_' + cls_name + func_name
        return _unpickle_method, (func_name, obj, cls)

    def _unpickle_method(func_name, obj, cls):
        for cls in cls.mro():
            try:
                func = cls.__dict__[func_name]
            except KeyError:
                pass
            else:
                break
        return func.__get__(obj, cls)

    _pickle(_MethodType, _pickle_method, _unpickle_method)


def try_import_for_setup(module_name):
    """
    Returns the imported module named :samp:`{module_name}`.
    Attempts to import the module named :samp:`{module_name}` and
    translates any raised :obj:`ImportError` into a :onj:`NotImplementedError`.
    This is useful in :samp:`setup`, so that a failed import results in
    the benchmark getting *skipped*.

    :rtype:`module`
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

    def find_module(self, fullname, path=None):
        if fullname == self._name:
            return self
        return None

    def load_module(self, fullname):
        import imp
        file, pathname, desc = imp.find_module(fullname, [self._root])
        return imp.load_module(fullname, file, pathname, desc)


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
