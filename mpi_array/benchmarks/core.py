"""
Core utilities for benchmark implementations.
"""
from __future__ import absolute_import
from ..license import license as _license, copyright as _copyright, version as _version

from .utils.misc import try_import_for_setup as _try_import_for_setup
from .. import logging as _logging

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


class Bench(object):
    """
    Base class for benchmarks.
    """

    #: Inter-locale cartesian communicator dims
    cart_comm_dims_dict = dict()

    def __init__(self):
        """
        Initialise, set :attr:`module` to :samp:`None`.
        """
        self._module = None
        self._root_logger = None
        self._rank_logger = None

    @property
    def root_logger(self):
        """
        A :obj:`logging.Logger` for root MPI process messages.
        """
        if self._root_logger is None:
            self._root_logger = \
                _logging.get_root_logger(str(__name__ + "." + self.__class__.__name__))
        return self._root_logger

    @property
    def rank_logger(self):
        """
        A :obj:`logging.Logger` for all rank MPI process messages.
        """
        if self._rank_logger is None:
            self._rank_logger = \
                _logging.get_rank_logger(str(__name__ + "." + self.__class__.__name__))
        return self._rank_logger

    @property
    def module(self):
        """
        The :obj:`module` used to create array instances.
        """
        return self._module

    @module.setter
    def module(self, module):
        self._module = module

    def try_import_for_setup(self, module_name):
        """
        Attempt to import module named :samp:`{module_name}`, return the module
        if it exists and is importable.

        :type module_name: :obj:`str`
        :param module_name: Attempt to import this module.
        :rtype: :obj:`module`
        :return: Module named :samp:`{module_name}`.
        :raises NotImplementedError: if there is an :obj:`ImportError`.

        .. seealso:: :func:`mpi_array.benchmarks.utils.try_import_for_setup`.
        """
        return _try_import_for_setup(module_name)

    def setup(self, shape):
        """
        Should be over-ridden in sub-classes.
        """
        pass

    def get_globale_shape(self, locale_shape):
        """
        Returns a *globale* array shape for the given shape of the *locale* array.

        :type locale_shape: sequence of :samp:`int`
        :param locale_shape: The shape of the array to be allocated on each *locale*.
        """
        locale_shape = tuple(locale_shape)
        if locale_shape not in self.cart_comm_dims_dict.keys():
            from ..comms import CartLocaleComms as _CartLocaleComms
            import numpy as _np
            import mpi4py.MPI as _mpi
            comms = _CartLocaleComms(ndims=len(locale_shape), peer_comm=_mpi.COMM_WORLD)
            self.cart_comm_dims_dict[locale_shape] = _np.asarray(comms.dims)
        return tuple(self.cart_comm_dims_dict[locale_shape] * locale_shape)

    def free_mpi_array_obj(self, a):
        """
        Free MPI communicators and MPI windows for the given object.

        :type a: free-able
        :param a: A :obj:`mpi_array.globale.gndarray` instance
           or a :obj:`mpi_array.comms.LocaleComms` instance.
        """
        if hasattr(a, "locale_comms"):
            a.locale_comms.free()
        if hasattr(a, "free"):
            a.free()


__all__ = [s for s in dir() if not s.startswith('_')]
