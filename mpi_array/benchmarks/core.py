"""
Benchmarks for array creation.
"""

from .utils import try_import_for_setup as _try_import_for_setup


class Bench(object):
    """
    Base class for benchmarks.
    """

    #: Inter-locale cartesian communicator dims
    cart_comm_dims = None

    def __init__(self):
        """
        Initialise, set :attr:`module` to :samp:`None`.
        """
        self._module = None

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
        if self.cart_comm_dims is None:
            from ..comms import CartLocaleComms as _CartLocaleComms
            import numpy as _np
            import mpi4py.MPI as _mpi
            comms = _CartLocaleComms(ndims=len(locale_shape), peer_comm=_mpi.COMM_WORLD)
            self.cart_comm_dims = _np.asarray(comms.dims)
        return tuple(self.cart_comm_dims * locale_shape)

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
