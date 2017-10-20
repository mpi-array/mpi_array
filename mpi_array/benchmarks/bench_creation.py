"""
Benchmarks for array creation.
"""


class NumpyCreateBench(object):
    """
    Comparison benchmarks for :func:`numpy.empty` and :func:`numpy.zeros`.
    """

    repeat = 16
    params = [[(1000000,), (10000000,), (100000000,)], ]
    param_names = ["shape"]

    def setup(self, shape):
        """
        Import :mod:`numpy` module as :samp:`self.module`.
        """
        import numpy
        self.module = numpy

    def time_empty(self, shape):
        """
        Time :func:`numpy.empty`.
        """
        self.module.empty(shape, dtype="int32")

    def time_zeros(self, shape):
        """
        Time :func:`numpy.zeros`.
        """
        self.module.zeros(shape, dtype="int32")


class MpiArrayCreateBench(NumpyCreateBench):
    """
    Benchmarks for :func:`mpi_array.empty` and :func:`mpi_array.zeros`.
    """

    repeat = NumpyCreateBench.repeat
    params = NumpyCreateBench.params

    def setup(self, shape):
        """
        Import :mod:`mpi_array` module as :samp:`self.module`.
        """
        import mpi_array
        self.module = mpi_array

    def time_empty(self, shape):
        """
        Time :func:`mpi_array.empty`.
        """
        with self.module.empty(shape, dtype="int32"):
            pass

    def time_zeros(self, shape):
        """
        Time :func:`mpi_array.zeros`.
        """
        with self.module.zeros(shape, dtype="int32"):
            pass
