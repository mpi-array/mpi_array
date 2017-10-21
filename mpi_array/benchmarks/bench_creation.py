"""
Benchmarks for array creation.
"""


class NumpyCreateBench(object):
    """
    Comparison benchmarks for :func:`numpy.empty` and :func:`numpy.zeros`.
    """

    repeat = 16
    params = [[(100, 100, 100,), ((1000, 100, 100,)), ((100, 1000, 1000,))], ]
    param_names = ["shape"]

    goal_time = 0.5
    warmup_time = 0.25

    def setup(self, shape):
        """
        Import :mod:`numpy` module and assign to :samp:`self.module`.
        """
        try:
            import numpy
            self.module = numpy
        except Exception:
            raise NotImplementedError("Error during numpy import.")

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

    def setup(self, shape):
        """
        Import :mod:`mpi_array` module and assign to :samp:`self.module`.
        """
        try:
            import mpi_array
            self.module = mpi_array
        except Exception:
            raise NotImplementedError("Error during mpi_array import.")


class MangoCreateBench(NumpyCreateBench):
    """
    Benchmarks for :func:`mango.empty` and :func:`mango.zeros`
    (`mango tomography software <https://physics.anu.edu.au/appmaths/capabilities/mango.php>`_).
    """

    def setup(self, shape):
        """
        Import :mod:`mango` module and assign to :samp:`self.module`.
        """
        try:
            import mango
            self.module = mango
        except Exception:
            raise NotImplementedError("Error during mango import.")
