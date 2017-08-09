#! /usr/bin/env python
"""
Prints mpi4py version, MPI vendor name, MPI vendor version
and MPI API version.
"""
import mpi4py
import mpi4py.MPI
import mpi_array
import array_split
import numpy

if __name__ == "__main__":
    print("numpy:")
    print("numpy.__version__       = %s" % (numpy.__version__,))
    print("numpy.__file__          = %s" % (numpy.__file__,))
    print("array_split:")
    print("array_split.__version__ = %s" % (array_split.__version__,))
    print("array_split.__file__    = %s" % (array_split.__file__,))
    print("mpi4py:")
    print("mpi4py.__version__      = %s" % (mpi4py.__version__,))
    print("mpi4py.__file__         = %s" % (mpi4py.__file__,))
    print("mpi4py.MPI.get_vendor() = %s" % (mpi4py.MPI.get_vendor(),))
    print("mpi4py.MPI.VERSION      = %s" % (mpi4py.MPI.VERSION,))
    print("mpi4py.MPI.Get_version()= %s" % (mpi4py.MPI.Get_version(),))
    print("mpi_array:")
    print("mpi_array.__version__   = %s" % (mpi_array.__version__,))
    print("mpi_array.__file__      = %s" % (mpi_array.__file__,))
