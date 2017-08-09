#! /usr/bin/env python
"""
Prints mpi4py version, MPI vendor name, MPI vendor version
and MPI API version.
"""

try:
    import mpi4py
    import mpi4py.MPI
except ImportError as e:
    mpi4py = e
try:
    import array_split
except ImportError as e:
    array_split = e
try:
    import numpy
except ImportError as e:
    numpy = e
try:
    import mpi_array
except ImportError as e:
    mpi_array = e

if __name__ == "__main__":
    if not isinstance(numpy, Exception):
        print("\nnumpy:")
        print("numpy.__version__       = %s" % (numpy.__version__,))
        print("numpy.__file__          = %s" % (numpy.__file__,))
    else:
        print("\nnumpy: error:")
        print(str(numpy))

    if not isinstance(array_split, Exception):
        print("\narray_split:")
        print("array_split.__version__ = %s" % (array_split.__version__,))
        print("array_split.__file__    = %s" % (array_split.__file__,))
    else:
        print("\narray_split: error:")
        print(str(array_split))
    
    if not isinstance(mpi4py, Exception):
        print("\nmpi4py:")
        print("mpi4py.__version__      = %s" % (mpi4py.__version__,))
        print("mpi4py.__file__         = %s" % (mpi4py.__file__,))
        print("mpi4py.MPI.get_vendor() = %s" % (mpi4py.MPI.get_vendor(),))
        print("mpi4py.MPI.VERSION      = %s" % (mpi4py.MPI.VERSION,))
        print("mpi4py.MPI.Get_version()= %s" % (mpi4py.MPI.Get_version(),))
    else:
        print("\nmpi4py: error:")
        print(str(mpi4py))

    if not isinstance(mpi_array, Exception):
        print("\nmpi_array:")
        print("mpi_array.__version__   = %s" % (mpi_array.__version__,))
        print("mpi_array.__file__      = %s" % (mpi_array.__file__,))
    else:
        print("\nmpi_array: error:")
        print(str(mpi_array))
