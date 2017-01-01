#! /usr/bin/env python
"""
Prints mpi4py version, MPI vendor name, MPI vendor version
and MPI API version.
"""
import mpi4py
import mpi4py.MPI

if __name__ == "__main__":
    print("mpi4py.__version__      = %s" % (mpi4py.__version__,))
    print("mpi4py.MPI.get_vendor() = %s" % (mpi4py.MPI.get_vendor(),))
    print("mpi4py.MPI.VERSION      = %s" % (mpi4py.MPI.VERSION,))
