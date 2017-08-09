
===========
`mpi_array`
===========

.. Start of sphinx doc include.
.. start long description.
.. start badges.

.. image:: https://travis-ci.org/mpi-array/mpi_array.svg?branch=dev
   :target: https://travis-ci.org/mpi-array/mpi_array
   :alt: Build Status
.. image:: https://coveralls.io/repos/github/mpi-array/mpi_array/badge.svg
   :target: https://coveralls.io/github/mpi-array/mpi_array
   :alt: Coveralls Status
.. image:: https://readthedocs.org/projects/mpi-array/badge/?version=latest
   :target: http://mpi-array.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/mpi-array/mpi_array/blob/dev/LICENSE.txt
   :alt: MIT License

.. end badges.

The `mpi_array <http://mpi-array.readthedocs.io/en/latest>`_ python package provides
a `numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_ API to a
`Partitioned Global Address Space <https://en.wikipedia.org/wiki/Partitioned_global_address_space>`_
array which utilizes
`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_
(via `mpi4py <http://pythonhosted.org/mpi4py/>`_) for parallelism.


Quick Start Example
===================

The following ``quickstart.py`` script creates a zero-initialised array and
performs some element assignments::

   import mpi_array as mpia
   
   # creates zero-initialized PGAS (distributed) array
   dary = mpia.zeros((1000, 1000, 1000), dtype="uint16") 
   
   # Add one to all elements of array
   dary += 1
   
   # Assign to slice
   dary[250:750, :, 250:750] = 8

   # ufuncs
   dary[...] = mpia.power(dary, 1.0/3.0)


The ``quickstart.py`` script can be executed serially
(single process) as::

   python quickstart.py

or in parallel (using 8 processes) as::

   mpirun -n 8 python quickstart.py


Related Work
============

Related distributed array implementations with python API:

- `Global Arrays <http://hpc.pnl.gov/globalarrays/>`_
  (`ga at github <https://github.com/GlobalArrays/ga>`_) with
  `ga4py <https://github.com/GlobalArrays/ga4py>`_ python
  API and `GAiN <http://hpc.pnl.gov/globalarrays/papers/scipy11_gain.pdf>`_
  numpy API
- `DistArray <http://distarray.readthedocs.io/en/latest/>`_
- `pnumpy <https://github.com/pletzer/pnumpy>`_
- `Spartan <https://www.cs.nyu.edu/media/publications/power_russell.pdf>`_
  (`at github here <https://github.com/spartan-array/spartan>`_)
- `caput.mpiarray <http://caput.readthedocs.io/en/latest/generated/caput.mpiarray.html>`_
- `dask.distributed <https://distributed.readthedocs.io/en/latest/>`_
- `bolt <http://bolt-project.org/>`_

Installation
============

Using ``pip`` from latest github source:

   ``pip install --user git+git://github.com/mpi-array/mpi_array.git#egg=mpi_array``


Requirements
============

Requires:

   - python-2 version `>= 2.7` or python-3 version `>= 3.3`,
   - `array_split <http://array-split.readthedocs.io/en/latest/>`_ version `>= 0.4.0`,
   - `numpy <http://docs.scipy.org/doc/numpy/>`_ version `>= 1.13` (for ``__array_ufunc__`` API),
   - an MPI implementation which supports at least MPI-3 (such as 
     `OpenMPI <http://openmpi.org/>`_ or `MPICH <http://mpich.org/>`_)
   - `mpi4py <http://pythonhosted.org/mpi4py/>`_ version `>= 2.0`.


Testing
=======

Run tests (unit-tests and doctest module docstring tests) using::

   python -m mpi_array.tests

or, from the source tree, run::

   python setup.py test

Run tests with parallelism::

   mpirun -n 8 python -m mpi_array.tests

Travis CI at:

    https://travis-ci.org/mpi-array/mpi_array/


Documentation
=============

Latest sphinx generated documentation at `readthedocs.org <readthedocs.org>`_:

    http://mpi-array.readthedocs.io/en/latest

and at github *gh-pages*:

    https://mpi-array.github.io/mpi_array/

Sphinx documentation can be built from the source::

   python setup.py build_sphinx
     
with the HTML generated in ``docs/_build/html``.


Latest source code
==================

Source at github:

    https://github.com/mpi-array/mpi_array

clone with::

    git clone https://github.com/mpi-array/mpi_array.git


License information
===================

See the file `LICENSE.txt <https://github.com/mpi-array/mpi_array/blob/dev/LICENSE.txt>`_
for terms & conditions, for usage and a DISCLAIMER OF ALL WARRANTIES.

.. end long description.
