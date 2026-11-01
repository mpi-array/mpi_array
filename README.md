# `mpi_array`

<!-- start long description -->
<!-- start badges -->

[![CI](https://github.com/mpi-array/mpi_array/actions/workflows/ci.yml/badge.svg)](https://github.com/mpi-array/mpi_array/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mpi-array/badge/?version=latest)](http://mpi-array.readthedocs.io/en/latest/?badge=latest)
[![MIT License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/mpi-array/mpi_array/blob/dev/LICENSE.txt)

<!-- end badges -->

The [mpi_array](http://mpi-array.readthedocs.io/en/latest) python package provides
a [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html) API to a
[Partitioned Global Address Space](https://en.wikipedia.org/wiki/Partitioned_global_address_space)
array which utilizes
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)
(via [mpi4py](http://pythonhosted.org/mpi4py/)) for parallelism.


## Quick Start Example

The following `quickstart.py` script creates a zero-initialised array and
performs some element-wise operations:

```python
import mpi_array as mpia

# Creates a zero-initialised PGAS (distributed) array,
# the elements are apportioned to the MPI processes.
dary = mpia.zeros((100, 100, 100), dtype="uint16")

# Add one to all elements of the array (in-place ufunc).
dary += 1

# Ufuncs create new distributed arrays.
roots = mpia.sqrt(dary + 3)

# Collective check (all processes) that all elements equal 2.
assert (roots == 2).all()

# Arrays hold MPI windows, free them (collectively) when done,
# or create them in a "with" statement.
roots.free()
dary.free()
```

Indexing and slice assignment (e.g. `dary[10:20] = 8`) are not implemented yet
and raise `NotImplementedError`.

The `quickstart.py` script can be executed serially
(single process) as:

```sh
python quickstart.py
```

or in parallel (using 8 processes) as:

```sh
mpirun -n 8 python -m mpi4py quickstart.py
```

(`python -m mpi4py` makes an unhandled exception on one process abort all processes.)


## Related Work

Related distributed array (PGAS) implementations with python API:

- [Global Arrays](http://hpc.pnl.gov/globalarrays/)
  ([ga at github](https://github.com/GlobalArrays/ga)) with
  [ga4py](https://github.com/GlobalArrays/ga4py) python
  API and [GAiN](http://hpc.pnl.gov/globalarrays/papers/scipy11_gain.pdf)
  numpy API
- [DistArray](http://distarray.readthedocs.io/en/latest/)
- [pnumpy](https://github.com/pletzer/pnumpy)
- [Spartan](https://www.cs.nyu.edu/media/publications/power_russell.pdf)
  ([at github here](https://github.com/spartan-array/spartan))
- [caput.mpiarray](http://caput.readthedocs.io/en/latest/generated/caput.mpiarray.html)
- [dask.distributed](https://distributed.readthedocs.io/en/latest/)
- [bolt](http://bolt-project.org/)
- [distnumpy](http://dl6.globalstf.org/index.php/joc/article/download/704/723) and
  [distnumpy at github](https://github.com/distnumpy/distnumpy)

Related parallel array implementations (not PGAS) with python API:

- [Bohrium](http://bohrium.readthedocs.io/)
- [Intel Distribution for Python](https://software.intel.com/en-us/distribution-for-python)


## Installation

Using `pip` from latest github source:

```sh
pip install --user git+git://github.com/mpi-array/mpi_array.git#egg=mpi_array
```


## Requirements

Requires:

- python version >= 3.10,
- [array_split](http://array-split.readthedocs.io/en/latest/) version >= 0.4.0,
- [psutil](https://pypi.python.org/pypi/psutil) >= 4.3.0
- [numpy](https://numpy.org/doc/stable/) version >= 2.0,
- an MPI implementation which supports at least MPI-3 (such as
  [OpenMPI](http://openmpi.org/), [MPICH](http://mpich.org/) or
  [Intel MPI](https://software.intel.com/en-us/intel-mpi-library))
- [mpi4py](https://mpi4py.readthedocs.io/) version >= 3.1.


## Testing

Run tests ([unittest](https://docs.python.org/3/library/unittest.html)
test-cases and [doctest](https://docs.python.org/3/library/doctest.html)
docstring test-cases) using:

```sh
python -m mpi_array.tests
```

Run tests with parallelism:

```sh
mpirun -n 8 python -m mpi4py -m mpi_array.tests
```

Launching with `python -m mpi4py` makes an unhandled exception on one rank abort all ranks.
Each test runs with a hang-detection timeout (`MPI_ARRAY_TEST_TIMEOUT` seconds, default 300):
a deadlocked test makes every rank print its stack trace and exit.
From the source tree, run the tests for 1, 2, 4 and 8 processes and print a summary:

```sh
python utils/run_mpi_tests.py
```

This also checks that programs which end while arrays are allocated (normally, or with an
unhandled exception) exit without hanging, see `mpi_array/exit_scenarios.py`.

The tests run in GitHub Actions (with MPICH and Open MPI), at:

> https://github.com/mpi-array/mpi_array/actions/workflows/ci.yml


## Documentation

Latest sphinx generated documentation at [readthedocs.org](https://readthedocs.org):

> http://mpi-array.readthedocs.io/en/latest

and at github *gh-pages*:

> https://mpi-array.github.io/mpi_array/

Sphinx documentation can be built from the source:

```sh
python setup.py build_sphinx
```

with the HTML generated in `docs/_build/html`.


## Latest source code

Source at github:

> https://github.com/mpi-array/mpi_array

clone with:

```sh
git clone https://github.com/mpi-array/mpi_array.git
```


## License information

See the file [LICENSE.txt](https://github.com/mpi-array/mpi_array/blob/dev/LICENSE.txt)
for terms & conditions, for usage and a DISCLAIMER OF ALL WARRANTIES.

<!-- end long description -->
