"""
Benchmarks for copying elements between arrays of different distributions
(:func:`mpi_array.copyto`) and for halo (ghost element) updates
(:meth:`mpi_array.globale.gndarray.update`).

The :samp:`*PlanBench` benchmarks time the calculation of the updates on one process,
for a number of locales which is a parameter (the distributions are created without
MPI communicators). The other benchmarks time the operations on the locales of the
processes which run them: with the :samp:`"process"` locale type every process is
a locale, so the number of locales grows with the number of processes.
"""
from __future__ import absolute_import
from ..license import license as _license, copyright as _copyright, version as _version

from .utils.misc import try_import_for_setup as _try_import_for_setup
from .core import Bench as _Bench

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def _get_num_locales(locale_type):
    """
    Returns the number of locales of the :samp:`{locale_type}` locale type
    over :attr:`mpi4py.MPI.COMM_WORLD`.
    """
    comms = _try_import_for_setup("mpi_array.comms")
    return comms.create_locale_comms(locale_type=locale_type)[0].num_locales


def _check_single_process():
    """
    Raises :obj:`NotImplementedError` (the benchmark is skipped) when there
    is more than one process: the planning benchmarks don't communicate, every
    process would time the same calculation.
    """
    mpi = _try_import_for_setup("mpi4py.MPI")
    if mpi.COMM_WORLD.size > 1:
        raise NotImplementedError("only runs for single process")


def _create_block_partition(shape, dims, halo=0):
    """
    Returns a :obj:`mpi_array.distribution.BlockPartition` of :samp:`{shape}`
    with :samp:`{dims}` blocks along each axis, the cartesian ranks are in C order.
    """
    distribution = _try_import_for_setup("mpi_array.distribution")
    np = _try_import_for_setup("numpy")
    num_locales = int(np.prod(dims))
    return \
        distribution.BlockPartition(
            globale_extent=distribution.GlobaleExtent(start=(0,) * len(shape), stop=shape),
            dims=dims,
            cart_coord_to_cart_rank=np.arange(num_locales).reshape(dims),
            halo=halo,
            inter_locale_rank_to_peer_rank=np.arange(num_locales)
        )


class CopytoBench(_Bench):
    """
    Benchmarks for :func:`mpi_array.copyto`. Each locale has :samp:`locale_shape` elements
    of the destination array, the globale array grows with the number of locales along
    axis 0. The :samp:`pattern` parameter selects the distributions:

    :samp:`"transpose"`
       The destination is partitioned into slabs along axis 0, the source into slabs
       along axis 1: every locale gets elements from every locale.
    :samp:`"same"`
       Both arrays have the (default) block distribution: the elements are copied
       within each locale.
    :samp:`"gather"`
       The destination is cloned (every locale gets the whole array), the source
       has the block distribution.
    """

    #: Number of repetitions to run each benchmark
    repeat = 5

    #: The locale type, distribution pattern and locale array shape parameters.
    params = \
        [
            ["process", "node"],
            ["transpose", "same", "gather"],
            [(8, 64, 8), (64, 64, 64)]
        ]

    #: The names of the parameters.
    param_names = ["locale_type", "pattern", "locale_shape"]

    # Goal time (seconds) for a single repeat.
    goal_time = 0.25

    # Execute benchmark for this time (seconds) as a *warm-up* prior to real timing.
    warmup_time = 0.25

    def __init__(self):
        """
        Initialise.
        """
        _Bench.__init__(self)
        self.dst = None
        self.src = None

    def setup(self, locale_type, pattern, locale_shape):
        """
        Creates the destination (:attr:`dst`) and source (:attr:`src`) arrays.
        """
        self.module = self.try_import_for_setup("mpi_array")
        num_locales = _get_num_locales(locale_type)
        shape = (num_locales * locale_shape[0],) + tuple(locale_shape[1:])
        if pattern == "transpose":
            dst_kwargs = {"dims": (0, 1, 1)}
            src_kwargs = {"dims": (1, 0, 1)}
        elif pattern == "same":
            dst_kwargs = {}
            src_kwargs = {}
        elif pattern == "gather":
            dst_kwargs = {"distrib_type": "cloned"}
            src_kwargs = {}
        else:
            raise ValueError("Invalid pattern=%r." % (pattern,))
        self.dst = self.module.empty(shape, dtype="float64", locale_type=locale_type, **dst_kwargs)
        self.src = self.module.empty(shape, dtype="float64", locale_type=locale_type, **src_kwargs)
        self.src.fill(1)

    def teardown(self, locale_type, pattern, locale_shape):
        """
        Frees the arrays created by :meth:`setup`.
        """
        for a in (self.dst, self.src):
            if a is not None:
                self.free_mpi_array_obj(a)
        self.dst = None
        self.src = None

    def time_copyto(self, locale_type, pattern, locale_shape):
        """
        Time :samp:`mpi_array.copyto(dst, src)`.
        """
        self.module.copyto(self.dst, self.src)


class HaloUpdateBench(_Bench):
    """
    Benchmarks for :meth:`mpi_array.globale.gndarray.update` of a 3-D block distributed
    array with a halo of :samp:`halo` elements. Each locale has :samp:`locale_shape`
    elements (without the halo).
    """

    #: Number of repetitions to run each benchmark
    repeat = 5

    #: The locale type, locale array shape and halo parameters.
    params = [["process", "node"], [(8, 8, 8), (64, 64, 64)], [1, 4]]

    #: The names of the parameters.
    param_names = ["locale_type", "locale_shape", "halo"]

    # Goal time (seconds) for a single repeat.
    goal_time = 0.25

    # Execute benchmark for this time (seconds) as a *warm-up* prior to real timing.
    warmup_time = 0.25

    def __init__(self):
        """
        Initialise.
        """
        _Bench.__init__(self)
        self.ary = None

    def setup(self, locale_type, locale_shape, halo):
        """
        Creates the array (:attr:`ary`).
        """
        self.module = self.try_import_for_setup("mpi_array")
        comms = self.try_import_for_setup("mpi_array.comms")
        dims = comms.get_cannonical_dims(_get_num_locales(locale_type), ndims=len(locale_shape))
        shape = tuple(int(d) * s for d, s in zip(dims, locale_shape))
        self.ary = \
            self.module.empty(shape, dtype="float64", locale_type=locale_type, dims=dims, halo=halo)
        self.ary.fill_h(1)

    def teardown(self, locale_type, locale_shape, halo):
        """
        Frees the array created by :meth:`setup`.
        """
        if self.ary is not None:
            self.free_mpi_array_obj(self.ary)
        self.ary = None

    def time_update(self, locale_type, locale_shape, halo):
        """
        Time :samp:`ary.update()`.
        """
        self.ary.update()


class RedistributePlanBench(object):
    """
    Benchmarks for the calculation of the updates which copy the elements of an
    array to an array of a different distribution
    (:obj:`mpi_array.update.UpdatesForRedistribute`), on one process
    for :samp:`num_locales` locales: the updates of the destination locale of the process
    (a locale in the middle). The :samp:`pattern` parameter selects the distributions:

    :samp:`"shifted"`
       1-D distributions, each destination locale gets elements from two source locales.
    :samp:`"transpose"`
       3-D block partitions, the destination is partitioned into slabs along axis 0,
       the source into slabs along axis 1: each destination locale gets elements from
       every source locale.
    """

    #: Number of repetitions to run each benchmark
    repeat = 3

    #: The pattern and number of locales parameters.
    params = [["shifted", "transpose"], [16, 64, 256]]

    #: The names of the parameters.
    param_names = ["pattern", "num_locales"]

    # Goal time (seconds) for a single repeat.
    goal_time = 0.25

    # Execute benchmark for this time (seconds) as a *warm-up* prior to real timing.
    warmup_time = 0.0

    def setup(self, pattern, num_locales):
        """
        Creates the destination (:attr:`dst`) and source (:attr:`src`) distributions.
        """
        _check_single_process()
        self.update = _try_import_for_setup("mpi_array.update")
        distribution = _try_import_for_setup("mpi_array.distribution")
        np = _try_import_for_setup("numpy")
        if pattern == "shifted":
            n = 64
            size = num_locales * n
            self.dst = \
                distribution.Distribution(
                    (size,),
                    [(slice(i * n, (i + 1) * n),) for i in range(num_locales)],
                    inter_locale_rank_to_peer_rank=np.arange(num_locales)
                )
            cuts = [0] + [n // 2 + i * n for i in range(num_locales)] + [size]
            self.src = \
                distribution.Distribution(
                    (size,),
                    [(slice(cuts[i], cuts[i + 1]),) for i in range(num_locales + 1)],
                    inter_locale_rank_to_peer_rank=np.arange(num_locales + 1)
                )
        elif pattern == "transpose":
            n = 8
            shape = (num_locales * n, num_locales * n, n)
            self.dst = _create_block_partition(shape, (num_locales, 1, 1))
            self.src = _create_block_partition(shape, (1, num_locales, 1))
        else:
            raise ValueError("Invalid pattern=%r." % (pattern,))

    def time_plan(self, pattern, num_locales):
        """
        Time the calculation of the updates.
        """
        self.update.UpdatesForRedistribute(
            self.dst,
            self.src,
            dst_inter_locale_rank=num_locales // 2
        )


class HaloPlanBench(object):
    """
    Benchmarks for the calculation of the halo updates
    (:meth:`mpi_array.globale.PerAxisRmaHaloUpdater.calc_halo_updates`) of a 3-D block
    partitioned array with a halo of 2 elements, on one process for :samp:`num_locales`
    locales: the updates of the locale of the process (a locale in the middle).
    """

    #: Number of repetitions to run each benchmark
    repeat = 3

    #: The number of locales parameter.
    params = [64, 256, 1024]

    #: The name of the parameter.
    param_names = ["num_locales"]

    # Goal time (seconds) for a single repeat.
    goal_time = 0.25

    # Execute benchmark for this time (seconds) as a *warm-up* prior to real timing.
    warmup_time = 0.0

    def setup(self, num_locales):
        """
        Creates the distribution (:attr:`distrib`).
        """
        _check_single_process()
        self.globale = _try_import_for_setup("mpi_array.globale")
        split = _try_import_for_setup("array_split.split")
        np = _try_import_for_setup("numpy")
        dims = tuple(int(d) for d in split.shape_factors(num_locales, 3))
        self.distrib = _create_block_partition(tuple(32 * d for d in dims), dims, halo=2)
        self.dtype = np.dtype("float64")

    def time_plan(self, num_locales):
        """
        Time the calculation of the halo updates.
        """
        self.globale.PerAxisRmaHaloUpdater(
            self.distrib,
            self.dtype,
            "C",
            None,
            None,
            inter_locale_rank=num_locales // 2
        ).calc_halo_updates()


__all__ = [s for s in dir() if not s.startswith('_')]
