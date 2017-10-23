"""
Manages finding, running and recoding benchmark results.

This module has shamelessly borrows from
the `airspeed velocity (asv) <http://asv.readthedocs.io/en/latest>`_
file `benchmark.py <https://github.com/spacetelescope/asv/blob/master/asv/benchmark.py>`_.
See the `airspeed velocity (asv) <http://asv.readthedocs.io/en/latest>`_
`LICENSE <https://github.com/spacetelescope/asv/blob/master/LICENSE.rst>`_.

"""
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import mpi4py.MPI as mpi
import sys
import datetime
try:
    import cProfile as profile
except BaseException:
    profile = None
from hashlib import sha256
import inspect
import itertools
import json
import os
import re
import textwrap
import timeit
from importlib import import_module
from .. import logging as _logging

from .utils import get_process_time_timer

__license__ = "https://github.com/spacetelescope/asv/blob/master/LICENSE.rst"


def _get_attr(source, name, ignore_case=False):
    if ignore_case:
        attrs = [getattr(source, key) for key in dir(source)
                 if key.lower() == name.lower()]

        if len(attrs) > 1:
            raise ValueError(
                "{0} contains multiple {1} functions.".format(
                    source.__name__, name))
        elif len(attrs) == 1:
            return attrs[0]
        else:
            return None
    else:
        return getattr(source, name, None)


def _get_all_attrs(sources, name, ignore_case=False):
    for source in sources:
        val = _get_attr(source, name, ignore_case=ignore_case)
        if val is not None:
            yield val


def _get_first_attr(sources, name, default, ignore_case=False):
    for val in _get_all_attrs(sources, name, ignore_case=ignore_case):
        return val
    return default


def get_setup_cache_key(func):
    if func is None:
        return None
    return '{0}:{1}'.format(inspect.getsourcefile(func),
                            inspect.getsourcelines(func)[1])


def get_source_code(items):
    """
    Extract source code of given items, and concatenate and dedent it.
    """
    sources = []
    prev_class_name = None

    for func in items:
        try:
            lines, lineno = inspect.getsourcelines(func)
        except TypeError:
            continue

        if not lines:
            continue

        src = "\n".join(line.rstrip() for line in lines)
        src = textwrap.dedent(src)

        class_name = None
        if inspect.ismethod(func):
            # Add class name
            if hasattr(func, 'im_class'):
                class_name = func.im_class.__name__
            elif hasattr(func, '__qualname__'):
                names = func.__qualname__.split('.')
                if len(names) > 1:
                    class_name = names[-2]

        if class_name and prev_class_name != class_name:
            src = "class {0}:\n    {1}".format(
                class_name, src.replace("\n", "\n    "))
        elif class_name:
            src = "    {1}".format(
                class_name, src.replace("\n", "\n    "))

        sources.append(src)
        prev_class_name = class_name

    return "\n\n".join(sources).rstrip()


class Benchmark(object):
    """
    Represents a single benchmark.
    """
    # The regex of the name of function or method to be considered as
    # this type of benchmark.  The default in the base class, will
    # match nothing.
    name_regex = re.compile('^$')

    def __init__(self, name, func, attr_sources):
        self.name = name
        self.func = func
        self.pretty_name = getattr(func, "pretty_name", name)
        self._attr_sources = list(attr_sources)

        self._setups = None
        self._teardowns = None
        self._setup_cache = None
        self.setup_cache_key = None
        self.setup_cache_timeout = None
        self.timeout = None
        self.code = None
        self.version = None
        self.type = None
        self.unit = None

        self._redo_setup_next = False

        self._params = None
        self.param_names = None
        self._current_params = None
        self._comm = None

    def initialise(self):
        """
        """
        if self._setups is None:
            if isinstance(self._attr_sources[-1], str):
                self._attr_sources[-1] = import_module(self._attr_sources[-1])
            self._setups = list(_get_all_attrs(self._attr_sources, 'setup', True))[::-1]
            self._teardowns = list(_get_all_attrs(self._attr_sources, 'teardown', True))
            self._setup_cache = _get_first_attr(self._attr_sources, 'setup_cache', None)
            self.setup_cache_key = get_setup_cache_key(self._setup_cache)
            self.setup_cache_timeout = _get_first_attr([self._setup_cache], "timeout", None)
            self.timeout = _get_first_attr(self._attr_sources, "timeout", 60.0)
            self.code = get_source_code([self.func] + self._setups + [self._setup_cache])
            if sys.version_info[0] >= 3:
                code_text = self.code.encode('utf-8')
            else:
                code_text = self.code
            code_hash = sha256(code_text).hexdigest()
            self.version = str(_get_first_attr(self._attr_sources, "version", code_hash))
            self.type = "base"
            self.unit = "unit"

            self._redo_setup_next = False

            self._params = _get_first_attr(self._attr_sources, "params", [])
            self.param_names = _get_first_attr(self._attr_sources, "param_names", [])
            self._current_params = ()

            # Enforce params format
            try:
                self.param_names = [str(x) for x in list(self.param_names)]
            except ValueError:
                raise ValueError("%s.param_names is not a list of strings" % (self.name,))

            try:
                self._params = list(self._params)
            except ValueError:
                raise ValueError("%s.params is not a list" % (self.name,))

            if self._params and not isinstance(self._params[0], (tuple, list)):
                # Accept a single list for one parameter only
                self._params = [self._params]
            else:
                self._params = [[item for item in entry] for entry in self._params]

            if len(self.param_names) != len(self._params):
                self.param_names = self.param_names[:len(self._params)]
                self.param_names += ['param%d' % (k + 1,) for k in range(len(self.param_names),
                                                                         len(self._params))]

            # Exported parameter representations
            self.params_repr = [[repr(item) for item in entry] for entry in self._params]

    @property
    def root_rank(self):
        """
        An :samp:`int` indicating the *root* rank process of :attr:`comm`.
        """
        return 0

    @property
    def comm(self):
        """
        The :obj:`mpi4pi.MPI.Comm` used for synchronization.
        """
        return self._comm

    @comm.setter
    def comm(self, comm):
        self._comm = comm

    def barrier(self):
        """
        Barrier.
        """
        self.comm.barrier()

    def bcast(self, value):
        """
        Broadcast value from :attr:`root_rank` to all ranks of :attr:`comm`.

        :rtype: :obj:`object`
        :return: value on rank :attr:`root_rank` rank process.
        """
        return self.comm.bcast(value, self.root_rank)

    @property
    def params(self):
        """
        The list of benchmark parameters.
        """
        self.initialise()
        return self._params

    @property
    def current_params(self):
        """
        The current set of parameters, set via :meth:`set_param_idx`.
        """
        return self._current_params

    def set_param_idx(self, param_idx):
        """
        Set the parameter combo via the index :samp:`{param_idx}`.

        :raises ValueError: if :samp:`param_idx` is out of range.
        """
        self.initialise()
        try:
            self._current_params, = itertools.islice(
                itertools.product(*self._params),
                param_idx, param_idx + 1)
        except ValueError:
            raise ValueError(
                "Invalid benchmark parameter permutation index: %r" % (param_idx,))

    def insert_param(self, param):
        """
        Insert a parameter at the front of the parameter list.
        """
        self.initialise()
        self._current_params = tuple([param] + list(self._current_params))

    def __repr__(self):
        return '<{0} {1}>'.format(self.__class__.__name__, self.name)

    def do_setup(self):
        self.initialise()
        try:
            for setup in self._setups:
                setup(*self._current_params)
        except NotImplementedError:
            # allow skipping test
            return True
        return False

    def redo_setup(self):
        self.initialise()
        if not self._redo_setup_next:
            self._redo_setup_next = True
            return
        self.do_teardown()
        self.do_setup()

    def do_teardown(self):
        for teardown in self._teardowns:
            teardown(*self._current_params)

    def do_setup_cache(self):
        if self._setup_cache is not None:
            return self._setup_cache()

    def do_run(self):
        return self.run(*self._current_params)

    def do_profile(self, filename=None):
        def method_caller():
            self.run(*self._current_params)

        if profile is None:
            raise RuntimeError("cProfile could not be imported")

        if filename is not None:
            if hasattr(method_caller, 'func_code'):
                code = method_caller.func_code
            else:
                code = method_caller.__code__

            self.redo_setup()

            profile.runctx(
                code, {'run': self.func, 'params': self._current_params},
                {}, filename)


class TimeBenchmark(Benchmark):
    """
    Represents a single benchmark for timing.
    """
    name_regex = re.compile(
        '^(Time[A-Z_].+)|(time_.+)$')

    def __init__(self, name, func, attr_sources):
        Benchmark.__init__(self, name, func, attr_sources)
        self.type = "time"
        self.unit = "seconds"
        self._attr_sources = attr_sources
        self._repeat = None
        self._number = None
        self._goal_time = None
        self._warmup_time = None
        self._timer = None
        self._wall_timer = None

    def _load_vars(self):
        self._repeat = _get_first_attr(self._attr_sources, 'repeat', 0)
        self._number = int(_get_first_attr(self._attr_sources, 'number', 0))
        self._goal_time = _get_first_attr(self._attr_sources, 'goal_time', 0.1)
        self._warmup_time = _get_first_attr(self._attr_sources, 'warmup_time', -1)
        self._timer = _get_first_attr(self._attr_sources, 'timer', get_process_time_timer())
        self._wall_timer = _get_first_attr(self._attr_sources, 'wall_timer', mpi.Wtime)

    @property
    def repeat(self):
        if self._repeat is None:
            self._load_vars()
        return self._repeat

    @property
    def number(self):
        if self._number is None:
            self._load_vars()
        return self._number

    @property
    def goal_time(self):
        if self._goal_time is None:
            self._load_vars()
        return self._goal_time

    @property
    def warmup_time(self):
        if self._warmup_time is None:
            self._load_vars()
        return self._warmup_time

    @property
    def timer(self):
        if self._timer is None:
            self._load_vars()
        return self._timer

    @property
    def wall_timer(self):
        if self._wall_timer is None:
            self._load_vars()
        return self._wall_timer

    def wall_time(self):
        """
        Return *current* time in seconds.
        """
        return self.wall_timer()

    def do_setup(self):
        result = Benchmark.do_setup(self)
        # For parameterized tests, setup() is allowed to change these
        self._load_vars()
        return result

    def run(self, *param):
        number = self.number
        repeat = self.repeat

        if repeat == 0:
            repeat = 10

        warmup_time = self.warmup_time
        if warmup_time < 0:
            if '__pypy__' in sys.modules:
                warmup_time = 1.0
            else:
                # Transient effects exist also on CPython, e.g. from
                # OS scheduling
                warmup_time = 0.1

        if param:
            def func():
                return self.func(*param)
        else:
            func = self.func

        timer = timeit.Timer(
            stmt=func,
            setup=self.redo_setup,
            timer=self.timer)

        samples, number, samples_pre_barrier, samples_post_barrier = \
            self.benchmark_timing(timer, repeat, warmup_time, number=number)

        l = [samples, samples_pre_barrier, samples_post_barrier]
        for i in range(len(l)):
            l[i] = [s / number for s in l[i]]
        return \
            {
                'samples': l[0],
                'number': number,
                'wall_samples_pre_barrier': l[1],
                'wall_samples_post_barrier': l[2]
            }

    def benchmark_timing(self, timer, repeat, warmup_time, number=0):
        goal_time = self.goal_time

        start_time = self.bcast(self.wall_time())

        max_time = start_time + min(warmup_time + 1.3 * repeat * goal_time,
                                    self.timeout - 1.3 * goal_time)

        def too_slow():
            # too slow, don't take more samples
            return self.bcast(self.wall_time()) > max_time

        if number == 0:
            # Select number & warmup.
            #
            # This needs to be done at the same time, because the
            # benchmark timings at the beginning can be larger, and
            # lead to too small number being selected.
            number = 1
            while True:
                self._redo_setup_next = False
                self.barrier()
                start = self.wall_time()
                timing = timer.timeit(number)
                self.barrier()
                end = self.wall_time()
                wall_time, timing = self.bcast((end - start, timing))
                actual_timing = max(wall_time, timing)

                if actual_timing >= goal_time:
                    if self.bcast(self.wall_time()) > start_time + warmup_time:
                        break
                else:
                    try:
                        p = min(10.0, max(1.1, goal_time / actual_timing))
                    except ZeroDivisionError:
                        p = 10.0
                    number = max(number + 1, int(p * number))

            if too_slow():
                return [timing], number
        elif warmup_time > 0:
            # Warmup
            while True:
                self._redo_setup_next = False
                timing = self.bcast(timer.timeit(number))
                if self.bcast(self.wall_time()) >= start_time + warmup_time:
                    break
            if too_slow():
                return [timing], number

        # Collect samples
        samples = []
        wall_samples_pre_barrier = []
        wall_samples_post_barrier = []
        for j in range(repeat):
            self.barrier()
            start = self.wall_time()
            timing = timer.timeit(number)
            end_pre_barrier = self.wall_time()
            self.barrier()
            end_post_barrier = self.wall_time()
            samples.append(timing)
            wall_samples_pre_barrier.append(end_pre_barrier - start)
            wall_samples_post_barrier.append(end_post_barrier - start)

            if too_slow():
                break

        return samples, number, wall_samples_pre_barrier, wall_samples_post_barrier


benchmark_types = [
    TimeBenchmark,
]


def disc_files(root, package=''):
    """
    Iterate over all .py files in a given directory tree.
    """
    if os.path.isdir(root):
        filename_list = os.listdir(root)
    elif os.path.isfile(root):
        root, filename = os.path.split(root)
        filename_list = [filename, ]
    else:
        raise ValueError("root=%s not an existing file or directory." % root)

    for filename in filename_list:
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            filename, ext = os.path.splitext(filename)
            if ext == '.py':
                module = import_module(package + filename)
                yield module
        elif os.path.isdir(path):
            for x in disc_files(path, package + filename + "."):
                yield x


def _get_benchmark(attr_name, module, klass, func):
    try:
        name = func.benchmark_name
    except AttributeError:
        name = None
        search = attr_name
    else:
        search = name.split('.')[-1]

    for cls in benchmark_types:
        if cls.name_regex.match(search):
            break
    else:
        return
    # relative to benchmark_dir
    mname = module.__name__.split('.', 1)[1]
    if klass is None:
        if name is None:
            name = ".".join([mname, func.__name__])
        sources = [func, module]
    else:
        instance = klass()
        func = getattr(instance, func.__name__)
        if name is None:
            name = ".".join([mname, klass.__name__, func.__name__])
        sources = [func, instance, module.__name__]
    return cls(name, func, sources)


def disc_benchmarks(root, package=None):
    """
    Discover all benchmarks in a given directory tree, yielding Benchmark
    objects

    For each class definition, looks for any methods with a
    special name.

    For each free function, yields all functions with a special
    name.
    """
    if package is None:
        package = os.path.basename(root)
    for module in disc_files(root, package + '.'):
        for attr_name, module_attr in (
            (k, v) for k, v in module.__dict__.items()
            if not k.startswith('_')
        ):
            if inspect.isclass(module_attr):
                for name, class_attr in inspect.getmembers(module_attr):
                    if (inspect.isfunction(class_attr) or
                            inspect.ismethod(class_attr)):
                        benchmark = _get_benchmark(name, module, module_attr,
                                                   class_attr)
                        if benchmark is not None:
                            yield benchmark
            elif inspect.isfunction(module_attr):
                benchmark = _get_benchmark(attr_name, module, None, module_attr)
                if benchmark is not None:
                    yield benchmark


def create_runner_argument_parser():
    """
    """
    import argparse
    ap = argparse.ArgumentParser("mpi-array-benchmarks")

    ap.add_argument(
        "-d", "--discover",
        action='store_true',
        help="Only discover benchmarks, do not run them."
    )
    ap.add_argument(
        "-q", "--quick",
        action='store_true',
        help="Quick benchmark run, only execute each benchmark once, skip repeats."
    )
    ap.add_argument(
        "-o", "--results_file",
        action='store',
        help="Name of the JSON file where benchmark results are to be stored.",
        default="mpia_bench_results.json"
    )
    ap.add_argument(
        "-b", "--benchmarks_file",
        action='store',
        help="Name of the JSON file where individual benchmark details are to be stored.",
        default="mpia_benchmarks.json"
    )
    ap.add_argument(
        "module_name",
        action="append",
        help="Name of modules to search for benchmarks.",
        default=["mpi_array.benchmarks"]
    )

    return ap


def root_and_package_from_name(module_name):
    """
    Returns root filename for the package/module named by :samp:`{module_name}`.
    """
    module = import_module(module_name)
    root = module.__file__
    dir, filename = os.path.split(root)
    if filename.find("__init__.py") == 0:
        root = dir
    return root, module.__name__


class BenchmarkRunner(object):
    """
    """

    def __init__(self, argv=None):
        """
        """
        if argv is None:
            argv = []
        arg_parser = self.create_argument_parser()
        self._args = arg_parser.parse_args(args=argv)
        self._root_logger = None
        self._rank_logger = None
        self._bench_module_names = None
        self._bench_results = None
        self._benchmarks = None

    @property
    def root_logger(self):
        """
        """
        if self._root_logger is None:
            self._root_logger = \
                _logging.get_root_logger(__name__ + "." + self.__class__.__name__, comm=self.comm)
        return self._root_logger

    @property
    def rank_logger(self):
        """
        """
        if self._rank_logger is None:
            self._rank_logger = \
                _logging.get_rank_logger(__name__ + "." + self.__class__.__name__, comm=self.comm)
        return self._rank_logger

    @property
    def comm(self):
        """
        A :obj:`mpi4py.MPI.Comm object.
        """
        return mpi.COMM_WORLD

    @property
    def root_rank(self):
        """
        An :obj:`int` indicating the rank of the master MPI (root) process.
        """
        return 0

    @property
    def is_root_rank(self):
        """
        A :obj:`bool`, if :samp:`True` this is running on the :attr:`root_rank` MPI rank.
        """
        return (self.comm.rank == self.root_rank)

    @property
    def do_quick_run(self):
        """
        A :obj:`bool`, if :samp:`True`, performs a *quick* run. Each benchmark
        is only executed once.
        """
        return self._args.quick

    @property
    def discover_only(self):
        """
        A :obj:`bool`, if :samp:`True` only *discover* tests and do not run them.
        """
        return self._args.discover

    @property
    def bench_results_file_name(self):
        """
        A :obj:`str`, name of file where benchmark results are written.
        """
        return self._args.results_file

    @property
    def bench_results(self):
        """
        A :obj:`list` of benchmark results.
        """
        return self._bench_results

    @property
    def benchmarks(self):
        """
        The :obj:`list` of :obj:`Benchmark` objects.
        """
        return self._benchmarks

    @property
    def benchmark_module_names(self):
        """
        The :obj:`list` of module names which are searched to discover benchmarks.
        """
        if self._bench_module_names is None:
            self._bench_module_names = \
                [
                    name for name in self._args.module_name
                    if not (os.path.split(name)[1].find("__main__") >= 0)
                ]
            if len(self._bench_module_names) <= 0:
                self._bench_module_names = ["mpi_array.benchmarks", ]
        return self._bench_module_names

    @property
    def benchmarks_file_name(self):
        """
        A :obj:`str`, name of file where benchmark results are written.
        """
        return self._args.benchmarks_file

    def create_argument_parser(self):
        """
        Creates :obj:`argparse.ArgumentParser` for handling command line.

        :rtype: :obj:`argparse.ArgumentParser`
        :return: A :obj:`argparse.ArgumentParser` for parsing command line.

        ..seealso:: :func:`create_runner_argument_parser`
        """
        return create_runner_argument_parser()

    def discover_benchmarks(self):
        """
        Find benchmarks, store :obj:`Benchmark` objects in :attr:`benchmarks`.
        """
        if self.is_root_rank:
            self._benchmarks = []
            for module_name in self.benchmark_module_names:
                root, package = root_and_package_from_name(module_name)
                self._benchmarks += [b for b in disc_benchmarks(root, package)]
        else:
            self._benchmarks = None

    def run_benchmarks(self):
        """
        """
        class QuickBenchmarkAttrs(object):
            """
            Over-rides for :attr:`repeat` and :attr:`number` for *quick* benchmark run.
            """
            repeat = 1
            number = 1

        self._bench_results = None
        benchmarks = self.benchmarks
        benchmarks = self.comm.bcast(benchmarks, self.root_rank)

        results = []
        if benchmarks is not None:
            for benchmark in benchmarks:
                if self.do_quick_run:
                    benchmark._attr_sources.insert(0, QuickBenchmarkAttrs)
                benchmark.comm = self.comm
                benchmark.initialise()
                if (benchmark.params is not None) and (len(benchmark.params) > 0):
                    param_iter = enumerate(itertools.product(*benchmark.params))
                else:
                    param_iter = [(None, None)]

                for param_idx, params in param_iter:
                    started_at = datetime.datetime.utcnow()
                    if param_idx is not None:
                        benchmark.set_param_idx(param_idx)
                    skip = benchmark.do_setup()

                    try:
                        if skip:
                            result = {"samples": None, "number": None}
                        else:
                            result = benchmark.do_run()
                    finally:
                        benchmark.do_teardown()

                    if not skip:
                        self.root_logger.info(
                            "%68s, %16.8f, %16.8f, %6d,%8d, %s",
                            benchmark.name,
                            min(result["samples"]),
                            max(result["samples"]),
                            benchmark.repeat,
                            result["number"],
                            [
                                ("%s=%s" % (benchmark.param_names[i], params[i]))
                                for i in range(len(params))
                            ] if params is not None else None
                        )
                    else:
                        self.root_logger.info("%68s, %16s", benchmark.name, "skipped...")

                    result["walltime_finished_at"] = str(datetime.datetime.utcnow())
                    result["walltime_started_at"] = str(started_at)
                    result["comm_name"] = self.comm.Get_name()
                    result["comm_size"] = self.comm.size
                    result["bench_name"] = benchmark.name
                    result["params"] = \
                        {
                            benchmark.param_names[i]: repr(benchmark.current_params[i])
                            for i in range(len(benchmark.param_names))
                    }

                    results.append(result)
        if self.root_rank == self.comm.rank:
            self._bench_results = results

    def run(self):
        """
        Discover and run benchmarks.
        """
        self.discover_benchmarks()
        if not self.discover_only:
            self.run_benchmarks()

    def write_bench_results(self, bench_results, bench_results_file_name):
        """
        Writes the benchmark results :samp:`{bench_results}` as :mod:`json`
        string to file named :samp:`{bench_results_file_name}`.
        """
        if (bench_results_file_name is not None) and (bench_results is not None):
            with open(bench_results_file_name, "wt") as fd:
                json.dump(bench_results, fd, indent=2, sort_keys=True)

    def write_benchmarks(self, benchmarks, benchmarks_file_name):
        """
        Writes individual :obj:`Benchmark` elements of :samp:`{benchmarks}`
        as :mod:`json` string to file named :samp:`{benchmarks_file_name}`.
        """
        if (benchmarks_file_name is not None) and (benchmarks is not None):
            benchmark_dicts = \
                [
                    dict(
                        (k, v) for (k, v) in benchmark.__dict__.items()
                        if isinstance(v, (str, int, float, list, dict, bool)) and not
                        k.startswith('_')
                    )
                    for benchmark in benchmarks
                ]
            with open(benchmarks_file_name, "wt") as fd:
                json.dump(benchmark_dicts, fd, indent=2, sort_keys=True)

    def run_and_write_results(self):
        """
        Discovers, runs and records benchmark results.
        """
        self.run()
        if self.bench_results is not None:
            self.write_bench_results(self.bench_results, self.bench_results_file_name)
        if self.benchmarks is not None:
            self.write_benchmarks(self.benchmarks, self.benchmarks_file_name)


def run_main(argv):
    """
    Runs the benchmarks.

    :type argv: :obj:`list` of :obj:`str`
    :param argv: The command line arguments (e.g. :samp:`sys.argv`).
    """
    runner = BenchmarkRunner(argv=argv)
    runner.run_and_write_results()


__all__ = [s for s in dir() if not s.startswith('_')]
