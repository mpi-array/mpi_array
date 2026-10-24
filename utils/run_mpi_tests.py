#!/usr/bin/env python
"""
Runs the :mod:`mpi_array` test suite under :samp:`mpirun` for several process counts
and prints a summary.

Each run is launched as::

   mpirun [impl-args] -n NP PYTHON -m mpi4py -m mpi_array.tests [TEST-ARGS]

* :samp:`python -m mpi4py` makes an unhandled exception on one rank abort the whole job
  (instead of leaving the other ranks blocked in a collective).
* Each test runs under a hang-detection timer (see :mod:`mpi_array.unittest`):
  a deadlocked test makes every rank dump its stacks and exit, so the run fails
  (reported as HANG) instead of blocking forever. The per-test timeout is
  set with :samp:`--timeout` (exported as :samp:`MPI_ARRAY_TEST_TIMEOUT`).
* :samp:`--job-timeout` is a backstop for hangs outside of a test (e.g. at exit).
* For Open MPI, :samp:`--oversubscribe` (allow more ranks than cores)
  and :samp:`--tag-output` (prefix every output line with the rank) are added.
  For MPICH-like launchers :samp:`-prepend-rank` is added.

Examples::

   python utils/run_mpi_tests.py                      # 1, 2, 4 and 8 processes
   python utils/run_mpi_tests.py -n 2 4 --timeout 60
   python utils/run_mpi_tests.py -n 4 -- mpi_array.globale_ufunc_test
   OMPI_MCA_osc='^ucx' python utils/run_mpi_tests.py  # silence Open MPI UCX OSC warnings

Use :samp:`--python` to choose the interpreter (e.g. a virtual environment
which has :mod:`array_split` installed). The exit status is zero only if every run passed.
"""
from __future__ import print_function

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

_RAN_RE = re.compile(r"Ran (\d+) tests? in ([0-9.]+)s")
# The runner logs "FAILED" and " (failures=1, errors=2)" as separate lines.
_FAILED_RE = re.compile(r"\|rank0\| \(((?:failures|errors)=\d+[^)]*)\)")
_TEST_RE = re.compile(r"\|rank(\d+)\|(ERROR|FAIL): (.+?)\s*$", re.MULTILINE)
_HANG_RE = re.compile(r"Timeout \((\d+:\d\d:\d\d(?:\.\d+)?)\)!")


def detect_launcher_args(mpirun):
    """
    Returns extra launcher arguments for the MPI implementation of :samp:`{mpirun}`.
    """
    try:
        out = subprocess.run(
            [mpirun, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise SystemExit("Could not run '%s --version': %s" % (mpirun, e))
    if ("Open MPI" in out) or ("OpenRTE" in out):
        return ["--oversubscribe", "--tag-output"]
    if ("HYDRA" in out) or ("MPICH" in out) or ("Intel" in out):
        return ["-prepend-rank"]
    return []


def kill_process_group(proc):
    """
    Kills the process group of :samp:`{proc}` (which was started in a new session).
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass


def run_once(args, launcher_args, np, log_file_name):
    """
    Runs the test suite once with :samp:`{np}` processes, output goes to :samp:`{log_file_name}`.

    :rtype: :obj:`dict`
    :return: Summary of the run.
    """
    cmd = \
        [args.mpirun] + launcher_args + list(args.mpirun_arg) + ["-n", str(np)] \
        + [args.python, "-m", "mpi4py", "-m", args.module] + list(args.test_args)
    env = dict(os.environ)
    env["MPI_ARRAY_TEST_TIMEOUT"] = str(args.timeout)
    job_timed_out = False
    t0 = time.time()
    with open(log_file_name, "w") as log_file:
        log_file.write("# %s\n" % " ".join(cmd))
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True
        )
        try:
            proc.wait(timeout=args.job_timeout)
        except subprocess.TimeoutExpired:
            job_timed_out = True
            kill_process_group(proc)
    elapsed = time.time() - t0

    with open(log_file_name, "r", errors="replace") as log_file:
        text = log_file.read()

    ran = _RAN_RE.search(text)
    failed = _FAILED_RE.search(text)
    hang = _HANG_RE.search(text)
    tests = []
    for m in _TEST_RE.finditer(text):
        name = "%s: %s" % (m.group(2), m.group(3))
        if name not in tests:
            tests.append(name)

    if job_timed_out:
        status = "JOB TIMEOUT (>%ss)" % args.job_timeout
    elif hang is not None:
        status = "HANG (a test exceeded %ss)" % args.timeout
    elif proc.returncode == 0:
        status = "PASS"
    elif failed is not None:
        status = "FAIL (%s)" % failed.group(1)
    elif ran is not None:
        status = "FAIL (exit status %s)" % proc.returncode
    else:
        status = "ERROR (exit status %s, no test summary)" % proc.returncode

    return {
        "np": np,
        "status": status,
        "passed": (status == "PASS"),
        "tests_run": int(ran.group(1)) if ran is not None else None,
        "elapsed": elapsed,
        "log": log_file_name,
        "failed_tests": tests,
    }


def main(argv=None):
    parser = \
        argparse.ArgumentParser(
            description="Run the mpi_array test suite under mpirun for several process counts.",
            epilog="Arguments after '--' are passed to the test module.",
        )
    parser.add_argument(
        "-n", "--np", type=int, nargs="+", default=[1, 2, 4, 8],
        help="Numbers of MPI processes (default: 1 2 4 8)."
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Per-test hang-detection timeout in seconds, 0 disables (default: 120)."
    )
    parser.add_argument(
        "--job-timeout", type=float, default=1800.0,
        help="Wall-clock limit in seconds for each mpirun invocation (default: 1800)."
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter to run the tests with (default: this interpreter)."
    )
    parser.add_argument("--mpirun", default="mpirun", help="MPI launcher (default: mpirun).")
    parser.add_argument(
        "--mpirun-arg", action="append", default=[],
        help="Extra launcher argument, can be repeated (e.g. --mpirun-arg=--bind-to=none)."
    )
    parser.add_argument(
        "--no-launcher-defaults", action="store_true",
        help="Don't add the implementation specific launcher arguments."
    )
    parser.add_argument("--module", default="mpi_array.tests", help="Test module to run.")
    parser.add_argument(
        "--log-dir", default=None,
        help="Directory for the per-run log files (default: a new temporary directory)."
    )
    parser.add_argument("test_args", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    launcher_args = [] if args.no_launcher_defaults else detect_launcher_args(args.mpirun)
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = tempfile.mkdtemp(prefix="mpi_array_tests_")
    elif not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    results = []
    for np in args.np:
        log_file_name = os.path.join(log_dir, "tests_np%d.log" % np)
        print("Running %s with %d process(es)..." % (args.module, np))
        sys.stdout.flush()
        result = run_once(args, launcher_args, np, log_file_name)
        print("    %s" % result["status"])
        sys.stdout.flush()
        results.append(result)

    print("")
    print("%5s  %-40s %6s %9s  %s" % ("ranks", "result", "tests", "time (s)", "log"))
    for r in results:
        print(
            "%5d  %-40s %6s %9.1f  %s"
            %
            (
                r["np"],
                r["status"],
                r["tests_run"] if r["tests_run"] is not None else "-",
                r["elapsed"],
                r["log"]
            )
        )
    for r in results:
        if len(r["failed_tests"]) > 0:
            print("")
            print("Failing tests with %d process(es):" % r["np"])
            for name in r["failed_tests"]:
                print("    %s" % name)

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
