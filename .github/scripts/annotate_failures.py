#!/usr/bin/env python
"""
Writes GitHub Actions error annotations for the failures reported by
:samp:`utils/run_mpi_tests.py`, so that they can be read without the job logs::

   python .github/scripts/annotate_failures.py HARNESS_OUTPUT_FILE [...]

Each file is the output of a :samp:`run_mpi_tests.py` run. Its summary (from the result
table on) is one annotation. For the log of each failed run or exit scenario, the errors
of the tests (with their tracebacks) and the stacks dumped by the hang detection (with
the frames of the :mod:`mpi_array` modules only) are others, or else the end of the log.
GitHub shows ten annotations for each step (:attr:`MAX_ANNOTATIONS`), and keeps 4096
characters of each message (:attr:`MAX_CHARS`).
"""
from __future__ import print_function

import os
import re
import sys

#: The number of annotations written.
MAX_ANNOTATIONS = 10

#: The number of characters of an annotation message.
MAX_CHARS = 4000

#: The number of lines at the end of a log in its annotation, when it has no errors or stacks.
MAX_LOG_LINES = 60

#: The number of lines of an error (with its traceback).
MAX_ERROR_LINES = 40

_LOG_RE = re.compile(r"(/\S+\.log)\s*$")

_STACKS_HEADER = "==== Stacks dumped by the hang detection"

_ERROR_RE = re.compile(r"\|rank(\d+)\|(ERROR|FAIL): (.*)$")

# The launcher's prefix (e.g. "[1,2]<stderr>:" or "[2] "), the time and the communicator.
_PREFIX_RE = re.compile(r"^(\[[\d,]+\](<\w+>:)?\s*)?\d\d:\d\d:\d\d\|[^|]*\|")

_FRAME_RE = re.compile(r'^(\s*File ")([^"]+)(", line \d+.*)$')


def escape_data(text):
    """
    Escapes the message of a workflow command.
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(text):
    """
    Escapes a property (e.g. the title) of a workflow command.
    """
    return escape_data(text).replace(":", "%3A").replace(",", "%2C")


def annotate(title, text):
    """
    Prints an error annotation (its first :attr:`MAX_CHARS` characters).
    """
    print("::error title=%s::%s" % (escape_property(title), escape_data(text[:MAX_CHARS])))


def package_path(path):
    """
    Returns :samp:`{path}` from its :mod:`mpi_array` package directory on,
    :samp:`None` if it isn't a module of the package (or of its test runner).
    """
    parts = re.split(r"[/\\]", path)
    indices = [i for i in range(len(parts) - 1) if parts[i] == "mpi_array"]
    if (len(indices) == 0) or (parts[-1] in ("unittest.py", "tests.py")):
        return None
    return "/".join(parts[indices[-1]:])


def compact_line(line):
    """
    Removes the launcher prefix, the time and the communicator of a log line, and shortens
    the path of a frame of the package.
    """
    line = _PREFIX_RE.sub("", line)
    m = _FRAME_RE.match(line)
    if m is not None:
        path = package_path(m.group(2))
        if path is not None:
            line = m.group(1) + path + m.group(3)
    return line


def error_blocks(log_lines):
    """
    Returns the lines of the errors and failures logged by the tests, each with the
    following lines of its rank (its traceback), once for each rank and test.
    """
    lines = []
    seen = set()
    for i in range(len(log_lines)):
        m = _ERROR_RE.search(log_lines[i])
        if m is None:
            continue
        key = (m.group(1), m.group(2), m.group(3))
        if key in seen:
            continue
        seen.add(key)
        rank = "|rank%s|" % m.group(1)
        block = [log_lines[i]]
        for line in log_lines[i + 1:]:
            if (
                (len(block) >= MAX_ERROR_LINES)
                or line.startswith(_STACKS_HEADER)
                or (_ERROR_RE.search(line) is not None)
            ):
                break
            if rank in line:
                if "=" * 20 in line:
                    break
                block.append(line)
        lines += [compact_line(line) for line in block if "-" * 20 not in line]
    return lines


def compact_stacks(log_lines):
    """
    Returns the stacks dumped by the hang detection in a log, with only their frames
    of the :mod:`mpi_array` modules, :samp:`None` if the log has none.
    """
    start = [i for i in range(len(log_lines)) if log_lines[i].startswith(_STACKS_HEADER)]
    if len(start) == 0:
        return None
    stacks = []
    for line in log_lines[start[0]:]:
        m = _FRAME_RE.match(line)
        if m is not None:
            if package_path(m.group(2)) is not None:
                stacks.append(compact_line(line))
        elif line.startswith("====") or line.startswith("Timeout"):
            stacks.append(line.replace(_STACKS_HEADER + ", ", "==== "))
    return stacks


def chunks(lines, max_chars=MAX_CHARS - 200):
    """
    Splits :samp:`{lines}` into lists of lines of at most :samp:`{max_chars}` characters.
    """
    result = [[]]
    size = 0
    for line in lines:
        if (size + len(line) + 1 > max_chars) and (len(result[-1]) > 0):
            result.append([])
            size = 0
        result[-1].append(line)
        size += len(line) + 1
    return result


def failed_logs(lines):
    """
    Returns the log files of the failed runs and exit scenarios in the harness output.
    """
    logs = []
    table = False
    for line in lines:
        if line.startswith("ranks  result"):
            table = True
            continue
        if table and (len(line.strip()) == 0):
            table = False
        m = _LOG_RE.search(line)
        if m is None:
            continue
        failed = ("PASS" not in line.split()[1:3]) if table else ("log " in line)
        if failed and (m.group(1) not in logs):
            logs.append(m.group(1))
    return logs


def log_annotations(log_file_name):
    """
    Returns the :samp:`(title, text)` annotations of a failed log.
    """
    with open(log_file_name, "r", errors="replace") as f:
        log_lines = f.read().splitlines()
    if len(log_lines) == 0:
        return []
    name = os.path.basename(log_file_name)
    header = compact_line(log_lines[0])
    annotations = []
    errors = error_blocks(log_lines)
    stacks = compact_stacks(log_lines)
    for kind, lines in (("errors", errors), ("stacks", stacks)):
        if lines:
            for chunk in chunks(lines):
                annotations.append(("%s, %s" % (name, kind), "\n".join([header] + chunk)))
    if len(annotations) == 0:
        tail = [compact_line(line) for line in log_lines[-MAX_LOG_LINES:]]
        annotations.append((name, "\n".join([header, "..."] + tail)))
    return annotations


def main(argv):
    """
    Writes the annotations of the harness output files :samp:`{argv}`.
    """
    annotations = []
    for file_name in argv:
        if not os.path.exists(file_name):
            continue
        with open(file_name, "r", errors="replace") as f:
            lines = f.read().splitlines()
        start = [i for i in range(len(lines)) if lines[i].startswith("ranks  result")]
        summary = lines[start[0]:] if len(start) > 0 else lines[-MAX_LOG_LINES:]
        annotations.append(("Summary of %s" % os.path.basename(file_name), "\n".join(summary)))
        for log_file_name in failed_logs(lines):
            if os.path.exists(log_file_name):
                annotations += log_annotations(log_file_name)
    for title, text in annotations[:MAX_ANNOTATIONS]:
        annotate(title, text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
