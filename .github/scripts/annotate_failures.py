#!/usr/bin/env python
"""
Writes GitHub Actions error annotations for the failures reported by
:samp:`utils/run_mpi_tests.py`, so that they can be read without the job logs::

   python .github/scripts/annotate_failures.py HARNESS_OUTPUT_FILE [...]

Each file is the output of a :samp:`run_mpi_tests.py` run. Its summary (from the result
table on) is one annotation, the end of the log of each failed run or exit scenario
is another (at most :attr:`MAX_ANNOTATIONS` in all, GitHub shows ten for each step).
"""
from __future__ import print_function

import os
import re
import sys

#: The number of annotations written.
MAX_ANNOTATIONS = 10

#: The number of lines at the end of a log in its annotation.
MAX_LOG_LINES = 80

#: The number of characters of an annotation message.
MAX_CHARS = 8000

_LOG_RE = re.compile(r"(/\S+\.log)\s*$")


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
    Prints an error annotation, the end of :samp:`{text}` if it is long.
    """
    if len(text) > MAX_CHARS:
        text = "...\n" + text[-MAX_CHARS:]
    print("::error title=%s::%s" % (escape_property(title), escape_data(text)))


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


def main(argv):
    """
    Writes the annotations of the harness output files :samp:`{argv}`.
    """
    num_annotations = 0
    for file_name in argv:
        if not os.path.exists(file_name):
            continue
        with open(file_name, "r", errors="replace") as f:
            lines = f.read().splitlines()
        start = [i for i in range(len(lines)) if lines[i].startswith("ranks  result")]
        summary = lines[start[0]:] if len(start) > 0 else lines[-MAX_LOG_LINES:]
        if num_annotations < MAX_ANNOTATIONS:
            annotate("Summary of %s" % os.path.basename(file_name), "\n".join(summary))
            num_annotations += 1
        for log_file_name in failed_logs(lines):
            if num_annotations >= MAX_ANNOTATIONS:
                break
            if not os.path.exists(log_file_name):
                continue
            with open(log_file_name, "r", errors="replace") as f:
                log_lines = f.read().splitlines()
            if len(log_lines) > MAX_LOG_LINES + 1:
                log_lines = [log_lines[0], "..."] + log_lines[-MAX_LOG_LINES:]
            text = "\n".join(log_lines)
            annotate(os.path.basename(log_file_name), text)
            num_annotations += 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
