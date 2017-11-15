"""

Module defining :mod:`mpi_array.benchmarks.utils.wlm` unit-tests.
Execute as::

   python -m mpi_array.benchmarks.utils.wlm_test


"""
from __future__ import absolute_import

import os as _os
import shutil as _shutil
import tempfile as _tempfile
import mpi4py.MPI as _mpi
import numpy as _np

from ...license import license as _license, copyright as _copyright, version as _version
from ... import unittest as _unittest
from ... import logging as _logging  # noqa: E402,F401
from . import wlm as _wlm

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


#: Template string for performing substitutions
WLM_SCRIPT_TEMPLATE =\
    """
#!/bin/bash
#
# PBS script for submission of mpi_array benchmark job
# ====================================================
#
#PBS -l mem=%(mem)s
#PBS -l ncpus=%(max_num_cpus)s
#PBS -l walltime=%(wall_time)s
#PBS -l jobfs=256GB
#PBS -l wd
#PBS -q %(queue)s

export OUT_DIR="."
for ncpus in %(num_cpus)s do;
    n_str=`printf "%%04d" ${ncpus}`
    mpiexec -n ${ncpus} python -m mpi_array.benchmarks \\
      -f '((.*Ufunc.*_add.*)|(.*Ufunc.*_multiply.*)|(.*Ufunc.*_divide.*))' \\
      -b ${OUT_DIR}/mpia_bench_ufunc_benchmarks_q_%(queue)s_${n_str}.json \\
      -o ${OUT_DIR}/mpia_bench_ufunc_results_q_%(queue)s_${n_str}.json
done
"""


class WlmScriptGeneratorTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :obj:`mpi_array.benchmarks.utils.wlm.WlmScriptGenerator`.
    """

    def setUp(self):
        self.rank_logger = _logging.get_rank_logger(name=self.id(), comm=_mpi.COMM_WORLD)
        self.tempdir = None
        if _mpi.COMM_WORLD.rank == 0:
            self.tempdir = _tempfile.mkdtemp(suffix=str(self.__class__.__name__))

    def tearDown(self):
        if self.tempdir is not None and _os.path.exists(self.tempdir):
            _shutil.rmtree(self.tempdir)

    def test_adjust_list_to_length(self):
        """
        Test :func:`mpi_array.benchmarks.utils.wlm.adjust_list_to_length`.
        """
        self.assertSequenceEqual(["hi", ], _wlm.adjust_list_to_length("hi", 1))
        self.assertSequenceEqual(["hi", ], _wlm.adjust_list_to_length(["hi", ], 1))
        self.assertSequenceEqual(["hi", "hi"], _wlm.adjust_list_to_length(["hi", ], 2))
        self.assertSequenceEqual(["hi", "hi", "hi"], _wlm.adjust_list_to_length(["hi", ], 3))
        self.assertSequenceEqual(
            ["hi", "bye", "bye"],
            _wlm.adjust_list_to_length(["hi", "bye"], 3)
        )
        self.assertSequenceEqual(
            ["hi", "bye", ],
            _wlm.adjust_list_to_length(["hi", "bye"], 2)
        )
        self.assertSequenceEqual(
            ["hi", ],
            _wlm.adjust_list_to_length(["hi", "bye"], 1)
        )

    def test_generate_scripts(self):
        """
        Test :meth:`mpi_array.benchmarks.utils.wlm.WlmScriptGenerator.generate_script_files`.
        """
        if self.tempdir is not None:
            subst_dict = \
                {
                    "num_cpus": [[1, 2, 4, 8, 16], 32, 48, 64],
                    "mem": ["32GB", "64GB", "96GB", "128GB"],
                    "wall_time": "00:59:59",
                    "queue": "express",
                }
            wlm_generator = \
                _wlm.WlmScriptGenerator(
                    subst_dict=subst_dict,
                    script_template=WLM_SCRIPT_TEMPLATE,
                    script_dir=self.tempdir
                )
            wlm_generator.generate_script_files()
            self.assertTrue(_os.path.exists(self.tempdir))
            script_file_list = \
                sorted([
                    _os.path.join(wlm_generator.script_dir, file_name)
                    for file_name in _os.listdir(wlm_generator.script_dir)
                ])
            self.assertEqual(4, len(script_file_list))
            for i in range(4):
                with open(script_file_list[i]) as f:
                    script_str = f.read()
                    num_cpus = subst_dict["num_cpus"][i]
                    max_num_cpus = _np.max(num_cpus)
                    mem = subst_dict["mem"][i]
                    wall_time = subst_dict["wall_time"]
                    queue = subst_dict["queue"]

                    self.assertLess(0, script_str.find("ncpus=%s" % max_num_cpus))
                    self.assertLess(0, script_str.find("mem=%s" % mem))
                    self.assertLess(0, script_str.find("walltime=%s" % wall_time))
                    self.assertLess(0, script_str.find("-q %s" % queue))
                    if isinstance(num_cpus, (list, tuple)):
                        self.assertLess(
                            0,
                            script_str.find("ncpus in " + " ".join([str(n) for n in num_cpus]))
                        )
                    else:
                        self.assertLess(0, script_str.find("ncpus in %s" % num_cpus))


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
