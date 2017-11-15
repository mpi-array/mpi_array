"""
Generate *workload-mangager* (e.g. PBS Pro, Slurm, Torque) benchmark batch scripts.
"""
from __future__ import absolute_import
import copy as _copy
import os as _os
import numpy as _np
from ...license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()


def adjust_list_to_length(obj, desired_length):
    """
    Returns a :obj:`list` of length :samp:`{desired_length}`,
    by extending or truncating the :samp:`obj` :obj:`list`.
    The sequence is extended by repeating the last element
    of :samp:`{obj}`.

    :type obj: sequence or :obj:`str`
    :param obj: The sequence to be extended/truncated.
       If :samp:`{obj}` is a string, it is converted to
       a list using :samp:`{obj} = [{obj}, ]`.
    :type desired_length: :obj:`int`
    :param desired_length: The length of the returned :obj:`list`.
    :rtype: :obj:`list`
    :return: A :obj:`list` of length :samp:`{desired_length}
       which contains the elements of the :samp:`{obj}` sequence.
    """
    if isinstance(obj, str):
        obj = [obj, ]
    ret_list = list(obj)
    if len(ret_list) < desired_length:
        ret_list = ret_list + [ret_list[-1], ] * (desired_length - len(ret_list))
    elif len(ret_list) > desired_length:
        ret_list = ret_list[:desired_length]

    return ret_list


class WlmScriptGenerator(object):
    """
    Generates multiple script files from a template.
    """

    def __init__(
        self,
        subst_dict,
        script_template,
        script_base_name="mpia_%(max_num_cpus)04d_bench",
        script_ext=".sh",
        script_dir="."
    ):
        """
        Initialise.
        """
        self._subst_dict = subst_dict
        self._script_template = script_template
        self._script_base_name = script_base_name
        self._script_ext = script_ext
        self._script_dir = script_dir

    @property
    def subst_dict(self):
        """
        A :obj:`dict` with the string substitutions for the :attr:`script_template` template.
        """
        return self._subst_dict

    @property
    def script_dir(self):
        """
        A :obj:`str` indicating the directory where script files are written.
        """
        return self._script_dir

    @property
    def script_template(self):
        """
        A :obj:`str` which gets written to file after substitutions from :attr:`subst_dict`.
        """
        return self._script_template

    @property
    def script_base_name(self):
        """
        A :obj:`str` with the *base name* part of the script file name.
        """
        return self._script_base_name

    @property
    def script_ext(self):
        """
        A :obj:`str` with the *extension* part of the script file name.
        """
        return self._script_ext

    def update_dict_max_num_cpus(self, subst_dict):
        """
        Add an entry for the :samp:`"max_num_cpus"` key in :samp:`{subst_dict}`
        if it does not already exist.

        :type subst_dict: :obj:`dict`
        :param subst_dict: Substitution dictionary.
        """
        subst_dict["max_num_cpus"] = _np.max(subst_dict["num_cpus"])

    def update_dict_num_cpus_string(self, subst_dict):
        """
        Replaces :samp:`"num_cpus"` list of values in :samp:`{subst_dict}`
        with a space separated string.

        :type subst_dict: :obj:`dict`
        :param subst_dict: Substitution dictionary.
        """
        if (
            ("num_cpus" in subst_dict)
            and
            (
                hasattr(subst_dict["num_cpus"], "__iter__")
                or
                hasattr(subst_dict["num_cpus"], "__getitem__")
            )
        ):
            subst_dict["num_cpus"] = " ".join([str(ncpu) for ncpu in list(subst_dict["num_cpus"])])

    def generate_script_file_name(self, subst_dict):
        """
        Returns the name of the script file.

        :type subst_dict: :obj:`dict`
        :param subst_dict: Substitution dictionary, substitutions made in :attr:`script_base_name`.
        :rtype: :obj:`str`
        :return: Name of script file.
        """
        return \
            _os.path.join(
                self.script_dir,
                (self.script_base_name % subst_dict) + self.script_ext
            )

    def generate_script_string(self, subst_dict):
        """
        Returns the text which is to be written to a script file.

        :type subst_dict: :obj:`dict`
        :param subst_dict: Substitution dictionary, substitutions made in :attr:`script_template`.
        :rtype: :obj:`str`
        :return: Text to be written to script file.
        """
        return (self.script_template % subst_dict)

    def generate_script_files(self):
        """
        Generates the script files.
        """
        subst_dict = _copy.copy(self.subst_dict)

        num_ncpus_list = subst_dict["num_cpus"]
        num_script_files = len(num_ncpus_list)
        mem_list = adjust_list_to_length(subst_dict["mem"], num_script_files)
        wall_time_list = adjust_list_to_length(subst_dict["wall_time"], num_script_files)

        for num_cpus, mem, wall_time in zip(num_ncpus_list, mem_list, wall_time_list):
            subst_dict["num_cpus"] = num_cpus
            subst_dict["mem"] = mem
            subst_dict["wall_time"] = wall_time
            self.update_dict_max_num_cpus(subst_dict)
            self.update_dict_num_cpus_string(subst_dict)
            script_str = self.generate_script_string(subst_dict)
            script_file_name = self.generate_script_file_name(subst_dict)
            with open(script_file_name, 'wt') as f:
                f.write(script_str)
