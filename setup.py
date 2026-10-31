#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import subprocess


def read_readme():
    """
    Reads part of the README.md for use as long_description in setup().
    """
    text = open("README.md", "rt").read()
    text_lines = text.split("\n")
    ld_i_beg = 0
    while text_lines[ld_i_beg].find("start long description") < 0:
        ld_i_beg += 1
    ld_i_beg += 1
    ld_i_end = ld_i_beg
    while text_lines[ld_i_end].find("end long description") < 0:
        ld_i_end += 1

    ld_text = "\n".join(text_lines[ld_i_beg:ld_i_end])

    return ld_text


def create_git_describe():
    cmd = ["git", "describe"]
    try:
        p = \
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            e = \
                subprocess.CalledProcessError(
                    returncode=p.returncode,
                    cmd=cmd
                )
            setattr(e, "output", " ".join([stdout.decode(), stderr.decode()]))

            raise e
        # Write the git describe to text file
        open(os.path.join("mpi_array", "git_describe.txt"), "wt").write(stdout.decode())
    except (Exception,) as e:
        # Try and make up a git-describe like string
        # (e.g. git is not installed, or not a git repository).
        print("Problem with '%s': %s: %s" % (" ".join(cmd), e, getattr(e, "output", "")))
        version_str = open(os.path.join("mpi_array", "version.txt"), "rt").read().strip()
        if ("TRAVIS_TAG" in os.environ.keys()) and (len(os.environ["TRAVIS_TAG"]) > 0):
            version_str = os.environ["TRAVIS_TAG"]
        else:
            if ("TRAVIS_BRANCH" in os.environ.keys()) and (len(os.environ["TRAVIS_BRANCH"]) > 0):
                version_str += os.environ["TRAVIS_BRANCH"]
            if ("TRAVIS_COMMIT" in os.environ.keys()) and (len(os.environ["TRAVIS_COMMIT"]) > 0):
                version_str += "-" + \
                    os.environ["TRAVIS_COMMIT"][0:min([7, len(os.environ["TRAVIS_COMMIT"])])]
        open(os.path.join("mpi_array", "git_describe.txt"), "wt").write(version_str)

create_git_describe()

_long_description = read_readme()

mpi4py_requires = ["mpi4py>=3.1", ]

if "READTHEDOCS" in os.environ.keys():
    # Skip mpi4py dependency, can't install an MPI implementation
    # in readthedocs.org virtual machine.
    mpi4py_requires = []

setup(
    name="mpi_array",
    version=open("mpi_array/version.txt", "rt").read().strip(),
    packages=find_packages(),
    # metadata for upload to PyPI
    author="Shane J. Latham",
    author_email="mpi.array@gmail.com",
    description=(
        "Python package providing PGAS multi-dimensional array with numpy-like API."
    ),
    long_description=_long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords=(
        "MPI mpi4py data-parallelism distributed numpy ndarray distributed-array"
        +
        " distributed-ndarray scipy domain-decomposition array-decomposition"
    ),
    url="http://github.com/mpi-array/mpi_array",   # project home page
    classifiers=[
        # How mature is this project? Common values are
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Utilities',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Natural Language :: English',
    ],
    python_requires=">=3.10",
    install_requires= \
        [
            "numpy>=2.0",
            "array_split>=0.4.0",
            "psutil>=4.3.0",
        ]
        + mpi4py_requires,
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme", "myst-parser", ],
    },
    package_data={
        "mpi_array": ["version.txt", "git_describe.txt", "copyright.txt", "license.txt"]
    },
    test_suite="mpi_array.tests",
    scripts=[os.path.join("utils", "mpi_array_info.py"),]
    # could also include download_url, etc.
)
