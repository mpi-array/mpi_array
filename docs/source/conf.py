"""
Sphinx configuration for the :mod:`mpi_array` documentation.

Build the HTML from the root of the source tree with::

   python -m sphinx -b html docs/source docs/_build/html

Add :samp:`-n` (nitpicky) to warn about references which aren't found.
"""
import os
import subprocess
import sys

conf_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(conf_dir, "..", ".."))
package_dir = os.path.join(root_dir, "mpi_array")

# Document the package of this source tree.
sys.path.insert(0, root_dir)


def read_package_file(file_name):
    """
    Returns the stripped text of a file of the :mod:`mpi_array` package directory.
    """
    with open(os.path.join(package_dir, file_name), "rt") as f:
        return f.read().strip()


def get_release(version):
    """
    Returns the :samp:`git describe` of the source, from :samp:`mpi_array/git_describe.txt`
    (written by :samp:`setup.py`) or from :samp:`git`, or :samp:`{version}` when neither
    is available.
    """
    try:
        return read_package_file("git_describe.txt")
    except OSError:
        pass
    try:
        return \
            subprocess.check_output(
                ["git", "describe", "--tags", "--always"],
                cwd=root_dir,
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
    except (OSError, subprocess.CalledProcessError):
        return version


# -- Project information ----------------------------------------------------------------------

project = "mpi_array"
author = "Shane J. Latham"
copyright = read_package_file("copyright.txt").replace("Copyright (C)", "").strip()
version = read_package_file("version.txt")
release = get_release(version)

# -- General configuration --------------------------------------------------------------------

needs_sphinx = "7.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    # Parses the README.md included in the docs.
    "myst_parser",
]
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
language = "en"
exclude_patterns = ["_templates*"]
pygments_style = "sphinx"
# The sections of the README.md included in the "about" page start at H2.
suppress_warnings = ["image.nonlocal_uri", "myst.header"]
todo_include_todos = True

# Generic types in the docstrings, which aren't objects to link to, and base classes
# which Python documents in other modules (nitpicky builds).
nitpick_ignore = \
    [
        ("py:class", name)
        for name in (
            "array_like", "callable", "file-like", "scalar", "sequence",
            "unittest.case.TestCase", "unittest.main.TestProgram",
            "unittest.runner.TextTestResult", "unittest.runner.TextTestRunner",
        )
    ]

# -- autodoc and autosummary ------------------------------------------------------------------

autosummary_generate = True
autodoc_default_options = {"show-inheritance": True}
# A class is described by its docstring and that of its __init__ (or __new__).
autoclass_content = "both"

# -- intersphinx ------------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
    "array_split": ("https://array-split.readthedocs.io/en/latest/", None),
}

# -- HTML output ------------------------------------------------------------------------------

try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
    html_theme_options = {"collapse_navigation": False}
except ImportError:
    html_theme = "alabaster"
htmlhelp_basename = "mpi_array_doc"

# -- Other output -----------------------------------------------------------------------------

latex_documents = [
    (root_doc, "mpi_array.tex", "mpi_array Documentation", author, "manual"),
]
man_pages = [
    (root_doc, "mpi_array", "mpi_array Documentation", [author], 1),
]
texinfo_documents = [
    (
        root_doc,
        "mpi_array",
        "mpi_array Documentation",
        author,
        "mpi_array",
        "Multi-dimensional distributed array.",
        "Miscellaneous",
    ),
]
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]
