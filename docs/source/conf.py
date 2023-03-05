"""Sphinx configuration."""

import inspect
import os
import sys

import respace

project = "ReSpace"
author = "Thomas Louf"
copyright = "2023, Thomas Louf"

# The full version, including alpha/beta/rc tags
release = str(respace.__version__)

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "myst_nb",
    "hoverxref.extension",
]

python_use_unqualified_type_names = True
autodoc_typehints_format = "short"
numpydoc_show_class_members = False

hoverxref_auto_ref = True
hoverxref_domains = ["py"]
hoverxref_role_types = dict.fromkeys(
    ["ref", "class", "func", "meth", "attr", "exc", "data", "obj"],
    "tooltip",
)
hoverxref_tooltip_lazy = True
# these have to match the keys on intersphinx_mapping, and those projects must be hosted on readthedocs.
hoverxref_intersphinx = [
    "python",
    "xarray",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/TLouf/respace",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    fn = os.path.relpath(fn, start=os.path.dirname(respace.__file__))

    return f"https://github.com/TLouf/respace/blob/master/src/respace/{fn}{linespec}"
