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
    "myst_parser",
]
autodoc_typehints = "description"

autosummary_generate = True

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/TLouf/respace",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
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

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(respace.__file__))

    return f"https://github.com/TLouf/respace/blob/master/respace/{fn}{linespec}"

