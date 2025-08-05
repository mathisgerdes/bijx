# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "bijx"
copyright = "2025, Mathis Gerdes"
author = "Mathis Gerdes"

from bijx import __version__

# Clean up development version strings like "0.1.dev57+g4e39c13.d20250126"
if "dev" in __version__:
    # For dev versions, just show "0.1-dev"
    base_version = __version__.split(".dev")[0]
    release = f"{base_version}-dev"
    version = base_version
else:
    release = __version__
    version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]


# Custom function to handle autodoc warnings and improve API cleanliness
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Skip all methods starting with _
    if name.startswith("_"):
        return True

    # Skip nnx.Module inherited methods that clutter the docs
    nnx_module_methods = {
        "set_attributes",
        "apply_gradients",
        "replace_params",
        "replace_grads",
        "split",
        "merge",
        "update",
        "pop",
        "tree_flatten",
        "tree_unflatten",
        "clear",
        "copy",
        "items",
        "keys",
        "values",
        "graphdef",
        "set_graphdef",
        "get_graphdef",
        "partition",
        "iter_modules",
        "train",
        "eval",
        "sow",
        "perturb",
        "iter_children",
    }
    if name in nnx_module_methods:
        return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)


# -- Autosummary settings ---
autosummary_generate = True
autosummary_ignore_module_all = False

# -- Numpydoc settings ---
numpydoc_show_class_members = False

# -- Autodoc settings for better API documentation ---
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Include class inheritance information
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# -- Favicon and logo settings ----------------------------------------------
html_favicon = "_static/icons/favicon.ico"
html_logo = "_static/icons/bijx.svg"

# -- Furo theme specific settings -------------------------------------------
html_theme_options = {
    "sidebar_hide_name": True,  # Hide project name since we have logo
    "source_repository": "https://github.com/mathisgerdes/bijx/",
    "source_branch": "master",
    "source_directory": "docs/source/",
}

# -- Custom CSS and JS files ------------------------------------------------
html_css_files = [
    "custom.css",
]

# -- MyST-NB settings --------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html
nb_execution_mode = "off"  # "auto"
nb_execution_timeout = 600
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "substitution",
]
myst_url_schemes = ["http", "https", "mailto"]
