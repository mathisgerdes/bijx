# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_pkg_version
from pathlib import Path

# Make the package source importable without installation
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent.parent  # <repo>/docs/source -> up twice to repo root
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

project = "bijx"
copyright = "2025, Mathis Gerdes"
author = "Mathis Gerdes"

# Resolve project version without importing the package to avoid heavy deps during config
try:
    _release = get_pkg_version("bijx")
except PackageNotFoundError:
    # Fallback when package is not installed in the docs build env
    _release = os.environ.get("BIJX_DOCS_VERSION", "0.0.dev0")
release = ".".join(_release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_math_dollar",  # Load first to process dollar math before autodoc
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


mathjax3_config = {
    "loader": {"load": ["[tex]/physics"]},
    "tex": {"packages": {"[+]": ["physics"]}},
}


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
autodoc_typehints_format = "short"

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

pygments_style = "tango"
pygments_dark_style = "monokai"

# -- Favicon and logo settings ----------------------------------------------
html_favicon = "_static/icons/favicon.ico"
html_logo = "_static/icons/bijx.svg"

# -- Furo theme specific settings -------------------------------------------
html_theme_options = {
    "sidebar_hide_name": True,  # Hide project name since we have logo
    "source_repository": "https://github.com/mathisgerdes/bijx/",
    "source_branch": "master",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/mathisgerdes/bijx",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
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

# MyST configuration (dollarmath disabled for docstrings, handled by sphinx-math-dollar)
myst_dmath_allow_space = True
myst_dmath_allow_digits = True
myst_dmath_allow_labels = True
myst_update_mathjax = False  # Let sphinx-math-dollar handle MathJax config

myst_heading_anchors = 3
