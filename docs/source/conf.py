# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'iamap'
copyright = '2024, TRESSON Paul, TULET Hadrien, LE COZ Pierre'
author = 'TRESSON Paul, TULET Hadrien, LE COZ Pierre'
release = '0.5.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import pydata_sphinx_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "myst_parser",
    "sphinx_favicon",
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_favicon = "./../../icons/encoder.svg"

