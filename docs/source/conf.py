# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re


metadata_file_path = os.path.join('..', '..', 'metadata.txt')
metadata_file_path = os.path.abspath(metadata_file_path)
with open(metadata_file_path, 'rt') as file:
    file_content = file.read()
try:
    versions_from_metadata = re.findall(r'version=(.*)', file_content)[0]
except Exception as e:
    raise Exception("Failed to read version from metadata!")

try:
    author_from_metadata = re.findall(r'author=(.*)', file_content)[0]
except Exception as e:
    raise Exception("Failed to read author from metadata!")

try:
    name_from_metadata = re.findall(r'name=(.*)', file_content)[0]
except Exception as e:
    raise Exception("Failed to read name from metadata!")

project = "iamap"
copyright = "2024, TRESSON Paul, TULET Hadrien, LE COZ Pierre"
author = author_from_metadata
# The short X.Y version
version = versions_from_metadata
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration





extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "./../../icons/favicon.svg"
