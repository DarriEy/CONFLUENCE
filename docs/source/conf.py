# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CONFLUENCE'
copyright = '2025, Darri Eythorsson'
author = 'Darri Eythorsson'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# List of Sphinx extensions to use
extensions = [
    'sphinx.ext.autodoc',    # Automatically include documentation from docstrings.
    'sphinx.ext.napoleon',   # Support for Google and NumPy style docstrings.
    'sphinx.ext.viewcode',   # Add links to highlighted source code.
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the theme to Read the Docs style
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


autodoc_mock_imports = ["rasterio"]