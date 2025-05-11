# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'CONFLUENCE'
copyright = '2025, Darri Eythorsson'
author = 'Darri Eythorsson'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'myst_parser',  # For Markdown support
]

# Theme
html_theme = 'sphinx_rtd_theme'

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Add support for both RST and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_context = {
    'display_github': True,
    'github_user': 'DarriEy',
    'github_repo': 'CONFLUENCE',
    'github_version': 'main/docs/source/',
}
