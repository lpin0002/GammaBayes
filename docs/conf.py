# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'GammaBayes'
copyright = '2023, Liam Pinchbeck'
author = 'Liam Pinchbeck'
release = '0.0.42'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx_rtd_theme','myst_parser','nbsphinx','sphinx.ext.napoleon']


master_doc = 'index'

nbsphinx_execute = 'never'  # Set to 'never' if you want to use static notebooks


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

source_suffix = '.rst'



nbsphinx_prolog = """
.. role:: raw-html(raw)
   :format: html

.. nbinfo::

   This page is a tutorial for the <a href="https://github.com/lpin0002/GammaBayes">GammaBayes</a> 
   and the original notebook can be found within the docs/tutorials folder within the GitHub repository.

"""