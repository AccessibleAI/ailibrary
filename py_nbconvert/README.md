This library is made to use `jupyter nbconvert` to execute or convert `ipynb` files.  

## Notes for this library
This library supports exporting to many different formats (see below). It expects to receive the path to a `ipynb` file relative to the workdir (`/cnvrg`).

For example, if you have a file at a certain path from the workdir, type out that path in the `--notebooks` field.

Multiple notebooks can be converted at once by adding all the paths/notebooks space-separated in the `--notebooks` field.

## Parameters

```--notebooks``` - string, required. The path/s to the notebook/s. More than one notebook can be converted by including all their paths in the field separated by spaces.

```--to``` - string (default = "notebook"). The type of format to convert the notebook into.
    The available options are:
    - notebook (Default): runs the notebook and saves output as a notebook.
    - html: exports the notebook as an html page.
    - latex: Latex export. This generates `NOTEBOOK_NAME.tex` file, ready for export. Images are output as .png files in a folder.
    - pdf: Generates a PDF via latex. 
    - slides: This generates a Reveal.js HTML slideshow.
    - markdown: Simple markdown output. Markdown cells are unaffected, and code cells indented 4 spaces. Images are output as .png files in a folder.
    - asciidoc: Ascii output. Images are output as .png files in a folder.
    - rst: Basic reStructuredText output. Useful as a starting point for embedding notebooks in Sphinx docs. Images are output as .png files in a folder.
    - script: Convert a notebook to an executable script. This is the simplest way to get a Python (or other language, depending on the kernel) script out of a notebook. If there were any magics in an Jupyter notebook, this may only be executable from a Jupyter session.

```--template``` - string (default = None). For some formats, you can choose a specific template to use.
    The available options are:
        - When using html format:
          - full (Default): A full static HTML render of the notebook. This looks very similar to the interactive view.
          - basic: Simplified HTML, useful for embedding in webpages, blogs, etc. This excludes HTML headers.
        - When using latex or pdf format:
          - article (Default): Latex article, derived from Sphinx’s howto template.
          - basic: Latex report, providing a table of contents and chapters.

```--inplace``` - boolean (default = False). Overwrites input notebook with output. Only relevant for converting to notebook.

```--allow-errors``` - boolean (default = False). Continues conversion if errors encountered. 

## Links
https://nbconvert.readthedocs.io/en/latest/usage.html#