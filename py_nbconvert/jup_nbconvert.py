"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

jup_nbconvert.py
==============================================================================
"""
import argparse

from NBProcessor import NBProcessor

def main(args):
	args.format = 'notebook' if args.format == 'None' else args.format
	if (args.template == 'None' & args.format == 'html'):
		args.template = 'full' 
	elif (args.template == 'None' & (args.format == 'latex' | args.format == 'pdf')):
		args.template = 'article'
	else:
		args.template = None

	converter = nbConverter(input=args.path,
							to=args.format,
							template=args.template,
							inplace=args.inplace,
							allow_errors=args.errors)
	converter.run()

if __name__ == '__main__':
	parser = argparse.ArgumentParser("""Pre-processing CSV""")

	parser.add_argument('--input', '--data', action='store', required=True, dest='path',
	                    help='''(string) path to ipynb file (required parameter). Can be multiple, specified with spaces in-between.''')

	parser.add_argument('--project_dir', action='store', dest='project_dir', help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--output_dir', action='store', dest='output_dir', help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--to', action='store', dest='format', default='notebook', help="""(string) The type of format to convert the notebook into.
						The available options are:
						- notebook (Default): runs the notebook and saves output as a notebook.
						- html: exports the notebook as an html page.
						- latex: Latex export. This generates `NOTEBOOK_NAME.tex` file, ready for export. Images are output as .png files in a folder.
						- pdf: Generates a PDF via latex. 
						- slides: This generates a Reveal.js HTML slideshow.
						- markdown: Simple markdown output. Markdown cells are unaffected, and code cells indented 4 spaces. Images are output as .png files in a folder.
						- asciidoc: Ascii output. Images are output as .png files in a folder.
						- rst: Basic reStructuredText output. Useful as a starting point for embedding notebooks in Sphinx docs. Images are output as .png files in a folder.
						- script: Convert a notebook to an executable script. This is the simplest way to get a Python (or other language, depending on the kernel) script out of a notebook. If there were any magics in an Jupyter notebook, this may only be executable from a Jupyter session.""")

	parser.add_argument('--template', action='store', dest='format', default='None', help="""(string) For some formats, you can choose a specific template to use.
						The available options are:
						 - When using html format:
						 	- full (Default): A full static HTML render of the notebook. This looks very similar to the interactive view.
							- basic: Simplified HTML, useful for embedding in webpages, blogs, etc. This excludes HTML headers.
						 - When using latex or pdf format:
						 	- article (Default): Latex article, derived from Sphinx’s howto template.
							- basic: Latex report, providing a table of contents and chapters.""")


	parser.add_argument('--inplace', action='store', dest='inplace', default=False, help="""Overwrites input notebook with output. Only relevant for converting to notebook. Default is False.""")

	parser.add_argument('--allow-errors', action='store', dest='errors', default=False, help="""Continues conversion if errors encountered. Default is False. """)

	args = parser.parse_args()
	main(args)
