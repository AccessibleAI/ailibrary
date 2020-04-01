"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

prep_lib.py
==============================================================================
"""
import argparse

from images_pre_processor import ImagesPreProcessor

def main(args):

	processor = ImagesPreProcessor()
	processor.run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser("""Pre-processing Images""")

	parser.add_argument('--path', '--data', action='store', required=True, dest='path',
	                    help='''(String) path to directory where the images are (required parameter).''')

	parser.add_argument('--project_dir', action='store', dest='project_dir', help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--output_dir', action='store', dest='output_dir', help="""--- For inner use of cnvrg.io ---""")



	args = parser.parse_args()
	main(args)
