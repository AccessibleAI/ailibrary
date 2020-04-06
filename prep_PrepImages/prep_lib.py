"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

prep_lib.py
==============================================================================
"""
import argparse
from images_pre_processor import ImagesPreProcessor


def main(args):

	processor = ImagesPreProcessor(args)
	processor.run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser("""Pre-processing Images""")

	parser.add_argument('--path', action='store', required=True, dest='path',
	                    help='''(String) path to directory where the images are (required parameter).''')

	parser.add_argument('--project_dir', action='store', dest='project_dir',
						help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--output_dir', action='store', dest='output_dir',
						help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--height', '--h', action='store', dest='height', default='None',
	                    help='''(int) (Default: None) new height for resizing.''')

	parser.add_argument('--width', '--w', action='store', dest='width', default='None',
	                    help='''(int) (Default: None) new width for resizing.''')

	parser.add_argument('--grayscale', '--g', action='store', dest='grayscale', default='None',
	                    help='''(bool) (Default: False) if the value is True, it turns all images to grayscale.''')

	parser.add_argument('--noise', action='store', dest='noise', default='None',
	                    help='''(String) (Default: None) Represents types of noises can be added to images. 
	                    Options are:''')

	parser.add_argument('--denoise', '--d', action='store', dest='denoise', default='False',
	                    help='''(bool) (Default: False) if the value is True, it runs de-noising algorithm over all images.''')

	parser.add_argument('--segmentation', '--s', action='store', dest='segmentation', default='False',
	                    help='''(bool) (Default: False) if the value is True, it runs segmentation algorithm over all images.''')

	parser.add_argument('--blur', '--b', action='store', dest='blur', default='False',
	                    help='''(bool) (Default: False) if the value is True, it runs blurring algorithm over all images.''')

	parser.add_argument('--zip_all', '--z', action='store', dest='zip_all', default='False',
	                    help='''(bool) (Default: False) if the value is True, it zips all the images to a zip named by the given directory name.''')

	args = parser.parse_args()
	main(args)
