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

	parser.add_argument('--grayscale', '--g', action='store', dest='grayscale', default='False',
	                    help='''(bool) (Default: False) if the value is True, it turns all images to grayscale.''')

	parser.add_argument('--noise', action='store', dest='noise', default='None',
	                    help='''(String) (Default: None) Represents types of noises can be added to images. 
	                    Options are: (1) gaussian (2) s&p (3) speckle (4) poisson.''')

	parser.add_argument('--blur', '--b', action='store', dest='blur', default='0',
	                    help='''(int) (Default: 0) Size of the squared kernel for gaussian blur.''')

	parser.add_argument('--convolve', action='store', dest='convolve', default='None',
	                    help='''(List) (Default: None) long list of lists of numbers which represents 2d squared array for the 
	                    convolution to apply.''')

	parser.add_argument('--zip', '--z', action='store', dest='zip_all', default='False',
	                    help='''(bool) (Default: False) if the value is True, it zips all the images to a zip named by the given directory name.''')

	parser.add_argument('--cnvrg_dataset_url', action='store', dest='cnvrg_ds', default='None',
	                    help='''(String) (Default: None) cnvrg dataset url to push to created images to.''')

	args = parser.parse_args()
	main(args)
