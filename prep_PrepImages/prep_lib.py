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

	parser.add_argument('--project_dir', action='store', dest='project_dir',
						help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--output_dir', action='store', dest='output_dir',
						help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--height', '-h', action='store', dest='height', default='None',
	                    help='''(int) (Default: None) new height for resizing.''')

	parser.add_argument('--width', '-w', action='store', dest='width', default='None',
	                    help='''(int) (Default: None) new width for resizing.''')

	parser.add_argument('--channels', '-c', action='store', dest='channels', default='None',
	                    help='''(int) (Default: None) new num of channels for resizing.''')

	parser.add_argument('--add_noise', '-a', action='store', dest='add_noise', default='False',
	                    help='''(bool) (Default: False) if the value is True, it adds noise to all images.''')

	parser.add_argument('--denoise', '-d', action='store', dest='denoise', default='False',
	                    help='''(bool) (Default: False) if the value is True, it runs de-noising algorithm over all images.''')

	parser.add_argument('--segmentation', '-s', action='store', dest='segmentation', default='False',
	                    help='''(bool) (Default: False) if the value is True, it runs segmentation algorithm over all images.''')

	parser.add_argument('--blur', '-b', action='store', dest='blur', default='False',
	                    help='''(bool) (Default: False) if the value is True, it runs blurring algorithm over all images.''')

	parser.add_argument('--zip_all', '-z', action='store', dest='zip_all', default='False',
	                    help='''(bool) (Default: False) if the value is True, it zips all the images to a zip named by the given directory name.''')

	parser.add_argument()

	args = parser.parse_args()
	main(args)
