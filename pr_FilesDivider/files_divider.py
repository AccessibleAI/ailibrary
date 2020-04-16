"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

prep_lib.py
==============================================================================
"""
import argparse
from utils import parse_list


def main(args):
	path = args.path
	file_type = args.files_type
	test_size = float(args.test_size)
	validation_size = float(args.validation_size)
	divide_files_to_two_directories = \
		True if args.divide_files_to_two_directories == 'True' else False
	divide_sub_directories_to_two_directories = \
		True if args.divide_sub_directories_to_two_directories == 'True' else False
	group_files_to_directories_by_prefix = \
		None if args.group_files_to_directories_by_prefix == 'None' \
			else parse_list(args.group_files_to_directories_by_prefix)


if __name__ == '__main__':
	parser = argparse.ArgumentParser("""Pre-processing CSV""")

	parser.add_argument(
		'--path', action='store', required=True, dest='path',
		help='''(string) path to directory contains the files (required parameter).''')

	parser.add_argument(
		'--files_type', action='store', required=True, dest='files_type',
		help='''(string) The type of the files (required parameter).''')

	parser.add_argument(
		'--test_size', action='store', default='0.2', dest='test_size',
		help='''(float) (Default: 0.2) size of the test set, float number in [0, 1].''')

	parser.add_argument(
		'--validation_size', action='store', default='0.', dest='test_size',
		help='''(float) (Default: 0.) size of the validation set, float number in [0, 1].''')

	parser.add_argument(
		'--divide_files_to_two_directories', action='store', default='False', dest='divide_files_to_two_directories',
		help='''(bool) (Default: False) If True -> it splits directory with files to two/three
		sub-directories depended by giving the --validation_size param.''')

	parser.add_argument(
		'--divide_sub_directories_to_two_directories', action='store', default='0.', dest='divide_sub_directories_to_two_directories',
		help='''(bool) (Default: False) If True -> it splits directory with sub-directories to
		two/three sub-directories depended by giving the --validation_size param where each contains
		the original sub directories with relative amount of files.''')

	parser.add_argument(
		'--group_files_to_directories_by_prefix', action='store', default='None', dest='group_files_to_directories_by_prefix',
		help='''(string) (Default: None)  Groups files to directories by given prefixes. Example: ['dog','cat'] ->
		would create 2 sub directories of 'dog' and 'cat' and each would contain all the images 
		starts with the directory name.''')

	args = parser.parse_args()
	main(args)
