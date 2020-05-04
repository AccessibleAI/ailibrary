"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

prep_lib.py
==============================================================================
"""
import argparse
import utils


def main(args):
	data_dir = args.path_to_data
	file_type = args.files_type
	test_size = float(args.test_size)
	validation_size = float(args.validation_size)
	divide_files_to_two_directories = True if args.divide_files_to_two_directories == 'True' else False
	divide_sub_directories_to_two_directories = True if args.divide_sub_directories_to_two_directories == 'True' else False
	group_files_to_directories_by_prefix = True if args.group_files_to_directories_by_prefix == 'True' else False

	if divide_files_to_two_directories:
		utils.divide_files_to_two_directories(
			data_dir=data_dir,
			file_type=file_type,
			test_size=test_size,
			validation_size=validation_size)
		return

	elif divide_sub_directories_to_two_directories:
		utils.divide_sub_directories_to_two_directories(
			data_dir=data_dir,
			file_type=file_type,
			test_size=test_size,
			validation_size=validation_size)
		return

	elif group_files_to_directories_by_prefix:
		utils.group_files_to_directories_by_prefix(
			data_dir=data_dir,
			file_type=file_type)
		return

	else:
		print("No operation has been conducted.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser("""Files Divider""")

	parser.add_argument(
		'--path_to_data', action='store', required=True, dest='path_to_data',
		help='''(string) path_to_data to directory contains the files (required parameter).''')

	parser.add_argument(
		'--files_type', action='store', required=True, dest='files_type',
		help='''(string) The type of the files (required parameter).''')

	parser.add_argument(
		'--test_size', action='store', default='0.2', dest='test_size',
		help='''(float) (Default: 0.2) size of the test set, float number in [0, 1].''')

	parser.add_argument(
		'--validation_size', action='store', default='0.', dest='validation_size',
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
		'--group_files_to_directories_by_prefix', action='store', default='False', dest='group_files_to_directories_by_prefix',
		help='''(bool) (Default: False)  Groups files to directories by given prefixes. Example: ['dog','cat'] ->
		would create 2 sub directories of 'dog' and 'cat' and each would contain all the images 
		starts with the directory name.''')

	args = parser.parse_args()
	main(args)
