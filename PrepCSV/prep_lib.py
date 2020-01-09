"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

prep_lib.py
==============================================================================
"""
import argparse

from CSVProcessor import CSVProcessor

def main(args):
	args.target = None if args.target == 'None' else args.target
	args.missing_values = None if args.missing_values == 'None' else args.missing_values
	args.scale_dict = None if args.scale_dict == 'None' else args.scale_dict
	args.normalize_list = None if args.normalize_list == 'None' else args.normalize_list
	args.one_hot_list = None if args.one_hot_list == 'None' else args.one_hot_list
	args.output = None if args.output == 'None' else args.output

	processor = CSVProcessor(path_to_csv=args.path,
							target_column=args.target,
							missing_dict=args.missing_values,
							scale_dict=args.scale_dict,
							normalize_list=args.normalize_list,
							one_hot_list=args.one_hot_list,
							output_name=args.output)
	processor.run()

if __name__ == '__main__':
	parser = argparse.ArgumentParser("""Pre-processing CSV""")

	parser.add_argument('--path', action='store', required=True, dest='path',
	                    help='''(string) path to csv file (required parameter).''')
	parser.add_argument('--target', action='store', default='None', dest='target',
	                    help='''(string) The name of the target column. By default it takes the rightmost column in the given csv.''')
	parser.add_argument('--missing_values', action='store', default='None', dest='missing_values',
	                    help='''(dict) Dictionary describes what to do with empty, nan or NaN values in specific column.
						The structure of the dictionary is **{"COLUMN_NAME": "OPERATION"}** (the column name and the operation must be considered as strings even if they are numbers, dont worry - it is re-converted).
						The available operations are:
						- **fill_X** - where X is an integer or float number which the user wants to set the empty values to.
						- **drop** - drops the **rows** which has empty values in the specific column.
						- **avg** - sets the empty values in the column to the average of the column (the other values must be integers or floats).
						- **med** - sets the empty values in the column to the median of the column (the other values must be integers or floats).
						- **randint_A_B** - sets the empty values in the column to a random integer between A and B.''')
	parser.add_argument('--scale_dict', action='store', default='None', dest='scale_dict',
	                    help='''(dict) Dictionary describes a range which the user wants to scale the values of the column to.
						The structure of the dictionary is **{COLUMN_NAME: RANGE}**, where **RANGE** looks like: **A:B** (A,B are integers or floats).''')
	parser.add_argument('--normalize_list', action='store', default='None', dest='normalize_list',
	                    help='''  ''')
	parser.add_argument('--one_hot_list', action='store', default='None', dest='one_hot_list',
	                    help='''  ''')
	parser.add_argument('--output', action='store', default='None', dest='output',
	                    help='''   ''')
	args = parser.parse_args()
	main(args)
