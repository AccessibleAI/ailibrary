"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

CnvrgPrep.py
==============================================================================
"""
import argparse
import pandas as pd


def _parse_list(list_as_string):
	"""
	Parses string of list to actual list.
	Returns either a list of all integers or list of all strings.
	"""
	if list_as_string == '[]':
		return []

	list_without_parenthesis = list_as_string.strip()[1: -1]
	parsed_list = [st.strip() for st in list_without_parenthesis.split(',')]

	# Check if the values are columns numbers.
	try:
		parsed_list = [int(st) for st in parsed_list]
	except ValueError:
		pass

	return parsed_list


def _parse_dict(dict_as_string):
	"""
	Parses strings of dict to actual dict.
	"""
	if dict_as_string == '{}':
		return {}

	import json
	parsed_dict = json.loads(dict_as_string)

	return parsed_dict


def _perform_one_hot_encoding(data, columns_list):
	"""
	Performs the one hot encoding.
	"""
	data = pd.get_dummies(data, columns=columns_list)
	return data


def _perform_scaling(data, columns_list):
	"""
	Performs the scaling.
	"""
	for col in columns_list:
		data[col] -= data[col].min()
		data[col] /= data[col].max()
	return data


def _perform_remapping(data, new_map):
	"""
	Performs the remapping.
	"""
	all_columns = data.columns
	for col in all_columns:
		data[col] = data[col].map(new_map)
	return data


def _perform_empty_cells(data, command):
	"""
	Performs dealing with empty cells and Nans.
	"""
	if command == 'r':
		data = data.dropna()

	elif command == 'z':
		data = data.fillna(0)

	else:  # Numerical value.
		numerical_value = float(command)
		data = data.fillna(numerical_value)

	return data


def _perform_index_col(data, index_col):
	"""
	Splits the given dataset to the features columns and the target column.
	"""

	# If it is a numerical index.
	try:
		index_col = int(index_col)
		target_col = data.iloc[:, index_col]
		target_col_header = target_col.index
		data = data.drop(columns=[target_col_header], axis=1)
	# If its a string name.
	except ValueError:
		target_col = data[index_col]
		data = data.drop(columns=[index_col], axis=1)
	return data, target_col


def _perform_dropping(data, drop_list):
	"""
	Drops the columns in the list of columns from the dataframe.
	"""
	return data.drop(columns=drop_list, axis=1)


def main(args):
	path = args.csv
	output_path = args.output
	index_col = args.index
	one_hot_list = _parse_list(args.one_hot)
	scale_list = _parse_list(args.scale)
	remap_dict = _parse_dict(args.remap)
	drop_list = _parse_list(args.drop)
	empty_command = args.empty

	data = pd.read_csv(path)
	data, index_col = _perform_index_col(data, index_col)
	data = _perform_dropping(data, drop_list)
	data = _perform_one_hot_encoding(data, one_hot_list)
	data = _perform_scaling(data, scale_list)
	# data = _perform_remapping(data, remap_dict)
	data = _perform_empty_cells(data, empty_command)

	data = pd.concat([data, index_col], axis=1)

	data.to_csv(output_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser("""Preprocessing CSV Script""")

	parser.add_argument('--csv', action='store', required=True, dest='csv',
	                    help='''(String) Path to local csv file.''')

	parser.add_argument('--output', action='store', default='Preprocessed.csv', dest='output',
	                    help='''(String) Path to the location the output file should be saved at. **Important** - should be ended with '.csv'.''')

	parser.add_argument('--index_col', action='store', default='-1', dest='index',
	                    help='''(String or Integer) The name or the column number of the index column.''')

	parser.add_argument('--one_hot', action='store', default='[]', dest='one_hot',
	                    help='''(list) a list contains all the columns names the user wants to one-hot-encode. Default value: '[]' (empty list).''')

	parser.add_argument('--scale', action='store', default='[]', dest='scale',
	                    help='''(list) a list contains all the columns names the user wants to scale. Default value: '[]' (empty list).''')

	parser.add_argument('--remap', action='store', default='{}', dest='remap',
	                    help='''(dict) a dictionary with from the structure ```{VALUE: NEW_VALUE}```, **Important** - it replaces all the values with the name value in the dataset without noticing 
	                    between columns. Default value: '{}' (empty dict).''')

	parser.add_argument('--drop', action='store', default='[]', dest='drop',
	                    help='''(list) a list contains all the columns names the user wants to drop. Default value: '[]' (empty list).''')

	parser.add_argument('--empty', action='store', default='z', dest='empty',
	                    help='''This field all deals with ```na``` or ```NaN``` values. 
						Multiple cases: 
						(-) If the user wishes to remove the lines contains those: ```--empty='r'```
						(-) If the user wishes to fill the lines contains those with 0: ```--empty='z'```
						(-) If the user wishes to fill the lines contains those with any other numeric value: ```--empty='NUMERIC_VAL'```
						(-) NOT AVAILABLE NOW: If the user wishes to replace the empty or nan values differently in each column, the user needs to give a dictionary like: ```--empty={COLUMN_NAME: 'VALUE'}```
						default value: 'z' (fill values with zeros).''')

	args = parser.parse_args()

	main(args)