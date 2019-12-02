"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

CnvrgPrep.py
==============================================================================
"""
import argparse
import numpy as np
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

	final_key = dict()
	import yaml
	parsed_dict = yaml.safe_load(dict_as_string)
	all_keys = parsed_dict.keys()
	for k in all_keys:
		true_key, true_value = k.split(':')
		true_key = true_key.strip()
		true_value = true_value.strip()
		final_key[true_key] = true_value
	return final_key


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
	if columns_list == '[all]':
		columns_list = list(data.columns)

	for col in columns_list:
		data[col] -= data[col].min()
		data[col] /= data[col].max()
	return data


def _perform_remapping(data, new_map):
	"""
	Performs the remapping.
	"""
	all_columns = data.columns
	set_of_keys = set(new_map.keys())
	for col in all_columns:
		if len(set_of_keys.intersection(set(data[col]))) > 0:
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

		if index_col == -1:
			index_col = data.shape[1] - 1

		target_col = data.iloc[:, index_col]
		data = data.drop(index=[index_col], axis=1)
		return data, target_col
	# If its a string name.
	except ValueError:
		target_col = data[index_col]
		data = data.drop(index=[index_col], axis=1)
		return data, target_col


def _perform_dropping(data, drop_list):
	"""
	Drops the columns in the list of columns from the dataframe.
	"""
	return data.drop(columns=drop_list, axis=1)


def _convert_all_to_numeric(data):
	columns = data.columns
	for col in columns:
		try:
			data[col] = np.float64(data[col])
		except TypeError or ValueError:
			raise Exception('The column {} contains not-numerical values.'.format(col))

	if True in np.isinf(data).any():
		raise Exception('Inf or -Ind found in data.')

	if True in np.isfinite(data).all():
		raise Exception('Not all values are finite. in data.')

	if True in data.isnull():
		raise Exception('Found NaN value in data.')

	return data


def main(args):
	# === Parse the input params.
	path = args.csv                                                                 # Parses the path.
	output_path = args.output                                                       # Parses the output path.
	index_col = args.index                                                          # Parses the target column index.
	one_hot_list = _parse_list(args.one_hot)                                        # Parses the list of columns for one-hot-encoding.
	remap_dict = _parse_dict(args.remap)                                            # Parses the dict of remapping values.
	drop_list = _parse_list(args.drop)                                              # Parses the list of columns to be dropped.
	empty_command = args.empty                                                      # Parses the command what to do with empty cells.
	scale_list = _parse_list(args.scale) if args.scale != '[all]' else args.scale   # Parses the list of columns to be scaled.

	# === Operations.
	data = pd.read_csv(path)                                                         # Reading data.
	data = _perform_remapping(data, remap_dict)                                      # Remapping values.
	data = _perform_empty_cells(data, empty_command)                                 # Perform operations on empty cells.
	data = _convert_all_to_numeric(data)                                             # Makes sure all value are numeric and finite.
	data, index_col = _perform_index_col(data, index_col)                            # Splits the index column and the rest of the feature columns.
	data = _perform_dropping(data, drop_list)                                        # Drops redundant columns.
	data = _perform_one_hot_encoding(data, one_hot_list)                             # Does one-hot-encoding.
	data = _perform_scaling(data, scale_list)                                        # Performs scaling to the selected columns.
	data = pd.concat([data, index_col], axis=1)                                      # attach the feature column to be the rightmost column.
	data.to_csv(output_path)                                                         # Saves the data.


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