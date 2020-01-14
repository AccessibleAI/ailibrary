"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

CSVProcessor.py
==============================================================================
"""
import json
import sys
import time

import yaml
import warnings
import numpy as np
import pandas as pd

class CSVProcessor:
	def __init__(self,
				 path_to_csv,
				 target_column=None,
				 missing_dict=None,
				 scale_dict=None,
				 normalize_list=None,
				 one_hot_list=None,
				 output_name=None):
		"""
		:param path_to_csv: string
		:param target_column: string
		:param missing_dict: dict
		:param scale_dict: dict
		:param normalize_list: list
		:param one_hot_list: list
		:param output_name: string
		"""
		self.__data = pd.read_csv(path_to_csv, index_col=0)
		self.__target_column = (target_column, self.__data[target_column]) if target_column is not None else (self.__data.columns[-1], self.__data[self.__data.columns[-1]])
		self.__features = [f for f in list(self.__data.columns) if f != self.__target_column[0]]
		self.__data = self.__data[self.__features]  #  remove the target column.

		self.__normalize_list = CSVProcessor.__parse_list(normalize_list) if isinstance(normalize_list, str) else normalize_list
		self.__one_hot_list = CSVProcessor.__parse_list(one_hot_list)  if isinstance(one_hot_list, str) else one_hot_list
		self.__missing_dict = CSVProcessor.__parse_dict(missing_dict) if isinstance(missing_dict, str) else missing_dict
		self.__scale_dict = CSVProcessor.__parse_dict(scale_dict) if isinstance(scale_dict, str) else scale_dict
		self.__output_name = output_name if output_name is not None else path_to_csv.split('.csv')[0] + '_processed.csv'

	def run(self):
		self.__handle_missing()
		self.__one_hot_encoding()
		self.__scale()
		self.__normalize()
		self.__set_target_column()
		self.__save()
		print('CSVProcessor: all tasks handled')
		self.__check_nulls_before_output()

	def __scale(self):
		if self.__scale_dict is not None:
			scale_all = False
			if set(self.__scale_dict.keys()) == set('all'): scale_all = True
			columns_to_scale = self.__features if scale_all is True else self.__scale_dict.keys()
			for col in columns_to_scale:
				curr_min, curr_max = (self.__data[col].min(), self.__data[col].max())
				new_min, new_max = (self.__data[col].min(), self.__data[col].max()) if scale_all else self.__scale_helper(self.__scale_dict[col])
				self.__data[col] = (((new_max - new_min) * (self.__data[col] - curr_min)) / (curr_max - curr_min)) + new_min
		print('CSVProcessor: scaling handled')

	def __normalize(self):
		if self.__normalize_list is not None:
			normalize_all = False
			if set(self.__normalize_list) == set('all'): normalize_all = True

			columns_to_scale = self.__features if normalize_all is True else self.__normalize_list
			for col in columns_to_scale:
				min_range, max_range = self.__data[col].min(), self.__data[col].max()
				self.__data[col] -= min_range
				self.__data[col] /= max_range
		print('CSVProcessor: normalization handled')


	def __one_hot_encoding(self):
		if self.__one_hot_list is not None:
			self.__data = pd.get_dummies(self.__data, columns=self.__one_hot_list)
		print('CSVProcessor: one-hot encoding handled')

	def __handle_missing(self):
		"""
		Options:
		1) fill_X (fill with value x)
		2) drop
		3) avg (fill with avg)
		4) med (short of median)
		5) rand_A_B (fill with random value in range [A,B]
		"""
		if self.__missing_dict is not None:
			handle_all, task_all = False, None
			if set(self.__missing_dict.keys()) == set('all'): handle_all, task_all = True, self.__missing_dict['all']
			column_to_handle = self.__features if handle_all is True else self.__missing_dict.keys()

			for col in column_to_handle:
				task = task_all if task_all is not None else self.__missing_dict[col][0]
				if task.startswith('fill_'):
					value = float(task[len('fill_'):]) if '.' in task[len('fill_'):] else int(task[len('fill_') :])
					self.__data[col] = self.__data[col].fillna(value)
				elif task.startswith('drop'):
					self.__data = self.__data[self.__data[col].notna()]
				elif task.startswith('avg'):
					self.__data[col] = self.__data[col].fillna(self.__data[col].mean())
				elif task.startswith('med'):
					self.__data[col] = self.__data[col].fillna(self.__data[col].median())
				elif task.startswith('randint_'):
					a, b = task[len('randint_'):].split('_')
					a, b = float(a) if '.' in a else int(a), float(b) if '.' in b else int(b)
					self.__data[col] = self.__data[col].fillna(np.random.randint(a, b))
				else: raise Exception('MissingValuesHandlingError: Undefined task.')
		print('CSVProcessor: missing values handled.')

	def __set_target_column(self):
		self.__data[self.__target_column[0]] = self.__target_column[1]
		print('CSVProcessor: target columns handled.')

	def __save(self):
		self.__data.to_csv(self.__output_name)
		print('CSVProcessor: model saved.')

	def __scale_helper(self, value):
		min_val, max_val = value.split(':') if isinstance(value, str) else value[0], value[1]
		min_val = float(min_val) if '.' in min_val else int(min_val)
		max_val = float(max_val) if '.' in max_val else int(max_val)
		return min_val, max_val

	def __check_nulls_before_output(self):
		# Check empty and nan values to warn the user.
		time.sleep(8)
		nulls_report = dict(self.__data.isnull().sum())
		features_with_null_values = [k for k, v in nulls_report.items() if v != 0]
		if len(features_with_null_values) != 0:
			warnings.warn("Null values or empty cells in the data set.", UserWarning)
		return

	@staticmethod
	def __parse_list(list_as_string):
		if list_as_string == '[]': return []

		list_without_parenthesis = list_as_string.strip()[1: -1]
		parsed_list = [st.strip() for st in list_without_parenthesis.split(',')]

		# Check if the values are columns numbers.
		try: parsed_list = [int(st) for st in parsed_list]
		except ValueError: pass

		return parsed_list

	@staticmethod
	def __parse_dict(dict_as_string):
		if dict_as_string == '{}': return {}
		final_key = dict()
		parsed_dict = eval(dict_as_string)
		if not isinstance(parsed_dict, dict): raise TypeError('Given a {} instead of dictionary.'.format(type(parsed_dict)))
		all_keys = parsed_dict.keys()
		for k in all_keys:
			true_key, true_value = k, parsed_dict[k].split(':')
			true_key = true_key.strip()
			final_key[true_key] = true_value
		return final_key
