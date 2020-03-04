"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

test_prepcsv.py
==============================================================================
"""
import json
import os
import string
import pandas
import random
import numpy as np

### Produce csv file for testing.
rows_num = 25
data = {}
summary = {}

### integers columns.
integers_columns_num = 3
for col in range(integers_columns_num):
	title = 'int_col_' + str(col)
	elements = [random.randint(1, 10) for i in range(rows_num)]
	data[title] = elements

	avg = sum(elements) / len(elements)
	num_of_elements = len(set(elements))
	summary[title] = {'avg': avg, 'num_of_elements': num_of_elements}

### strings columns.
strings_columns_num = 3
values = [(random.choice(string.ascii_letters)*3).upper() for i in range(10)]
for col in range(strings_columns_num):
	title = 'str_col_' + str(col)
	elements = [random.choice(values) for i in range(rows_num)]
	data[title] = elements

	num_of_elements = len(set(elements))
	summary[title] = {'num_of_elements': num_of_elements}

### column with empty values.
empty_values_columns_num = 1
num_of_empty_cells = 6
for col in range(empty_values_columns_num):
	title = 'empty_val_' + str(col)
	elements = [random.randint(1, 10) for i in range(rows_num)]
	rand_indexes = [random.randint(0, rows_num) for i in range(num_of_empty_cells)]
	for ind in range(len(elements)):
		if ind in rand_indexes: elements[ind] = np.nan

	data[title] = elements
	num_of_elements = len(set(elements))
	summary[title] = {'num_of_elements': num_of_elements}


### target column.
title = 'target_col'
elements = [random.choice([0, 0.3, 0.6, 1]) for i in range(rows_num)]
data[title] = elements


df = pandas.DataFrame.from_dict(data)
df.to_csv(os.getcwd() + "/_data_for_testing.csv")

with open(os.getcwd() + '/_results.json', 'w') as f:
	json.dump(summary, f)

