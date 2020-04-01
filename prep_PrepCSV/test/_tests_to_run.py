### COMMAND: RESULT
import json
import os
import pandas
import warnings

base = "python3 {script}\
		--csv={csv}\
		--target_column_name={target}\
		--columns_with_missing_values={missing}\
		--columns_to_scale={scale}\
		--columns_to_normalize={normal}\
		--columns_to_dummy={dummy}\
		--output_file_path={output}\
		"

LOCAL_DIR = 'test/'
SCRIPT = '/prep_lib.py'
DEF_CSV = os.getcwd() + '/_data_for_testing.csv'
DEF_OUTPUT = os.getcwd() + '/_output.csv'
RESULTS_FILE = os.getcwd() + '/_results.json'

with open(RESULTS_FILE, 'r') as f:
	expected_results = json.load(f)

def __results(test_title, errors):
	print("=== Test: {} ===".format(test_title))
	if len(errors) == 0: print('\t\t \x1b[6;30;42m' + 'PASSED' + '\x1b[0m')

	for error in errors:
		print("\t\t \x1b[0;34;41m{}".format(error) + '\x1b[0m')

def __test1():
	test_results = []
	os.chdir('..')
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='None',
						  scale='None',
						  normal='[int_col_0,int_col_1]',
						  dummy='None',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	if not(max(output['int_col_0'].values) == 1. and min(output['int_col_0'].values) == 0.): test_results.append("normalization problem")
	if not(max(output['int_col_1'].values) == 1. and min(output['int_col_1'].values) == 0.): test_results.append("normalization problem")

	__results("basic - normalization", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

def __test2():
	test_results = []
	os.chdir('..')
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='None',
						  scale='None',
						  normal='None',
						  dummy='[int_col_0,int_col_1]',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	a = ['1' for c in output.columns if 'int_col_0' in c]
	b = ['1' for c in output.columns if 'int_col_1' in c]
	if not(len(a) == expected_results['int_col_0']['num_of_elements']): test_results.append("dummying problem")
	if not(len(b) == expected_results['int_col_1']['num_of_elements']): test_results.append("dummying problem")

	__results("basic - dummy", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

def __test3():
	test_results = []
	os.chdir('..')
	scaling = "[['int_col_0',12,80],['int_col_1',32.5,78.5]]"
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='None',
						  scale=scaling,
						  normal='None',
						  dummy='None',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	if not(min(output['int_col_0'].values) == 12.0 and max(output['int_col_0'].values) == 80.0): test_results.append("scaling problem")
	if not(min(output['int_col_1'].values) == 32.5 and max(output['int_col_1'].values) == 78.5): test_results.append("scaling problem")
	__results("basic - scaling", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

def __test4():
	test_results = []
	os.chdir('..')
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='[[empty_val_0,fill_0]]',
						  scale='None',
						  normal='None',
						  dummy='None',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	if not(False in set(output['empty_val_0'].isnull().values)): test_results.append("fill_X problem")
	__results("basic - filling empty values - fill_X", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

def __test5():
	test_results = []
	os.chdir('..')
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='[[empty_val_0,drop]]',
						  scale='None',
						  normal='None',
						  dummy='None',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	if not(False in set(output['empty_val_0'].isnull().values)): test_results.append("dropping problem")
	__results("basic - filling empty values - drop", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

def __test6():
	test_results = []
	os.chdir('..')
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='[[empty_val_0,avg]]',
						  scale='None',
						  normal='None',
						  dummy='None',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	if not(False in set(output['empty_val_0'].isnull().values)): test_results.append("avg problem")
	__results("basic - filling empty values - avg", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

def __test7():
	test_results = []
	os.chdir('..')
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='[[empty_val_0,med]]',
						  scale='None',
						  normal='None',
						  dummy='None',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	if not(False in set(output['empty_val_0'].isnull().values)): test_results.append("med problem")
	__results("basic - filling empty values - median", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

def __test8():
	test_results = []
	os.chdir('..')
	command = base.format(script=os.getcwd() + SCRIPT,
						  csv=DEF_CSV,
						  target='target_col',
						  missing='[[empty_val_0,randint_100_200]]',
						  scale='None',
						  normal='None',
						  dummy='None',
						  output=DEF_OUTPUT)
	os.system(command)
	output = pandas.read_csv(DEF_OUTPUT, index_col=0)
	### Checks:
	if output.columns[-1] != 'target_col': test_results.append("target_call_is_not_last_one")
	if not(False in set(output['empty_val_0'].isnull().values)): test_results.append("random element problem")
	__results("basic - filling empty values - random element", test_results)
	os.remove(DEF_OUTPUT)
	os.chdir(LOCAL_DIR)

if __name__ == '__main__':
	warnings.filterwarnings("ignore", category=UserWarning)

	try:
		__test1()   ### normalization.
		__test2()   ### dummy.
		__test3()   ### scaling.
		__test4()   ### fill_X.
		__test5()   ### drop.
		__test6()   ### avg.
		__test7()   ### med.
		__test8()   ### randInt.

	except FileNotFoundError:
		print("\x1b[0;34;41m The script run failed, therefore the processed csv hasn't been created. \x1b[0m")
