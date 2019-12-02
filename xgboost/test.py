"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

test.py
==============================================================================
"""
import os
import random
import sys
import yaml
import argparse

ENV = 'python3'
TEST_MODE = ' --test_mode=True '
SCRIPT_FILE = 'xgb.py'
DEF_DATA_PATH = "../_testing_data/default.csv"

X_VAL = ' --x_val=5 '
OUTPUT_FILE_NAME = 'model.sav'

# Output messages - Inform success or fail.
MODEL_CREATED = "*** Test Result: Model created successfully. ***"

with open(os.getcwd() + '/_params') as file:
    params_dict = yaml.load(file, Loader=yaml.FullLoader)


"""
Tests.
"""

# Test 1 - Run file without data set.
# os.system("{} {}".format(ENV, SCRIPT_FILE))

# Test 2 - Run file with data only (not other params).
def _test_file_with_data_only():
    print("=== Test {} - Running script with data only (no cross validation).")
    os.system("{env} {script} --data={data} {test}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE))
    _get_all_results()

# Test 3 - Run file with data only (not other params) with cross validation
def _test_file_with_data_only_and_cross_validation():
    print("=== Test {} - Running script with data only (with cross validation).")
    os.system("{env} {script} --data={data} {x_val} {test}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, x_val=X_VAL))
    _get_all_results()

# Test 4 - Run file with data and some of the params.
def _test_running_with_data_and_some_params():
    print("=== Test {} - Running script with data and 50% of params without cross validation.")
    params_list = params_dict.keys()
    for it in range(5):
        curr_params = random.sample(params_list, int(0.5 * len(params_list)))
        command_for_params = __create_command(curr_params)
        print("Iteration number: {}, Run with params: {}".format(it, curr_params))
        os.system("{env} {script} --data={data} {test} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, params_command=command_for_params))
        _get_all_results()

# Test 4 - Run file with data and some of the params.
def _test_running_with_data_and_some_params_and_cross_validation():
    print("=== Test {} - Running script with data and 50% of params with cross validation.")
    params_list = params_dict.keys()
    for it in range(5):
        curr_params = random.sample(params_list, int(0.5 * len(params_list)))
        command_for_params = __create_command(curr_params)
        print("Iteration number: {}, Run with params: {}".format(it, curr_params))
        os.system("{env} {script} --data={data} {test} {x_val} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, x_val=X_VAL, params_command=command_for_params))
        _get_all_results()

# Test 5 - Run file with data and all of the params.
def _test_running_with_data_and_all_params():
    print("=== Test {} - Running script with data and ALL of params (without cross validation).")
    params_list = params_dict.keys()
    command_for_params = __create_command(params_list)
    os.system("{env} {script} --data={data} {test} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, params_command=command_for_params))
    _get_all_results()

# Test 6 - Run file with data and all of the params with cross validation.
def _test_running_with_data_and_all_params_and_cross_validation():
    print("=== Test {} - Running script with data and ALL of params (with cross validation).")
    params_list = params_dict.keys()
    command_for_params = __create_command(params_list)
    os.system("{env} {script} --data={data} {test} {x_val} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, x_val=X_VAL, params_command=command_for_params))
    _get_all_results()


"""
Helpers.
"""
def _get_all_results():
    __show_result_model_created()

def __create_command(params_list):
    full_command = ""
    for param in params_list:
        full_command += '--{param}={value} '.format(param=param, value=params_dict[param])
    return full_command

def __show_result_model_created():
    if OUTPUT_FILE_NAME in os.listdir(os.getcwd()):
        print(MODEL_CREATED)
        os.remove(OUTPUT_FILE_NAME)



