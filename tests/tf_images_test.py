"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

tf_images_test.py
==============================================================================
"""
import os
import random
import sys
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algo', '--algorithm', action='store', dest='algo', required=True)
parser.add_argument('-d', '--data', action='store', dest='data', default='default')
args = parser.parse_args()

# model_name : ( path_to_script, directory_name )
scripts = {
    'densenet201': ('tf2_densenet201/densenet201.py', 'tf2_densenet201'),
    'mobilenetv2': ('tf2_mobilenetv2/mobilenetv2.py', 'tf2_mobilenetv2'),
    'resnet50': ('tf2_resnet50/resnet50.py', 'tf2_resnet50'),
    'vgg16': ('tf2_vgg16/vgg16.py', 'tf2_vgg16'),
}

curr_dir = os.getcwd().split('tf_images_test.py')[0]

SCRIPT_FILE, DIRECTORY = scripts[args.algo]
SCRIPT_FILE = curr_dir + '/' + SCRIPT_FILE
DIRECTORY = curr_dir + '/' + DIRECTORY
DEF_DATA_PATH = "tf_image_test_data/" if args.data == 'default' else args.data

ENV = 'python3'
TEST_MODE = ' --test_mode=True '
X_VAL = ' --x_val=8 '
OUTPUT_FILE_NAME = 'model.sav'

# Output messages - Inform success or fail.
MODEL_CREATED = "*** Sub-Test Result: Model created successfully. ***"
TESTING_STARTS = "\n***** Testing Starts *****\n"
ENDED_SUCCESSFULLY = "\n***** Test Result: Ended. Look for errors! *****\n"

with open(DIRECTORY + '/' + '_params') as file:
    params_dict = yaml.load(file, Loader=yaml.FullLoader)

"""
Tests.
"""
# Test - Run file with data only (not other params).
def _test_file_with_data_only(title=''):
    print("=== Test {title} - Running script with data only (no cross validation).".format(title=title))
    os.system("{env} {script} --data={data} {test}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE))
    _get_all_results()

# Test - Run file with data only (not other params) with cross validation
def _test_file_with_data_only_and_cross_validation(title=''):
    print("=== Test {title} - Running script with data only (with cross validation).".format(title=title))
    os.system("{env} {script} --data={data} {x_val} {test}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, x_val=X_VAL))
    _get_all_results()

# Test - Run file with data and some of the params.
def _test_running_with_data_and_some_params(title=''):
    print("=== Test {title} - Running script with data and 50% of params without cross validation.".format(title=title))
    params_list = params_dict.keys()
    for it in range(5):
        curr_params = random.sample(params_list, int(0.5 * len(params_list)))
        command_for_params = __create_command(curr_params)
        print("Iteration number: {}, Run with params: {}".format(it, curr_params))
        os.system("{env} {script} --data={data} {test} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, params_command=command_for_params))
        _get_all_results()

# Test - Run file with data and some of the params.
def _test_running_with_data_and_some_params_and_cross_validation(title=''):
    print("=== Test {title} - Running script with data and 50% of params with cross validation.".format(title=title))
    params_list = params_dict.keys()
    for it in range(5):
        curr_params = random.sample(params_list, int(0.5 * len(params_list)))
        command_for_params = __create_command(curr_params)
        print("Iteration number: {}, Run with params: {}".format(it, curr_params))
        os.system("{env} {script} --data={data} {test} {x_val} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, x_val=X_VAL, params_command=command_for_params))
        _get_all_results()

# Test - Run file with data and all of the params.
def _test_running_with_data_and_all_params(title=''):
    print("=== Test {title} - Running script with data and ALL of params (without cross validation).".format(title=title))
    params_list = params_dict.keys()
    command_for_params = __create_command(params_list)
    os.system("{env} {script} --data={data} {test} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, params_command=command_for_params))
    _get_all_results()

# Test - Run file with data and all of the params with cross validation.
def _test_running_with_data_and_all_params_and_cross_validation(title=''):
    print("=== Test {title} - Running script with data and ALL of params (with cross validation).".format(title=title))
    params_list = params_dict.keys()
    command_for_params = __create_command(params_list)
    os.system("{env} {script} --data={data} {test} {x_val} {params_command}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, test=TEST_MODE, x_val=X_VAL, params_command=command_for_params))
    _get_all_results()

"""
Helpers.
"""
def _get_all_results():
    """
    The operations should be done at the end of each test method.
    """
    __show_result_model_created()

def __create_command(params_list):
    """
    Receives a list of params names and creates the line:
    "--param=value --param2=value2"
    with the values from the params_dict.
    """
    full_command = ""
    for param in params_list:
        full_command += '--{param}={value} '.format(param=param, value=params_dict[param])
    return full_command

def __show_result_model_created():
    """
    Checks if the model file has been created in the directory, prints confirmation and remove it.
    """
    if OUTPUT_FILE_NAME in os.listdir(os.getcwd()):
        print(MODEL_CREATED)
        os.remove(OUTPUT_FILE_NAME)

# All test runner.
def run():
    """
    Run all the test methods.
    """
    print(TESTING_STARTS)

    _test_file_with_data_only('1.1')
    _test_file_with_data_only_and_cross_validation('1.2')

    _test_running_with_data_and_some_params('2.1')
    _test_running_with_data_and_some_params_and_cross_validation('2.2')

    _test_running_with_data_and_all_params('3.1')
    _test_running_with_data_and_all_params_and_cross_validation('3.2')

    print(ENDED_SUCCESSFULLY)

run()

