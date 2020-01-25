"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_tf_images_test.py
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
    'densenet201': ('tf2_densenet201/inceptionv3.py', 'tf2_densenet201'),
    'mobilenetv2': ('tf2_mobilenetv2/mobilenetv2.py', 'tf2_mobilenetv2'),
    'resnet50': ('tf2_deep_resnet50/resnet50.py', 'tf2_deep_resnet50'),
    'vgg16': ('tf2_vgg16/vgg16.py', 'tf2_vgg16'),
    'inceptionv3': ('tf2_inceptionv3/inceptionv3.py', 'tf2_inceptionv3')
}

curr_dir = os.getcwd().split('_tf_images_test.py')[0]

SCRIPT_FILE, DIRECTORY = scripts[args.algo]
SCRIPT_FILE = curr_dir + '/' + SCRIPT_FILE
DIRECTORY = curr_dir + '/' + DIRECTORY
DEF_DATA_PATH = "data_image_test/" if args.data == 'default' else args.data

ENV = 'python3'
PARAMS = ' --test_mode=True --epochs=5 --batch_size=28 --image_width=75 --image_height=75'
OUTPUT_FILE_NAME = 'model.h5'
OUTPUT_DICT_NAME = 'labels_dict.json'

# Output messages - Inform success or fail.
MODEL_CREATED = "*** Sub-Test Result: Model created successfully. ***"
TESTING_STARTS = "\n***** Testing Starts *****\n"
ENDED_SUCCESSFULLY = "\n***** Test Result: Ended. Look for errors! *****\n"

"""
Tests.
"""
# Test - Run file with data only (not other params).
def _test_file_with_data_only(title=''):
    print("=== Test {title} - Running script with data only (no cross validation).".format(title=title))
    os.system("{env} {script} --data={data} {params}".format(env=ENV, script=SCRIPT_FILE, data=DEF_DATA_PATH, params=PARAMS))
    _get_all_results()

"""
Helpers.
"""
def _get_all_results():
    """
    The operations should be done at the end of each test method.
    """
    __show_result_model_created()


def __show_result_model_created():
    """
    Checks if the model file has been created in the directory, prints confirmation and remove it.
    """
    if OUTPUT_FILE_NAME in os.listdir(os.getcwd()) and OUTPUT_DICT_NAME in os.listdir(os.getcwd()):
        print(MODEL_CREATED)
        os.remove(OUTPUT_FILE_NAME)
        os.remove(OUTPUT_DICT_NAME)

# All test runner.
def run():
    """
    Run all the test methods.
    """
    print(TESTING_STARTS)

    _test_file_with_data_only('1/1')

    print(ENDED_SUCCESSFULLY)

run()

