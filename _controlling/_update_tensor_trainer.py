"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_update_sktrainer.py
==============================================================================
"""
import os
import shutil

PATH = 'ailibrary/'
NAME = 'TensorTrainer.py'
GOOD_FILE = 'TEST_DEEP/_src/TensorTrainer.py'

paths = []
os.chdir('..')

for d in os.listdir(os.getcwd()):
	if os.path.isdir(d):
		for f in os.listdir(d):
			if f.endswith(NAME):
				full_path_to_f =  d + '/' + f
				paths.append(full_path_to_f)

for script in paths:
	if script != GOOD_FILE:
		shutil.copy(GOOD_FILE, script)
		print('updated: {}'.format(script))
