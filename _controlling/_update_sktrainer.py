"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_update_sktrainer.py
==============================================================================
"""
import os
import shutil

PATH = 'ailibrary/'
NAME1 = 'SKTrainer.py'

GOOD_FILE = 'xgboost/SKTrainer.py'

paths = []
os.chdir('..')

for d in os.listdir(os.getcwd()):
	if os.path.isdir(d):
		for f in os.listdir(d):
			if f.endswith(NAME1):
				full_path_to_f =  d + '/' + f
				paths.append(full_path_to_f)

for script in paths:
	if script != GOOD_FILE:
		shutil.copy(GOOD_FILE, script)
		print('updated: {}'.format(script))
