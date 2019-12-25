"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_update_sktrainer.py
==============================================================================
"""
import os

PATH = 'ailibrary/'
NAME = 'SKTrainer.py'
GOOD_FILE = 'ailibrary/xgboost/SKTrainer.py'

paths = []

for d in os.listdir(os.getcwd()):
	if os.path.isdir(d):
		for f in os.listdir(d):
			if f.endswith(NAME):
				full_path_to_f =  d + '/' + f
				paths.append(full_path_to_f)

for script in paths:
	f = open(script, 'r')
	lines = f.readlines()
	for i, line in enumerate(lines):
		if line.strip().startswith('version:'):
			lines[i] = 'version: ' + NEW_VERSION + '\n'
	os.remove(script)

	f = open(script, 'w+')
	f.writelines(lines)
	f.close()
