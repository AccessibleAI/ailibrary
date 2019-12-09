"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_update_yml.py
==============================================================================
"""
import os

PATH = 'ailibrary/'

A = '1'
B = '1'
C = '13'
NEW_VERSION = A + '.' + B + '.' + C

yml_files = []

for d in os.listdir(os.getcwd()):
	if os.path.isdir(d):
		for f in os.listdir(d):
			if f.endswith('yml'):
				full_path_to_f =  d + '/' + f
				yml_files.append(full_path_to_f)

for yml_file in yml_files:
	f = open(yml_file, 'r')
	lines = f.readlines()
	for i, line in enumerate(lines):
		if line.strip().startswith('version:'):
			lines[i] = 'version: ' + NEW_VERSION + '\n'
	os.remove(yml_file)

	f = open(yml_file, 'w+')
	f.writelines(lines)
	f.close()
