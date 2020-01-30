"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_update_yml.py
==============================================================================
"""
import os

PATH = 'ailibrary/'

TOP_VER = '1'
MID_VER = '1'
LEAST_VER = '44'
NEW_VERSION = TOP_VER + '.' + MID_VER + '.' + LEAST_VER

yml_files = []

os.chdir('..')

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
	print('Updated: ', yml_file)
	os.remove(yml_file)

	f = open(yml_file, 'w+')
	f.writelines(lines)
	f.close()
