"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_update_yml.py
==============================================================================
"""
import os

PATH = 'ailibrary/'

packages = {
	'json5': ('>=', '0.8.5'),
	'numpy': ('>=', '1.18.0'),
	'pandas': ('>=', '0.25.0'),
	'scikit-learn': ('>=', '0.21.3'),
	'scipy': ('>=', '1.3.1'),
	'xgboost': ('>=', '0.90'),
	'cnvrg': ('>=', '0.1.7.6'),
	'tensorflow': ('==', '2.1.0'),

}

print("=== Running: updating requirements.txt files ===")

req_files = []

os.chdir('..')

for d in os.listdir(os.getcwd()):
	if os.path.isdir(d):
		for f in os.listdir(d):
			if f.endswith('requirements.txt'):
				full_path_to_f =  d + '/' + f
				req_files.append(full_path_to_f)

for req_file in req_files:
	f = open(req_file, 'r')
	lines = f.readlines()
	for i, line in enumerate(lines):
		for package, version in packages.items():
			if package in line:
				lines[i] = package + version[0] + version[1] + '\n'
	print('Updated: ', req_file)
	os.remove(req_file)

	f = open(req_file, 'w+')
	f.writelines(lines)
	f.close()


print("=== Finished. ===")