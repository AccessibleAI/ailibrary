"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

_update_yml.py
==============================================================================
"""
import os
import platform

PATH = 'ailibrary/'
slash = '\\' if platform.system() == 'Windows' else '/'


def get_new_version(previous_version: str):
	if previous_version.startswith('version:'):
		previous_version = previous_version[len('version:'):].strip()

	top_ver = previous_version[:previous_version.find('.')]
	previous_version = previous_version[previous_version.find('.') + 1:]
	mid_ver = previous_version[:previous_version.find('.')]
	previous_version = previous_version[previous_version.find('.') + 1:]
	least_ver = previous_version[previous_version.find('.') + 1:]

	if int(least_ver) + 1 > 99:
		mid_ver = str(int(mid_ver) + 1)
		least_ver = (str(1))

	return top_ver + "." + mid_ver + "." + str(int(least_ver) + 1)


yml_files = []

os.chdir('..')

for d in os.listdir(os.getcwd()):
	if os.path.isdir(d):
		for f in os.listdir(d):
			if f.endswith('yml'):
				full_path_to_f = os.getcwd() + slash + d + slash + f
				yml_files.append(full_path_to_f)

for yml_file in yml_files:
	f = open(yml_file, 'r')
	lines = f.readlines()
	for i, line in enumerate(lines):
		if line.strip().startswith('version:'):
			prev_version = line.strip()
			new_version = get_new_version(prev_version)
			lines[i] = 'version: ' + new_version + '\n'
	f.close()

	new_f = open(yml_file, 'w')
	new_f.writelines(lines)
	new_f.close()

	print('Updated: ', yml_file)