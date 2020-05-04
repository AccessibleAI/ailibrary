import os
import shutil
from time import sleep
import subprocess
import platform
slash = '\\' if platform.system() == 'Windows' else '/'
path_to_data = os.getcwd() + slash + 'data'


def all_tests() -> None:
	_before_test()
	_test_1_1()
	_test_1_2()
	_test_2_1()
	_test_2_2()
	_test_3_1()
	_after_test()


def _before_test() -> None:
	try:
		os.mkdir(path_to_data)
	except FileExistsError:
		pass

	# before testing:
	for f in os.listdir(path_to_data):
		if os.path.isdir(path_to_data + slash + f):
			shutil.rmtree(path_to_data + slash + f, ignore_errors=True)
		else:
			os.remove(path_to_data + slash + f)
	return


def _after_test() -> None:
	shutil.rmtree(path_to_data)
	return


def _test_1_1() -> None:
	# (1.1) dividing the directory 'random_data_1' to two sub_directories.
	try:
		os.mkdir(path_to_data + slash + 'random_data_1')
		for i in range(100):
			with open(path_to_data + slash + 'random_data_1' + slash + 'file_' + str(i) + '.in', 'w'):
				pass
	except FileExistsError:
		pass

	os.chdir('..')
	command_1_1 = _build_command(path=path_to_data + slash + 'random_data_1', divide_to_two_dirs=True)
	os.system(command_1_1)
	sleep(2)
	assert len(os.listdir(path_to_data + slash + 'random_data_1')) == 2, "num of sub_directories is: {}".format(len(os.listdir(path_to_data + slash + 'random_data_1')))
	assert sorted(os.listdir(path_to_data + slash + 'random_data_1')) == sorted(['test', 'train']), "the sub_dirs are: " + str(os.listdir(path_to_data + slash + 'random_data_1'))
	shutil.rmtree(path_to_data + slash + 'random_data_1')
	_print_success_msg('1.1')


def _test_1_2() -> None:
	# (1.2) dividing the directory 'random_data_1' to two sub_directories.
	try:
		os.mkdir(path_to_data + slash + 'random_data_2')
		for i in range(100):
			with open(path_to_data + slash + 'random_data_2' + slash + 'file_' + str(i) + '.in', 'w'):
				pass
	except FileExistsError:
		pass

	command_1_2 = _build_command(path=path_to_data + slash + 'random_data_2', val_size=0.1, divide_to_two_dirs=True)
	subprocess.call(command_1_2)
	sleep(2)
	assert len(os.listdir(path_to_data + slash + 'random_data_2')) == 3, "num of sub_directories is: {}".format(len(os.listdir(path_to_data + slash + 'random_data_2')))
	assert sorted(os.listdir(path_to_data + slash + 'random_data_2')) == sorted(['train', 'validation', 'test'])
	shutil.rmtree(path_to_data + slash + 'random_data_2')
	_print_success_msg('1.2')


def _test_2_1() -> None:
	# (2.1) dividing the directory 'random_data_1' to two sub_directories.
	try:
		os.mkdir(path_to_data + slash + 'random_data_1')
		for d in ['a', 'b', 'c']:
			os.mkdir(path_to_data + slash + 'random_data_1' + slash + d)
			for i in range(50):
				with open(path_to_data + slash + 'random_data_1' + slash + d + slash + 'file_' + str(i) + '.in', 'w'):
					pass
	except FileExistsError:
		pass

	command_2_1 = _build_command(path=path_to_data + slash + 'random_data_1', divide_sub_dirs_to_two=True)
	os.system(command_2_1)
	sleep(2)
	assert len(os.listdir(path_to_data + slash + 'random_data_1')) == 2, "num of sub_directories is: {}".format(len(os.listdir(path_to_data + slash + 'random_data_1')))
	assert sorted(os.listdir(path_to_data + slash + 'random_data_1')) == sorted(['test', 'train']), "the sub_dirs are: " + str(os.listdir(path_to_data + slash + 'random_data_1'))
	shutil.rmtree(path_to_data + slash + 'random_data_1')
	_print_success_msg('2.1')


def _test_2_2() -> None:
	# (2.2) dividing the directory 'random_data_1' to three sub_directories.
	try:
		os.mkdir(path_to_data + slash + 'random_data_1')
		for d in ['a', 'b', 'c']:
			os.mkdir(path_to_data + slash + 'random_data_1' + slash + d)
			for i in range(50):
				with open(path_to_data + slash + 'random_data_1' + slash + d + slash + 'file_' + str(i) + '.in', 'w'):
					pass
	except FileExistsError:
		pass

	command_2_2 = _build_command(path=path_to_data + slash + 'random_data_1', divide_sub_dirs_to_two=True, val_size=0.1)
	os.system(command_2_2)
	sleep(2)
	assert len(os.listdir(path_to_data + slash + 'random_data_1')) == 3, "num of sub_directories is: {}".format(len(os.listdir(path_to_data + slash + 'random_data_1')))
	assert sorted(os.listdir(path_to_data + slash + 'random_data_1')) == sorted(['test', 'train', 'validation']), "the sub_dirs are: " + str(os.listdir(path_to_data + slash + 'random_data_1'))
	shutil.rmtree(path_to_data + slash + 'random_data_1')
	_print_success_msg('2.2')


def _test_3_1() -> None:
	# (3.1) dividing the directory 'random_data_1' to two sub_directories.
	try:
		os.mkdir(path_to_data + slash + 'random_data_1')
		for d in ['a', 'b', 'c']:
			for i in range(50):
				with open(path_to_data + slash + 'random_data_1' + slash + d + '_file_' + str(i) + '.in', 'w'):
					pass
	except FileExistsError:
		pass

	command_3_1 = _build_command(path=path_to_data + slash + 'random_data_1', group_by_prefix=True)
	os.system(command_3_1)
	sleep(2)
	assert sorted(os.listdir(path_to_data + slash + 'random_data_1')) == sorted(['a', 'b', 'c']), "the sub_dirs are: " + str(os.listdir(path_to_data + slash + 'random_data_1'))
	assert len(os.listdir(path_to_data + slash + 'random_data_1')) == 3, "num of sub_directories is: {}".format(len(os.listdir(path_to_data + slash + 'random_data_1')))
	shutil.rmtree(path_to_data + slash + 'random_data_1')
	_print_success_msg('3.1')


def _build_command(
				path: str, type: str = '.in', test_size: float = 0.2,
				val_size: float = 0., divide_to_two_dirs: bool = False,
				divide_sub_dirs_to_two: bool = False, group_by_prefix: bool = False) -> str:
	command = "python files_divider.py " \
			  "--path_to_data=\"{path}\" " \
			  "--files_type={type} " \
			  "--test_size={test_size} " \
			  "--validation_size={val_size} " \
			  "--divide_files_to_two_directories={divide_to_two_dirs} " \
			  "--divide_sub_directories_to_two_directories={divide_sub_dirs_to_two} " \
			  "--group_files_to_directories_by_prefix={group_by_prefix}".format(
																				path=path,
																				type=type,
																				test_size=test_size,
																				val_size=val_size,
																				divide_to_two_dirs=divide_to_two_dirs,
																				divide_sub_dirs_to_two=divide_sub_dirs_to_two,
																				group_by_prefix=group_by_prefix)
	return command


def _print_success_msg(test_num: str) -> None:
	print('\t\t \x1b[6;30;42m' + 'PASSED {test_num}'.format(test_num=test_num) + '\x1b[0m')


if __name__ == '__main__':
	all_tests()
