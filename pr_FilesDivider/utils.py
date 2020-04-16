import os
import shutil


def parse_list(list_as_string):
	list_as_string = list_as_string.trim()[1: -1]
	return [s.trim() for s in list_as_string.split(',')]


def divide_files_to_two_directories():
	"""
	dividing single directory to two directory:
		ex: dir -> train (x % of the files)
				-> test (100-x % of the files)
	"""


def divide_sub_directories_to_two_directories(data_dir, files_type, test_size, validation_size):
	"""
	dividing divided directory to two sub-also-divided-directories:
		ex: dir/a           dir/train -> dir/train/a , dir/train/b , dir/train/c
			dir/b     ->    dir/test  -> dir/test/a  , dir/test/b  , dir/test/c
			dir/c
	"""
	all_sub_dirs = [
		d for d in os.listdir(data_dir)
		if os.path.isdir(data_dir + '/' + d)
		and not d.startswith('.cnvrg')]
	print("All sub directories found: ", all_sub_dirs)

	with_validation = False
	if validation_size > 0.:
		with_validation = True

	# Creating the new directories.
	os.mkdir(data_dir + '/train')
	os.mkdir(data_dir + '/test')
	for d in all_sub_dirs:
		os.mkdir(data_dir + '/train/' + d)
		os.mkdir(data_dir + '/test/' + d)

	if with_validation:
		os.mkdir(data_dir + '/validation')
		for d in all_sub_dirs:
			os.mkdir(data_dir + '/validation/' + d)

	# Moving images.
	for d in all_sub_dirs:
		print("Dividing: {}".format(d))
		files_in_d = [img for img in os.listdir(data_dir + '/' + d) if img.endswith(files_type)]

		if with_validation:
			test_set_size = int(len(files_in_d) * test_size)
			validation_set_size = int(len(files_in_d) * validation_size)
			test_set = files_in_d[:test_set_size]
			validation_set = files_in_d[test_set_size:test_set_size + validation_set_size]
			train_set = files_in_d[test_set_size + validation_set_size:]
		else:
			test_set_size = int(len(files_in_d) * test_size)
			test_set = files_in_d[:test_set_size]
			validation_set = []
			train_set = files_in_d[test_set_size:]

		# Moving the train set.
		for img in train_set:
			shutil.move(src=data_dir + '/' + d + '/' + img, dst=data_dir + '/train/' + d + '/' + img)
		# Moving the train set.
		for img in test_set:
			shutil.move(src=data_dir + '/' + d + '/' + img, dst=data_dir + '/test/' + d + '/' + img)

		if with_validation:
			# Moving the train set.
			for img in validation_set:
				shutil.move(src=data_dir + '/' + d + '/' + img, dst=data_dir + '/validation/' + d + '/' + img)

		print("\t\tDivided: {} successfully!".format(d))
		# Deletes the previous directories.
		shutil.rmtree(data_dir + '/' + d)


def group_files_to_directories_by_prefix():
	"""
	group files by prefix to directories.
	ex: dir/dog_1.jpg       dir/dog  -> dir/dog/dog_1.jpg , dir/dog/dog_2.jpg
		dir/dog_2.jpg   ->
		dir/cat_1.jpg       dir/cat  -> dir/cat/cat_1.jpg , dir/cat/cat_2.jpg
		dir/cat_2.jpg
	"""
	pass
