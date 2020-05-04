import os
import shutil

import platform
slash = '\\' if platform.system() == 'Windows' else '/'


def parse_list(list_as_string):
	list_as_string = list_as_string.trim()[1: -1]
	return [s.trim() for s in list_as_string.split(',')]


def divide_files_to_two_directories(data_dir, file_type, test_size, validation_size):
	"""
	dividing single directory to two directory:
		ex: dir -> train (x % of the files)
				-> test (100-x % of the files)
	"""
	with_validation = True if validation_size > 0. else False

	all_sub_dirs = ['train', 'validation', 'test'] if with_validation else ['train', 'test']

	# Creating the new directories.
	for d in all_sub_dirs:
		try:
			os.mkdir(data_dir + slash + d)
		except FileExistsError:
			pass

	files = [img for img in os.listdir(data_dir) if img.endswith(file_type)]

	if with_validation:
		test_set_size = int(len(files) * test_size)
		validation_set_size = int(len(files) * validation_size)
		test_set = files[:test_set_size]
		validation_set = files[test_set_size:test_set_size + validation_set_size]
		train_set = files[test_set_size + validation_set_size:]
	else:
		test_set_size = int(len(files) * test_size)
		test_set = files[:test_set_size]
		validation_set = []
		train_set = files[test_set_size:]

	# Moving the images.
	for img in files:
		if img in train_set:
			shutil.move(src=data_dir + slash + img, dst=data_dir + slash + 'train' + slash + img)

		elif img in test_set:
			shutil.move(src=data_dir + slash + img, dst=data_dir + slash + 'test' + slash + img)

		elif img in validation_set:
			shutil.move(src=data_dir + slash + img, dst=data_dir + slash + 'validation' + slash + img)

		else:
			pass


def divide_sub_directories_to_two_directories(data_dir, file_type, test_size, validation_size):
	"""
	dividing divided directory to two sub-also-divided-directories:
		ex: dir/a           dir/train -> dir/train/a , dir/train/b , dir/train/c
			dir/b     ->    dir/test  -> dir/test/a  , dir/test/b  , dir/test/c
			dir/c
	"""
	all_sub_dirs = [
		d for d in os.listdir(data_dir)
		if os.path.isdir(data_dir + slash + d)
		and not d.startswith('.cnvrg')]
	print("All sub directories found: ", all_sub_dirs)

	with_validation = False
	if validation_size > 0.:
		with_validation = True

	# Creating the new directories.
	os.mkdir(data_dir + slash + 'train')
	os.mkdir(data_dir + '/test')
	for d in all_sub_dirs:
		os.mkdir(data_dir + slash + 'train' + slash + d)
		os.mkdir(data_dir + slash + 'test' + slash + d)

	if with_validation:
		os.mkdir(data_dir + slash + 'validation')
		for d in all_sub_dirs:
			os.mkdir(data_dir + slash + 'validation' + slash + d)

	# Moving images.
	for d in all_sub_dirs:
		print("Dividing: {}".format(d))
		files_in_d = [img for img in os.listdir(data_dir + slash + d) if img.endswith(file_type)]

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
			shutil.move(src=data_dir + slash + d + slash + img, dst=data_dir + slash + 'train' + slash + d + slash + img)

		# Moving the test set.
		for img in test_set:
			shutil.move(src=data_dir + slash + d + slash + img, dst=data_dir + slash + 'test' + slash + d + slash + img)

		if with_validation:
			# Moving the validation set.
			for img in validation_set:
				shutil.move(src=data_dir + slash + d + slash + img, dst=data_dir + slash + 'validation' + slash + d + slash + img)

		print("\t\tDivided: {} successfully!".format(d))
		# Deletes the previous directories.
		shutil.rmtree(data_dir + slash + d)


def group_files_to_directories_by_prefix(data_dir, file_type):
	"""
	group files by prefix to directories.
	ex: dir/dog_1.jpg       dir/dog  -> dir/dog/dog_1.jpg , dir/dog/dog_2.jpg
		dir/dog_2.jpg   ->
		dir/cat_1.jpg       dir/cat  -> dir/cat/cat_1.jpg , dir/cat/cat_2.jpg
		dir/cat_2.jpg
	"""
	all_sub_dirs = []

	def make_dir(name: str):
		os.mkdir(data_dir + slash + name)

	def move_to_dir(dir_name, to_move):
		shutil.move(src=data_dir + slash + to_move, dst=data_dir + slash + dir_name + slash + to_move)

	all_files = os.listdir(data_dir)
	for ind, file in enumerate(all_files):
		if file.endswith(file_type):
			label = file.split('_')[0]
			if label in all_sub_dirs:
				move_to_dir(dir_name=label, to_move=file)

			else:
				all_sub_dirs.append(label)
				make_dir(label)
				move_to_dir(dir_name=label, to_move=file)

		if ((ind / len(all_files)) * 100) % 10 == 0:
			print("Divided {}% of the files".format(((ind / len(all_files)) * 100)))

	print("All sub directories found: ", all_sub_dirs)

