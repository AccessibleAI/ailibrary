"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

utils.py
==============================================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray


def types_casting(args):
	args.height = None if args.height == 'None' else int(args.height)
	args.width = None if args.width == 'None' else int(args.width)
	args.grayscale = (args.grayscale == 'True')
	args.noise = None if args.noise == 'None' else args.noise
	args.blur = int(args.blur)
	args.zip_all = (args.zip_all == 'True')
	args.cnvrg_ds = None if args.cnvrg_ds == 'None' else args.cnvrg_ds
	args.convolve = None if args.convolve is 'None' else _parse_multi_dimensional_list(args.convolve)


def _parse_multi_dimensional_list(md_list_str):
	open_close_dict = dict()
	brackets_counter, curr_open = 0, -1   ## +1 -> [  ||| -1 -> ]
	for i in range(1, len(md_list_str) - 1):   ## skips the open and close of all.
		if md_list_str[i] == '[':
			brackets_counter += 1
			curr_open = i
		elif md_list_str[i] == ']':
			brackets_counter -= 1
			if brackets_counter == 0:
				open_close_dict[curr_open] = i
		else:
			pass

	md_list = []
	for k, v in open_close_dict.items():
		nlist = md_list_str[k + 1: v]
		md_list.append(np.array(_parse_string_to_list(nlist)))

	return np.array(md_list)


def _parse_string_to_list(list_as_string):
	to_return = []
	buf = ''
	for i in range(len(list_as_string)):
		if list_as_string[i] == ',':
			to_return.append(float(buf))
			buf = ''
		else:
			buf += list_as_string[i]
	to_return.append(float(buf))
	return to_return


def get_generator(dir_path, grayscale=False):
	"""
	returns generator object of images.
	"""
	paths = os.listdir(dir_path)
	for path in paths:
		full_path = dir_path + '/' + path
		try:
			img_obj = plt.imread(full_path)
			if grayscale:
				img_obj = rgb2gray(img_obj)
			img = np.asarray(img_obj).astype(np.float64)
			yield (full_path, img)
		except OSError:  # catching file which is not an image.
			pass
