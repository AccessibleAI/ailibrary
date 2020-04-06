"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

utils.py
==============================================================================
"""
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def types_casting(args):
	args.height = None if args.height == 'None' else int(args.height)
	args.width = None if args.width == 'None' else int(args.width)
	args.grayscale = (args.grayscale == 'True')
	args.noise = None if args.noise == 'None' else args.noise
	args.denoise = (args.denoise == 'True')
	args.segmentation = (args.segmentation == 'True')
	args.blur = int(args.blur)
	args.zip_all = (args.zip_all == 'True')
	args.cnvrg_ds = None if args.cnvrg_ds == 'None' else args.cnvrg_ds


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
