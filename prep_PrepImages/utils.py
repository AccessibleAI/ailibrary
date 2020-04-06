"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

utils.py
==============================================================================
"""
import os
from PIL import Image


def types_casting(args):
	args.height = None if args.height == 'None' else int(args.height)
	args.width = None if args.width == 'None' else int(args.width)
	args.grayscale = (args.grayscale == 'True')
	args.noise = None if args.noise == 'None' else args.noise
	args.denoise = (args.denoise == 'True')
	args.segmentation = (args.segmentation == 'True')
	args.blur = (args.blur == 'True')
	args.zip_all = (args.zip_all == 'True')


def get_generator(dir_path):
	"""
	returns generator object of images.
	"""
	paths = os.listdir(dir_path)
	for path in paths:
		full_path = dir_path + '/' + path
		try:
			img = Image.open(full_path)
			yield (full_path, img)
		except OSError:  # catching file which is not an image.
			pass
