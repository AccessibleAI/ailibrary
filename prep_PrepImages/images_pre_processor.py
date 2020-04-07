"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

==============================================================================
"""
import os
import numpy as np

from pathlib import Path
from imageio import imsave
from skimage.util import random_noise
from utils import types_casting, get_generator

MEAN_GAUSS = 0
VAR_GAUSS = 1
RGB_CHANNELS = 3
GRAYSCALE_CHANNELS = 1

class ImagesPreProcessor:

	def __init__(self, args):
		types_casting(args)
		# paths to images.
		self.__path = args.path
		# resizing.
		self.__height, self.__width = args.height, args.width
		# channels.
		self.__grayscale = args.grayscale
		# add noise.
		self.__noise_arg = args.noise
		# blurring.
		self.__gaussian_blur_kernel_size = args.blur
		# convolution.
		self.__convolve_arg = args.convolve
		# zip.
		self.__zip_arg = args.zip_all
		# cnvrg dataset.
		self.__cnvrg_ds = args.cnvrg_ds

	def run(self):
		self.__load_images()
		self.__resize()
		self.__noise()
		self.__blur()   ## doesn't work
		self.__convolution()   ## doesn't work
		self.__zip()
		self.__push_to_cnvrg_dataset()

		print("All tasks are done.")

	def __load_images(self):
		print("Loading images ...")
		print('Changing to grayscale (if required) ...')
		self.__gen = get_generator(self.__path, self.__grayscale)

	def __resize(self):
		print("Resizing (if required) ...")
		self.__operate_on_all_images(self.__resize_image)

	def __noise(self):
		print('Adding noising (if required) ...')
		self.__operate_on_all_images(self.__add_noise_to_image)

	def __blur(self):
		print('Blurring (if required) ...')
		self.__operate_on_all_images(self.__do_blurring)

	def __convolution(self):
		print('Convolving (if required) ...')
		self.__operate_on_all_images(self.__convolve)

	def __zip(self):
		if self.__zip_arg:
			print('Zipping ...')
			os.chdir(self.__path)
			dir_name = Path(self.__path).parts[-1]
			zip_name = self.__path + '/' + dir_name + '.zip'
			os.system('zip -r {zip_name} *'.format(zip_name=zip_name, dir_name=dir_name))
			self.__zip_created = zip_name
		else:
			self.__zip_created = None

	def __push_to_cnvrg_dataset(self):
		if self.__cnvrg_ds is not None:
			print('Pushing dataset to cnvrg url: {}'.format(self.__cnvrg_ds))
			os.chdir(self.__path)
			to_push = '.' if self.__zip_created is None else self.__zip_created
			os.system('cnvrg data put {url} {to_push}'.format(url=self.__cnvrg_ds, to_push=to_push))

	###############################################################

	def __operate_on_all_images(self, func):
		for (path, img) in self.__gen:
			img = func(img)
			if isinstance(img, np.ndarray):
				imsave(path, img)
		self.__gen = get_generator(self.__path, self.__grayscale)

	def __resize_image(self, image):
		is_rgb = (len(image.shape) == 3)
		is_grayscale = (len(image.shape) == 2)
		is_rgba = (len(image.shape) == 4)

		if is_rgb:
			w, h, c = image.shape
			new_w = w if self.__width is None else self.__width
			new_h = h if self.__height is None else self.__height
			resized = image.resize((new_w, new_h, c))
			return resized

		elif is_grayscale:
			w, h = image.shape
			new_w = w if self.__width is None else self.__width
			new_h = h if self.__height is None else self.__height
			resized = image.resize((new_w, new_h))
			return resized

		elif is_rgba:
			raise ValueError("Still Doesn't work for rgba")

		else:
			raise ValueError('Unrecognized num of channels.')

	def __add_noise_to_image(self, img):
		if self.__noise_arg is not None:
			if self.__noise_arg == 'gaussian':
				img = random_noise(img, mode='gaussian', mean=MEAN_GAUSS, var=VAR_GAUSS)
			elif self.__noise_arg == 's&p':
				img = random_noise(img, mode='s&p')
			elif self.__noise_arg == 'speckle':
				img = random_noise(img, mode='speckle')
			elif self.__noise_arg == 'poisson':
				img = random_noise(img, mode='poisson')
			else:
				raise ValueError('Unsupported type of noise.')
			return img

	def __do_blurring(self, image):
		## doesn't work so far
		return image

	def __convolve(self, image):
		rgb_img = (len(image.shape) == 3)
		grayscale_img = (len(image.shape) == 1)

		# if grayscale_img: return np.convolve(image, self.__convolve_arg)
		# elif rgb_img:
		# 	for c in range(RGB_CHANNELS):
		# 		image[:,:,c] = np.convolve(image[:,:,c], self.__convolve_arg)
		# else: raise Exception("Can't perform convolution.")

		return image
