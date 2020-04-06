"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

CSVProcessor.py
==============================================================================
"""
import os

import cv2
import numpy as np

from imageio import imsave
from skimage.util import random_noise
from utils import types_casting, get_generator


class ImagesPreProcessor:
	MEAN_GAUSS = 0
	VAR_GAUSS = 1

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
		# de-noising.
		self.__denoise_arg = args.denoise
		# segmentation.
		self.__segmentation_arg = args.segmentation
		# blurring.
		self.__gaussian_blur_kernel_size = args.blur
		# zip.
		self.__zip_arg = args.zip_all
		# cnvrg dataset.
		self.__cnvrg_ds = args.cnvrg_ds

	def run(self):
		self.__load_images()
		self.__resize()
		self.__noise()
		self.__denoise()  ## doesn't work
		self.__segmentation()  ## doesn't work
		self.__blur()
		self.__zip()   ## doesn't work
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

	def __denoise(self):
		print('De-noising (if required) ...')
		self.__operate_on_all_images(self.__de_noising)

	def __segmentation(self):
		print('Segmenting (if required) ...')
		self.__operate_on_all_images(self.__do_segmentation)

	def __blur(self):
		print('Blurring (if required) ...')
		self.__operate_on_all_images(self.__do_blurring)

	def __zip(self):
		print('Zipping (if required) ...')
		self.__zip_images()

	def __push_to_cnvrg_dataset(self):
		if self.__cnvrg_ds is not None:
			print('Pushing dataset to cnvrg url: {}'.format(self.__cnvrg_ds))
			os.chdir(self.__path)
			os.system('cnvrg data put {url} .'.format(url=self.__cnvrg_ds))

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
				img = random_noise(img, mode='gaussian', mean=ImagesPreProcessor.MEAN_GAUSS, var=ImagesPreProcessor.VAR_GAUSS)
			elif self.__noise_arg == 's&p':
				img = random_noise(img, mode='s&p')
			elif self.__noise_arg == 'speckle':
				img = random_noise(img, mode='speckle')
			elif self.__noise_arg == 'poisson':
				img = random_noise(img, mode='poisson')
			else:
				raise ValueError('Unsupported type of noise.')
			return img

	def __de_noising(self, image):
		pass

	def __do_segmentation(self, image):
		pass

	def __do_blurring(self, image):
		# kernel_size = (self.__gaussian_blur_kernel_size, self.__gaussian_blur_kernel_size)
		# image = cv2.medianBlur(image, self.__gaussian_blur_kernel_size)
		return image

	def __zip_images(self):
		pass
