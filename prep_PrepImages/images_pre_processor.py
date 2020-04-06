"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

CSVProcessor.py
==============================================================================
"""
import cnvrg
from utils import types_casting, get_generator
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
import numpy as np
from imageio import imsave



class ImagesPreProcessor:
	MEAN_GAUSS = 0
	VAR_GAUSS = 25

	def __init__(self, args):
		types_casting(args)
		# paths to images.
		self.__path = args.path
		# resizing.
		self.__height, self.__width = args.height, args.width
		# channels.
		self.__grayscale = args.grayscale
		# add noise.
		self.__noise = args.noise
		# de-noising.
		self.__denoise = args.denoise
		# segmentation.
		self.__segmentation = args.segmentation
		# blurring.
		self.__blur = args.blur
		# zip.
		self.__zip = args.zip_all

	def run(self):
		print("Loading images ...")
		self.__load_images()

		print("Resizing (if required) ...")
		self.__operate_on_all_images(self.__resizing)

		print('Changing to grayscale (if required) ...')
		self.__operate_on_all_images(self.__convert_grayscale)

		print('Adding noising (if required) ...')
		self.__operate_on_all_images(self.__add_noise)

		print('De-noising (if required) ...')
		self.__operate_on_all_images(self.__de_noising)

		print('Segmenting (if required) ...')
		self.__operate_on_all_images(self.__do_segmentation)

		print('Blurring (if required) ...')
		self.__operate_on_all_images(self.__do_blurring)

		print('Zipping (if required) ...')
		self.__zip_images()

		print("All tasks are done.")

	def __load_images(self):
		self.__gen = get_generator(self.__path)

	def __operate_on_all_images(self, func):
		for (path, img) in self.__gen:
			img = func(img)
			if isinstance(img, np.ndarray):
				imsave(path, img)
		self.__gen = get_generator(self.__path)

	def __resizing(self, image):
		is_rgb = (len(image.shape) == 3)
		is_grayscale = (len(image.shape) == 1)

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

		else:
			raise Exception('Unrecognized num of channels.')


	def __add_noise(self, img):
		if self.__noise is not None:
			if self.__noise == 'gaussian':
				img = ImagesPreProcessor.__noise_gaussian(img)
			elif self.__noise == 'salt&pepper':
				img = ImagesPreProcessor.__noise_salt_pepper(img)
			elif self.__noise == 'poisson':
				img = ImagesPreProcessor.__noise_poisson(img)
			else:
				raise ValueError('Unsupported type of noise.')
			return img

	@staticmethod
	def __noise_gaussian(image):
		"""
		mean = 0, variance = 1.
		"""
		x, y, z = image.shape
		gauss = np.random.normal(
			ImagesPreProcessor.MEAN_GAUSS,
			ImagesPreProcessor.VAR_GAUSS,
			(x, y, z))
		gauss = gauss.reshape((x, y, z))
		as_np_array = image + gauss
		as_np_array = np.ceil(as_np_array).astype(np.float64)
		return as_np_array

	@staticmethod
	def __noise_salt_pepper(image):
		return image

	@staticmethod
	def __noise_poisson(image):
		return image

	def __de_noising(self, image):
		pass

	def __do_segmentation(self, image):
		pass

	def __do_blurring(self, image):
		pass

	def __zip_images(self):
		pass

	def __convert_grayscale(self, image):
		if self.__grayscale:
			return image.convert('L')

