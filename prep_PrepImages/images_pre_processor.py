"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

CSVProcessor.py
==============================================================================
"""
import cnvrg
from utils import types_casting


class ImagesPreProcessor:
	def __init__(self, args):
		args = types_casting(args)
		# resizing.
		self.__height, self.__width, self.__channels = args.height, args.width, args.channels
		# add noise.
		self.__add_noise = args.add_noise
		# de-noising.
		self.__denoise = args.denoise
		# segmentation.
		self.__segmentation = args.segmentation
		# blurring.
		self.__blur = args.blur
		# zip.
		self.__zip = args.zip_all


	def run(self):
		pass
