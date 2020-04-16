"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

nbConverter.py
==============================================================================
"""
import cnvrg
import nbconvert
import subprocess
from cnvrg import Experiment
from cnvrg.modules import UserError, CnvrgError


class NbConverter:
	def __init__(self,
				 input,
				 to,
				 template,
				 inplace,
				 allow_errors):
		self.__cnvrg_env = True  # When testing locally, it is turned False.
		self.input = input
		self.to = to
		self.template = template
		self.inplace = inplace
		self.allow_errors = allow_errors

		try:
			self.__experiment = Experiment()
		except:
			self.__cnvrg_env = False

		if self.__cnvrg_env:
			self.__experiment.log_param("template", template)

	def run(self):
		if self.__cnvrg_env:
			self.__experiment.log("Configuring nbconvert options")
		run_string = ''
		if self.allow_errors is False:
			if self.template is None:
				if self.to != 'notebook':
					run_string = "jupyter nbconvert --to {} {}'".format(self.to, self.input)
				elif self.inplace is True and self.to == 'notebook':
					run_string = "jupyter nbconvert --inplace --to {} {}".format(self.to, self.input)
				else:
					run_string = "jupyter nbconvert --to notebook {}".format(self.input)
			else:
				run_string = "jupyter nbconvert --to {} -template {} {}".format(self.to, self.template, self.input)
		else:
			if self.template is None:
				if self.to != 'notebook':
					run_string = "jupyter nbconvert --allow-errors --to {} {}".format(self.to, self.input)
				elif self.inplace is True and self.to == 'notebook':
					run_string = "jupyter nbconvert --allow-errors --inplace --to {} {}".format(self.to, self.input)
				else:
					run_string = "jupyter nbconvert --allow-errors --to notebook {}".format(self.input)
			else:
				run_string = "jupyter nbconvert --allow-errors --to {} -template {} {}".format(self.to, self.template, self.input)
		log_string = "Running command: {}".format(run_string)
		run_list = run_string.split(' ')
		dir='/cnvrg'
		if self.__cnvrg_env:
			self.__experiment.log(log_string)
		try:
			subprocess.run(run_list, cwd=dir)
		except OSError:
			print('jupyter nbconvert was unsuccessful. Please check your file path and parameters.')
			exit(1)
		if self.__cnvrg_env:
			self.__experiment.log("Conversion finished")
