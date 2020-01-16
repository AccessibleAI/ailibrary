"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

SKTrainerClustering.py
==============================================================================
"""
from cnvrg import Experiment


class SKTrainerClustering:
	def __init__(self, model, train_set, test_set, output_model_name, testing_mode):
		self.__model = model
		self.__x_train = train_set
		self.__train_set_size = len(self.__x_train)
		self.__x_test = test_set
		self.__test_set_size = len(self.__x_test)
		self.__testing_mode = testing_mode
		self.__features = list(self.__x_train.columns)
		self.__metrics = {'model': output_model_name}
		self.__experiment = Experiment()

	def run(self):
		pass

	def __plot_all(self, y_test_pred):
		pass

	def __train_without_cross_validation(self):
		pass

	def __plot_centroids(self):
		pass

