"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

SKTrainerClustering.py
==============================================================================
"""
import os
import pickle
import numpy as np
import pandas as pd

from cnvrg import Experiment
from cnvrg.charts import MatrixHeatmap


class SKTrainerClustering:
	def __init__(self, model, train_set, test_set, output_model_name, testing_mode):
		self.__model = model
		self.__x_train, _ = (train_set, None) if len(train_set) == 1 else train_set
		self.__train_set_size = len(self.__x_train)
		self.__x_test, self.__y_test = (test_set, None) if len(train_set) == 1 else train_set
		self.__test_set_size = len(self.__x_test)
		self.__testing_mode = testing_mode
		self.__features = list(self.__x_train.columns)
		self.__metrics = {'model': output_model_name}
		self.__labeled = len(train_set) == 2 or len(test_set) == 2  # if any of the sets includes target column.
		# self.__experiment = Experiment()
		self.__experiment = Experiment.init("test_charts")

	def run(self):
		if self.__labeled: self.__train_with_target()
		else:              self.__train_without_target()


	def __plot_all(self, y_test_pred=None):
		self.__plot_correlation_matrix()
		self.__plot_centroids()
		pass

	def __train_with_target(self):
		pass

		self.__plot_all()

	def __train_without_target(self):
		pass

		self.__plot_all()

	def __plot_centroids(self):
		pass

	def __save_model(self):
		output_model_name = self.__metrics['model']
		output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + output_model_name if os.environ.get("CNVRG_PROJECT_PATH") \
																				is not None else output_model_name
		pickle.dump(self.__model, open(output_file_name, 'wb'))

	def __plot_correlation_matrix(self):
		data = pd.concat([pd.concat([self.__x_train, self.__x_test], axis=0), pd.concat([self.__y_train, self.__y_test], axis=0)], axis=1)
		correlation = data.corr()
		self.__experiment.log_chart("correlation", [MatrixHeatmap(np.round(correlation.values, 2))],
									x_ticks=correlation.index.tolist(), y_ticks=correlation.index.tolist())