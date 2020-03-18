"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

time_history.py
==============================================================================
"""
import tensorflow as tf
import time

class TimeHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.times = []

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)