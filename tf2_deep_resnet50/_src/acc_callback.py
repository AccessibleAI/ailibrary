import tensorflow as tf
import cnvrg


class MySuperDetailedAccuracy(tf.keras.callbacks.Callback):

	def set_experiment(self, exp):
		self.exp = exp
		self.accuracies = []

	def on_batch_begin(self, batch, logs=None):
		print("Start of batch number: {}.".format(batch))

	def on_batch_end(self, batch, logs=None):
		to_log = logs['accuracy']
		self.accuracies.append(to_log)
		print("End of batch number: {}, Accuracy measured: {}.".format(batch, str(to_log)))

	def on_epoch_begin(self, epoch, logs=None):
		print("Start of epoch number: {}".format(epoch))

	def on_epoch_end(self, epoch, logs=None):
		print("End of epoch number: {}, Accuracy measured: {}.".format(epoch, logs['accuracy']))

	def on_train_end(self, logs=None):
		for ind in range(len(self.accuracies)):
			self.accuracies[ind] = float(self.accuracies[ind])
		self.exp.log_metric("Accuracies", self.accuracies)
