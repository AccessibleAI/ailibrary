"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

predict.py
-----------------------
This file performs prediction using a .sav file.
Using with sk-learn models.
==============================================================================
"""
import torch
import keras
import pickle
import argparse

from cnvrg import Endpoint
from flask import current_app as app


def _load_model():
	"""
	Load models of types: keras|tensorflow|xgboost|sklearn|pytorch.
	"""
	try:
		path = app.config["model_path"]
		model_type = app.config["model_type"]
	except Exception:
		raise Exception('CnvrgError: Unrecognized flask.')

	if model_type == 'sklearn':
		return pickle.load(open(path, 'rb'))
	elif model_type == 'keras':
		return keras.models.load_model(path)
	elif model_type == 'tensorflow':
		return tf.keras.models.load_model(path)
	elif model_type == 'xgboost':
		return pickle.load(open("model.sav", 'rb'))
	elif model_type == 'pytorch':
		return torch.load(path)
	else:
		raise Exception('CnvrgError: unrecognized model type.')

loaded_model = _load_model()


def _preprocess(input):
	"""
	In order to have a functional and working predict file,
	Fill this method.
	:param input:
	:return:
	"""
	return 'None'


def predict(*args):
	"""
	:param args: should get 4 floats.
	:return: prediction (string).
	"""

	to_predict = _preprocess(args)

	prediction = loaded_model.predict(to_predict)

	e = Endpoint()
	e.log_metric("Prediction", prediction)
	return prediction


