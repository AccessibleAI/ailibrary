"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

predict.py
-----------------------
This file performs prediction using a .sav file.
Using with sk-learn models.
==============================================================================
"""
import pickle
import argparse

from cnvrg import Endpoint

loaded_model = pickle.load(open("model.sav", 'rb'))

def _preprocess(input):
	return 'None'

def predict(*args):
	"""
	:param args: should get 4 floats.
	:return: prediction (string).
	"""

	to_predict = args
	to_predict = _preprocess(to_predict)

	prediction = loaded_model.predict(to_predict)

	e = Endpoint()
	e.log_metric("Prediction", prediction)
	return prediction


