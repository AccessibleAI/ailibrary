import os
import argparse
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = "0"



# parser = argparse.ArgumentParser(description="""Preprocessor""")
# parser.add_argument('-p','--phrase', action='store', dest='phrase', required=True,
# 					help="""string. phrase for sentiment analysis.""")



# args = parser.parse_args()

# phrase = args.phrase

model = keras.models.load_model('model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sentiment_label = ['positive','negative']

def predict(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return(sentiment_label[prediction])



