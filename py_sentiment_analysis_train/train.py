import pandas as pd
import argparse

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

import pickle



parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('-f','--filename', action='store', dest='filename', required=True,
					help="""string. csv twitter labeled train data file""")


parser.add_argument('-o','--output_token_file', action='store', dest='output_token_file',default='tokenizer.pickle',required=False,
					help="""string. filename for saving the tokenizer""")


parser.add_argument('-m','--output_model_file', action='store', dest='output_model_file' ,default='model.h5' ,required=False,
					help="""string. filename for saving the model""")


parser.add_argument('-t','--text_column', action='store', dest='text_column',default='text' ,required=False,
					help="""string. name of text column""")


parser.add_argument('-s','--label_column', action='store', dest='label_column',default='sentiment' ,required=False,
					help="""string. name of label column""")

parser.add_argument('--epochs', action='store', dest='epochs',default=5 ,required=False,
					help="""int. number of training epochs to run""")

parser.add_argument('--batch_size', action='store', dest='batch_size',default=32 ,required=False,
					help="""int. size of each batch to train on""")



args = parser.parse_args()
FILENAME = args.filename
output_token_file = args.output_token_file
output_model_file = args.output_model_file
text_column = args.text_column
label_column = args.label_column

print(args)

df = pd.read_csv(FILENAME)



tweet_df = df[[text_column,label_column]]

tweet_df = tweet_df[tweet_df[label_column] != 'neutral']

sentiment_label = tweet_df[label_column].factorize()

tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

with open('/cnvrg/{}'.format(output_token_file), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary())


history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)


model.save('/cnvrg/{}'.format(output_model_file))







