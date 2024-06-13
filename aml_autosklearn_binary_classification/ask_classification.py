import argparse
import sklearn.metrics
import autosklearn.classification
import pandas as pd
import numpy
import pickle

parser = argparse.ArgumentParser(description="""Kaggle Dataset Connector""")
parser.add_argument('--csv_filename', action='store', dest='csv_filename', required=True, help="""--- The name of the dataset ---""")
parser.add_argument('--dataset_name', action='store', dest='dataset_name', required=True, help="""--- The name of the dataset ---""")

args = parser.parse_args()
csv_filename = args.csv_filename
dataset_name = args.dataset_name



# PREPROCESS THE CSV AND DIVIDE INTO X,y
df = pd.read_csv(csv_filename)
lables=df.iloc[:,-1:].values
lables = lables.ravel()
df = df.iloc[: , :-1]
features = df.to_numpy()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, lables, random_state=42)

# Fit the classifier
print(f'Training the model on the "{dataset_name}" dataset')
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_classification_tmp',
)
automl.fit(X_train, y_train, dataset_name=dataset_name)


# Show all the models 
print(automl.show_models())

# Get the score of the final ensemble
predictions = automl.predict(X_test)
# print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
print(f'cnvrg_tag_accuracy: {accuracy}')
with open('/cnvrg/model.pickle', 'wb') as f:
    s = pickle.dump(automl, f)
