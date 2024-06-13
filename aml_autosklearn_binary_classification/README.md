This library does binary classification on any given dataset

## Parameters

```--csv_filename``` - string, required. The path to the dataset's csv file.

```--dataset_name``` - string, required. The name of the dataset.


## CSV Required Format
The input csv file should have every feature as a column and the last column needs to contain the lables (0 or 1).

For example:

|1.0|85.0|66.0|29.0|0.0|26.6|0.35|31.0|1|
|----:|-------------:|------:|---------:|------:|-------:|------:|-------:|------:|
|8.0|183.0|64.0|0.0|0.0|23.3|0.672|32.0|0|
|1.0|89.0|66.0|23.0|94.0|28.1|0.16|21.0|0|
|0.0|137.0|40.0|35.0|168.0|43.1|2.288|33.0|1|
|5.0|116.0|74.0|0.0|0.0|25.6|0.201|30.0|1|
|3.0|78.0|50.0|32.0|88.0|31.0|0.248|26.0|0|

## Output
After the library will run it will output an accuracy score model pickle file named "model.pickle" which can be loaded later on by using:
```py
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
```

Get predcitions using:
```
model.predict(<values>)
```

