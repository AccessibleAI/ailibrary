# PrepCSV

## General


## Notes for this library
Command to run:  
```python3 prep_lib.py --path=PATH --target=TARGET --missing=MISSING --scale=SCALE --normalize=NORMALIZE --one_hot=ONE_HOT --output=OUTPUT```

## Parameters
1) ```--path``` - (string) path to csv file (required parameter).

2) ```--target``` - (string) The name of the target column. By default it takes the rightmost column in the given csv.

3) ```--missing``` - (dict) Dictionary describes what to do with empty, nan or NaN values in specific column.  
The structure of the dictionary is **{"COLUMN_NAME": "OPERATION"}** (the column name and the operation must be considered as strings even if they are numbers, dont worry - it is re-converted).\
The available operations are:
- **fill_X** - where X is an integer or float number which the user wants to set the empty values to.
- **drop** - drops the **rows** which has empty values in the specific column.
- **avg** - sets the empty values in the column to the average of the column (the other values must be integers or floats).
- **med** - sets the empty values in the column to the median of the column (the other values must be integers or floats).
- **randint_A_B** - sets the empty values in the column to a random integer between A and B.

4) ```--scale``` - (dict) Dictionary describes a range which the user wants to scale the values of the column to.  
The structure of the dictionary is **{COLUMN_NAME: RANGE}**, where **RANGE** looks like: **A:B** (A,B are integers or floats).

5) ```--normalize``` - (list) list of columns names the user wants to scale to [0, 1] range.

6) ```--one_hot``` - (list) list of columns names the user wants to perform one hot encoding on.

7) ```--output``` - (string) path for the output the csv file. By default it takes the given file path and add `_processed` to the name of it.
