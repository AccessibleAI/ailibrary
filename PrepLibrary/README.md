# Pre Processing Library

This library process csv files according to user commands.

The options the library support:

1) One Hot Encoding.

2) Filling empty cells.

3) Scaling columns to [0, 1] range.

4) Remap string values (NOT AVAILABLE NOW)

5) Replace NULL values.

6) Dropping columns.

## API:
The user delivers requests via the command line. 

The parameters which the user is required to fill are:

1) ```--csv``` - (String) Path to local csv file. This param is required.

2) ```--output``` - (String) Path to the location the output file should be saved at. **Important** - should be ended with '.csv'. Default value: 'Preprocessed.csv'.

3) ```--index_col``` - (String or Integer) The name or the column number of the index column. By default it takes the rightmost column in the given file.

4) ```--one_hot``` - (list) a list contains all the columns names the user wants to one-hot-encode. Default value: '[]' (empty list).

5) ```--scale``` - (list) a list contains all the columns names the user wants to scale. Default value: '[]' (empty list).

6) ```--remap``` - NOT AVAILABLE NOW (dict) a dictionary with from the structure ```{VALUE: NEW_VALUE}```, **Important** - it replaces all the values with the name value in the dataset without noticing between columns. Default value: '{}' (empty dict).

7) ```--drop``` - (list) a list contains all the columns names the user wants to drop. Default value: '[]' (empty list).

8) ```--empty``` - This field all deals with ```na``` or ```NaN``` values. 
                Multiple cases: 
                
(-) If the user wishes to remove the lines contains those: ```--empty='r'```

(-) If the user wishes to fill the lines contains those with 0: ```--empty='z'```

(-) If the user wishes to fill the lines contains those with any other numeric value: ```--empty='NUMERIC_VAL'```

(-) NOT AVAILABLE NOW: If the user wishes to replace the empty or nan values differently in each column, the user needs to give a dictionary like: ```--empty={COLUMN_NAME: 'VALUE'}```

default value: 'z' (fill values with zeros).

## Example
```
python3 ./CnvrgPrep.py --csv='local/my_dataset.csv' --output='local/processed.csv' --index_col='6' --one_hot='['OfficeLocation']' --scale='['Age', 'Salary']' --remap={'R&D':'1', 'Sales':'2'} --empty='r --drop='interns'
```


