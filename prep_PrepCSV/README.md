This library is made for pre-processing csv files.  

## Notes for this library
The library enables you to deals with empty values, scale or normalize features and do one-hot encoding.

## Parameters

```--csv``` - string, required. The path to the csv file.

```--target_column_name``` - string (default = None). The name of the target column. By default it takes the rightmost column in the given csv.

```--columns_with_missing_values``` - 2d list (default = None). list of sub lists which looks like: [COL_NAME,Operation], avoid spaces!!!.
The structure of the dictionary is **{"COLUMN_NAME": "OPERATION"}**. The column name and the operation must be considered as strings even if they are numbers.
The available operations are:
- **fill_X** - where X is an integer or float number which the user wants to set the empty values to.
- **drop** - drops the rows which have empty values in the specific column.
- **avg** - sets the empty values in the column to the average of the column (the other values must be integers or floats).
- **med** - sets the empty values in the column to the median of the column (the other values must be integers or floats).
- **randint_A_B** - sets the empty values in the column to a random integer between A and B.

```--columns_to_scale``` - 2d list (default = None). list of lists where each sublist looks like: [COL_NAME,lower_range,higher_range], avoid spaces!!!.

```--columns_to_normalize``` - list (default = None). A list of column names the user wants to scale to the range [0, 1], avoid spaces!!!.

```--columns_to_dummy``` - list (default = None). A list of column names the user wants to perform one-hot encoding on.

```--output_file_path``` - str (default = None). The path for the output the csv file. By default it takes the given file path and adds `_processed` to the file name.

```--visualize``` - bool (default = False). Indicates whether to plot visualizations or not.

## Examples
1) Tips Data Set
Input data set:  

|     |   total_bill |   sex |   smoker |   day |   time |   size |   tip > 12% |
|----:|-------------:|------:|---------:|------:|-------:|-------:|------------:|
|   0 |        16.99 |     1 |        1 |     1 |      2 |      2 |           0 |
|   1 |        10.34 |     0 |        1 |     1 |      2 |      3 |           1 |
|   2 |        21.01 |     0 |        1 |     1 |      2 |      3 |           1 |
|   3 |        23.68 |     0 |        1 |     1 |      2 |      2 |           1 |
|   4 |        24.59 |     1 |        1 |     1 |      2 |      4 |           1 |
|   5 |        25.29 |     0 |        1 |     1 |      2 |      4 |           1 |
|   6 |         8.77 |     0 |        1 |     1 |      2 |      2 |           1 |

Given the command: ```--data="~/tips.csv" --scale="{'total_bill': '100:1000'}" --one_hot=[time,day] --normalize=[size]```  
Output data set:

|     |   total_bill |   sex |   smoker |     size |   time_1 |   time_2 |   day_1.0 |   day_6.0 |   day_7.0 |   tip > 12% |
|----:|-------------:|------:|---------:|---------:|---------:|---------:|----------:|----------:|----------:|------------:|
|   0 |      362.421 |     1 |        1 | 0.166667 |        0 |        1 |         1 |         0 |         0 |           0 |
|   1 |      237.055 |     0 |        1 | 0.333333 |        0 |        1 |         1 |         0 |         0 |           1 |
|   2 |      438.207 |     0 |        1 | 0.333333 |        0 |        1 |         1 |         0 |         0 |           1 |
|   3 |      488.542 |     0 |        1 | 0.166667 |        0 |        1 |         1 |         0 |         0 |           1 |
|   4 |      505.698 |     1 |        1 | 0.5      |        0 |        1 |         1 |         0 |         0 |           1 |
|   5 |      518.894 |     0 |        1 | 0.5      |        0 |        1 |         1 |         0 |         0 |           1 |
|   6 |      207.457 |     0 |        1 | 0.166667 |        0 |        1 |         1 |         0 |         0 |           1 |

2) Churn From Banks Data Set (with empty values)  
Input data:  

|   RowNumber |   CustomerId | Surname   |   CreditScore | Geography   | Gender   |   Age |   Tenure |   Balance |   NumOfProducts |   HasCrCard |   IsActiveMember |   EstimatedSalary |   Exited |
|------------:|-------------:|:----------|--------------:|:------------|:---------|------:|---------:|----------:|----------------:|------------:|-----------------:|------------------:|---------:|
|           1 |     15634602 | Hargrave  |           619 | France      | Female   |    42 |        2 |         0 |               1 |           1 |                1 |          101349   |        1 |
|           2 |     15647311 | Hill      |           608 | Spain       | Female   |    41 |        1 |   **nan** |               1 |           0 |                1 |          112543   |        0 |
|           3 |     15619304 | Onio      |           502 | France      | Female   |    42 |        8 |    159661 |               3 |           1 |                0 |         **nan**   |        1 |
|           4 |     15701354 | **nan **  |           699 | France      | Female   |    39 |        1 |         0 |               2 |           0 |                0 |           93826.6 |        0 |
|           5 |     15737888 | Mitchell  |           850 | Spain       | Female   |    43 |        2 |    125511 |               1 |           1 |                1 |           79084.1 |        0 |

Given command: ```--data="~/creditcard_empty.csv" --missing="{'Surname':'drop', 'Balance':'fill_0', 'EstimatedSalary':'randint_0_5'}```  

|   RowNumber |   CustomerId | Surname   |   CreditScore | Geography   | Gender   |   Age |   Tenure |   Balance |   NumOfProducts |   HasCrCard |   IsActiveMember |   EstimatedSalary |   Exited |
|------------:|-------------:|:----------|--------------:|:------------|:---------|------:|---------:|----------:|----------------:|------------:|-----------------:|------------------:|---------:|
|           1 |     15634602 | Hargrave  |           619 | France      | Female   |    42 |        2 |         0 |               1 |           1 |                1 |          101349   |        1 |
|           2 |     15647311 | Hill      |           608 | Spain       | Female   |    41 |        1 |         0 |               1 |           0 |                1 |          112543   |        0 |
|           3 |     15619304 | Onio      |           502 | France      | Female   |    42 |        8 |    159661 |               3 |           1 |                0 |               2   |        1 |
|           5 |     15737888 | Mitchell  |           850 | Spain       | Female   |    43 |        2 |    125511 |               1 |           1 |                1 |           79084.1 |        0 |