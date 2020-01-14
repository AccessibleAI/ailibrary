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


2) House Prices Data Set  
Input data:  

|    |   bedrooms |   bathrooms |   sqft_living |   sqft_lot |   floors |   waterfront |   view |   condition |   grade |   sqft_above |   sqft_basement |   yr_built |   price |
|---:|-----------:|------------:|--------------:|-----------:|---------:|-------------:|-------:|------------:|--------:|-------------:|----------------:|-----------:|--------:|
|  0 |          3 |        1    |          1180 |       5650 |        1 |            0 |      0 |           3 |       7 |         1180 |               0 |       1955 |  221900 |
|  1 |          3 |        2.25 |          2570 |       7242 |        2 |            0 |      0 |           3 |       7 |         2170 |             400 |       1951 |  538000 |
|  2 |          2 |        1    |           770 |      10000 |        1 |            0 |      0 |           3 |       6 |          770 |               0 |       1933 |  180000 |
|  3 |          4 |        3    |          1960 |       5000 |        1 |            0 |      0 |           5 |       7 |         1050 |             910 |       1965 |  604000 |
|  4 |          3 |        2    |          1680 |       8080 |        1 |            0 |      0 |           3 |       8 |         1680 |               0 |       1987 |  510000 |

Given command: ```--data="~/house_data.csv" --scale="{'grade': '1.5:10'}" --one_hot=[bedrooms,floors,condition] --normalize=[sqft_living,yr_built] --target=price```  
Output data:  

|    |   bathrooms |   sqft_living |   sqft_lot |   waterfront |   view |   grade |   sqft_above |   sqft_basement |   yr_built |   bedrooms_0 |   bedrooms_1 |   bedrooms_2 |   bedrooms_3 |   bedrooms_4 |   bedrooms_5 |   bedrooms_6 |   bedrooms_7 |   bedrooms_8 |   bedrooms_9 |   bedrooms_10 |   bedrooms_11 |   bedrooms_33 |   floors_1.0 |   floors_1.5 |   floors_2.0 |   floors_2.5 |   floors_3.0 |   floors_3.5 |   condition_1 |   condition_2 |   condition_3 |   condition_4 |   condition_5 |   price |
|---:|------------:|--------------:|-----------:|-------------:|-------:|--------:|-------------:|----------------:|-----------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------:|
|  0 |        1    |     0.0657312 |       5650 |            0 |      0 | 5.75    |         1180 |               0 |  0.0272953 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             0 |            1 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             1 |             0 |             0 |  221900 |
|  1 |        2.25 |     0.16839   |       7242 |            0 |      0 | 5.75    |         2170 |             400 |  0.0253102 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             0 |            0 |            0 |            1 |            0 |            0 |            0 |             0 |             0 |             1 |             0 |             0 |  538000 |
|  2 |        1    |     0.0354505 |      10000 |            0 |      0 | 5.04167 |          770 |               0 |  0.0163772 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             0 |            1 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             1 |             0 |             0 |  180000 |
|  3 |        3    |     0.123338  |       5000 |            0 |      0 | 5.75    |         1050 |             910 |  0.0322581 |            0 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             0 |            1 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             0 |             0 |             1 |  604000 |
|  4 |        2    |     0.102659  |       8080 |            0 |      0 | 6.45833 |         1680 |               0 |  0.0431762 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             0 |            1 |            0 |            0 |            0 |            0 |            0 |             0 |             0 |             1 |             0 |             0 |  510000 |

3) Churn From Banks Data Set (with empty values)  
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