# Data Validation for Recommender System
## _Making sure that the input file is correct and gives reasonable results_

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Data Validation librarty serves as a validation tool that has to be run before using the algorithms for the Recommender System to make sure that the input gets converted into a standard format which can be used further for other algorithms.

## Features
- Upload a file containing the ratings data and save its path
- Checks on whether the data is sanitised
- conversion code to map the data into a pre-ordained format
- output files consisting of the mapped file and the user/item dictionaries

## Arguments
- `--input_path` refers to the name and path of the file which contains the ratings. 
    | user_id | item_id  | rating  |
    | :---:   | :-: | :-: |
    | John | 217 | 4 |
    | Patrick | 4 | 1 |
    | Patrick | 56 | 2 |
   - For best reuslts, the ratings must lie within 0-10. However, implicit ratings (0,1) will work as well.
   - The file must be comma delimited and the headings must be in lower case.
   - The file path would contain the name of the dataset as well which contains the file. It would look like this :- `\dataset_name\input_file.csv`. Ensure that the dataset is of the text format.
   - The values can contain float values as well as integer values.
   - This library is the pre-requisite for all the other algorithms
- Model Artifacts
     -  `--item_dict1.csv` refers to the item mapping file that contains the real and converted item ids.
        | originalitem_id  | item_id  |
        | :---:   | :-: |
        | 217 | 3 |
        | 4 | 1 |
        | 56 | 2 |
       - `--user_dct1` refers to the user mapping file that contains the real and converted user ids.
            | originaluserid  | user_id  |
            | :---:   | :-: |
            | John | 0 |
            | Patrick | 1 |
            | Bernie | 2 |
       - `--ratings_translated.csv` refers to the file that contains the converted user-ids and item-ids alongside the ratings, to be used as an input for the train test split module.
            | user_no | item_no  | rating  |
            | :---:   | :-: | :-: |
            | 0 | 3 | 4 |
            | 1 | 1 | 1 |
            | 1 | 2 | 2 |

## How to run
```
cnvrg run  --datasets='[{id:"movies_rec_sys",commit:"f6e1126cedebf23e1463aee73f9df08783640400"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 data_validation.py --filename /data/movies_rec_sys/ratings_2.csv
```