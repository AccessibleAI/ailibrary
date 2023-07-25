# Train Test Split for Recommender System
## _Splitting the data into train and test samples to get evaluate the model and avoid overfitting_

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Train Test Split library serves as a module for a stratified splitting of the data into test and train samples to make sure that each user's information is split proportionally into both test and train datasets.

## Features
- Either run the library in conjunction with the data_validation library or upload a ratings file with properly enumerated user and item ids (integer values starting from 0)
- Makes sure that the split happens at the user level and not an overall level
- outputs both train and test files 

## Arguments
- `--filename` refers to the name and path of the file which contains the ratings and user/item ids in the correct format.
    | user_no | item_no  | rating  |
    | :---:   | :-: | :-: |
    | 0 | 3 | 4 |
    | 1 | 1 | 1 |
    | 1 | 2 | 1 |
   - The user_no and item_no must start at 0. 
   - For best reuslts, the ratings must lie within 0-10. However, implicit ratings (0,1) will work as well.
   - The file must be comma delimited and the headings must be in lower case.
   - The file path would contain the name of the dataset as well which contains the file. It would look like this :- `\dataset_name\input_file.csv`. Ensure that the dataset is of the text format.
   - The values can contain float values as well as integer values.
   - This library is the pre-requisite for all the other algorithms
 - Model Artifacts
        `--train_whole.csv` refers to the training dataset, around 75% of the data.
        `--test_whole` refers to the test dataset, around 25% of the data
        Both will look similar, with the exception that data in one won't be present in the other.
    | user_no | item_no  | rating  |
    | :---:   | :-: | :-: |
    | 0 | 3 | 4 |
    | 1 | 1 | 1 |
    | 1 | 2 | 2 |

## How to run
```
cnvrg run  --datasets='[{id:"movies_rec_sys",commit:"f6e1126cedebf23e1463aee73f9df08783640400"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 TTS.py --filename /input/data_validation/ratingstranslated.csv
```