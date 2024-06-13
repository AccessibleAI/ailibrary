# Compare Function , for Recommender System
_Compares the best model out of all possible runs of all 6 models in the recommendation system blueprint. Chooses the model with the maximum value of the composite metric. The Composite Metric is defined later: -
_
[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Compare library is a comparison library in recommendation system blueprint to get the select the best model and then transfer its artifacts over to the predict function deployed as a webapp,.

## Features
- No parameters required. Just need to set the condition in the 'Flows'/'Experiments' according to the correct metric
- Even if two separate iterations are run of a particular model, the compare selects the best out of those iterations as well as other models.
- It works on the environment variables and selects the model combination with the "passed condition variable as "TRUE".
- It finally outputs the 'recommend.csv' file of the best model combination.

## Arguments
None

## Model Artifacts 
- `--recommend.csv` refers to the file which contains items not rated by the user, making them ideal for final recommendation. This file can be rather large, depending on the number of distinct items and users in the original file.
    | user_no | item_no  | score  |
    | :---:   | :-: | :-: |
    | 0 | 5 | 2.6 |
    | 1 | 3 | 1.7 |
    | 1 | 6 | 4.4 |

## How to run
```
cnvrg run  --datasets='[{id:"movies_rec_sys",commit:"f6e1126cedebf23e1463aee73f9df08783640400"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 test.py
```