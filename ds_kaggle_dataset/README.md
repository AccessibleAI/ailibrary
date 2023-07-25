This library is made to download datasets from Kaggle

## Parameters

```--kaggle_dataset_name``` - string, required. The name of the dataset.

```--target_path``` - string (default = /cnvrg/output). The path to save the dataset files to

```--cnvrg_dataset``` - string (default = None). If provided then the files that were downloaded from Kaggle will be uploaded straight into the provided dataset, if the dataset does not exist a new one will be created

```--file_name``` - string (default = None). If provided then the library will download this specific file from the dataset

## Finding the dataset name
To grab a dataset name from Kaggle, navigate into your desired dataset page on Kaggle, afterwards copy the dataset name from the end of the url.
Paste the Kaggle dataset name into the `kaggle_dataset_name` field.

## Authentication

You can get your Kaggle API credentials by going into the user profile and under "Account" press "Create API Token".

This will download a "kaggle.json" file with your credentials.

It is recommended to use environment variables as authentication method. This library expects the following env variables:
    
* `KAGGLE_KEY` - The Kaggle API key
* `KAGGLE_USERNAME` - The Kaggle API username

The environment variables can be stored securely in the project secrets in cnvrg. 


