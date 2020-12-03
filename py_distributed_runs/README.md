Run transformation in distributed way
## Parameters

### cnvrg.io parameters

```--s3_bucket``` - str, required. Path to the S3 bucket to fetch new changes from

```--target_dataset``` - str, required. Dataset name to upload the new changes to 

```--max_groups``` - str, required. Max parallel compute  

```--items_per_group``` - str, required. Items per execution 

```--script``` - str, required. script to run on the transformations 
