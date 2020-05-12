This library is made to run a batch prediction using a CSV with an existing cnvrg batch prediction endpoint.

## Notes for this library
This library will run a batch prediction on a supplied CSV. It will submit each row of the CSV as a prediction to an existing batch prediction endpoint and save the resulting inferences to a new CSV and upload the outut CSV to a dataset of your choosing.

## Parameters

```--endpoint_id``` - string, required. The ID of the batch predict endpoint you are using. The endpoint must already be deployed.

```--input_file``` - string, required. The path of a CSV file with the data for the batch prediction. For example, `/data/dataset/data.csv`.

```--output_file``` - string, required. The path of the CSV output file. For example, `/cnvrg/output.csv`.

```--dataset``` - string, required. The name of an existing dataset that the output CSV will be uploaded to.
