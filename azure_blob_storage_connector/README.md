
## Cnvrg Azure-client for AI-LIBRARY

This library is made to download/upload blobs from AzureBlobStorage.

Install the cnvrg-sdk 

```pip install -i https://test.pypi.org/simple/ cnvrg-new```

Python: 3.7+

## Authorization
Authorization is performed using environment variables under **project -> settings -> secrets**

![image](https://github.com/snirbenyosef/snirlabpub/raw/master/img.png)

## Mandatory envs
**AZURE_STORAGE_ACCOUNT_NAME** - the name of the storage account

#### Set the environment variables depends on your Auth choice:
**AZURE_STORAGE_CONNECTION_STRING** - the connection string to your storage account
**AZURE_STORAGE_ACCESS_KEY** - the storage account access key


## Parameters

``` --container_name ``` - string Azure Container name

``` --file_name ``` - string The file name

``` --ouput ``` - string (default = env["CNVRG_WORKDIR"]).Define where to download files or upload from (Default is project's default)

```--cnvrg_dataset``` - string, the name of the dataset

```--prefix``` - string, part of a string name to upload

#### Only One Action
```--download``` - flag, download a single file by file name
```
python3 azure-connector.py --download --container_name="cnvrg-container" --output="/path/to/dir" --file_name="file.jpg"
```

```--batch_download``` - flag, part of a string name to upload
```
python3 azure-connector.py --batch_download  --container_name="cnvrg-container" --output="/path/to/dir"
```

```--upload``` - flag, part of a string name to upload
```
python3 azure-connector.py --upload --container_name="cnvrg-container" --output="/path/to/dir" --file_name="file.jpg"
```

```--batch_upload``` - flag, part of a string name to upload
```
python3 azure-connector.py --batch_upload --container_name="cnvrg-container" --output="/path/to/dir"
```
![image](https://github.com/snirbenyosef/snirlabpub/raw/master/imageai.png)


### Examples

Batch Download with a prefix (.png)
```
python3 azure-connector.py --batch_download --prefix=".png" --container_name="cnvrg-container" --output="/path/to/dir"
```

Batch Download and upload to dataset

```
python3 azure-connector.py --batch_download --dataset="dataset_name" --container_name="cnvrg-container" --output="/path/to/dir"
```

Batch Download with a prefix (img_test_)

```
python3 azure-connector.py --batch_upload --prefix="img_test_" --container_name="cnvrg-container" --output="/path/to/dir"
```