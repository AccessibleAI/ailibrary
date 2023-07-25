
## Cnvrg S3-client for AI-LIBRARY

This library is made to download ojectfiles from S3.

Install the cnvrg-sdk 

```pip install -i https://test.pypi.org/simple/ cnvrg-new```

## Authorization
Authorization is performed using environment variables under **project -> settings -> secrets**

<img width="1234" alt="secrets" src="https://user-images.githubusercontent.com/88431066/138663758-a71b796b-ad33-46a3-9566-255a8e674c30.png">


**AWS_ACCESS_KEY_ID** - Specifies an AWS access key

**AWS_SECRET_ACCESS_KEY** - Specifies the secret key associated with the access key. This is essentially the "password" for the access key.

## Parameters

``` bucketname ``` - string s3 bucketname

``` file ``` - string The file that inisde the bucket

``` localdir ``` - string (default = /cnvrg/output). When project is connected to git - The path of the folder with experiment artifacts to be synced as the commit for the experiment. (Default is project's default)

```cnvrg_dataset``` - string, The path to save the dataset files.

![image](https://user-images.githubusercontent.com/88431066/138676166-3aa696bb-f43f-4d12-80a8-2b88945787dc.png)


#### Download objects

```bash
$ python s3-connector.py --endpoint https://s3.amazonaws.com download --bucketname cnvrg-bucket --file file.csv --localdir /cnvrg/output --cnvrg_data <datasetname>

usage: s3-connector.py download [-h] [--nopbar] [-l LOCALDIR] [-o] [-v VERSIONID] (-f FILENAME | -p PREFIX) bucket

positional arguments:
  bucket                Bucket Name

optional arguments:
  -h, --help            show this help message and exit
  --bucketname              Disable progress bar
  -l LOCALDIR, --localdir LOCALDIR
                        Local directory to save downloaded file. Default current directory
  -o, --overwrite       Overwrite local destination file if it exists. Default false
  -v VERSIONID, --versionid VERSIONID
                        Object version id
  -f FILENAME, --file FILENAME
                        Download a specific file
  -p PREFIX, --prefix PREFIX
                        Download recursively all files with a prefix.
```

