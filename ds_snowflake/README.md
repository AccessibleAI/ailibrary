SQL Connector library provides an easy way to connect to different SQL databases using ODBC. 
This connector allows you to connect to DB, run queries and analyze results. It is supported in both Python environments or cnvrg Flows

In addition, you can create CSVs, Dataframes and store them to a versioned dataset in cnvrg. 


## Parameters
---

```--query``` - str, required. The Snowflake query to be executed
```--output_file``` - str, optional. Filename to store the query as a CSV

## Config parameters
---

```--warehouse``` - Snowflake warehouse name 

```--account``` - account name

```--database``` - database name

```--schema``` - schema name


## Authentication
---
It is recommended to use environment variables as authentication method. This library expects the following env variables:

* `SNOWFLAKE_PASSWORD` - account
* `SNOWFLAKE_USER` - username

You can also set additional parameters as environment variables and not pass them as arguments:

* `SNOWFLAKE_WAREHOUSE` - warehouse
* `SNOWFLAKE_ACCOUNT` - username

The environment variables can be stored securely in the project settings in cnvrg. 

You can also pass credentials as arguments: `user` and `password`

