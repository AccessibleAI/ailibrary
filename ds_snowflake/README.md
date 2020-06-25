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

## Prerequisites
---
The following library need to be installed before using the library:
```python
pip install snowflake-connector-python
```
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

```python
from cnvrg import Library
library = Library('cnvrg/snowflake_connector')
library.load()
library.connect(warehouse="SNOWFLAKE_WAREHOUSE",account="SNOWFLAKE_ACCOUNT", database="SNOWFLAKE_DATABASE",schema="SNOWFLAKE_SCHEMA")
```

## Using the Library
---

### Executing a query

Using the `library.query(query)` will return a cursor object, which can be later used to retrieve the relevant results
```python
results = library.query("SELECT * FROM users")
results.fetchall()
```
### Create a Dataframe from query
```python
df = library.to_df("SELECT * FROM users")
```
### Create a csv file from query
This will create a csv file (with the given filename path) with the results
```python
library.to_csv("SELECT * FROM users","results.csv")
```
### Close Connection
This will close the connection
```python
library.close_connection()
```