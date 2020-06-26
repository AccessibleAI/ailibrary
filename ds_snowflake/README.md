Snowflake Connector library provides an easy way to connect to Snowflake server. 
This connector allows you to connect to Snowflake, run queries and analyze results. It is supported in both Python environments or cnvrg Flows

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

<code>pip install snowflake-connector-python</code>

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

Connect to Snowflake server:<br>
<code>from cnvrg import Library</code><br>
<code>library = Library('cnvrg/snowflake_connector')</code><br>
<code>library.load()</code><br>
<code>library.connect(warehouse="SNOWFLAKE_WAREHOUSE",account="SNOWFLAKE_ACCOUNT", database="SNOWFLAKE_DATABASE",schema="SNOWFLAKE_SCHEMA")</code><br>

## Using the Library
---

### Executing a query

Using the `library.query(query)` will return a cursor object, which can be later used to retrieve the relevant results

Example:<br> 
<code>results = library.query("SELECT * FROM users")</code><br>
<code>results.fetchall()</code><br>

### Create a Dataframe from query

Example:<br>
<code>df = library.to_df("SELECT * FROM users")</code>

### Create a csv file from query
Create a csv file (with the given filename path) with the results

Example:<br>
<code>library.to_csv("SELECT * FROM users","results.csv")</code>

### Close Connection
Closes the connection

<code>library.close_connection()</code>