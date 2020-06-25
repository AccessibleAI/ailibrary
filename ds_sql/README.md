SQL Connector library provides an easy way to connect to different SQL databases using ODBC. 
This connector allows you to connect to DB, run queries and analyze results. It is supported in both Python environments or cnvrg Flows

In addition, you can create CSVs, Dataframes and store them to a versioned dataset in cnvrg. 


## Parameters
---

```--query``` - str, required. The SQL query to be executed

## Config parameters
---

```--database``` - database name 

```--server``` - Host (ip/domain) of your PostgreSQL database

```--driver``` - ODBC Driver, for example: `ODBC Driver 17 for SQL Server Environment` 


## Authentication
---
It is recommended to use environment variables as authentication method. This library expects the following env variables:
* `SQL_PWD` - password
* `SQL_UID` - uid for the database

The environment variables can be stored securely in the project settings in cnvrg. 

You can also pass credentials as arguments: `uid` and `pwd`
