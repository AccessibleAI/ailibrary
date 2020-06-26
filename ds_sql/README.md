SQL Connector library provides an easy way to connect to different SQL databases using ODBC. 
This connector allows you to connect to DB, run queries and analyze results. It is supported in both Python environments or cnvrg Flows

In addition, you can create CSVs, Dataframes and store them to a versioned dataset in cnvrg. 


## Parameters
---

```--query``` - str, required. The SQL query to be executed

## Prerequisites
---
The following prerequisites need to be installed before using the library:<br>
- [install the ODBC driver](https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15)

- Install pyodbc library<br>
<code>pip install pyodbc</code>

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

Connect to the SQL Server:<br>
<code>from cnvrg import Library</code><br>
<code>library = Library('cnvrg/sql_connector')</code><br>
<code>library.load()</code><br>
<code>library.connect(driver="DRIVER VERSION",server="SERVER", database="DATABASE")</code><br>

## Using the Library
---

### Executing a query

Using the `library.query(query)` will return a cursor object, which can be later used to retrieve the relevant results<br>
Example:<br>
<code>results = library.query("SELECT * FROM users")</code>
<br><code>results.fetchall()<br></code>

### Create a Dataframe from query
Example:<br>
<code>df = library.to_df("SELECT * FROM users")</code>
<br>
[You can send additional parameters](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql_query.html)
<code>df = library.to_df("SELECT * FROM users",index_col=[surname,firstname])</code><br>

### Create a csv file from query
Creates a csv file (with the given filename path) with the results

Example:<br>
<code>library.to_csv("SELECT * FROM users","results.csv")</code>

### Close Connection
Closes the connection
<br>
<code>
library.close_connection()
</code>