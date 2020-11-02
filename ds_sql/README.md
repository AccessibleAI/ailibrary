SQL Connector library provides an easy way to connect to different SQL databases using ODBC. 
This connector allows you to connect to DB, run queries and analyze results. It is supported in both Python environments or cnvrg Flows

In addition, you can create CSVs, Dataframes and store them to a versioned dataset in cnvrg. 

## Prerequisites
---
The following prerequisites need to be installed before using the library:

* ODBC driver: [Install the ODBC driver](https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver15)

* Install pyodbc library
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>pip install pyodbc</code></pre></div>


## Running in interactive mode (Notebooks / IDE)
---
<div style='font-size:0.9rem; font-weight:bold;'>Loading the library</div>
<p></p>
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>from cnvrg import Library
library = Library('cnvrg/sql_connector')
library.load()</code></pre></div>
<p></p>
<div style='font-size:0.9rem; font-weight:bold;'>Connecting to the data source</div>
<p></p>
Connect to your data source using the following single line of code. It is recommended to store 
credentials as secrets.


<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<pre><code class='python'>library.connect(driver="DRIVER VERSION",
                server="SERVER", 
                database="DATABASE")</code></pre></div>
<p></p>
<div style='font-size:0.9rem; font-weight:bold;'>Executing a query</div>
<p></p>
Using the `library.query(query)` will return a cursor object, which can be later used to retrieve the relevant results

<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<pre><code class='python'>results = library.query("SELECT * FROM users")
results.fetchall()</code></pre></div>
<p></p>
<div style='font-size:0.9rem; font-weight:bold;'>Load as Dataframe / CSV</div>
<p></p>
You could also run the query and retrieve it as dataframe / CSV file automatically using the following lines of code:
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<pre><code class='python'># Create a dataframe from query in a single line

df = library.to_df("SELECT * FROM users")

# Create a csv file (with the given filename path) with the results

library.to_csv("SELECT * FROM users","results.csv")</code></pre></div>
<p></p>
<div style='font-size:0.9rem; font-weight:bold;'>Push the dataframe back to the sql server</div>
<p></p>
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<p></p>
<pre>
<code class='python'>library.to_sql(df=df,table_name='CARS',if_exists='replace', index = False))</code></pre></div>
<p></p>

<div style='font-size:0.9rem; font-weight:bold;'>Close Connection</div>
<p></p>
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<p></p>
<pre>
<code class='python'>library.close_connection()</code></pre></div>
<p></p>

## Running as an executable (Flow / Job)
---
You can also run this library as part of a Flow that will fetch data and store it as a 
dataset in cnvrg.io. This is useful for data/ML pipelines that are running recurringly or on trigger.

<div style='font-size:0.9rem; font-weight:bold;'>Executable Parameters</div>
<p></p>
```--query``` - str, required. The Snowflake query to be executed
<p></p>
<div style='font-size:0.9rem; font-weight:bold;'>Config & Auth Parameters</div>
<p></p>
```--database``` - database name 

```--server``` - Host (ip/domain) of your PostgreSQL database

```--driver``` - ODBC Driver, for example: `ODBC Driver 17 for SQL Server Environment` 

<p></p>
It is recommended to use environment variables as authentication method. This library expects the following env variables:

* `SQL_PWD` - password
* `SQL_UID` - uid for the database

The environment variables can be stored securely in the project settings in cnvrg. 

You can also pass credentials as arguments: `uid` and `pwd`
