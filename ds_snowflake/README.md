Snowflake Connector library provides an easy way to connect to Snowflake server. 
This connector allows you to connect to Snowflake, run queries and analyze results. It is supported in both Python environments or cnvrg Flows

In addition, you can create CSVs, Dataframes and store them to a versioned dataset in cnvrg. 

## Prerequisites
---
The following library need to be installed before using the library:

<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>pip install snowflake-connector-python</code></pre></div>


## Running in interactive mode (Notebooks / IDE)
---
<div style='font-size:0.9rem; font-weight:bold;'>Loading the library</div>
<p></p>
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>from cnvrg import Library
library = Library('cnvrg/snowflake_connector')
library.load()</code></pre></div>
<p></p>
<div style='font-size:0.9rem; font-weight:bold;'>Connecting to the data source</div>
<p></p>
Connect to your data source using the following single line of code. It is recommended to store 
credentials as environment variables.

<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<pre><code class='python'>library.connect(warehouse="SNOWFLAKE_WAREHOUSE",
                account="SNOWFLAKE_ACCOUNT",
                database="SNOWFLAKE_DATABASE",
                schema="SNOWFLAKE_SCHEMA")</code></pre></div>
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
<div style='font-size:0.9rem; font-weight:bold;'>Close Connection</div>
<p></p>
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<pre>
<code class='python'>library.close_connection()</code></pre></div>

## Running as an executable (Flow / Job)

You can also run this library as part of a Flow that will fetch data and store it as a 
dataset in cnvrg.io. This is useful for data/ML pipelines that are running recurringly or on trigger.

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
