
# Redshift Connector
Redshift Connector library provides an easy way to connect to AWS Redshfit database using ODBC. This connector allows you to connect to DB, run queries and analyze results. It is supported in both Python environments or cnvrg Flows

In addition, you can create CSVs, Dataframes and store them to a versioned dataset in cnvrg.
 

## Prerequisites

Authentication to the Redshift cluster is done with the Redshift username and passwords. In `Project Settings -> Secrets` add the following two environment variables:

* `REDSHFIT_PWD` - password
* `REDSHIFT_UID` - uid for the database



## Running in interactive mode (Notebooks / IDE)
#### Loading the library

<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>from cnvrg import Library
library = Library('cnvrg/redshift_connector')
library.load()
conn = library.run()</code></pre></div>

#### Connecting to the data source
Connect to your data source using the following single line of code. You will need to store your 
credentials as secrets.


<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>library.connect(host=str,
	database=str, 
	port=int, //default 5439
	query=str,
	csv=bool, //default True
	df=bool, //default False
	upload=bool, //default True
	filename=str, //default: redshift_extracted_query.csv
	output_dir=str, //default: /cnvrg/output
	cnvrg_ds=str, //default: redshift_ds
	)</code></pre></div>

#### Executing a query

Using the `library.query(query)` will return a cursor object, which can be later used to retrieve the relevant results


<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>results = library.query("SELECT * FROM users")
results.fetchall()</code></pre></div>

### Load as Dataframe / CSV
You could also run the query and retrieve it as dataframe / CSV file automatically using the following lines of code:
#### Create a dataframe from query in a single line
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>df = library.to_df("SELECT * FROM users")</code></pre></div>

#### Create a csv file (with the given filename path) with the results

<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>library.to_csv("SELECT * FROM users","results.csv")</code></pre></div>


### Close Connection
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>library.close_connection()</code></pre></div>


## Running as an executable (Flow / Job)
You can also run this library as part of a Flow that will fetch data and store it as a 
dataset in cnvrg.io. This is useful for data/ML pipelines that are running recurringly or on trigger.

### Executable Parameters
`--host=HOST_URL`
`--database=DB_NAME` 
`--port=PORT_NUM` default: `5439`
`--query='REDSHIFT_QUERY`
`--df` save query to Pandas DataFrame object, default: `False`
`--csv or --no-csv` save query to csv file. If the parameter isn't specified, the query will be saved to csv. 
`--filename=FILE.csv` default: `redshift_extracted_query.csv`
`--output_dir=/PATH/TO/FILE` default: `/cnvrg/output`
`--upload=Bool` upload csv file to a dataset, default: `True`
`--cnvrg_ds=CNVRG_DS_ID` default: `redshift_ds`
