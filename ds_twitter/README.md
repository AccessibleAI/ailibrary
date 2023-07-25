#  cnvrg pull_twitter_data

This README describes the pull_twitter_data library

## How It Works

Running the library with your twitter developer bearer token an a search string,
will search twitter for twitts from the last 3 days, containing the given search string.
the search resutl will be saves as a csv file in a new cnvrg dataset.

## Running

python3 pull_twitter_data.py --token [YOUR_TWITTER_DEVELOPER_BEARER_TOKEN] --term [YOUR_SEARCH_STRING]


## Additional optional parameters:



'--datase': dataset for saving the result.  default=[by default we don't save to datset]

'--output_file': filename for saving the data. default='twitts.csv'

'--max_twitts': max num of twitts. default=500

'--end_point':  twitter api endpoint. default='recent'

'--days_back':  num of days back for recent twitts. default=3
 

## See Also
* [LINKS](../README.md)

