from argparse import ArgumentParser, SUPPRESS
import bigquery_connector
import sys
    
if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--project_dir', action='store', dest='project_dir',
                        help="""--- For inner use of cnvrg.io ---""")
    parser.add_argument('--output_dir', action="store", dest='output_dir', type=str, default='')
    parser.add_argument('--query', action="store", dest='query', type=str, default= None)
    parser.add_argument('--csv', action="store", dest='csv', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--cnvrg_ds', action="store", dest='filename', type=str, default='')
    parser.add_argument('--filename', action="store", dest='filename', type=str, default='')
    parser.add_argument('--df', action="store", dest='df', type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()
    if args.query is None:
        print("Query can't be empty")
        sys.exit(1)
    if args.df:
        bigquery_connector.to_df(args.query)
    if args.csv:
        bigquery_connector.to_csv(args.query, args.filename, args.output_dir)
    if args.query:
        bigquery_connector.run(args.query)
    if args.cnvrg_ds:
        bigquery_connector.upload_to_ds(args.cnvrg_ds, args.query, args.filename, args.output_dir)
        
        
