from argparse import ArgumentParser, SUPPRESS
import cnvrg_redshift
from pathlib import Path
#TODO --- ADD your imports Classes and methods

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--host', action="store", dest='host', type=str, default='')
    parser.add_argument('--database', action="store", dest='database', type=str, default='')
    parser.add_argument('--trusted_connection', action="store", dest='trusted_connection', type=str, default=True)
    parser.add_argument('--port', action="store", dest='port', type=int)
    parser.add_argument('--query', action="store", dest='query', type=str, default='')
    parser.add_argument('--csv', dest='csv', default=False, action='store_true')
    parser.add_argument('--no-csv', action='store_false', dest='csv')
    parser.add_argument('--no-upload', action="store", dest='upload', default=True, nargs='?', const=False) #if not set, csv will upload
    parser.add_argument('--filename', action="store", dest='filename', type=str, default='redshift_extracted_query.csv')
    parser.add_argument('--df', action="store", dest='df', default=False, nargs='?', const=True)
    parser.add_argument('--cnvrg_ds', action='store', dest='cnvrg_ds', type=str, default='redshift_ds')
    parser.add_argument('--output_dir', action='store', dest='output_dir', type=str, default='/cnvrg/output')


    args = parser.parse_args()
    
    if args.output_dir[-1] == '/':
        args.output_dir = args.output_dir[:-1]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    full_path= args.output_dir + '/' + args.filename
    
    redshift = cnvrg_redshift.connect(host=args.host, db=args.database, port=args.port, trusted=args.trusted_connection)

    if args.csv:
        print(args.csv)
        print(args.upload)
        cnvrg_redshift.to_csv(conn=redshift, query=args.query, filename=full_path)
        if args.upload:
            cnvrg_redshift.upload_to_ds(args.cnvrg_ds, filename=args.filename, output_dir=args.output_dir)
        exit(0)
    if args.df:
        df = cnvrg_redshift.to_df(conn=redshift, query=args.query)
        #TODO: need to add df params, and save to file vefore exit
        exit(0)