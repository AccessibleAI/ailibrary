from cnvrg import Endpoint
import argparse
import json

parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('--endpoint_id', action="store", dest='slug', type=str, default='')
#parser.add_argument('--model_id', action="store", dest='model', type=str, default='')

args = parser.parse_args()
endpoint_id = args.slug
#model_id = args.model

#fetch endpoint details
endpoint = Endpoint(endpoint_id)
if endpoint is None:
    print(f"Can't find Endpoint {endpoint_id}.")
    exit(1)

#print(f"Rolling back Model {model_id} in Endpoint {endpoint_id}.")
#resp = endpoint.rollback(model_id)

print(f"Rolling back Model in Endpoint {endpoint_id}.")
resp = endpoint.rollback()
if resp.get("status") == 200:
    print("Model rolled back successfully.")
else:
    print("Could not roll back model.")

