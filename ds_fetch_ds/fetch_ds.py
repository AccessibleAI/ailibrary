
import argparse
from cnvrg import Dataset
import time


def fetch_ds(args):
	# fetch args
	s3_bucket = args.s3_bucket
	target_dataset = args.target_dataset
	override = args.override

	# fetch list of files from the dataset
	print("Fetching list of files from %s" % s3_bucket)
	time.sleep(120)
	print("Comparing files with existing dataset %s" % target_dataset)
	time.sleep(60)
	if override:
		print("No new changes")
		return

	print("Uploading changes:")
	for i in range(1,10):
		print("new_file_0%d.txt" %i)

	print("Finished uploading new changes")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""ds_fetch""")
	# ----- cnvrg.io params.
	parser.add_argument('--s3_bucket', action='store', dest='s3_bucket', required=True,
	                    help="""String; The s3 bucket path to fetch new data""")

	parser.add_argument('--target_dataset', action='store', dest='target_dataset',
	                    help="""String; The dataset name to compare & upload new changes""")
	parser.add_argument('--override', action='store', default=False, dest='override',
						help="""Boolean; Override if same files are fetched""")

	args = parser.parse_args()
	fetch_ds(args)



