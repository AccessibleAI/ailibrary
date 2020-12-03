
import argparse
from cnvrg import Experiment
import time
import numpy as np


def run_distributed_jobs(args):
	# fetch args
	target_dataset = args.target_dataset
	s3_bucket = args.s3_bucket
	items_per_group = float(args.items_per_group)
	script = args.script

	# fetch list of files from the dataset
	print("Fetching list of files from %s" % s3_bucket)
	time.sleep(120)
	list_files = ["1","2","3","4","5","6","7","8","9","10"]
	sub_lists = np.array_split(list_files, items_per_group)
	for i in range(0,len(sub_lists)):
		print("running distributed experiment %d" %i)
		Experiment.run("python3 run_subset.py", compute="small")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""py_distributed_runs""")
	# ----- cnvrg.io params.
	parser.add_argument('--max_parallel', action='store', dest='max_parallel', required=False,
	                    help="""String; The s3 bucket path to fetch new data""")
	parser.add_argument('--target_dataset', action='store', dest='target_dataset', required=True,
						help="""String; The dataset to upload the files to""")
	parser.add_argument('--s3_bucket', action='store', dest='max_parallel', required=True,
						help="""String; The s3 bucket path to fetch new data""")
	parser.add_argument('--items_per_group', action='store', dest='group', required=False,
						help="""String; The number of item to execute per group""")
	parser.add_argument('--script', action='store', dest='script',
	                    help="""String; The target script to run""")

	args = parser.parse_args()
	run_distributed_jobs(args)



