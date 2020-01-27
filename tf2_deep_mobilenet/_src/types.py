"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

types.py
==============================================================================
"""

def _cast(args):
	args.test_size = float(args.test_size)
	args.epochs = int(args.epochs)
	args.batch_size = int(args.batch_size)
	args.image_width = int(args.image_width)
	args.image_height = int(args.image_height)
	args.conv_width = int(args.conv_width)
	args.conv_height = int(args.conv_height)
	args.pool_width = int(args.pool_width)
	args.pool_height = int(args.pool_height)
	args.dropout = float(args.dropout)

	return args
