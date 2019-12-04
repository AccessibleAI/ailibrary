"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

casting.py
==============================================================================
"""

def cast_types(args):
	# test_split
	args.test_size = float(args.test_size)

	# epochs.
	args.epochs = int(args.epochs)

	# batch_size.
	args.batch_size = int(args.batch_size)

	# img_width.
	args.image_width = int(args.image_width)

	# img_height.
	args.image_height = int(args.image_height)

	# conv_width.
	args.conv_width = int(args.conv_width)

	# conv_height.
	args.conv_height = int(args.conv_height)

	# pool_width.
	args.pool_width = int(args.pool_width)

	# pool_height.
	args.pool_height = int(args.pool_height)
	# -------- #
	return args
