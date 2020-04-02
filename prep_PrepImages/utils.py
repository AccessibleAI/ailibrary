"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

utils.py
==============================================================================
"""

def types_casting(args):
	args.height = None if args.height == 'None' else int(args.height)
	args.width = None if args.width == 'None' else int(args.width)
	args.channels = None if args.channels == 'None' else int(args.channels)
	args.add_noise = (args.add_noise == 'True')
	args.denoise = (args.denoise == 'True')
	args.segmentation = (args.segmentation == 'True')
	args.blur = (args.blur == 'True')
	args.zip_all = (args.zip_all == 'True')
