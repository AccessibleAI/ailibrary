from argparse import ArgumentParser, SUPPRESS

#TODO --- ADD your imports Classes and methods

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # TODO --- ADD your parameters
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', action='store', help="name for output model will result in <value>.tflite",
                      required=True, type=str)
    parser.add_argument('-i', '--input', action='store', help="name of input model <name>.h5",
                      required=True, type=str)

    parser.add_argument('--project_dir', action='store', dest='project_dir',
                        help="""--- For inner use of cnvrg.io ---""")

    parser.add_argument('--output_dir', action='store', dest='output_dir',
                        help="""--- For inner use of cnvrg.io ---""")

    args = parser.parse_args()

    print(args)
    print("TEMPLATE AI COMPONENT WORKS!!")
    #TODO --- execut your AI Component code
    #main(args)