import tensorflow as tf
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', action='store', help="name for output model will result in <value>.tflite",type=str)
    parser.add_argument('-i', '--input', action='store', help="name of input model <name>.h5",
                      required=True, type=str)

    parser.add_argument('--project_dir', action='store', dest='project_dir',
                        help="""--- For inner use of cnvrg.io ---""")

    parser.add_argument('--output_dir', action='store', dest='output_dir',
                        help="""--- For inner use of cnvrg.io ---""")

    args = parser.parse_args()

    print(args)

    #TODO - add args parse for [ in_model , out_model)


    input_model = args.input
    output_model = args.model

    model = tf.keras.models.load_model(input_model+".h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(output_model+".tflite", "wb").write(tflite_model)


    ##TODO ADVANCED! Option to include testing of the in and out models to validate accuracy against specified test data


