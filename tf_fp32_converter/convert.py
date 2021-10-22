import tensorflow as tf
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from keras.models import load_model


def save_frozen_pb(model, mod_path):
    # Convert Keras model to ConcreteFunction

    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    # Generate frozen pb
    tf.io.write_graph(graph_or_graph_def=frozen_model.graph,
                      logdir=".",
                      name=mod_path,
                      as_text=False)



if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', action='store', help="name for output model will result in <value>.pb",
                      required=True, type=str)
    parser.add_argument('-i', '--input', action='store', help="name of input model <name>.h5",
                      required=True, type=str)

    parser.add_argument('--project_dir', action='store', dest='project_dir',
                        help="""--- For inner use of cnvrg.io ---""")

    parser.add_argument('--output_dir', action='store', dest='output_dir',
                        help="""--- For inner use of cnvrg.io ---""")

    parser.add_argument('--compute', action='store', dest='compute template',
                        help="""--- For inner use of cnvrg.io ---""")
    args = parser.parse_args()

    print(args)

    #TODO - add args parse for [ in_model , out_model)


    input_model = str(args.input)
    output_model = str(args.model)
    model = load_model('/cnvrg/'+input_model+'.h5')
    save_frozen_pb(model, '/cnvrg/'+output_model+'.pb')




    ##TODO ADVANCED! Option to include testing of the in and out models to validate accuracy against specified test data


