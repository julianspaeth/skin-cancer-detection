
# with some inspiration from https://gist.github.com/249505f540a5e53a48b0c1a869d370bf.git

import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.contrib.slim.python.slim.nets import inception_v3

from image_tools.preprocess import preprocess
from tensorflow.python.tools import optimize_for_inference_lib

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant

    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('\\')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True


    # build graph

    x = tf.placeholder(dtype=tf.float32, shape=[1, 542, 718, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[1, 2], name='label')

    x_preprocessed = preprocess(x)

    net, endpoints = inception_v3.inception_v3(inputs=x_preprocessed, num_classes=2, is_training=True,
                                               dropout_keep_prob=0.8)

    print(endpoints['Predictions'].graph == tf.get_default_graph())

    print("PREDICTION NAME : " , endpoints['Predictions'])

    saver = tf.train.Saver()

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=endpoints['Predictions'].graph) as sess:
        # We import the meta graph in the current default Graph
        sess.run(tf.global_variables_initializer())

        # We restore the weights
        print(input_checkpoint)
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # output_graph ="./frozen.pb"
        #
        # # Finally we serialize and dump the output graph to the filesystem
        # print("output path: {}".format(output_graph))
        # with tf.gfile.GFile(output_graph, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        #optimizing graph

        # input_graph_def = tf.GraphDef()
        #
        # with tf.gfile.Open(output_graph, "r") as f:
        #     data = f.read()
        #     input_graph_def.ParseFromString(data)

        output_graph_def_opt = optimize_for_inference_lib.optimize_for_inference(
            output_graph_def,
            ["input"],  # an array of the input node(s)
            [output_node_names],  # an array of output nodes
            tf.float32.as_datatype_enum)

        # Save the optimized graph
        output_graph_opt_path = "./frozen.pb"# output_graph.split(".")[0] + "_opt" + "." +  output_graph.split(".")[-1]
        with tf.gfile.FastGFile(output_graph_opt_path, "w") as f:
            f.write(output_graph_def_opt.SerializeToString())

    return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names)