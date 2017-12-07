import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models

def export(model_data_path, export_dir):

    height = 228
    width = 304
    channels = 3
    batch_size = 1

    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    net = models.ResNet50UpProj({'data': input_node}, batch_size)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)      
        
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:               
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])

    print('Saving the graph')
    builder.save()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_data_path', help='Parameters for the model')
	parser.add_argument('export_model_dir', help='directory for exported model')
	args = parser.parse_args()

	export(args.model_data_path, args.export_model_dir)





