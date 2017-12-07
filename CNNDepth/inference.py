import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

def runInference(graph_dir, image_path):
	height = 228
	width = 304

	img = Image.open(image_path)
	img = img.resize([width,height], Image.ANTIALIAS)
	img = np.array(img).astype('float32')
	img = np.expand_dims(np.asarray(img), axis = 0)

	with tf.Session() as sess:
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], graph_dir)
		pred_tensor = sess.graph.get_tensor_by_name('ConvPred/ConvPred:0')
		pred = sess.run(pred_tensor, feed_dict={'Placeholder:0': img})
		fig = plt.figure()
		ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
		fig.colorbar(ii)
		plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_dir', help='dir of the graph')
    parser.add_argument('image_path', help='Directory of images to predict')
    args = parser.parse_args()

    runInference(args.graph_dir, args.image_path)

