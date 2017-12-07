# Testing intermediate layers
# H.H.
# Oct 2017

import os
import glob
import cv2
import sys
caffe_root = '/home/carrot/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu()

#Size of images
IMAGE_WIDTH = 304
IMAGE_HEIGHT = 228

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

#Read model architecture and trained model's weights
net = caffe.Net('/home/carrot/NYU/DD_deploy2.prototxt',
                '/home/carrot/NYU/snaps/_iter_10.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data_images': net.blobs['data_images'].data.shape})
transformer.set_transpose('data_images', (2,0,1))

img_path = "/home/carrot/NYU/images/validation/00000_image.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

net.blobs['data_images'].data[...] = transformer.preprocess('data_images', img)
end = 'unpoolA'
out = net.forward(end=end)

# Testing layers
c1=net.blobs['depth_0'].data
c2=net.blobs['unpoolA'].data
c3=net.blobs['res5c_branch2b'].data
W = net.params['depthA_1_2'][0].data[...]
W2 = net.params['depth_0'][0].data[...]


out = net.blobs['depthA_1_1'].data
D = net.blobs['depthA_1_2'].data
print c3
print "depth_0"
print c1
print "unpool"
print c2
print "weights"
print W
print "OUT"
print out
print "depthA"
print D
print "W2"
print W2
