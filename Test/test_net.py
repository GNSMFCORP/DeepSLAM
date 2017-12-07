# Testing Images with caffe model
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

def transform_back(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

#Test images path
test_img_paths = [img_path for img_path in glob.glob("/home/carrot/NYU/validation/images/*jpg")]

#Read model architecture and trained model's weights
net = caffe.Net('/home/carrot/NYU/DD_deploy2.prototxt',
                '/home/carrot/NYU/snaps/_iter_5000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data_images': net.blobs['data_images'].data.shape})
transformer.set_transpose('data_images', (2,0,1))

i=0
for img_path in train_img_paths:
    #img_path = "/home/carrot/NYU/images/validation/00000_image.jpg"
    
    #reading an image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    #feeding through net
    net.blobs['data_images'].data[...] = transformer.preprocess('data_images', img)
    out = net.forward(end='depth_out')

    #resizing and saving
    transform_back(out, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    cv2.imwrite('/home/carrot/NYU/images/test_out/'+'{0:05}'.format(i)+'_depth'+'.jpg',depth_*255/20)
    i=i+1


