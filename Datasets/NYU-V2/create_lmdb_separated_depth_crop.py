# H.H. Oct 2017
# Augmenting and creating LMDB for NYU-V2

import os
import glob
import random
import numpy as np
import sys
caffe_root = '/home/carrot/caffe/'
sys.path.insert(0, caffe_root + 'python')
import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

import skimage.io as io
import h5py

# data path
path_to_depth = './nyu_depth_v2_labeled.mat'

# read mat file
f = h5py.File(path_to_depth)

# read all images original format is [3 x 640 x 480], uint8
i=0

#Size of images
IMAGE_WIDTH = 304
IMAGE_HEIGHT = 228

#Size of depths
DEPTH_WIDTH = 160
DEPTH_HEIGHT = 128

# creating multiple crops
def create_crops(img,type='image'):
    if type == 'image':
        finalH=IMAGE_HEIGHT
        finalW=IMAGE_WIDTH
        sizes=[480,360]
    else:
        finalH=DEPTH_HEIGHT
        finalW=DEPTH_WIDTH
        sizes=[240,180]

    height, width = img.shape[:2]
    flag=0
    # scaled widths

                
    if height>width:
        flag = 1
    
    reses =[]
    first_crops =[]
    second_crops = []
    if flag:
        for s in sizes:
            reses.append( cv2.resize(img,(s, s*height/width), interpolation = cv2.INTER_CUBIC) )
    else:
        for s in sizes:
            reses.append( cv2.resize(img,(s*width/height,s), interpolation = cv2.INTER_CUBIC) )
    '''
    for res in reses:
        h, w = res.shape[:2]
        
        if flag:
            l=w
        else:
            l=h

        first_crops.append( res[0:l, 0:l] )
        first_crops.append( res[(h-l):h, (w-l):w] )
        first_crops.append( res[(h/2-l/2):(h/2+l/2), (w/2-l/2):(w/2+l/2)] )
    '''
    for crop in reses:
        h, w = crop.shape[:2] # first one with second height
        second_crops.append( crop[0:finalH, 0:finalW] )
        second_crops.append( crop[(h-finalH):h, 0:finalW] )
        second_crops.append( crop[0:finalH, (w-finalW):w] )
        second_crops.append( crop[(h-finalH):h, (w-finalW):w] )
        second_crops.append( crop[(h/2-(finalH/2)):(h/2+(finalH/2)), (w/2-(finalW/2)):(w/2+(finalW/2))] )
        second_crops.append( cv2.resize(crop,(finalW,finalH), interpolation = cv2.INTER_CUBIC) )

    return second_crops

#def Augment_img():



def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label,form='image'):
    if form == 'image':
        #image is numpy.ndarray format. BGR instead of RGB
        return caffe_pb2.Datum(
            channels=3,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            label=label,
            data= img)#np.rollaxis(img, 2).tostring())
    elif form == 'depth':
        #depth is numpy.ndarray WxH 
        return caffe_pb2.Datum(
            channels=1,
            width=DEPTH_WIDTH,
            height=DEPTH_HEIGHT,
            label=label,
            data= img)#np.rollaxis(img, 2).tostring())
    else:
        print "WRONG INPUT!"


train_lmdb_images = '/home/carrot/NYU/train_lmdb_images'
train_lmdb_depths = '/home/carrot/NYU/train_lmdb_depths'
validation_lmdb_images = '/home/carrot/NYU/validation_lmdb_images'
validation_lmdb_depths = '/home/carrot/NYU/validation_lmdb_depths'

os.system('rm -rf  ' + train_lmdb_images)
os.system('rm -rf  ' + validation_lmdb_images)
os.system('rm -rf  ' + train_lmdb_depths)
os.system('rm -rf  ' + validation_lmdb_depths)

#Shuffle train_data
#random.shuffle(train_data)

print 'Creating train_lmdb_image'
i=0
j=0
in_db = lmdb.open(train_lmdb_images, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for img in f['images']:
	#print i
        if j%6 ==0:
            j=j+1
            continue
        # reshape
        img_ = np.empty([480, 640, 3])
        img_[:,:,0] = img[0,:,:].T
        img_[:,:,1] = img[1,:,:].T
        img_[:,:,2] = img[2,:,:].T
        img_ = img_.astype(np.uint8)

        #img_ = transform_img(img_, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        img_list =create_crops(img_,type='image')
        for img_ in img_list:
            # making datum
            img_=np.rollaxis(img_, 2).tostring()
            datum = make_datum(img_, 0)
            in_txn.put('{:0>5d}'.format(i), datum.SerializeToString())
            i=i+1
            print ('Finished processing image {}'.format(i))
        j=j+1
in_db.close()

print 'Creating train_lmdb_depth'
i=0
j=0
in_db = lmdb.open(train_lmdb_depths, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
    for depth in f['depths']:
        #print i
        if j%6 ==0:
            j=j+1
            continue
        # reshape
        depth_ = np.empty([480, 640, 1])
        depth_[:,:,0] = depth[:,:].T
        
        depth_list =create_crops(depth_,type='image')
        for depth_ in depth_list:
            print depth_.shape[:2]
            # resize
            depth_ = cv2.resize(depth_, (160, 128), interpolation = cv2.INTER_CUBIC)
            depth_=depth_.tostring()
            # make datum
            datum = make_datum(depth_, 0,form='depth')
            in_txn.put('{:0>5d}'.format(i), datum.SerializeToString())
            i=i+1
            print ('Finished processing image {}'.format(i))
        j=j+1
in_db.close()


print 'Creating validation_lmdb_images'
i=0
j=0
in_db = lmdb.open(validation_lmdb_images, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for img in f['images']:
        #print i
        if j%6 !=0:
            j=j+1
            continue
        # reshape
        img_ = np.empty([480, 640, 3])
        img_[:,:,0] = img[0,:,:].T
        img_[:,:,1] = img[1,:,:].T
        img_[:,:,2] = img[2,:,:].T
        img_ = img_.astype(np.uint8)

        img_list =create_crops(img_,type='image')
        for img_ in img_list:
            # making datum
            img_=np.rollaxis(img_, 2).tostring()
            datum = make_datum(img_, 0)
            in_txn.put('{:0>5d}'.format(i), datum.SerializeToString())
            i=i+1
            print ('Finished processing image {}'.format(i))
        j=j+1

in_db.close()
print 'Creating validation_lmdb_depth'
i=0
j=0
in_db = lmdb.open(validation_lmdb_depths, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
    for depth in f['depths']:
        #print i
        if j%6 !=0:
            j=j+1
            continue
        # reshape
        depth_ = np.empty([480, 640, 1])
        depth_[:,:,0] = depth[:,:].T
        depth_list =create_crops(depth_,type='image')
        for depth_ in depth_list:
            # resize
            depth_ = cv2.resize(depth_, (160, 128), interpolation = cv2.INTER_CUBIC)
            depth_=depth_.tostring()
            # make datum
            datum = make_datum(depth_, 0,form='depth')
            in_txn.put('{:0>5d}'.format(i), datum.SerializeToString())
            i=i+1
            print ('Finished processing image {}'.format(i))
        j=j+1
in_db.close()

print '\nFinished processing all images'

