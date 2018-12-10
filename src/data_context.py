
from __future__ import print_function
import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import urllib
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#from helpers import augmentation, make_img_overlay, label_to_img, img_float_to_uint8, extract_labels, img_crop, value_to_class

from keras_pred import make_img_overlay, label_to_img
from image_processing import img_float_to_uint8, img_crop
from image_augmentation import *
from data_extraction import extract_labels
from helpers import value_to_class

import cv2 as cv2



def img_crop_context(im, w, h, w_context):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    #print('IMGWIDTH: ', imgwidth)
    #print('IMGHEIGHT: ',  imgheight)
    is_2d = len(im.shape) < 3
    
    # creates the image with reflected edges
    im_reflect = cv2.copyMakeBorder(im, w_context, w_context, w_context, w_context, cv2.BORDER_REFLECT)

    for i in range(0+w_context,imgheight+w_context,h): # iterates through the 0th axis
        for j in range(0+w_context,imgwidth+w_context,w): # iterates through the 1th axis
            
            l = j - w_context # left
            r = j + w + w_context # right
            t = i - w_context # bottom
            b = i + h + w_context # top
            

            if is_2d:
                im_patch = im_reflect[l:r, t:b]
            else:
                im_patch = im_reflect[l:r, t:b, :]
            list_patches.append(im_patch)
    return list_patches


def value_to_class_context(patch, IMG_PATCH_SIZE, CONTEXT_SIZE):
    
    patch_center = patch[CONTEXT_SIZE:CONTEXT_SIZE+IMG_PATCH_SIZE,CONTEXT_SIZE:CONTEXT_SIZE+IMG_PATCH_SIZE]
    v = numpy.mean(patch)

    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def extract_data_context(filename, num_images, IMG_PATCH_SIZE, CONTEXT_SIZE, datatype):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        if datatype == 'train':
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
        elif datatype == 'test':
            imageid = "/test_%d" % i
            image_filename = filename + imageid + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    # makes a list of all patches for the image at each index
    img_patches = [img_crop_context(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_images)]
    # "unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    # i = antall bilder, j = hvilken patch
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    #print("data",data.shape)
    #shape of returned = (width_image/num_patches * height_image/num_patches*num_images), patch_size, patch_size, 3
    return numpy.asarray(data)


def extract_aug_data_and_labels_context(filename, num_images, IMG_PATCH_SIZE, CONTEXT_SIZE):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    gt_imgs = []
    pathlist = Path(filename).glob('**/*.png')
    #goes through all the augmented images in image directory
    # must pair them with all the augmented groundtruth images
    for path in pathlist:
        # because path is object not string
        image_path = str(path)
        lhs,rhs = image_path.split("/images")
        img = mpimg.imread(image_path)
        imgs.append(img)
        gt_path = lhs + '/groundtruth' + rhs
        g_img = mpimg.imread(gt_path)
        gt_imgs.append(g_img)

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
    # makes a list of all patches for the image at each index
    img_patches = [img_crop_context(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_images)]
    
    # "unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    
    # i = antall bilder, j = hvilken patch
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data_gt = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data_gt[i])) for i in range(len(data_gt))])

    return numpy.asarray(data), labels.astype(numpy.float32)

'''
# Extract label images
def extract_labels_context(filename, num_images, IMG_PATCH_SIZE, CONTEXT_SIZE):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop_context(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_images)]
    data_gt = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class_context(data_gt[i], IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(len(data_gt))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)
'''


def load_data_context(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, TESTING_SIZE, augment=False, MAX_AUG=1, augImgDir='', data_dir='', groundTruthDir=''):

    if augment == False:
        print('No augmenting of traing images')
        print('\nLoading training images')
        x_train = extract_data_context(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE,  'train')
        #print(x_train[:10])

        print('Loading training labels')
        y_train = extract_labels(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE)
        #print(y_train[:20])
    elif augment == True:
        print('Augmenting traing images...')
        augmentation(data_dir, augImgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG)
        x_train, y_train = extract_aug_data_and_labels_context(augImgDir, TRAINING_SIZE*(MAX_AUG+1), IMG_PATCH_SIZE, CONTEXT_SIZE)

    
    print('Loading test images\n')
    x_test = extract_data_context(test_data_filename,TESTING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, 'test')
    #print(x_test[:10])

    print('Train data shape: ',x_train.shape)
    print('Train labels shape: ',y_train.shape)
    print('Test data shape: ',x_test.shape)

    [cl1,cl2] = numpy.sum(y_train, axis = 0, dtype = int)
    print('Number of samples in class 1 (background): ',cl1)
    print('Number of samples in class 2 (road): ',cl2, '\n')


    return x_train, y_train, x_test






############################################## PREDICTIONS ##############################################


def get_prediction_context(img, model, IMG_PATCH_SIZE, CONTEXT_SIZE):
    
    # Turns the image into its data patches
    data = numpy.asarray(img_crop_context(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE))
    #shape ((38*38), 16,16,3)

    # Data now is a vector of the patches from one single image in the testing data
    output_prediction = model.predict_classes(data)
    #predictions have shape (1444,), a prediction for each patch in the image

    return output_prediction


'''def label_to_img_context(imgwidth, imgheight, w, h, output_prediction):

    # Defines the "black white" image that is to be saved
    predict_img = numpy.zeros([imgwidth, imgheight])

    # Fills image with the predictions for each patch, so we have a int at each position in the (608,608) array
    ind = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            predict_img[j:j+w, i:i+h] = output_prediction[ind]
            ind += 1

    return predict_img'''

# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay_context(filename, image_idx, datatype, model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH):

    i = image_idx
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')

    img = mpimg.imread(image_filename)


    # Returns a vector with a prediction for each patch
    output_prediction = get_prediction_context(img, model, IMG_PATCH_SIZE, CONTEXT_SIZE) 
    
    # Returns a representation of the image as a 2D vector with a label at each pixel
    img_prediction = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    

    oimg = make_img_overlay(img, img_prediction, PIXEL_DEPTH)

    return oimg


def get_predictionimage_context(filename, image_idx, datatype, model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')      

    # loads the image in question
    img = mpimg.imread(image_filename)
    #data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    output_prediction = get_prediction_context(img, model, IMG_PATCH_SIZE, CONTEXT_SIZE)
    predict_img = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    
    # Changes into a 3D array, to easier turn into image
    predict_img_3c = numpy.zeros((img.shape[0],img.shape[1], 3), dtype=numpy.uint8)
    predict_img8 = img_float_to_uint8(predict_img, PIXEL_DEPTH)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)

    return imgpred

