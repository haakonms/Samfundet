from __future__ import print_function
import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import urllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import shutil
from data_extraction import *
#from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img











def load_data(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, TESTING_SIZE, augment=False, MAX_AUG=1, augImgDir='', data_dir='', groundTruthDir=''):

    if augment == False:
        print('No augmenting of traing images')
        print('\nLoading training images')
        x_train = extract_data(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE,  'train')
        #print(x_train[:10])

        print('Loading training labels')
        y_train = extract_labels(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE)
        #print(y_train[:20])
    elif augment == True:
        print('Augmenting traing images...')
        augmentation(data_dir, augImgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG)
        x_train, y_train = extract_aug_data_and_labels(augImgDir, TRAINING_SIZE*(MAX_AUG+1), IMG_PATCH_SIZE)

    
    print('Loading test images\n')
    x_test = extract_data(test_data_filename,TESTING_SIZE, IMG_PATCH_SIZE, 'test')
    #print(x_test[:10])

    print('Train data shape: ',x_train.shape)
    print('Train labels shape: ',y_train.shape)
    print('Test data shape: ',x_test.shape)

    [cl1,cl2] = np.sum(y_train, axis = 0, dtype = int)
    print('Number of samples in class 1 (background): ',cl1)
    print('Number of samples in class 2 (road): ',cl2, '\n')


    return x_train, y_train, x_test






def load_data_img(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE, new_dim_train):
    x_test_img = extract_data_pixelwise(test_data_filename, TESTING_SIZE,  'test', new_dim_train)
    x_test_img = np.transpose(x_test_img, (0, 3, 1, 2))
    print('Test data shape: ',x_test_img.shape)

    x_train_img = extract_data_pixelwise(train_data_filename, TRAINING_SIZE,  'train', new_dim_train)
    x_train_img = np.transpose(x_train_img, (0, 3, 1, 2))
    print('Train data shape: ',x_train_img.shape)
    y_train_img = extract_labels_pixelwise(train_labels_filename, TRAINING_SIZE, new_dim_train)
    y_train_img = np.transpose(y_train_img, (0, 3, 1, 2))
    print('Train labels shape: ',y_train_img.shape)


    road = np.sum(y_train_img[:,1,:,:], dtype = int)
    background = np.sum(y_train_img[:,0,:,:], dtype = int)
    print('Number of samples in class 1 (background): ',road)
    print('Number of samples in class 2 (road): ',background, '\n')


    return x_train_img, y_train_img, x_test_img


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))



# def make_img_overlay(img, predicted_img):
#     w = img.shape[0]
#     h = img.shape[1]
#     color_mask = np.zeros((w, h, 3), dtype=np.uint8)
#     color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

#     img8 = img_float_to_uint8(img)
#     background = Image.fromarray(img8, 'RGB').convert("RGBA")
#     overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
#     new_img = Image.blend(background, overlay, 0.2)
#     return new_img








''' Not finished
def save_overlay_and_prediction_pixel(filename, image_idx,datatype,model,IMG_PATCH_SIZE,PIXEL_DEPTH, prediction_training_dir):
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

    # Returns a vector with a prediction for each patch
    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN)
    # Returns a representation of the image as a 2D vector with a label at each pixel
    img_prediction = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    # Changes into a 3D array, to easier turn into image
    predict_img_3c = np.zeros((img.shape[0],img.shape[1], 3), dtype=np.uint8)
    predict_img8 = img_float_to_uint8(img_prediction, PIXEL_DEPTH)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)
    oimg = make_img_overlay(img, img_prediction, PIXEL_DEPTH)

    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
    imgpred.save(prediction_training_dir + "predictimg_" + str(i) + ".png")

    return
'''












