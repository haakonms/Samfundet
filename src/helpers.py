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
import cv2

from PIL import Image
from pathlib import Path
import shutil
from data_extraction import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from image_processing import *
from data_extraction import *



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



def make_img_binary(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, output_prediction):

    # Defines the "black white" image that is to be saved
    predict_img = np.zeros([imgwidth, imgheight])

    # Fills image with the predictions for each patch, so we have a int at each position in the (608,608) array
    ind = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            predict_img[j:j+w, i:i+h] = output_prediction[ind]
            ind += 1

    return predict_img

def get_prediction(img, model, IMG_PATCH_SIZE):
    
    # Turns the image into its data patches
    data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    #shape ((38*38), 16,16,3)

    # Data now is a vector of the patches from one single image in the testing data
    output_prediction = model.predict_classes(data)
    #predictions have shape (1444,), a prediction for each patch in the image

    return output_prediction


def get_predictionimage(filename, image_idx, datatype, model, IMG_PATCH_SIZE, PIXEL_DEPTH):

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

    output_prediction = get_prediction(img, model, IMG_PATCH_SIZE)
    predict_img = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    
    # Changes into a 3D array, to easier turn into image
    predict_img_3c = np.zeros((img.shape[0],img.shape[1], 3), dtype=np.uint8)
    predict_img8 = img_float_to_uint8(predict_img, PIXEL_DEPTH)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)

    return imgpred


def make_img_overlay(img, predicted_img, PIXEL_DEPTH):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8) #samme størrelse som bildet
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH #0 eller 3 Endrer bare R i rgb, altså gjør bildet 

    img8 = img_float_to_uint8(img, PIXEL_DEPTH)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, datatype, model, IMG_PATCH_SIZE, PIXEL_DEPTH):

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
    output_prediction = get_prediction(img, model, IMG_PATCH_SIZE) 
    
    # Returns a representation of the image as a 2D vector with a label at each pixel
    img_prediction = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    oimg = make_img_overlay(img, img_prediction, PIXEL_DEPTH)

    return oimg

def save_overlay_and_prediction(filename, image_idx,datatype,model,IMG_PATCH_SIZE,PIXEL_DEPTH, prediction_training_dir):
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
    output_prediction = get_prediction(img, model, IMG_PATCH_SIZE)
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

def get_pred_postprocessed(filename, image_idx, datatype, IMG_PATCH_SIZE):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        #filename = prediction_test_dir + "predictimg_" + str(i) + ".png"
        imageid = "predictimg_%d" % i
        image_filename = filename + imageid + ".png"
    else:
        print('Error: Enter test or train')      
    #print(image_filename)
    # loads the image in question
    #img = mpimg.imread(image_filename)
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    p_img = post_process(img)
    #data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    label_patches = img_crop(p_img, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
    data = np.asarray(label_patches)
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    img_post = Image.fromarray(p_img)

    return labels, img_post








