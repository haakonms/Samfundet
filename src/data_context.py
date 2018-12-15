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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2 as cv2
import random

from image_processing import *
from image_augmentation import *
from data_extraction import *
from helpers import *




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
    v = np.mean(patch)

    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]





def extract_data_context(filename, num_images, IMG_PATCH_SIZE, CONTEXT_SIZE, datatype, val_img=[]):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)

    # print(val_img)
    # print(train_img)
    # print(val_img.shape)
    # print(train_img.shape)


    #for i in range(1, num_images+1):
    for i in train_img:
        if datatype == 'train':
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
        elif datatype == 'test':
            imageid = "/test_%d" % i
            image_filename = filename + imageid + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            t_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    if datatype == 'train':
        for i in val_img:
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
            if os.path.isfile(image_filename):
                #print ('Loading ' + image_filename)
                img = mpimg.imread(image_filename)
                v_imgs.append(img)
            else:
                print ('File ' + image_filename + ' does not exist')


    num_t_images = len(t_imgs)
    num_v_images = len(v_imgs)
    IMG_WIDTH = t_imgs[0].shape[0]
    IMG_HEIGHT = t_imgs[0].shape[1]
    #N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    # makes a list of all patches for the image at each index
    train_img_patches = [img_crop_context(t_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_t_images)]
    val_img_patches = [img_crop_context(v_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_v_images)]
    # "unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    # i = antall bilder, j = hvilken patch
    train_data = [train_img_patches[i][j] for i in range(len(train_img_patches)) for j in range(len(train_img_patches[i]))]
    val_data = [val_img_patches[i][j] for i in range(len(val_img_patches)) for j in range(len(val_img_patches[i]))]
    #print("data",data.shape)
    #shape of returned = (width_image/num_patches * height_image/num_patches*num_images), patch_size, patch_size, 3
    return np.asarray(train_data), np.asarray(val_data)




def extract_aug_data_and_labels_context(filename, num_images, IMG_PATCH_SIZE, CONTEXT_SIZE, val_img=[]):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    gt_imgs = []
    #pathlist = Path(filename).glob('**/*.png')
    glob_path = Path(filename)
    pathlist = [str(pp) for pp in glob_path.glob('**/*.png')]
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
    data_gt = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data_gt[i])) for i in range(len(data_gt))])

    return np.asarray(data), labels.astype(np.float32)

# Extract label images
def extract_labels_context(filename, num_images, IMG_PATCH_SIZE, val_img=[]):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)


    for i in train_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            t_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    for i in val_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            v_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')


    num_t_images = len(t_imgs)
    num_v_images = len(v_imgs)
    t_patches = [img_crop(t_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_t_images)]
    v_patches = [img_crop(v_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_v_images)]

    t_data = np.asarray([t_patches[i][j] for i in range(len(t_patches)) for j in range(len(t_patches[i]))])
    v_data = np.asarray([v_patches[i][j] for i in range(len(v_patches)) for j in range(len(v_patches[i]))])
    

    t_labels = np.asarray([value_to_class(np.mean(t_data[i])) for i in range(len(t_data))])
    v_labels = np.asarray([value_to_class(np.mean(v_data[i])) for i in range(len(v_data))])

    # Convert to dense 1-hot representation.
    return t_labels.astype(np.float32), v_labels.astype(np.float32)





def load_data_context(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, VALIDATION_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, TESTING_SIZE, saltpepper = 0.004, augment=False, MAX_AUG=1, augImgDir='', data_dir='', groundTruthDir='', newaugment=True):


    idx = random.sample(range(1, 100), VALIDATION_SIZE)


    if augment == False:
        print('No augmenting of traing images')
        print('\nLoading training images')
        x_train, x_val = extract_data_context(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE,  'train', idx)
        #print(x_train[:10])
        #x_train = np.asarray(x_train)

        print('Loading training labels')
        y_train, y_val = extract_labels_context(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE, idx)

    elif augment == True:
        print('Augmenting traing images...')
        if newaugment == True:
            augmentation(data_dir, augImgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG, idx)
        x_train, y_train = extract_aug_data_and_labels_context(augImgDir, TRAINING_SIZE*(MAX_AUG+1), IMG_PATCH_SIZE, CONTEXT_SIZE)
        _, x_val = extract_data_context(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE,  'train', idx)
        _, y_val = extract_labels_context(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE, idx)

    
    x_train = sp_noise(x_train, amount=saltpepper)

    print('Loading test images\n')
    x_test, _ = extract_data_context(test_data_filename,TESTING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, 'test')
    #print(x_test[:10])

    print('Train data shape: ',x_train.shape)
    print('Train labels shape: ',y_train.shape)
    print('Test data shape: ',x_test.shape)
    print('Val data shape: ',x_val.shape)
    print('Val labels shape: ',y_val.shape)

    [cl1,cl2] = np.sum(y_train, axis = 0, dtype = int)
    print('Number of samples in class 1 (background): ',cl1)
    print('Number of samples in class 2 (road): ',cl2, '\n')


    return x_train, y_train, x_test, x_val, y_val






############################################## PREDICTIONS ##############################################


def get_prediction_context(img, model, IMG_PATCH_SIZE, CONTEXT_SIZE):
    
    # Turns the image into its data patches
    data = np.asarray(img_crop_context(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE))
    #shape ((38*38), 16,16,3)

    # Data now is a vector of the patches from one single image in the testing data
    output_prediction = model.predict_classes(data)
    #predictions have shape (1444,), a prediction for each patch in the image

    return output_prediction


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
    predict_img_3c = np.zeros((img.shape[0],img.shape[1], 3), dtype=np.uint8)
    predict_img8 = img_float_to_uint8(predict_img, PIXEL_DEPTH)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)

    return imgpred


