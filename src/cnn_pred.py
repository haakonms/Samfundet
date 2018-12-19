''' Functions for making prediction images from the CNN model '''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from PIL import Image
import matplotlib.image as mpimg

from data_extraction import value_to_class, img_crop_context, img_crop, img_float_to_uint8, post_process


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


def get_prediction_context(img, model, IMG_PATCH_SIZE, CONTEXT_SIZE):
    # Turns the image into its data patches
    data = np.asarray(img_crop_context(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE))

    # Data now is a vector of the patches from one single image in the testing data
    output_prediction = model.predict_classes(data)

    return output_prediction



def get_prediction_with_overlay_context(filename, image_idx, datatype, model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH):
    # Get prediction overlaid on the original image for given input file

    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % image_idx
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

    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % image_idx
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')      

    img = mpimg.imread(image_filename)

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


def get_pred_postprocessed(filename, image_idx, datatype, IMG_PATCH_SIZE):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        #filename = prediction_test_dir + "predictimg_" + str(i) + ".png"
        imageid = "gt_pred_%d" % i
        image_filename = filename + imageid + ".png"
    else:
        print('Error: Enter test or train')      

    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    p_img = post_process(img)

    img_post = Image.fromarray(p_img)

    return img_post