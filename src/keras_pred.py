import numpy as np
from image_processing import img_float_to_uint8, post_process, img_crop
from data_extraction import *
from PIL import Image
import cv2


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

#######

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