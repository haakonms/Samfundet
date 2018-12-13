from PIL import Image
import numpy as np
import matplotlib.image as mpimg
from helpers import *
from image_processing import *

def get_prediction_pixel(img, model, NEW_DIM_TRAIN):
    
    #img has shape (608, 608, 3)
    a = img
    #image = resize(img, (NEW_DIM_TRAIN , NEW_DIM_TRAIN,3))
    image= a.resize((NEW_DIM_TRAIN , NEW_DIM_TRAIN))#, refcheck=False)
    #img has shape (224, 224, 3)
    #image = img
    # Turns the image into matrix
    data = np.asarray(image)
    temp = np.zeros((1,NEW_DIM_TRAIN,NEW_DIM_TRAIN,3))
    temp[0,:,:,:] = data
    data = np.transpose(temp, (0, 3, 1, 2))
    newdata = np.divide(data,255.0)
    # now img has shape (1, 3, 224, 224)
    #print("data",data.shape)
    # makes predictions on the image
    output_prediction = model.predict(newdata)
    new_out = np.multiply(output_prediction,255.0)
    output_prediction = new_out[:,0,:,:]
    #output_prediction = output_prediction[:,0,:,:] # (1,224,224)

    #print('output_prediction: ', output_prediction.shape)
    #output_prediction = np.squeeze(output_prediction, axis=0) #(1,224,224)
    #print('output_prediction: ', output_prediction.shape)

    # output_prediction has shape (1,224,224), a prediction for each pixel in the reshaped image

    return output_prediction


def make_img_overlay_pixel(img, predicted_img, PIXEL_DEPTH):
    #w = img.shape[0]
    #h = img.shape[1]
    w, h = img.size
    #print(w,h)
    #pred_img = Image.fromarray(predicted_img)
    #pred_img = Image.fromarray(np.uint8(predicted_img*255))
    #print(shape.pred_img)
    #predicted_img = pred_img.resize((w,w))
    #predicted_img = np.asarray(predicted_img)
    #print('predicted img',predicted_img.shape)
    predicted_img = np.asarray(predicted_img)
    color_mask = np.zeros((w, h, 3), dtype=np.uint8) #samme størrelse som bildet
    color_mask[:,:,0] = predicted_img[:,:,0]*PIXEL_DEPTH #0 eller 3 Endrer bare R i rgb, altså gjør bildet 

    img8 = img_float_to_uint8(img, PIXEL_DEPTH)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img



def get_predictionimage_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN):

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

    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN) #(1,224,224)
    predict_img = output_prediction
    #predict_img = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    
    # Changes into a 3D array, to easier turn into image
    predict_img_3c = np.zeros((predict_img.shape[1],predict_img.shape[2], 3), dtype=np.uint8)
    predict_img8 = img_float_to_uint8(predict_img, PIXEL_DEPTH)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)
    imgpredict = imgpred.resize((608,608))

    return imgpredict

def get_pred_and_ysubmit_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN, IMG_PATCH_SIZE, prediction_test_dir):

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
    print(image_filename)
    # loads the image in question
    #img = mpimg.imread(image_filename)
    img = Image.open(image_filename)
    #data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    #arrimg = np.array(img)
    #print(arrimg)
    #img = np.divide(arrimg,255.0)
    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN) #(1,224,224)
    predict_img = output_prediction

    #predict_img = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    predict_img = np.transpose(predict_img, (1, 2, 0)) #(224,224,1)
    #print(predict_img.shape)
    # Changes into a 3D array, to easier turn into image
    predict_img_3c = np.zeros((predict_img.shape[0],predict_img.shape[1], 3), dtype=np.uint8)
    predict_img8 = np.squeeze(img_float_to_uint8(predict_img, PIXEL_DEPTH))
    #print(predict_img8)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8
    #np.uint8
    #imgpred = Image.fromarray(np.multiply(predict_img_3c,255.0))
    imgpred = Image.fromarray(predict_img_3c)
    #imgpred.save(prediction_test_dir + "small_" + str(i) + ".png")
    imgpredict = imgpred.resize((608,608))
    imgpredict.save(prediction_test_dir + "gtimg_" + str(i) + ".png")

    img = mpimg.imread(prediction_test_dir + "gtimg_" + str(i) + ".png")


    label_patches = img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
    data = np.asarray(label_patches)#([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    #print("bilde",imgpredict.shape)
    '''imgpredarr = np.asarray(imgpredict)
    imgpredarr = np.transpose(imgpredarr, (0, 3, 1, 2))
    print("array", imgpredarr.shape)
    labels = np.zeros((1,608,608,2))

    foreground_threshold = 0.5
    labels[imgpredarr > foreground_threshold] = [1,0]
    labels[imgpredarr <= foreground_threshold] = [0,1]'''

    return labels, imgpredict

def get_prediction_with_overlay_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN):

    i = image_idx
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')

    #img = mpimg.imread(image_filename) # Reads out the original image
    img = Image.open(image_filename)
    #print(img.shape)

    # Returns a matrix with a prediction for each pixel
    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN) #(1,224,224)
    output_prediction = np.transpose(output_prediction, (1, 2, 0)) #(224,224,1)


    predict_img_3c = np.zeros((output_prediction.shape[0],output_prediction.shape[1], 3), dtype=np.uint8)
    predict_img8 = np.squeeze(img_float_to_uint8(output_prediction, PIXEL_DEPTH))       
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)
    imgpredict = imgpred.resize((400,400))
    
    #gtimg = get_predictionimage_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN)
    #wpred,hpred = imgpredict.size
    #w,h = img.size
    #print("wpred: ", wpred, "hpred: ", hpred, "w", w, "h: ", h)
    #img = mpimg.imread(image_filename) # Reads out the original image
    oimg = make_img_overlay_pixel(img, imgpredict, PIXEL_DEPTH)

    return oimg, imgpredict