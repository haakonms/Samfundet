from PIL import Image
import numpy as np
import matplotlib.image as mpimg
from helpers import *
from image_processing import *

def get_prediction_pixel(img, model, NEW_DIM_TRAIN):
    

    a = img

    image= a.resize((NEW_DIM_TRAIN , NEW_DIM_TRAIN))#, refcheck=False)

    data = np.asarray(image)
    temp = np.zeros((1,NEW_DIM_TRAIN,NEW_DIM_TRAIN,3))
    temp[0,:,:,:] = data
    #print(data.shape)
    #data = np.transpose(temp, (0, 3, 1, 2))
    #newdata = np.divide(temp,255.0)

    output_prediction = model.predict(temp)
    new_out = np.multiply(output_prediction,255.0)
    #output_prediction = new_out[:,0,:,:]
    output_prediction = new_out[:,:,:,0]


    return output_prediction

def label_to_img_unet(imgwidth, imgheight, w, h, output_prediction,datatype):
    # W = h = IMGPATCHSIZE
    # Defines the "black white" image that is to be saved
    predict_img = np.zeros([imgwidth, imgheight,3],dtype=np.uint8)

    #is_2d = output_prediction.shape[2] < 3
    #print(range(0:h:imgheight))
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            
            #already made black and white
            meanval = np.mean(output_prediction[j:j+w, i:i+h,0])
            if meanval>=128:
                val = 255
            else:
                val = 0
            predict_img[j:j+w, i:i+h,0] = val#np.mean(output_prediction[j:j+w, i:i+h,0])
            predict_img[j:j+w, i:i+h,1] = val#np.mean(output_prediction[j:j+w, i:i+h,1])
            predict_img[j:j+w, i:i+h,2] = val#np.mean(output_prediction[j:j+w, i:i+h,2])
            #list_patches.append(im_patch)
    #print(predict_img)
    return predict_img
    

    # Fills image with the predictions for each patch, so we have a int at each position in the (608,608) array
    ind = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            predict_img[j:j+w, i:i+h] = output_prediction[ind]
            ind += 1

    return predict_img

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
    color_mask[:,:,0] = predicted_img[:,:,0]#*PIXEL_DEPTH #0 eller 3 Endrer bare R i rgb, altså gjør bildet 

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
    #print(image_filename)
    # loads the image in question
    #img = mpimg.imread(image_filename)
    img = Image.open(image_filename)
    #data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    #arrimg = np.array(img)
    #print(arrimg)
    #img = np.divide(arrimg,255.0)
    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN) #(1,224,224)
    #predict_img = output_prediction

    #predict_img = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    output_prediction = np.transpose(output_prediction, (1, 2, 0)) #(224,224,1)
    #print(predict_img.shape)
    #print(newPred.shape)
    predict_img = np.asarray(output_prediction)

    # Changes into a 3D array, to easier turn into image
    predict_img_3c = np.zeros((predict_img.shape[0],predict_img.shape[1], 3), dtype=np.uint8)
    predict_img8 = np.squeeze(img_float_to_uint8(predict_img, PIXEL_DEPTH))
    #predict_img_3c = np.zeros((newPred.shape[0],newPred.shape[1], 3), dtype=np.uint8)
    #predict_img8 = np.squeeze(img_float_to_uint8(newPred, PIXEL_DEPTH))
    predict_img8[predict_img8 >= 128] = 255 
    predict_img8[predict_img8 < 128] = 0 
    #print(predict_img8)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8
    #np.uint8

    #imgpred = Image.fromarray(np.multiply(predict_img_3c,255.0))
    imgpred = Image.fromarray(predict_img_3c)
    #imgpred.save(prediction_test_dir + "small_" + str(i) + ".png")
    imgpredict = imgpred.resize((608,608))
    #imgpredict = np.asarray(imgpredict)
    #newPred = label_to_img_unet(imgpredict.shape[0], imgpredict.shape[1],IMG_PATCH_SIZE, IMG_PATCH_SIZE, predict_img,datatype)
    #print(newPred)
    #img = Image.fromarray(newPred)
    #img.save(prediction_test_dir + "gtimg_" + str(i) + ".png")

    #img = mpimg.imread(prediction_test_dir + "gtimg_" + str(i) + ".png")

    #overlay = make_img_overlay_pixel(img, imgpredict, PIXEL_DEPTH)
    #overlay.save(prediction_test_dir + "overlay_" + str(i) + ".png")



    return imgpredict,img#, overlay#,labels

def get_prediction_with_overlay_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE):

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
    #print(output_prediction.shape)
    output_prediction = np.transpose(output_prediction, (1, 2, 0)) #(224,224,1)
    


    predict_img_3c = np.zeros((output_prediction.shape[0],output_prediction.shape[1], 3), dtype=np.uint8)
    predict_img8 = np.squeeze(img_float_to_uint8(output_prediction, PIXEL_DEPTH))
    predict_img8[predict_img8 >= 128] = 255 
    predict_img8[predict_img8 < 128] = 0       
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    newPred = label_to_img_unet(predict_img_3c.shape[0], predict_img_3c.shape[1],IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction,datatype)
    imgpred = Image.fromarray(newPred)
    #imgpredict = imgpred.resize((400,400))
    
    #gtimg = get_predictionimage_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN)
    #wpred,hpred = imgpredict.size
    #w,h = img.size
    #print("wpred: ", wpred, "hpred: ", hpred, "w", w, "h: ", h)
    #img = mpimg.imread(image_filename) # Reads out the original image
    oimg = make_img_overlay_pixel(img, imgpred, PIXEL_DEPTH)

    return oimg, imgpred

def get_pred_postprocessed_unet(filename, image_idx, datatype, IMG_PATCH_SIZE):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        #filename = prediction_test_dir + "predictimg_" + str(i) + ".png"
        imageid = "patch_gtimg_%d" % i
        image_filename = filename + imageid + ".png"
    else:
        print('Error: Enter test or train')      
    #print(image_filename)
    # loads the image in question
    #img = mpimg.imread(image_filename)
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    #rgbimg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #print(rgbimg.shape)
    p_img = post_process(img)
    #data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]


    label_patches = img_crop(p_img, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
    data = np.asarray(label_patches)#([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    #print("bilde",imgpredict.shape)
    img_post = Image.fromarray(p_img)


    return labels, img_post