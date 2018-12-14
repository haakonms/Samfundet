from __future__ import print_function
import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import shutil
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from mask_to_submission import *
from helpers import *
from F1_metrics import *
from Unet import *
from nyUnet import *
from image_processing import *
from image_augmentation import *
from F1_metrics import *
#from data_context import *
from data_extraction import *
from prediction import *
from keras_pred import *
from unet_pred import *

import code
import tensorflow.python.platform

import numpy as np

import tensorflow as tf
from scipy import misc, ndimage
import shutil

import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from pathlib import Path
from sklearn.utils import class_weight


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 5
TESTING_SIZE = 50
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 1
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
MAX_AUG = 1
NEW_DIM_TRAIN = 400

# The size of the patches each image is split into. Should be a multiple of 4, and the image
# size would be a multiple of this. For this assignment to get the delivery correct it has to be 16
IMG_PATCH_SIZE = 16
INPUT_CHANNELS = 3

# Extract data into numpy arrays, divided into patches of 16x16
data_dir = 'data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/' 
test_data_filename = data_dir + 'test_set_images'

# Directive for storing the augmented training images
imgDir = data_dir + 'training/augmented/images'
groundTruthDir = data_dir + 'training/augmented/groundtruth'




# Loading the data, and set wheter it is to be augmented or not
#x_train, y_train, x_test = load_data(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, TESTING_SIZE,
 #         augment=True, MAX_AUG=MAX_AUG, augImgDir=imgDir , data_dir=data_dir, groundThruthDir =groundThruthDir) # The last 3 parameters can be blank when we dont want augmentation


#x_train_img, y_train_img, x_test_img = load_data_img(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE, NEW_DIM_TRAIN)
x_train_img, y_train_img, x_test_img = load_data_unet(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE, NEW_DIM_TRAIN,
  augment=True, MAX_AUG=MAX_AUG, augImgDir=imgDir , data_dir=data_dir, groundTruthDir =groundTruthDir)

x_train = x_train_img
y_train = y_train_img
x_test = x_test_img

#print(y_train)

print(y_train.shape)
print(x_train.shape)


# Class weigths
#classes = np.array([0,1])
#class_weights = class_weight.compute_class_weight('balanced',classes,y_train[:,1])
#print(class_weights) 
# {0:0.66819193, 1:1.98639715} Class 1 (road) weights mer enn class 0 (foreground)
#class_weights = {0:1, 1:4}
#class_weights = (1,15)
#print('Class weights: ',class_weights) 

# input image dimensions
#img_rows, img_cols = BATCH_SIZE, BATCH_SIZE
img_rows = x_train[0].shape[1]
img_cols = img_rows
print(img_rows)
#input_shape = (NUM_CHANNELS, img_rows, img_cols) 
yweight = y_train[:,:,:,0]
yweight = yweight.flatten()
print(np.unique(yweight), sum(yweight))
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(yweight),
                                                 yweight)
#class_weights = (15,1)
print('Class weights: ',class_weights) 
#model = Unet( nClasses =NUM_LABELS , input_width=NEW_DIM_TRAIN , input_height=NEW_DIM_TRAIN , nChannels=NUM_CHANNELS )
#model = ZF_UNET_224(dropout_val=0.2, weights=None, input_shape=NEW_DIM_TRAIN)
#model = ZF_UNET_224(weights='generator',input_shape=NEW_DIM_TRAIN)
#model = ZF_UNET_224(class_weights,NEW_DIM_TRAIN)
inputs = Input((NEW_DIM_TRAIN, NEW_DIM_TRAIN,INPUT_CHANNELS))
model = create_model(inputs)
model.summary()



# min_delta = 0.000000000001

# #stopper = EarlyStopping(monitor='val_loss',min_delta=min_delta,patience=2) 

# class OptimizerChanger(keras.callbacks.EarlyStopping):
#     def __init__(self, on_train_end, **kwargs):

#         self.do_on_train_end = on_train_end
#         super(OptimizerChanger,self).__init__(**kwargs)

#     def on_train_end(self, logs=None):
#         super(OptimizerChanger,self).on_train_end(self,logs)
#         self.do_on_train_end()

# def do_after_training():
#     #warining, this creates a new optimizer and,
#     #at the beginning, it might give you a worse training performance than before
#     model.compile(optimizer = 'SGD', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#     model.fit(x_train, y_train,
#           batch_size=BATCH_SIZE,
#           epochs=NUM_EPOCHS,
#           shuffle = True,
#           verbose=1,
#           validation_split = 0.1
#           #class_weight = class_weights
#           )

# changer = OptimizerChanger(on_train_end= do_after_training, 
#                            monitor='val_loss',
#                            min_delta=min_delta,
#                            patience=2)




#sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1)

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          shuffle = True,
          verbose=1,
          validation_split = 0.1,
          #callbacks = [earlystop],
          class_weight = class_weights
          )
          #validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
'''model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=25000, epochs=NUM_EPOCHS, verbose=1)
'''

#y_submit = model.predict(x_test_img)
#print('y_submit: ', y_submit.shape)
#print('antall vei / antall bakgrunn: ', np.sum(y_submit[:,0,:,:]))
#y_predi = np.argmax(y_submit, axis=3)
#y_testi = np.argmax(y_test, axis=3)
#print(y_testi.shape,y_predi.shape)

#for i in range(10):
#  img = (trainpred[i] + 1)*(255.0/2)

prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
y_submit = np.zeros((((608//IMG_PATCH_SIZE)**2)*TESTING_SIZE,2))
for i in range(1,TESTING_SIZE+1):
  #y_submit[((608//IMG_PATCH_SIZE)**2)*(i-1):((608//IMG_PATCH_SIZE)**2)*i,:], gtimg = get_pred_and_ysubmit_pixelwise(test_data_filename, i, 'test', model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE,prediction_test_dir)
  gtimg,orImg = get_pred_and_ysubmit_pixelwise(test_data_filename, i, 'test', model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE,prediction_test_dir)
  gtimg.save(prediction_test_dir + "gtimg_" + str(i) + ".png")
  #overlay.save(prediction_test_dir + "overlay_" + str(i) + ".png")
  gtarr = np.asarray(gtimg)
  #print(gtarr)
  label_patches = img_crop(gtarr, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
  data = np.asarray(label_patches)#([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
  labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
  
  newPred = label_to_img_unet(gtarr.shape[0], gtarr.shape[1],IMG_PATCH_SIZE, IMG_PATCH_SIZE, gtarr,'test')
  #print(newPred)
  img = Image.fromarray(newPred)
  #imageid = "/test_%d" % i
  #image_filename = filename + imageid + imageid + ".png"
  #orImg = Image.open(image_filename)
  img.save(prediction_test_dir + "patch_gtimg_" + str(i) + ".png")
  y_submit[((608//IMG_PATCH_SIZE)**2)*(i-1):((608//IMG_PATCH_SIZE)**2)*i,:] = labels
  overlay = make_img_overlay_pixel(orImg, newPred, PIXEL_DEPTH)
  overlay.save(prediction_test_dir + "overlay_" + str(i) + ".png")
  

print('y_submit: ', y_submit.shape)
print('antall vei / antall bakgrunn: ', np.sum(y_submit[:,0]))

prediction_training_dir = "predictions_training/"
#image_filenames = []
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)
for i in range(1, TRAINING_SIZE+1):
    oimg, imgpred = get_prediction_with_overlay_pixelwise(train_data_filename, i, 'train', model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE)
    #save_overlay_and_prediction(train_data_filename, i, 'train', model, IMG_PATCH_SIZE, PIXEL_DEPTH, prediction_training_dir)
    #oimg = get_prediction_with_overlay(train_data_filename, i, 'train', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

    #imgpred = get_predictionimage(train_data_filename, i, 'train', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    imgpred.save(prediction_training_dir + "predictimg_" + str(i) + ".png")


'''
#image_filenames=[]
prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
for i in range(1,TESTING_SIZE+1):
    test_data_filename = data_dir + 'test_set_images'

    oimg = get_prediction_with_overlay(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    oimg.save(prediction_test_dir + "overlay_" + str(i) + ".png")

    filename = prediction_test_dir + "predictimg_" + str(i) + ".png"
    imgpred = get_predictionimage(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    imgpred.save(filename)
    #print(filename)
    #image_filenames.append(filename)

'''
#submission_filename = 'keras_submission'
#pred_to_submission(submission_filename,*image_filenames)    

# Make submission file
prediction_to_submission2('submission_keras.csv', y_submit)



'''
trainpred = model.predict(x_train_img)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)
print(y_testi.shape,y_predi.shape)
>>>>>>> 2b40ae872028ee8b402de5ec4895d715e2de46c1





'''
'''
y_validation_train = model.predict_classes(x_train)
tp, tn, fp, fn = f1_values(y_train, y_validation_train)
f1 = f1_score(tp, fp, fn)
print("f1", f1)

y_submit = model.predict_classes(x_test)
print('Size of predictions: ',y_submit.shape)
print('Number of road patches: ',sum(y_submit))
'''

'''
prediction_training_dir = "predictions_training/"
#image_filenames = []
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)
for i in range(1, TRAINING_SIZE+1):
    oimg = get_prediction_with_overlay(train_data_filename, i, 'train', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

    imgpred = get_predictionimage(train_data_filename, i, 'train', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    imgpred.save(prediction_training_dir + "predictimg_" + str(i) + ".png")



#image_filenames=[]
prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
for i in range(1,TESTING_SIZE+1):
    test_data_filename = data_dir + 'test_set_images'

    oimg = get_prediction_with_overlay(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    oimg.save(prediction_test_dir + "overlay_" + str(i) + ".png")

    filename = prediction_test_dir + "predictimg_" + str(i) + ".png"
    imgpred = get_predictionimage(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, PIXEL_DEPTH)
    imgpred.save(filename)
    #print(filename)
    #image_filenames.append(filename)


#submission_filename = 'keras_submission'
#pred_to_submission(submission_filename,*image_filenames)    

# Make submission file
prediction_to_submission('submission_keras.csv', y_submit)

'''

