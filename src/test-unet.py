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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from pathlib import Path
from sklearn.utils import class_weight


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 4
TESTING_SIZE = 4
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 5
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
MAX_AUG = 3
NEW_DIM_TRAIN = 224

# The size of the patches each image is split into. Should be a multiple of 4, and the image
# size would be a multiple of this. For this assignment to get the delivery correct it has to be 16
IMG_PATCH_SIZE = 16


# Extract data into numpy arrays, divided into patches of 16x16
data_dir = 'data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/' 
test_data_filename = data_dir + 'test_set_images'

# Directive for storing the augmented training images
imgDir = data_dir + 'training/augmented/images'
groundThruthDir = data_dir + 'training/augmented/groundtruth'




# Loading the data, and set wheter it is to be augmented or not
#x_train, y_train, x_test = load_data(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, TESTING_SIZE,
 #         augment=True, MAX_AUG=MAX_AUG, augImgDir=imgDir , data_dir=data_dir, groundThruthDir =groundThruthDir) # The last 3 parameters can be blank when we dont want augmentation


x_train_img, y_train_img, x_test_img = load_data_img(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE, NEW_DIM_TRAIN)

x_train = x_train_img
y_train = y_train_img
x_test = x_test_img




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
input_shape = (img_rows, img_cols, NUM_CHANNELS) 

#model = Unet( nClasses =NUM_LABELS , input_width=NEW_DIM_TRAIN , input_height=NEW_DIM_TRAIN , nChannels=NUM_CHANNELS )
model = ZF_UNET_224(dropout_val=0.2, weights=None)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          shuffle = True,
          verbose=1,
          validation_split = 0.1
          #class_weight = class_weights
          )
          #validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
'''model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=25000, epochs=NUM_EPOCHS, verbose=1)

trainpred = model.predict(x_train_img)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)
print(y_testi.shape,y_predi.shape)

for i in range(10):
  img = (trainpred[i] + 1)*(255.0/2)



y_validation_train = model.predict_classes(x_train)
tp, tn, fp, fn = f1_values(y_train, y_validation_train)
f1 = f1_score(tp, fp, fn)
print("f1", f1)

y_submit = model.predict_classes(x_test)
print('Size of predictions: ',y_submit.shape)
print('Number of road patches: ',sum(y_submit))


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


