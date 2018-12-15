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
#from Unet import *
from unetModel import *
from image_processing import *
from image_augmentation import *
from F1_metrics import *
#from data_context import *
from data_extraction import *
from prediction import *
#from keras_pred import *
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


x_train, y_train, x_test, x_val, y_val = load_data_unet(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE,VALIDATION_SIZE, NEW_DIM_TRAIN,
  saltpepper = 0.05, augment=True, MAX_AUG=MAX_AUG, augImgDir=imgDir , data_dir=data_dir, groundTruthDir =groundTruthDir,newaugment=True)


print(y_train.shape)
print(x_train.shape)


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

print('Class weights: ',class_weights) 

inputs = Input((NEW_DIM_TRAIN, NEW_DIM_TRAIN,INPUT_CHANNELS))
model = create_model(inputs)
model.summary()




#sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1)

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_val, y_val),
          epochs=NUM_EPOCHS,
          shuffle = True,
          verbose=1,
          #validation_split = 0.1,
          #callbacks = [earlystop],
          class_weight = class_weights
          )


prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
y_submit = np.zeros((((608//IMG_PATCH_SIZE)**2)*TESTING_SIZE,2))
for i in range(1,TESTING_SIZE+1):
  gtimg, orImg = get_pred_and_ysubmit_pixelwise(test_data_filename, i, 'test', model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE,prediction_test_dir)
  gtimg.save(prediction_test_dir + "gtimg_" + str(i) + ".png")
  gtarr = np.asarray(gtimg)
  label_patches = img_crop(gtarr, IMG_PATCH_SIZE, IMG_PATCH_SIZE)
  data = np.asarray(label_patches)
  labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
  newPred = label_to_img_unet(gtarr.shape[0], gtarr.shape[1],IMG_PATCH_SIZE, IMG_PATCH_SIZE, gtarr,'test')
  img = Image.fromarray(newPred)
  img.save(prediction_test_dir + "patch_gtimg_" + str(i) + ".png")
  y_submit[((608//IMG_PATCH_SIZE)**2)*(i-1):((608//IMG_PATCH_SIZE)**2)*i,:] = labels
  overlay = make_img_overlay_pixel(orImg, img, PIXEL_DEPTH)
  overlay.save(prediction_test_dir + "overlay_" + str(i) + ".png")
  

print('y_submit: ', y_submit.shape)
print('antall vei / antall bakgrunn: ', np.sum(y_submit[:,0]))

prediction_training_dir = "predictions_training/"
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)
for i in range(1, TRAINING_SIZE+1):
    oimg, imgpred = get_prediction_with_overlay_pixelwise(train_data_filename, i, 'train', model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE)
    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
    imgpred.save(prediction_training_dir + "predictimg_" + str(i) + ".png")

  

# Make submission file
prediction_to_submission2('submission_keras.csv', y_submit)



