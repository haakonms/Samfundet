from __future__ import print_function
import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import shutil
import sys
import urllib

from PIL import Image
from mask_to_submission import *
from helpers import *
from image_processing import *
from image_augmentation import *
from F1_metrics import *
from data_context import *
from data_extraction import *
from prediction import *
from unet_pred import *

import code
import tensorflow.python.platform

import numpy as np

import tensorflow as tf
from scipy import misc, ndimage
import shutil

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from pathlib import Path
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score



NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 5
TESTING_SIZE = 50
VALIDATION_SIZE = 0  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 
NUM_EPOCHS = 1
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
MAX_AUG = 1

# The size of the patches each image is split into. Should be a multiple of 4, and the image
# size would be a multiple of this. For this assignment to get the delivery correct it has to be 16
IMG_PATCH_SIZE = 16
CONTEXT_SIZE = 16


# Extract data into numpy arrays, divided into patches of 16x16
data_dir = 'data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/' 
test_data_filename = data_dir + 'test_set_images'

# Directive for storing the augmented training images
imgDir = data_dir + 'training/augmented/images'
groundTruthDir = data_dir + 'training/augmented/groundtruth'


# Loading the data, and set wheter it is to be augmented or not
x_train, y_train, x_test, _, _ = load_data_context(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, VALIDATION_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, TESTING_SIZE,
          saltpepper = 0.00, augment=False, MAX_AUG=MAX_AUG, augImgDir=imgDir , data_dir=data_dir, groundTruthDir =groundTruthDir, newaugment=True) # The last 3 parameters can be blank when we dont want augmentation


# Shuffle the training data
ind_list = [i for i in range(y_train.shape[0])]
shuffle(ind_list)
x_train  = x_train[ind_list, :,:,:]
y_train = y_train[ind_list,:]


# Class weigths
classes = np.array([0,1])
class_weights = class_weight.compute_class_weight('balanced',classes,y_train[:,1])
print('Class weights: ',class_weights) 


# Input image dimensions
img_rows = x_train[0].shape[1]
img_cols = img_rows
input_shape = (img_rows, img_cols, NUM_CHANNELS) 


# CNN model
ordering = 'channels_last'

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=input_shape, padding="same", data_format=ordering))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format=ordering))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), activation='relu', padding="same", data_format=ordering))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format=ordering))
model.add(Dropout(0.1))

model.add(Conv2D(252, (3, 3), activation='relu', padding="same", data_format=ordering))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format=ordering))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(NUM_LABELS, activation='softmax'))

model.summary()


model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adam(),
          metrics=['accuracy'])


# Checkpoint
filepath="weights/weights.best.hdf5" # Set to preferred path
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]


# Train the model
model.fit(x_train, y_train,
          #validation_data=(x_val, y_val),
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          shuffle = True,
          verbose=1,
          validation_split = 0.1,
          class_weight = class_weights,
          callbacks = callbacks_list)


# Predict
print("Predicting")
y_submit = model.predict_classes(x_test)

# Make submission file without postprocessing
#prediction_to_submission('submission_keras.csv', y_submit)


print("Making images")
# Make images, both with predictions and overlay
prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
for i in range(1,TESTING_SIZE+1):
    test_data_filename = data_dir + 'test_set_images'

    oimg = get_prediction_with_overlay_context(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH)
    oimg.save(prediction_test_dir + "overlay_" + str(i) + ".png")

    filename = prediction_test_dir + "predictimg_" + str(i) + ".png"
    imgpred = get_predictionimage_context(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH)
    imgpred.save(filename)


# Postprocessing
new_test_filename = data_dir + 'test_set_post_images/'
if not os.path.isdir(new_test_filename):
    os.mkdir(new_test_filename)

y_submit_post = np.zeros((((608//IMG_PATCH_SIZE)**2)*TESTING_SIZE,2))

print("Postprocessing")
for i in range(1,TESTING_SIZE+1):
    y_submit_post[((608//IMG_PATCH_SIZE)**2)*(i-1):((608//IMG_PATCH_SIZE)**2)*i,:], p_img = get_pred_postprocessed(prediction_test_dir, i, 'test',IMG_PATCH_SIZE)
    filename = new_test_filename + "processedimg_" + str(i) + ".png"
    p_img.save(filename)

# Make csv file for postprocessed predictions
prediction_to_submission2('submission_keras_test.csv', y_submit_post)
