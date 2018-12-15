from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import sys
import urllib
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow.python.platform
import tensorflow as tf
from scipy import misc, ndimage
import shutil
from sklearn.utils import class_weight, shuffle


from mask_to_submission import *
from helpers import *
from image_processing import *
from image_augmentation import *
from F1_metrics import *
from data_extraction import *
from prediction import *
from unet_pred import *
from unetModel import *


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


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

classes = np.array([0,1])
class_weights = class_weight.compute_class_weight('balanced',classes,y_train[:,:,:,0].flatten())
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


list_filename = []
prediction_test_dir = data_dir + "predictions_test/"


if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)

for i in range(1,TESTING_SIZE+1):
  gt_pred, orImg = get_pred_img_pixelwise(test_data_filename, i, 'test', model, PIXEL_DEPTH, NEW_DIM_TRAIN,prediction_test_dir)
  gt_filename = prediction_test_dir + "gt_pred_" + str(i) + ".png"
  list_filename.append(gt_filename)
  gt_pred.save(gt_filename)
  overlay2 = make_img_overlay_pixel(orImg, gt_pred, PIXEL_DEPTH)
  overlay2.save(prediction_test_dir + "overlay_" + str(i) + ".png")
  
masks_to_submission("kerasMask.csv", *list_filename)

prediction_training_dir = data_dir + "predictions_training/"

if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)
for i in range(1, TRAINING_SIZE+1):
    oimg, imgpred = get_prediction_with_overlay_pixelwise(train_data_filename, i, 'train', model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE)
    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
    imgpred.save(prediction_training_dir + "predictimg_" + str(i) + ".png")

new_test_filename = data_dir + 'test_set_post_images/'
post_processed_list = []
if not os.path.isdir(new_test_filename):
    os.mkdir(new_test_filename)



for i in range(1,TESTING_SIZE+1):
    p_img = get_postprocessed_unet(prediction_test_dir, i, 'test')
    filename = new_test_filename + "processedimg_" + str(i) + ".png"
    post_processed_list.append(filename)
    p_img.save(filename)
    pred = Image.open(filename)
    pred = pred.convert('RGB')
    imageid = "/test_%d" % i
    image_filename = test_data_filename + imageid + imageid + ".png"
    overlay = make_img_overlay_pixel(orImg, pred, PIXEL_DEPTH)
    overlay.save(new_test_filename + "overlay_" + str(i) + ".png")

masks_to_submission("kerasPostprocessedMask.csv", *post_processed_list)


