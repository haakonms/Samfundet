from __future__ import print_function

#import matplotlib
#matplotlib.use('Agg')


import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import shutil
import sys
import urllib

#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
from PIL import Image
from mask_to_submission import *
from helpers import *
from image_processing import *
from image_augmentation import *
from F1_metrics import *
from data_context import *
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
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
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
x_train, y_train, x_test = load_data_context(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, TESTING_SIZE,
          augment=True, MAX_AUG=MAX_AUG, augImgDir=imgDir , data_dir=data_dir, groundTruthDir =groundTruthDir) # The last 3 parameters can be blank when we dont want augmentation


#x_train_img, y_train_img, x_test_img = load_data_img(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE)

#x_train = x_train_img
#y_train = y_train_img
#x_test = x_test_img




# Class weigths
#classes = np.array([0,1])
#class_weights = class_weight.compute_class_weight('balanced',classes,y_train[:,1])
#print(class_weights) 
# {0:0.66819193, 1:1.98639715} Class 1 (road) weights mer enn class 0 (foreground)
#class_weights = {0:1, 1:4}
class_weights = (1,3)
print('Class weights: ',class_weights) 

# input image dimensions
#img_rows, img_cols = BATCH_SIZE, BATCH_SIZE
img_rows = x_train[0].shape[1]
img_cols = img_rows
#print(img_rows)
input_shape = (img_rows, img_cols, NUM_CHANNELS) 

ordering = 'channels_last'

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape, padding="same", data_format=ordering)) #32 is number of outputs from that layer, kernel_size is filter size, 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", data_format=ordering))
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", data_format=ordering))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format=ordering))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same", data_format=ordering))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same", data_format=ordering))
#model.add(Conv2D(64, (3, 3), activation='relu', padding="same", data_format=ordering))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format=ordering))
model.add(Dropout(0.25))

model.add(Conv2D(128, (2, 2), activation='relu', padding="same", data_format=ordering))
model.add(Conv2D(128, (2, 2), activation='relu', padding="same", data_format=ordering))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format=ordering))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_LABELS, activation='softmax'))

model.summary()



use_model = False
model_filename = 'weights/weights.best.context2.aug3.con16.hdf5'

if use_model == True:

    model.load_weights(model_filename)


#model.load_weights(model_filename)
model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adam(),
          metrics=['accuracy'])



if use_model == False:

    # Split train/test
    seed = 1

    train_rate = 0.80
    index_train = np.random.choice(x_train.shape[0],int(x_train.shape[0]*train_rate),replace=False)
    index_val  = list(set(range(x_train.shape[0])) - set(index_train))
                                
    x, y = shuffle(x_train,y_train)
    x_train, y_train = x[index_train],y[index_train]
    x_val, y_val = x[index_val],y[index_val]
    print('train shape: ',x_train.shape, y_train.shape)
    print('val shape: ',x_val.shape, y_val.shape)


    # F1
    class Metrics(Callback):
      def on_train_begin(self, logs={}):
        self.val_f1s = []
        #self.val_recalls = []
        #self.val_precisions = []

      def on_epoch_end(self, epoch, logs={}):
        #val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        y_validation = self.validation_data[1]
        y_validation = y_validation[:,1]
        y_validation = np.squeeze(y_validation)
        y_pred = np.asarray(self.model.predict_classes(self.validation_data[0]))
        #print('y_validation: ', y_validation.shape)

        #print('y_pred: ', y_pred.shape)
        #tp, tn, fp, fn = f1_values(y_pred, y_validation)
        #_val_f1 = 
        _val_f1 = f1_score(y_validation, y_pred, average='weighted')
        #_val_recall = recall_score(val_targ, val_predict,average='micro')
        #_val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        #self.val_recalls.append(_val_recall)
        #self.val_precisions.append(_val_precision)
        #print(' — val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1: %f' %(_val_f1))
        return
     
    # class Metrics(Callback):
    #     def on_epoch_end(self, batch, logs={}):
    #         predict = np.asarray(self.model.predict(self.validation_data[0]))
    #         targ = self.validation_data[1]
    #         self.f1s=f1(targ, predict)
    #         return

    metrics = Metrics()



    # Checkpoint
    filepath="weights/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [metrics,checkpoint]


    # Train the model
    print("X", x_train.shape, "y", y_train.shape)
    #print(y_train[:10]) # kolonne 0 sier om den er foreground eller ikke, kolonne 1 sier om den er road eller ikke
    # Altså når man lager weights med den første kolonnen, vil man få klasse 1 = road og klasse 0 = background
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              shuffle = True,
              verbose=1,
              #validation_split = 0.1,
              class_weight = class_weights,
              callbacks = callbacks_list)
              #validation_data=(x_test, y_test))
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    '''model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=25000, epochs=NUM_EPOCHS, verbose=1)'''


# Make submission file
y_submit = model.predict_classes(x_test)
prediction_to_submission('submission_keras.csv', y_submit)


y_train_val = model.predict_classes(x_train)
tp, tn, fp, fn = f1_values(y_train, y_train_val)
f1 = f1_measure(tp, fp, fn)
print("f1", f1)
'''

#y_testelitt = model.predict_classes(x_train)

#print(y_testelitt.shape)

y_submit = model.predict_classes(x_test)

print('Size of predictions: ',y_submit.shape)
print('Number of road patches: ', np.sum(y_submit))


prediction_training_dir = "predictions_training/"
#image_filenames = []
if not os.path.isdir(prediction_training_dir):
  os.mkdir(prediction_training_dir)
for i in range(1, TRAINING_SIZE+1):
    oimg = get_prediction_with_overlay_context(train_data_filename, i, 'train', model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH)
    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

    imgpred = get_predictionimage_context(train_data_filename, i, 'train', model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH)
    imgpred.save(prediction_training_dir + "predictimg_" + str(i) + ".png")

'''
#image_filenames=[]
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
    #print(filename)
    #image_filenames.append(filename)


#submission_filename = 'keras_submission'
#pred_to_submission(submission_filename,*image_filenames)    









