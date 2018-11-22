from __future__ import print_function
import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from mask_to_submission import *
from helpers import *

import code
import tensorflow.python.platform

import numpy as np

import tensorflow as tf
from scipy import misc, ndimage

import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from pathlib import Path
from sklearn.utils import class_weight



NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 50
TESTING_SIZE = 50
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 20
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
MAX_AUG = 10

# The size of the patches each image is split into. Should be a multiple of 4, and the image
# size would be a multiple of this. For this assignment to get the delivery correct it has to be 16
IMG_PATCH_SIZE = 16


# Extract data into numpy arrays, divided into patches of 16x16
data_dir = 'data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/' 
test_data_filename = data_dir + 'test_set_images'
x_train, y_train, x_test = load_data(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, TESTING_SIZE)


#############################################
seed = 0
datagenImg = ImageDataGenerator(
        rotation_range=20, #in radians
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.25,
        zoom_range=0.2,
        channel_shift_range=10,
        horizontal_flip=True,
        vertical_flip=True)
datagenGT = ImageDataGenerator(
        rotation_range=20, #in radians
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.25,
        zoom_range=0.2,
        channel_shift_range=10,
        horizontal_flip=True,
        vertical_flip=True)

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.15,
                     zoom_range=0.1,
                     channel_shift_range=10,
                     horizontal_flip=True,
                     vertical_flip=True)
imgDir = data_dir + 'training/augmented/images'
groundThruthDir = data_dir + 'training/augmented/groundthruth'

# Create target directory & all intermediate directories if don't exists
try:
  os.makedirs(imgDir)
  os.makedirs(groundThruthDir)
  print("Directory " , imgDir ,  " Created ")
except FileExistsError:
    print("Directory " , imgDir ,  " already exists")  


image_datagen = ImageDataGenerator(**data_gen_args)
ground_thruth_datagen = ImageDataGenerator(**data_gen_args)
for i in range(1,TRAINING_SIZE+1):
  imageid = "satImage_%.3d" % i
  image_filename = train_data_filename + imageid + ".png"
  groundthruth_filename = train_labels_filename + imageid + ".png"
  trainImg = load_img(image_filename)
  trainLabel = load_img(groundthruth_filename)
  img_arr = img_to_array(trainImg)
  img_arr = img_arr.reshape((1,) + img_arr.shape)
  gT_arr = img_to_array(trainLabel)
  gT_arr = gT_arr.reshape((1,) + gT_arr.shape)
  #for j in range(5):
    #image_datagen.flow_from_directory(img_arr,batch_size=1, save_to_dir=imgDir, save_prefix=imageid,save_format='png', seed=j)
    #ground_thruth_datagen.flow_from_directory(gT_arr,batch_size=1, save_to_dir=groundThruthDir, save_prefix=imageid,save_format='png', seed=j)
  j = 0
  for batch in datagenImg.flow(
    img_arr,
    batch_size=1, 
    save_to_dir=imgDir, 
    save_prefix=imageid,
    save_format='png', 
    seed=j):
    j +=1
    if j>=MAX_AUG:
      break
  j = 0
  for batch in datagenGT.flow(
    gT_arr,
    batch_size=1, 
    save_to_dir=groundThruthDir, 
    save_prefix=imageid,
    save_format='png', 
    seed=j):
    j +=1
    if j>MAX_AUG:
      break






print('\nLoading training images')
#x_train = extract_data(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE,  'train')
#print(x_train[:10])

print('Loading training labels')
#y_train = extract_labels(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE)


x_train, y_train = extract_aug_data_and_labels(imgDir, TRAINING_SIZE*MAX_AUG, IMG_PATCH_SIZE)

print('Loading test images\n')
x_test = extract_data(test_data_filename,TESTING_SIZE, IMG_PATCH_SIZE, 'test')
#print(x_test[:10])

print('Train data shape: ',x_train.shape)
print('Train labels shape: ',y_train.shape)
print('Test data shape: ',x_test.shape)






# Increase the dataset

train_datagen = ImageDataGenerator(
        rotation_range=10, #in radians
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.1,
        channel_shift_range=10,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator()

'''x_batch, y_batch = train_datagen.flow(x_train, y_train, batch_size=9).next()
x_train.append(x_batch)
y_train.append(y_batch)
print('Train data shape: ',x_train.shape)
print('Train labels shape: ',y_train.shape)'''

'''X_batch, y_batch = train_datagen.flow(
	x=x_train, 
	y=y_train,
	batch_size = 2
	)'''
train_datagen.fit(x_train)


#fit_generator(train_datagen, samples_per_epoch=len(train), epochs=10)

validation_generator = test_datagen.flow(
    x=x_test,
    batch_size=BATCH_SIZE,
    )


# Class weigths
classes = np.array([0,1])
class_weights = class_weight.compute_class_weight('balanced',classes,y_train[:,1])



# input image dimensions
img_rows, img_cols = BATCH_SIZE, BATCH_SIZE
input_shape = (img_rows, img_cols, NUM_CHANNELS) 

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) #32 is number of outputs from that layer, kernel_size is filter size, 
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(NUM_LABELS, activation='softmax'))

# Compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          shuffle = True,
          verbose=1,
          validation_split = 0.1,
          class_weight = class_weights)
          #validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
'''model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=25000, epochs=NUM_EPOCHS, verbose=1)'''

y_submit = model.predict_classes(x_test)
print(y_submit.shape)
print(sum(y_submit))

#image_filenames=[]
prediction_test_dir = "predictions_test/"
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

with open('submission_keras.csv', 'w') as f:
        f.write('id,prediction\n')
        #for i in range(72200):
        i=0;
        for j in range(1,50+1):
          for k in range(0,593,16):
            for l in range(0,593,16): 
              strj = ''
            
              if len(str(j))<2:
                strj='00'
              elif len(str(j))==2:
                  strj='0'

              text = strj + str(j) + '_' + str(k) + '_' + str(l) + ',' + str(y_submit[i])
              f.write(text)
              f.write('\n')
              i=i+1;






