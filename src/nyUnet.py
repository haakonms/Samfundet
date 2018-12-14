import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import random
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data_extraction import *
from image_augmentation import *
from pathlib import Path
from data_context import sp_noise

from keras.utils.data_utils import get_file

def load_data_unet(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE,VALIDATION_SIZE, new_dim_train,saltpepper = 0.004,augment=False,MAX_AUG=1, augImgDir='', data_dir='', groundTruthDir=''):
    
    idx = random.sample(range(1, 100), VALIDATION_SIZE)
    if augment == False:
        print('No augmenting of training images')
        print('\nLoading training images')
        x_train, x_val = extract_data_pixelwise(train_data_filename,TRAINING_SIZE,  'train', new_dim_train,idx)
        print('Train data shape: ',x_train.shape)
        y_train, y_val = extract_labels_pixelwise(train_labels_filename,TRAINING_SIZE, new_dim_train,idx)
        print('Train labels shape: ',y_train.shape)
    elif augment == True:
        print('Augmenting training images...')

        augmentation(data_dir, augImgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG,idx)
        x_train, y_train = extract_aug_data_and_labels_pixelwise(augImgDir)#, TRAINING_SIZE*(MAX_AUG+1))
        _, x_val = extract_data_pixelwise(train_data_filename,TRAINING_SIZE, 'train', new_dim_train,idx)
        _, y_val = extract_labels_pixelwise(train_labels_filename,TRAINING_SIZE, new_dim_train,idx)

    x_train = sp_noise(x_train, amount=saltpepper)
    x_test,_ = extract_data_pixelwise(test_data_filename,TESTING_SIZE,  'test', new_dim_train)
    print('Test data shape: ',x_test.shape)
    road = np.sum(y_train[:,:,:,1], dtype = int)
    background = np.sum(y_train[:,:,:,0], dtype = int)
    print('Number of samples in class 1 (background): ',road)
    print('Number of samples in class 2 (road): ',background, '\n')


    return x_train, y_train, x_test, x_val, y_val

NEW_DIM_TRAIN = 400
INPUT_CHANNELS = 3
IMAGE_ORDERING = 'channels_first'

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def create_model(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)
    
    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    ################################
    # Changed from outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9) to see if the channels where the problem 
    # That fuckes up the predictions
    # May need to change to outputs = Conv2D(None, (1, 1), activation='sigmoid') (c9) as both PILLOW and matplot lib
    # don't use a number to represent the channels for black and white Imanges... But this wouldn't compile the model
    ##################################
    outputs = Conv2D(2, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=input_img, outputs=outputs)
    return model