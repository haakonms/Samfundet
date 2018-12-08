import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from keras.utils.data_utils import get_file


# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 2
# Pretrained weights
#ZF_UNET_224_WEIGHT_PATH = 'https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/releases/download/v1.0/zf_unet_224.h5'


def ZF_UNET_224(dropout_val=0.2, weights=None):
    #if K.image_dim_ordering() == 'th':
     #   inputs = Input(( 224, 224, INPUT_CHANNELS))
      #  axis = 1
    #else:
    inputs = Input((INPUT_CHANNELS, None, None))
    axis = 1
    filters = 32
    IMAGE_ORDERING = 'channels_first'

    conv_224 = Conv2D(filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(inputs)
    pool_112 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), data_format=IMAGE_ORDERING)(conv_224)

    conv_112 = Conv2D(2*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(pool_112)
    pool_56 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), data_format=IMAGE_ORDERING)(conv_112)

    conv_56 = Conv2D(4*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(pool_56)
    pool_28 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), data_format=IMAGE_ORDERING)(conv_56)

    conv_28 = Conv2D(8*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(pool_28)
    pool_14 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), data_format=IMAGE_ORDERING)(conv_28)

    conv_14 = Conv2D(16*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(pool_14)
    pool_7 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), data_format=IMAGE_ORDERING)(conv_14)

    conv_7 = Conv2D(32*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(pool_7)
   
    up_14 = concatenate([UpSampling2D(size=(2, 2),data_format=IMAGE_ORDERING)(conv_7), conv_14], axis=axis)
    up_conv_14 = Conv2D(16*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(up_14)

    up_28 = concatenate([UpSampling2D(size=(2, 2),data_format=IMAGE_ORDERING)(up_conv_14), conv_28], axis=axis)
    up_conv_28 = Conv2D(8*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(up_28)

    up_56 = concatenate([UpSampling2D(size=(2, 2),data_format=IMAGE_ORDERING)(up_conv_28), conv_56], axis=axis)
    up_conv_56 = Conv2D(4*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(up_56)

    up_112 = concatenate([UpSampling2D(size=(2, 2),data_format=IMAGE_ORDERING)(up_conv_56), conv_112], axis=axis)
    up_conv_112 = Conv2D(2*filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(up_112)

    up_224 = concatenate([UpSampling2D(size=(2, 2),data_format=IMAGE_ORDERING)(up_conv_112), conv_224], axis=axis)
    up_conv_224 = Conv2D(filters, (3, 3), padding='same', data_format=IMAGE_ORDERING)(up_224)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1),data_format=IMAGE_ORDERING)(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)


    model = Model(inputs, conv_final, name="ZF_UNET_224")
    return model

