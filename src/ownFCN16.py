	
from keras.models import *
from keras.layers import *

IMAGE_ORDERING = 'channels_first'

def test16(nC, input_shape):
	
	img_input = Input(shape=(3,input_shape,input_shape)) ## Assume 224,224,3
	#permute = Permute((3, 1, 2), input_shape=(3,input_shape,input_shape))
	
	## Block 1
	conv1_1 = Conv2D(64, (4, 4), activation='relu', padding='same', name='conv1_1', data_format=IMAGE_ORDERING )(img_input)
	conv1_2 = Conv2D(64, (4, 4), activation='relu', padding='same', name='conv1_2', data_format=IMAGE_ORDERING )(conv1_1)
	pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1', data_format=IMAGE_ORDERING )(conv1_2)
	#f1 = x

	# Block 2
	conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(pool1)
	conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(conv2_1)
	pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(conv2_2)
	#f2 = x

	# Block 3
	conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(pool2)
	conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(conv3_1)
	conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(conv3_2)
	pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(conv3_3)
	#pool3 = x

	# Block 4
	conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(pool3)
	conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(conv4_1)
	conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(conv4_2)
	pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(conv4_3)## (None, 14, 14, 512) 

	pool411 = Conv2D( nClasses, (1, 1) , activation='relu' , padding='same', name="pool4_1", data_format=IMAGE_ORDERING)(pool4)

	# Block 5
	conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
	conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(conv5_1)
	conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(conv5_2)
	pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(conv5_3)## (None, 7, 7, 512)

	n = 4096

	conv6 = Conv2D(n, (7, 7), activation='relu', padding='same', name='conv6', data_format=IMAGE_ORDERING )(pool5)

	conv7 = Conv2D(n, (1, 1), activation='relu', padding='same', name='conv7', data_format=IMAGE_ORDERING )(conv6)

	dense1 = Conv2D(nClasses, kernel_size=(2,2), strides=(2,2), use_bias=False, data_format=IMAGE_ORDERING )(conv7)

	dense2 = Conv2DTranspose(nClasses, kernel_size=(2,2), strides=(2,2), use_bias=False, data_format=IMAGE_ORDERING )(conv7)

	

	



	model = Model(img_input, pool5)

	return model
