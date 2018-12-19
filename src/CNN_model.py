from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K

def create_model_cnn(input_shape, num_classes):
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
	model.add(Dense(num_classes, activation='softmax'))

	return model