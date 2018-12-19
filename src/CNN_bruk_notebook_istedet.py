
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import tensorflow.python.platform
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.utils import class_weight, shuffle


from mask_to_submission import masks_to_submission
from cnn_pred import get_prediction_with_overlay_context, get_predictionimage_context, get_pred_postprocessed
from data_extraction import load_data_context
from cnn_model import create_model_cnn


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 3
TESTING_SIZE = 5
VALIDATION_SIZE = 0  # Size of the validation set.

BATCH_SIZE = 16 
NUM_EPOCHS = 2
MAX_AUG = 1
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


model = create_model_cnn(input_shape, NUM_LABELS)

model.summary()

model.compile(loss=categorical_crossentropy,
          optimizer=Adam(),
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


print("Making images")
list_filename = []
# Make images, both with predictions and overlay
prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
for i in range(1,TESTING_SIZE+1):
    test_data_filename = data_dir + 'test_set_images'

    oimg = get_prediction_with_overlay_context(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH)
    oimg.save(prediction_test_dir + "overlay_" + str(i) + ".png")

    gt_filename = prediction_test_dir + "predictimg_" + str(i) + ".png"
    imgpred = get_predictionimage_context(test_data_filename, i, 'test', model, IMG_PATCH_SIZE, CONTEXT_SIZE, PIXEL_DEPTH)
    list_filename.append(gt_filename)
    imgpred.save(gt_filename)


masks_to_submission('submission_forsok.csv', *list_filename)

'''
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
'''

