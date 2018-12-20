''' This script recreates the best result we achieved, F1 = 0.869 in CrowdAI '''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import tensorflow.python.platform
import tensorflow as tf
from keras.layers import Input
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from data_extraction import extract_data_pixelwise
from unet_model import create_model_unet
from unet_pred import get_pred_img_pixelwise, make_img_overlay_pixel
from mask_to_submission import masks_to_submission 


''' Global definitions '''
NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255 
NUM_LABELS = 2 # Two classes, road and not-road
TESTING_SIZE = 50
IMG_DIMENSION = 400 # The images are resized to this



''' Directives for where the training images are stored '''
test_data_filename = 'data/test_set_images'
prediction_test_dir = 'predictions_test/'
weight_path = 'weights/UNET_best_weights.hdf5' 
submission_path = 'submission.csv'



''' Loading testing data to do predictions on '''
x_test, _ = extract_data_pixelwise(test_data_filename, TESTING_SIZE, 'test', IMG_DIMENSION)
print('\nData loaded.')



''' Loading the model, compile using the Adam optimizer '''
inputs = Input((IMG_DIMENSION, IMG_DIMENSION, NUM_CHANNELS))
model = create_model_unet(inputs)
print('Model loaded.')
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])



''' Loading the pre-trained weights into the model '''
model.load_weights(weight_path)
print('Weights loaded, creating predictions...\n')



''' Creating predictions and overlay images on the testing set '''
filenames = []
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)

for i in range(1,TESTING_SIZE+1):
  if (i%np.floor(TESTING_SIZE/10) == 0):
    print(str(int(np.floor(i/np.floor(TESTING_SIZE/10))*10)), '% done');
  
  groundtruth_prediction, original_img = get_pred_img_pixelwise(test_data_filename, i, 'test', model, PIXEL_DEPTH, IMG_DIMENSION, prediction_test_dir)
  gt_filename = prediction_test_dir + "gt_pred_" + str(i) + ".png"
  filenames.append(gt_filename)
  groundtruth_prediction.save(gt_filename)
  
  overlay = make_img_overlay_pixel(original_img, groundtruth_prediction, PIXEL_DEPTH)
  overlay.save(prediction_test_dir + "overlay_" + str(i) + ".png")
  
masks_to_submission(submission_path, *filenames)
print('\nFinished creating predictions! Submission file saved to', submission_path)
print('Have a nice day :)\n')
print('Finished.\n\n')
