
import numpy as np
from data_context import load_data_context

TRAINING_SIZE = 100
TESTING_SIZE = 50
IMG_PATCH_SIZE = 16
MAX_AUG = 2
CONTEXT_SIZE = 16



# Extract data into numpy arrays, divided into patches of 16x16
# Where we get the images from
data_dir = 'data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/' 
test_data_filename = data_dir + 'test_set_images'


# Where we get the augmented images from
# Directive for storing the augmented training images
imgDir = data_dir + 'training/augmented/images'
groundTruthDir = data_dir + 'training/augmented/groundtruth'


array_Dir = 'data_arrays/'


def save_imgs_as_array():


    x_train, y_train, x_test = load_data_context(train_data_filename, train_labels_filename, test_data_filename, 
        TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, TESTING_SIZE, augment=True, MAX_AUG=MAX_AUG, augImgDir=imgDir, 
        data_dir=data_dir, groundTruthDir=groundTruthDir)



    np.save(array_dir+'x_train.npy', x_train)
    np.save(array_dir+'y_train.npy', y_train)
    np.save(array_dir+'x_test.npy', x_test)
    print('saved')


def load_img_arrays():
    x_train = np.load(array_dir+'x_train.npy')
    y_train = np.load(array_dir+'y_train.npy')
    x_test = np.load(array_dir+'x_test.npy')

    print('Train data shape: ',x_train.shape)
    print('Train labels shape: ',y_train.shape)
    print('Test data shape: ',x_test.shape)

    return x_train, y_train, x_test





save_imgs_as_array()

x_train, y_train, x_test = load_img_arrays()










