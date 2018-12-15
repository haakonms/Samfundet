import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import random
from PIL import Image
import matplotlib.image as mpimg
from pathlib import Path
from image_processing import img_crop
from image_augmentation import *


def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def load_data_unet(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE,VALIDATION_SIZE, 
    new_dim_train,saltpepper = 0.004,augment=False,MAX_AUG=1, augImgDir='', data_dir='', groundTruthDir='',newaugment=True):
    
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
        if newaugment == True:
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


def extract_data_pixelwise(filename,num_images, datatype, new_dim_train,val_img=[]):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)
    for i in train_img:
        if datatype == 'train':
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
        elif datatype == 'test':
            imageid = "/test_%d" % i
            image_filename = filename + imageid + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            img = Image.open(image_filename)
            img = np.asarray(img)

            t_imgs.append(img)

        else:
            print ('File ' + image_filename + ' does not exist')

    if datatype == 'train':
        for i in val_img:
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
            if os.path.isfile(image_filename):
                img = Image.open(image_filename)
                img = np.asarray(img)
                v_imgs.append(img)
            else:
                print ('File ' + image_filename + ' does not exist')
    
    t_arr =np.array(t_imgs)
    v_arr = np.array(v_imgs) 

    return t_arr,v_arr


# Extract label images
def extract_labels_pixelwise(filename, num_images,new_dim_train, val_img=[]):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    """ We want the images with depth = 2, one for each class, one of the depths is 1 and the other 0"""
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)

    gt_imgs = []
    for i in train_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            img = Image.open(image_filename)
            img = np.asarray(img)
            t_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    for i in val_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            img = Image.open(image_filename)
            img = np.asarray(img)
            v_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    t_imgs = np.array(t_imgs)
    v_imgs = np.array(v_imgs)
    num_t_images = len(t_imgs)
    num_v_images = len(v_imgs)

    t_labels = np.zeros((num_t_images,new_dim_train,new_dim_train,2))
    v_labels = np.zeros((num_v_images,new_dim_train,new_dim_train,2))

    foreground_threshold = 0.5
    t_labels[t_imgs > foreground_threshold] = [1,0]
    t_labels[t_imgs <= foreground_threshold] = [0,1]

    v_labels[v_imgs > foreground_threshold] = [1,0]
    v_labels[v_imgs <= foreground_threshold] = [0,1]

    # Convert to dense 1-hot representation.
    return t_labels.astype(np.float32),v_labels.astype(np.float32)

def extract_aug_data_and_labels_pixelwise(filename):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    gt_imgs = []
    pathlist = Path(filename).glob('**/*.png')
    # goes through all the augmented images in image directory
    # must pair them with all the augmented groundtruth images
    for path in pathlist:
        # because path is object not string
        image_path = str(path)
        lhs,rhs = image_path.split("/images")
        img = Image.open(image_path)
        img = np.asarray(img)
        imgs.append(img)
        gt_path = lhs + '/groundtruth' + rhs
        g_img = Image.open(gt_path)
        g_img = np.asarray(g_img)
        gt_imgs.append(g_img)

    num_images = len(imgs)
    gt_imgs = np.array(gt_imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    labels = np.zeros((num_images,IMG_WIDTH,IMG_HEIGHT,2))
    foreground_threshold = 0.5
    labels[gt_imgs > foreground_threshold] = [1,0]
    labels[gt_imgs <= foreground_threshold] = [0,1]
    imgarr = np.array(imgs)

    return imgarr, labels.astype(np.float32)