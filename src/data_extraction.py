#from __future__ import print_function
#import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import sys
#import urllib
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
#from PIL import Image
#from pathlib import Path
#import shutil
from pathlib import Path
from image_processing import img_crop
#from helpers import value_to_class


def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]



def extract_data(filename, num_images, IMG_PATCH_SIZE, datatype):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        if datatype == 'train':
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
        elif datatype == 'test':
            imageid = "/test_%d" % i
            image_filename = filename + imageid + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    # makes a list of all patches for the image at each index
    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    # "unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    # i = antall bilder, j = hvilken patch
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    #print("data",data.shape)
    #shape of returned = (width_image/num_patches * height_image/num_patches*num_images), patch_size, patch_size, 3
    return np.asarray(data)






def extract_aug_data_and_labels(filename, num_images, IMG_PATCH_SIZE):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    gt_imgs = []
    pathlist = Path(filename).glob('**/*.png')
    #goes through all the augmented images in image directory
    # must pair them with all the augmented groundtruth images
    for path in pathlist:
        # because path is object not string
        image_path = str(path)
        lhs,rhs = image_path.split("/images")
        img = mpimg.imread(image_path)
        imgs.append(img)
        gt_path = lhs + '/groundtruth' + rhs
        g_img = mpimg.imread(gt_path)
        gt_imgs.append(g_img)


    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
    # makes a list of all patches for the image at each index
    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    # "unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    # i = antall bilder, j = hvilken patch
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data_gt = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data_gt[i])) for i in range(len(data_gt))])

    return np.asarray(data), labels.astype(np.float32)


# Extract label images
def extract_labels(filename, num_images, IMG_PATCH_SIZE):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)



def extract_data_pixelwise(filename, num_images, datatype, new_dim_train):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        if datatype == 'train':
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
        elif datatype == 'test':
            imageid = "/test_%d" % i
            image_filename = filename + imageid + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            #img = mpimg.imread(image_filename)
            img = Image.open(image_filename)
            if datatype == 'train':
                #img = img.resize((new_dim_train , new_dim_train))
                if i == 1:
                    img.save('traintest.png')
            img = np.asarray(img)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    return np.asarray(imgs)


# Extract label images
def extract_labels_pixelwise(filename, num_images, new_dim_train):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    """ We want the images with depth = 2, one for each class, one of the depths is 1 and the other 0"""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            #img = mpimg.imread(image_filename)
            img = Image.open(image_filename)
            #print(img.shape)
            #img.resize((new_dim_train , new_dim_train))
            #img = img.resize((new_dim_train , new_dim_train))
            if i == 1:
                img.save('labeltest.png')
            img = np.asarray(img)
            #print(img.shape)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    gt_imgs = np.asarray(gt_imgs)
    #print(len(gt_imgs))
    labels = np.zeros((num_images,new_dim_train,new_dim_train,2))

    foreground_threshold = 0.5
    labels[gt_imgs > foreground_threshold] = [1,0]
    labels[gt_imgs <= foreground_threshold] = [0,1]

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)

def extract_aug_data_and_labels_pixelwise(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    gt_imgs = []
    pathlist = Path(filename).glob('**/*.png')
    #goes through all the augmented images in image directory
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
    #w,h = imgs[0].size
    #print(w,h)
    gt_imgs = np.asarray(gt_imgs)
    print("gt", gt_imgs.shape)#, "img", imgs.shape)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    #N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
    # makes a list of all patches for the image at each index
    #img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    # "unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    # i = antall bilder, j = hvilken patch
    #data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    #gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    #data_gt = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    #labels = np.asarray([value_to_class(np.mean(data_gt[i])) for i in range(len(data_gt))])
    labels = np.zeros((num_images,IMG_WIDTH,IMG_HEIGHT,2))

    foreground_threshold = 0.5
    labels[gt_imgs > foreground_threshold] = [1,0]
    labels[gt_imgs <= foreground_threshold] = [0,1]

    return np.asarray(imgs), labels.astype(np.float32)