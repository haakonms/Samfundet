#from __future__ import print_function
#import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import sys
#import urllib
import numpy as np
#import matplotlib.image as mpimg
#from PIL import Image
#from pathlib import Path
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def augmentation(data_dir, imgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG, val_img = []):

    all_img = range(1,TRAINING_SIZE+1)
    train_img = np.setdiff1d(all_img, val_img)

    seed = 0
    datagenImg = ImageDataGenerator(
            rotation_range=180, #in radians
            zoom_range=0.4,
            fill_mode= 'reflect',
            #brightness_range=(0,2))
            vertical_flip=True,
            horizontal_flip=True
            #shear_range=0.25,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            #channel_shift_range=10,
            )
    datagenGT = ImageDataGenerator(
            rotation_range=180, #in radians
            zoom_range=0.4,
            fill_mode= 'reflect',
            #brightness_range=(0,2))
            vertical_flip=True,
            horizontal_flip=True
            #shear_range=0.25,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            #channel_shift_range=10,
            )

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


    if os.path.exists(imgDir):
        shutil.rmtree(imgDir)
        print("Directory " , imgDir ,  " already exists, overwritten")
    os.makedirs(imgDir)
    if os.path.exists(groundTruthDir):
        shutil.rmtree(groundTruthDir)
        print("Directory " , groundTruthDir ,  " already exists, overwritten")
    os.makedirs(groundTruthDir)


    #image_datagen = ImageDataGenerator(**data_gen_args)
    #ground_truth_datagen = ImageDataGenerator(**data_gen_args)

    #moving original pictures to augmentet position
    for i in train_img:
      imageid = "satImage_%.3d" % i
      image_filename = train_data_filename + imageid + ".png"
      gt_filename = train_labels_filename + imageid + ".png"
      image_dest = imgDir + "/" + imageid + ".png"
      gt_dest = groundTruthDir + "/" + imageid + ".png"
      #print(image_dest,gt_dest)
      shutil.copyfile(image_filename, image_dest)
      shutil.copyfile(gt_filename, gt_dest)

    for i in train_img:
      imageid = "satImage_%.3d" % i
      image_filename = train_data_filename + imageid + ".png"
      groundtruth_filename = train_labels_filename + imageid + ".png"
      trainImg = load_img(image_filename)
      trainLabel = load_img(groundtruth_filename,color_mode='grayscale')
      img_arr = img_to_array(trainImg)
      img_arr = img_arr.reshape((1,) + img_arr.shape)
      gT_arr = img_to_array(trainLabel)
      gT_arr = gT_arr.reshape((1,) + gT_arr.shape)
      #for j in range(5):
        #image_datagen.flow_from_directory(img_arr,batch_size=1, save_to_dir=imgDir, save_prefix=imageid,save_format='png', seed=j)
        #ground_truth_datagen.flow_from_directory(gT_arr,batch_size=1, save_to_dir=groundTruthDir, save_prefix=imageid,save_format='png', seed=j)
      j = 0
      seed_array = np.random.randint(10000, size=(1,MAX_AUG), dtype='l')
      for batch in datagenImg.flow(
        img_arr,
        batch_size=1, 
        save_to_dir=imgDir, 
        save_prefix=imageid,
        save_format='png', 
        seed=seed_array[j]):
        j +=1
        if j>=MAX_AUG:
          break
      j = 0
      for batch in datagenGT.flow(
        gT_arr,
        batch_size=1, 
        save_to_dir=groundTruthDir, 
        save_prefix=imageid,
        save_format='png', 
        seed=seed_array[j]):
        j +=1
        if j>=MAX_AUG:
          break