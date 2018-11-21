from __future__ import print_function
import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import urllib
import numpy
import matplotlib.image as mpimg
from PIL import Image


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    #print('IMGWIDTH: ', imgwidth)
    #print('IMGHEIGHT: ',  imgheight)
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
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

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)


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
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)







def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

def img_float_to_uint8(img, PIXEL_DEPTH):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def make_img_binary(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

#######

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])

    #for i in range(0,imgheight,h):
    #    for j in range(0,imgwidth,w):
            #if labels[idx][0] > 0.5:
            #    l = 1
            #else:
            #    l = 0
    #        array_labels[j:j+w, i:i+h] = l
    #        idx = idx + 1
    
    #labels = numpy.array(labels)

    array_labels = numpy.reshape(labels,(-1,int(imgwidth)))
    print(array_labels.shape)


    return array_labels


def get_prediction(img, model, IMG_PATCH_SIZE):
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

    output_prediction = model.predict_classes(data)
    #print(output_prediction[])

    #data_node = tf.constant(data)
    #output = tf.nn.softmax(model(data_node))
    #output_prediction = s.run(output)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    return img_prediction

def get_predictionimage(filename, image_idx, datatype, model, i, IMG_PATCH_SIZE,PIXEL_DEPTH):

    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')      

    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img, model, IMG_PATCH_SIZE)
    img_prediction[img_prediction == 0] = 100
    img_prediction[img_prediction == 1] = 0
    img_prediction[img_prediction == 100] = 1

    #print(img_prediction)
    #cimg = concatenate_images(img, img_prediction)

    nChannels = len(img_prediction.shape)
    
    w = img_prediction.shape[0]
    h = img_prediction.shape[1]
    if nChannels != 3:
        img_prediction_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        img_prediction8 = img_float_to_uint8(img_prediction, PIXEL_DEPTH)          
        img_prediction_3c[:,:,0] = img_prediction8
        img_prediction_3c[:,:,1] = img_prediction8
        img_prediction_3c[:,:,2] = img_prediction8


    imgpred = Image.fromarray(img_prediction_3c)

    #filename = prediction_training_dir + "prediction_" + str(i) + ".png"
    #.save(filename)

    return imgpred



#########

def pred_to_submission_strings(y_test):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def pred_to_submission(submission_filename, y_test):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))



