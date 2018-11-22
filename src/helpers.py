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

    # makes a list of all patches for the image at each index
    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    # "unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    # i = antall bilder, j = hvilken patch
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    #shape of returned = (width_image/num_patches * height_image/num_patches*num_images), patch_size, patch_size, 3
    return numpy.asarray(data)

    def extract_aug_data_and_labels(filename, num_images, img_patch_s):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    gt_imgs = []
    pathlist = Path(filename).glob('**/*.png')
    #goes through all the augmented images in image directory
    # must pair them with all the augmented groundthruth images
    for path in pathlist:
        # because path is object not string
        image_path = str(path)
        lhs,rhs = image_path.split("/images")
        #print("lhs",lhs,"rhs",rhs)
        img = mpimg.imread(image_path)
        imgs.append(img)
        gt_path = lhs + '/groundthruth' + rhs
        g_img = mpimg.imread(gt_path)
        gt_imgs.append(g_img)


    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/img_patch_s)*(IMG_HEIGHT/img_patch_s)

    img_patches = [img_crop(imgs[i], img_patch_s, img_patch_s) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    gt_patches = [img_crop(gt_imgs[i], img_patch_s, img_patch_s) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    return numpy.asarray(data),labels.astype(numpy.float32)


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


def load_data(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, TESTING_SIZE):

    print('\nLoading training images')
    x_train = extract_data(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE,  'train')
    #print(x_train[:10])

    print('Loading training labels')
    y_train = extract_labels(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE)
    #print(y_train[:20])

    print('Loading test images\n')
    x_test = extract_data(test_data_filename,TESTING_SIZE, IMG_PATCH_SIZE, 'test')
    #print(x_test[:10])

    print('Train data shape: ',x_train.shape)
    print('Train labels shape: ',y_train.shape)
    print('Test data shape: ',x_test.shape)

    [cl1,cl2] = numpy.sum(y_train, axis = 0, dtype = int)
    print('Number of samples in class 1 (background): ',cl1)
    print('Number of samples in class 2 (road): ',cl2, '\n')


    return x_train, y_train, x_test




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
def label_to_img(imgwidth, imgheight, w, h, output_prediction):

    # Defines the "black white" image that is to be saved
    predict_img = numpy.zeros([imgwidth, imgheight])

    # Fills image with the predictions for each patch, so we have a int at each position in the (608,608) array
    ind = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            predict_img[j:j+w, i:i+h] = output_prediction[ind]
            ind += 1

    return predict_img

def get_prediction(img, model, IMG_PATCH_SIZE):
    
    # Turns the image into its data patches
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    #shape ((38*38), 16,16,3)

    # Data now is a vector of the patches from one single image in the testing data
    output_prediction = model.predict_classes(data)
    #predictions have shape (1444,), a prediction for each patch in the image

    return output_prediction


def get_predictionimage(filename, image_idx, datatype, model, IMG_PATCH_SIZE, PIXEL_DEPTH):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')      

    # loads the image in question
    img = mpimg.imread(image_filename)
    #data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    output_prediction = get_prediction(img, model, IMG_PATCH_SIZE)
    predict_img = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    
    # Changes into a 3D array, to easier turn into image
    predict_img_3c = numpy.zeros((img.shape[0],img.shape[1], 3), dtype=numpy.uint8)
    predict_img8 = img_float_to_uint8(predict_img, PIXEL_DEPTH)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)

    return imgpred


def make_img_overlay(img, predicted_img, PIXEL_DEPTH):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img, PIXEL_DEPTH)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, datatype, model, IMG_PATCH_SIZE, PIXEL_DEPTH):

    i = image_idx
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')

    img = mpimg.imread(image_filename)


    # Returns a vector with a prediction for each patch
    output_prediction = get_prediction(img, model, IMG_PATCH_SIZE) 
    
    # Returns a representation of the image as a 2D vector with a label at each pixel
    img_prediction = label_to_img(img.shape[0],img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    

    oimg = make_img_overlay(img, img_prediction, PIXEL_DEPTH)

    return oimg



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



