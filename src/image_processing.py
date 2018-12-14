import numpy as np
import cv2


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

def img_float_to_uint8(img, PIXEL_DEPTH):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def post_process(img):
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,33))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(33,1))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(33,1))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,33))
    img1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
    img2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)

    img_open = cv2.bitwise_or(img1, img2)
    img3 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel3)
    img4 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel4)
    img_close = cv2.bitwise_or(img3, img4)
    
    return img1


