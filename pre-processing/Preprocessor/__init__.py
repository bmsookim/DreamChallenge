import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.info('a')

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
image I/O func.
"""
def dcm2cvimg(dcm, proc_num=0):
    """
    tmp_img_path = 'tmp/tmp'+str(proc_num) + '.png'

    #TODO: imporve performance (no writing tmp image)
    plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
    plt.axis('off')
    plt.savefig(tmp_img_path, bbox_inches="tight", pad_inches=0)
    plt.clf()

    img = cv2.imread(tmp_img_path)
    """

    arr = dcm.pixel_array
    img = cv2.convertScaleAbs(arr, alpha=(255.0/arr.max(axis=1).max(axis=0)))

    return img

def read_img(path):
    return cv2.imread(path)

def write_img(path, image):
    cv2.imwrite(path, image)

"""
adjust image
"""
def flip(img, direction='H'):
    if direction == 'H':
        d_code = 1
    elif direction == 'V':
        d_code = 0
    else:
        raise ValueError("Invalid flip-direction: {0}".format(direction))

    return cv2.flip(img, d_code)

# size parameter is tuple(W, H)
def resize(img, size=(1024,1024)):
    return cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)

def trim(img):
    ret,thresh = cv2.threshold(img,0,255,0)
    _,contours,__ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)

    # Get largest contour - extract breast
    cnt=contours[max_index]
    mask = np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)

    x,y,w,h = cv2.boundingRect(cnt)

    # 1st final image with breast ROI extracted
    img = cv2.bitwise_and(img, img, mask=mask)
    trimmed = img[y:y+h, x:x+w]

    return trimmed

def padding(img):
    max_size = len(img)
    empty = np.zeros([max_size, max_size], dtype=img.dtype)

    for i in range(len(img)):
        empty[i + max_size - len(img)][max_size - len(img[0]):] = img[i]

    return empty
