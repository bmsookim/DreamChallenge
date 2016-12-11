import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.info('a')

import sys
import numpy as np
import cv2
import extractor


"""
image I/O func.
"""
def read_img(path):
    return cv2.imread(path)

def write_img(path, image):
    cv2.imwrite(path, image)

"""
image converter
"""
def dcm2cvimg(data, proc_num=0, lut_min=0, lut_max=255):
    arr = data.pixel_array
    gray = cv2.convertScaleAbs(arr, alpha=(255.0/arr.max(axis=1).max(axis=0)))

    return gray

def img2gray(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return gray

def gray2rgb(im):
    gray = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    return gray


"""
adjust image
"""
def flip(img, config):
    direction = config['direction']

    if direction == 'H':
        d_code = 1
    elif direction == 'V':
        d_code = 0
    else:
        raise ValueError("Invalid flip-direction: {0}".format(direction))

    return cv2.flip(img, d_code)

# size parameter is tuple(W, H)
def resize(img, config=None, size=(1024,1024)):
    interpolation = getattr(cv2, 'INTER_' + config['interpolation'])

    if config is not None:
        size = (config['size']['width'], config['size']['height'])

    return cv2.resize(img, size, interpolation = interpolation)

def trim(im, config):
    ret,thresh = cv2.threshold(im,0,255,0)
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    max_index = np.argmax(areas)
    areas.remove(max(areas))

    cnt=contours[max_index]
    mask = np.zeros(im.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)

    im = cv2.bitwise_and(im, im, mask=mask)

    x,y,w,h = cv2.boundingRect(cnt)

    return im[y:y+h, x:x+w]

def padding(img, config):
    size = max(len(img), len(img[0]))
    empty= np.zeros(img.shape, dtype=img.dtype)

    for i in range(len(img)):
        for j in range(len(img[0])):
            empty[i][j] = img[i][j]

    return empty

def crop(im, config):
    method = config['method']
    if method == 'centered':
        coor = extractor.find_center(im)
        im   = extractor.crop(coor, im)
    elif method == 'roi-boundary':
        nz_coor = extractor.find_nonezero(im)
        nz_coor = extractor.merge_coord(nz_coor)
        im = extractor.crop_inner(nz_coor, im, config['min'], config['padding'])
    else:
        raise AttributeError("Invalid Method: {0}".format(method))
        sys.exit(-1)

    return im

def colormap(img, config=None, color_map='BONE'):
    if config is not None:
        color_map = getattr(cv2,'COLORMAP_' + config)
    else:
        color_map = getattr(cv2,'COLORMAP_' + color_map)
    img = cv2.applyColorMap(img, color_map)

    return img
