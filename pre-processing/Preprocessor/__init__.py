import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.info('a')

import sys
import numpy as np
import cv2

"""
image I/O func.
"""
def dcm2cvimg(data, proc_num=0, lut_min=0, lut_max=255):
    arr = data.pixel_array
    gray = cv2.convertScaleAbs(arr, alpha=(255.0/arr.max(axis=1).max(axis=0)))

    """
    arr = data.pixel_array

    wc = (arr.max() + arr.min()) / 2.0
    ww = arr.max()  - arr.min()  + 1.0

    if ('WindowCenter' in data) and ('WindowWidth' in data):
        wc = data.WindowCenter
        ww = data.WindowWidth
        try: wc = wc[0]
        except: pass
        try: ww = ww[0]
        try: pass

    minval = wc - 0.5 - (ww - 1.0) / 2.0
    maxval = wc - 0.5 + (ww - 1.0) / 2.0

    min_mask = (minval >= arr)
    to_scale = (arr > minval) & (arr < maxval)
    max_mask = (arr >= maxval)

    if min_mask.any(): arr[min_mask] = lut_min
    if to_scale.any(): arr[to_scale] = ((arr[to_scale] - (wc - 0.5)) /
                                        (ww - 1.0) + 0.5) * lut_range + lut_min

    if max_mask.any(): arr[max_mask] = lut_max

    arr = np.rint(arr).astype(np.uint8)

    col_row_string = ' '.join(reversed(map(str, arr.shape)))
    bytedata_string = '\n'.join(('P5',
                            col_row_string,
                            str(arr.max()),
                            arr.tostring()))
    """
    return gray

def img2gray(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return gray

def gray2rgb(im):
    gray = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    return gray

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
    return cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)

def trim(im):
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

# TODO: revove flip
def padding(img):
    size = max(len(img), len(img[0]))
    empty= np.zeros(img.shape, dtype=img.dtype)

    for i in range(len(img)):
        for j in range(len(img[0])):
            empty[i][j] = img[i][j]

    return empty

def colormap(img, color_map='BONE'):
    color_map_flag = getattr(cv2, 'COLORMAP_' + color_map)
    img = cv2.applyColorMap(img, color_map_flag)

    return img
