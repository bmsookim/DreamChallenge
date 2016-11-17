import numpy as np
import cv2

def arr2cvimg(pixel_array):
    #img = cv2.imdecode(pixel_array, cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)

    return img

def flip(img, direction='H'):
    if direction == 'H':
        d_code = 1
    elif direction == 'V':
        d_code = 0
    else:
        raise ValueError("Invalid flip-direction: {0}".format(direction))

    return cv2.flip(img, d_code)

# size parameter is tuple(W, H)
def resize(img, size=(256,256)):
    return cv2.resize(img, size)

def trim(arr):
    return cv2.convertScaleAbs(arr, alpha=(255.0/arr.max(axis=1).max(axis=0)))

method = {
        'resize': resize,
        'flip'  : flip,
        'trim'  : trim,
        #TODO: 'fill'  : fill
}
