import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
image I/O func.
"""
def dcm2cvimg(dcm, proc_num=0):
    tmp_img_path = 'tmp/tmp'+str(proc_num) + '.png'

    #TODO: imporve performance (no writing tmp image)
    plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
    plt.axis('off')
    plt.savefig(tmp_img_path, bbox_inches="tight", pad_inches=0)
    plt.clf()

    img = cv2.imread(tmp_img_path)
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
def resize(img, size=(800,800)):
    return cv2.resize(img, size)

def trim(arr):
    return cv2.convertScaleAbs(arr, alpha=(255.0/arr.max(axis=1).max(axis=0)))

method = {
        'resize': resize,
        'flip'  : flip,
        'trim'  : trim,
        #TODO: 'fill'  : fill
}
