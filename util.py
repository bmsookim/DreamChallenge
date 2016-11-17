# default packages
import os

import logging
import logging.config

# installed packages
import yaml
import cv2
import numpy as np


"""
File system utils
"""
def mkdir(dir_path):
    sub_path = os.path.dirname(dir_path)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

"""
Logger
"""
def build_logger(default_path='config/logging.yaml',
        default_level = logging.DEBUG,
        env_key='LOG_CFG'):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        for handler in config['handlers'].keys():
            if 'filename' in config['handlers'][handler]:
                mkdir(config['handlers'][handler]['filename'])
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=devault_level)

    return logging

"""
CV2
"""
def imshow_gray(img, title, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.title(title)
    plt.imshow(img.astype('uint8').squeeze(), cmap = 'gray')
    plt.gca().axis('off')

def imshow(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
