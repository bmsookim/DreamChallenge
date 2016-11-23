# default packages
import os
import multiprocessing

from itertools import izip_longest, ifilter

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
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

"""
DS functions
"""
def split_dict(d, cnt):
    result = [d.iteritems()] * cnt
    g = (dict(ifilter(None, v))
            for v in izip_longest(*result))

    return list(g)

"""
Env.
"""
def get_cpu_cnt():
    return multiprocessing.cpu_count()

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
