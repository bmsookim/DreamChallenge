import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import scipy.sparse
import scipy.io as sio

os.environ['GLOG_minloglevel'] = '3'

def factory(method, config):
    if method == 'mass':
        return MassExtractor(config)
    elif method == 'calcification':
        pass
    else:
        raise AttributeError('Invalid Method: {0}'.format(method))
        sys.exit(-1)

class MassExtractor():
    def __init__(self, config, gpu_id=0):
        self.CLASSES = ('__background__','mass')
        self.NETS    = {'MAMMO': ('MAMMO', 'MAMMO_faster_rcnn_final.caffemodel')}

        self.prototxt = os.path.join('model', 'proto', 'faster_rcnn_test.pt')
        self.caffemodel = os.path.join('model', 'MAMMO_mass.caffemodel')

        if not os.path.isfile(self.caffemodel):
            raise IOError(('{:s} not found').format(self.caffemodel))

        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = gpu_id
        cfg.TEST.HAS_RPN = True
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

        self.CONF_THRESH= config['threshold']
        self.NMS_THRESH = 0.3

    def get_mask(self, im):
        scores, boxes = im_detect(self.net, im)
        """Detect object classes in an image using pre-computed object proposals."""
        # Visualize detections for each class


        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            return self.vis_detections(im, cls, dets, thresh=self.CONF_THRESH)


    def vis_detections(self, im, class_name, dets, thresh=.5):
        """Draw detected bounding boxes."""
        maskImage = np.zeros(im.shape, np.uint8)
        og_im = np.copy(im)


        inds = np.where(dets[:, -1] >= thresh)[0]

        if len(inds) == 0:
            #cv2.imwrite(maskimagepath, maskImage)
            return maskImage

        im = im[:, :, (2, 1, 0)]

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            maskImage[int(bbox[1]):int (bbox[3]), int(bbox[0]):int(bbox[2])] = np.copy(og_im[int(bbox[1]):int (bbox[3]), int(bbox[0]):int(bbox[2])])

        #cv2.imwrite('./test.png', maskImage)
        return maskImage

"""
Inner Cropping
"""
def find_center(im):
    coor = {'X': {'max': None, 'min': None},
            'Y': {'max': None, 'min': None} }
    meta = dict()
    meta['size']   = {'X': im.shape[1],
                      'Y': im.shape[0]}
    meta['center'] = {'X': meta['size']['X']/2,
                      'Y': meta['size']['Y']/2}

    base   = 'X' if meta['size']['Y'] > meta['size']['X'] else 'Y'
    target = 'Y' if meta['size']['Y'] > meta['size']['X'] else 'X'
    trans  = meta['size'][base] / 2

    coor[target]['min'] = int(meta['center'][target] - trans)
    coor[target]['max'] = int(meta['center'][target] + trans)
    coor[base]['min']   = 0
    coor[base]['max']   = meta['size'][base]

    return coor

def crop(coor, im):
    im = im[coor['Y']['min']:coor['Y']['max'],
            coor['X']['min']:coor['X']['max']]

    return im

#TODO: change function name (+ ROI)
def find_nonezero(im):
    # ignore raw image (channel index:0)
    nz = np.nonzero(im[:,:,1])

    if len(nz[0]) > 0 and len(nz[1]) > 0:
        coor = dict()

        coor['Y']= {
            'min': min(nz[0]),  # min Y
            'max': max(nz[0])  # max Y
        }
        coor['X']= {
            'min': min(nz[1]),  # min X
            'max': max(nz[1])   # max X
        }
    else:
        coor = None

    return coor

def merge_coord(*args):
    coor = {'X': {'max': None, 'min': None},
            'Y': {'max': None, 'min': None} }

    for axis in ['X', 'Y']:
        max_coor = -1
        min_coor = sys.maxint

        _max = None
        _min = None
        for ch in args:
            if ch is None:continue

            _max = ch[axis]['max']
            _min = ch[axis]['min']

            if _max > max_coor:
                max_coor = _max
            if _min < min_coor:
                min_coor = _min

        coor[axis]['max'] = _max
        coor[axis]['min'] = _min

    return coor


def crop_inner(coor, im, min_size=1024, padding=10):
    # height, width, channel
    h, w = im.shape[0], im.shape[1]

    coord = {
        'Y': cal_crop_boundary(coor['Y'], h, min_size, padding, 'Y'),
        'X': cal_crop_boundary(coor['X'], w, min_size, padding, 'X')
    }

    diff_y = coord['Y'][1] -coord['Y'][0]
    diff_x = coord['X'][1] -coord['X'][0]

    if diff_x < diff_y:
        coord['X'][1] = coord['X'][0] + diff_y
        if coord['X'][1] > w : coord['X'][1] = w
    if diff_y < diff_x:
        coord['Y'][1] = coord['Y'][0] + diff_x
        if coord['Y'][1] > h : coord['Y'][1] = h

    im = im[coord['Y'][0]:coord['Y'][1],
            coord['X'][0]:coord['X'][1]]

    return im

def cal_crop_boundary(coor, og_size, min_size, padding=20, laterality='X'):
    #No min/max----------------------------

    if coor['max'] == None:
        start= og_size * 0.2
        end  = og_size * 0.8

        return [int(start), int(end)]
    #----------------------------
    start = coor['min']
    end   = coor['max']
    diff  = end-start

    if diff < min_size:
        trans = (min_size - diff)/2
        #start
        start = start-trans

        if start < 0:
            trans = (-1 * start) + trans
            start = 0
        end = end + trans
        if end > og_size    :end = og_size
        if end < coor['max']:end = coor['max']

    # apply padding
    if start - padding < 0 :
        start = 0
    else:
        start -= padding

    if end + padding > og_size:
        end = og_size
    else:
        end += padding

    return [int(start), int(end)]
