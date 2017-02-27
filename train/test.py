import os.path as osp
import sys
import csv

# add dicom-preprocessing to PYTHONPATH
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)

preprocessor_path = osp.join(this_dir, "..", "dicom-preprocessing")
add_path(preprocessor_path)


import lutorpy as lua
import yaml
import ConfigParser
import glob
import numpy as np
import cv2
from pprint import pprint

from util import util
from util import option

from ImageTools import tools as imTool
from ImageTools import coords as coordTool
from ImageTools import roi
from ImageTools import crop

from DataLoader import loader
from DataLoader import sampler

from procs import Proc

"""
PYTHON
"""
# Load configuration & merge with arguments
args = option.args
with open(args.config, 'rt') as f:
    config = yaml.safe_load(f.read())
config = option.merge_args2config(args, config)
pprint(config)

# Load data
sampler = getattr(sampler, config["sampler"])
data_all, key_all, _ = loader.load(config, sampler)
# build preprocessor
PROC_NUM = 0
preprocessor = Proc([data_all], PROC_NUM, config, 'test')
preprocessor.build_extractor(PROC_NUM)
preprocessor.build_sim_model(PROC_NUM)

"""
STATIC VARIABLE
"""
IM_MEAN = 0.496
IM_STD  = 0.229

"""
TORCH (LUA)
"""
# build trained model
require('torch')
require('paths')
require('optim')
require('nn')
require('image')
require('cutorch')

models      = require('networks/init')
#opts        = require('opts_test')
opts        = require('opts')
checkpoints = require('checkpoints')
ffi         = require('ffi')
sys         = require('sys')

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

opt = opts._parse(sys.argv)
torch.manualSeed(opt['manualSeed'])
cutorch.manualSeedAll(opt['manualSeed'])

# get model file path
checkpoint = checkpoints.best(opt)
model, criterion = models.setup(opt, checkpoint)
model._evaluate()

"""
RUN 'inference' phase
"""
IM_SIZE = config['modules']['resize']['size']

f = open('/output/predictions.tsv', 'w')
fieldnames = ['subjectId', 'laterality', 'confidence']
writer = csv.DictWriter(f, fieldnames = fieldnames, delimiter='\t')
writer.writeheader()

write_set = set()
# each subject and exam
for k in key_all:
    s_id, e_id = k
    dcm_dict = data_all[k]['dcm']
    exam_dict= data_all[k]['exam']

    dcm_info = dict()
    dcm_info['s_id'] = s_id
    dcm_info['e_id'] = e_id

    # each laterality
    for l in dcm_dict.keys():
        scores = list()
        dcm_info['laterality'] = l
        dcm_info['cancer'] = exam_dict['cancer' + l]

        try:
            imgs, rois, roi_diff, roi_sim, roi_score = preprocessor.process_laterality(dcm_dict[l], dcm_info, exam_dict)
            filtered_roi = preprocessor.process_roi_filtering(roi_sim, roi_score)

            processed_im = list()
            for r in ['mass']:
                for view, idx in filtered_roi[r]:
                    """
                    im_og = imgs[view]['gray']
                    roi   = rois[view][r][idx]
                    """
                    im_og = imgs[view]['roi'][r][idx]
                    imTool.write_im("./a.png", im_og)

                    processed_im.append(im_og)

            for im in processed_im:
                #im =  cv2.imread('/preprocessedData/dreamCh/test/0/1626_1_CC_R_0.png')
                # convert numpy image to torch cuda tensor
                im      = np.array([[
                    im[:,:,0],
                    im[:,:,1],
                    im[:,:,2],
                ]])
                im      = im.astype(np.float64)
                # normalization for compatibility with TORCH
                im      = np.divide(im, 255)
                # color normalization
                #im      = np.subtract(im, IM_MEAN)

                # infer
                im_t    = torch.fromNumpyArray(im)
                infer   = model._forward(im_t._cuda())._float()
                # TORCH: calculate score in each image
                exp     = torch.exp(infer)
                exp_sum = exp._sum()
                exp     = torch.div(exp, exp_sum)
                exp   = exp.asNumpyArray()
                score = exp[0][1]
                scores.append(score)
            if len(scores) == 0:
                scores.append(.2)
            # calculate score for subject&exam
            scores = np.array(scores)

            score_avg = np.average(scores)
            score_min = scores.min()
            score_max = scores.max()
            score_sum = scores.sum()

            if score_max - score_min < .2:
                score_fin = score_max
            else:
                score_fin = score_avg
        except Exception:
            raise
            score_fin = .2

        # write result
        write_key = (s_id.strip(), l)
        print dcm_info
        print s_id, l, score_fin
        if write_key not in write_set:
            write_set.add(write_key)
            writer.writerow({
                'subjectId': s_id,
                'laterality': l,
                'confidence': score_fin
            })

f.close()
