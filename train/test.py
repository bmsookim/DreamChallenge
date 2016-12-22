import lutorpy
import csv
import numpy as np
import dicom
import yaml
import glob
import sys as py_sys

import ConfigParser
import argparse
"""
Build  Classifier
"""
require('torch')
require('paths')
require('optim')
require('nn')
require('image')
require('cutorch')
models = require('networks/init')
opts = require('opts')
checkpoints = require('checkpoints')
ffi = require('ffi')
sys = require('sys')

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

opt = opts.parse(sys.argv)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

# get model file path
model_path = glob.glob('/modelState/**/model*.t7')[0]
model = torch.load(model_path)
opt = opts.parse(sys.argv)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

"""
Build preprocessor
"""
# program argument
parser = argparse.ArgumentParser()
#### Environment
parser.add_argument('-p', '--processor',
        type=int,
        required=False,
        default = 1,
        help='how many use processors')
parser.add_argument('-g', '--gpu',
        type = int,
        required=False,
        default=1,
        help='use GPU? in ROI extraction')
parser.add_argument('-q', '--queue',
        required=True,
        help='running configuration file')
parser.add_argument('-e', '--exams_metadata',
        type = int,
        required=False,
        help='if using exams_metadata')
parser.add_argument('-f', '--form',
        type = string,
        required=False,
        help='if using exams_metadata')
parser.add_argument('-v', '--valid',
        type = string,
        required=False,
        help='if using exams_metadata')
#### Dataset
parser.add_argument('-c', '--corpus',
        required=True,
        help='will be used in preprocessing phase')
parser.add_argument('-d', '--dataset',
        required=True,
        help='dataset in corpus')

args = parser.parse_args()

with open('../pre-processing/config/train.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())

py_sys.path.insert(0, '../pre-processing')
from DREAM_DM_preprocessing import App
from Preprocessor import extractor
from Dataloader import loader

s_dict = loader.build_image_walker('/metadata/images_crosswalk.tsv')

preprocessor = App(args=args, config=config)
ext = preprocessor.build_extractor()

result = dict()
for (s_id, exam_idx) in s_dict.keys():
    k = (s_id, exam_idx)

    dicom_dict = s_dict[k]
    for v in dicom_dict.keys():
        for l in dicom_dict[v].keys():
            info = dicom_dict[v][l]
            dcm = dicom.read_file('/inferenceData/' + info['fname'])
            im = preprocessor.preprocessing_dcm(dcm, l, ext)
            im = im.reshape(1, 3, 256, 256)
            t = torch.fromNumpyArray(im)
            cudat = t._cuda()
            yt = model._forward(cudat)

            exp = torch.exp(yt)
            exp_sum = exp._sum()
            exp = torch.div(exp, exp_sum)
            score = exp[0][1]

            if s_id not in result:
                result[s_id] = dict()
            if l not in result[s_id]:
                result[s_id][l] = list()

            result[s_id][l].append(float(score))

f = open('/output/predictions.tsv', 'w')
fieldnames = ['subjectId', 'laterality', 'confidence']
writer = csv.DictWriter(f, fieldnames = fieldnames, delimiter='\t')
writer.writeheader()

for s_id in result.keys():
    for l in result[s_id].keys():
        scores = np.array(result[s_id][l])
        max_score = scores.max()
        writer.writerow({'subjectId' : s_id, 'laterality' : l, 'confidence' : max_score})
f.close()
