import cv2
import glob
import os
from os import walk

data_dir = '../preprocessedData/dreamCh/pilot/'

lst = os.listdir(data_dir)

for patient in lst:
    if (patient != 'metadata.tsv'):
        exam_lst = os.listdir(data_dir + patient)
        for exam in exam_lst:
            for lat in ['L', 'R']:
                for view in ['CC', 'MLO']:
                    direct = os.path.join(data_dir, patient, exam, view, lat) + '.png'
                    img = cv2.imread(direct)
                    img.shape()
                    # print (lat + view + ' : ' + data_dir + patient + view + lat + '.png')


## Please ..
## 1. Fix the order into [L/R] / [CC.png, MLO.png]

