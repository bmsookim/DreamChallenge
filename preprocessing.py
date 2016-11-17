from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# default packages
import glob
import sys

# installed packages
import yaml
import dicom

# implemented packages
import util

import Preprocess
from Preprocess import alignment
from Preprocess import matcher


logger = util.build_logger()

class Preprocessor(object):
    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.config = kwargs['config']

        # verify argument
        #if args.corpus not in config['data'] and args.corpus != 'all':
        if args.dataset not in config['data'][args.corpus]:
            logger.error('Invalid data corpusr: {0}'.format(args.corpus))
            sys.exit(-1)

        self.data_dir = self.config['data'][args.corpus][args.dataset]

    def load_data(self):
        args = self.args

        logger.info('load dcm files in {0}'.format(self.data_dir))

        self.dcm_file_paths = list()
        for dcm_file_path in glob.glob('/'.join([self.data_dir, '*.dcm'])):
            self.dcm_file_paths.append(dcm_file_path)

    """
    Preprocessing Main Pipeline (end-point)
    """
    def preprocessing(self):
        # find patient pair
        p_dict = self.build_patient_dict()

        # each patient
        for p_id in p_dict.keys():
            target_dir = '/'.join([
                self.config['resultDir'],
                self.args.corpus,self.args.dataset,
                p_id])

            # each view
            p = True
            for view in p_dict[p_id].keys():
                if p:
                    p = False
                    continue
                for direction in p_dict[p_id][view].keys():
                    dcm = dicom.read_file(p_dict[p_id][view][direction])
                    self.preprocessing_dcm(dcm, target_dir)

    # TODO:below function only works in dreamCh dcm format
    def build_patient_dict(self):
        patient_dict = dict()

        # pid: patient id
        # d  : direction [L | R]
        # v  : view [CC | MLO | ..]
        for dcm_file_path in self.dcm_file_paths:
            dcm = dicom.read_file(dcm_file_path)
            pid = dcm.PatientID
            (d, v) = dcm.SeriesDescription.split(' ', 1)

            if pid not in patient_dict:
                patient_dict[pid] = dict()
            if v not in patient_dict[pid]:
                patient_dict[pid][v] = dict()

            patient_dict[pid][v][d] = dcm_file_path

        return patient_dict

    def preprocessing_dcm(self, dcm, target_dir):
        img = Preprocess.dcm2cvimg(dcm)
        (d, v) = dcm.SeriesDescription.split(' ', 1)

        # execute pipeline methods by configuration
        for method in self.config['preprocessing']['modify']['pipeline']:
            if method == 'flip' and  d == 'R': continue

            method_f = getattr(Preprocess, method)
            img = method_f(img)

        # save preprocessed dcm image
        path = '/'.join([target_dir, v, d + '.png'])
        util.mkdir(path)
        Preprocess.write_img(path, img)


    def alignment_both(self, l_img, r_img):
        # feature extraction & matching
        features, matches, masks = matcher.flann(l_img, r_img)

        # adjust image
        img1, img2 = alignment.adjust(features, matches, masks)
        return img2

    def extract_roi(self, img):
        # from hwejin
        pass

if __name__ == '__main__':
    import ConfigParser
    import argparse

    # program argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus',
            required=True,
            help='will be used in preprocessing phase')
    parser.add_argument('-d', '--dataset',
            required=True,
            help='dataset in corpus')
    args = parser.parse_args()

    # load program configuration
    with open('config/preprocessing.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    preprocessor = Preprocessor(args=args, config=config)
    preprocessor.load_data()
    preprocessor.preprocessing()
