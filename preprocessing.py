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
            p_dir = '/'.join([
                self.config['resultDir'],
                self.args.corpus,self.args.dataset,
                p_id])

            # each view
            for view in p_dict[p_id].keys():
                for direction in p_dict[p_id][view].keys():
                    dcm = dicom.read_file(p_dict[p_id][view][direction])
                    self.preprocessing_dcm(dcm)


                """
                # if has both direction
                if 'R' in p_dict[p_id][view] and 'L' in p_dict[p_id][view]:
                    l_dcm = dicom.read_file(p_dict[p_id][view]['L'])
                    r_dcm = dicom.read_file(p_dict[p_id][view]['R'])
                    self.preprocessing_dcms(l_dcm, r_dcm)
                """

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

    def preprocessing_dcm(self, dcm):
        img = Preprocess.arr2cvimg(dcm.pixel_array)

        # execute pipeline methods by configuration
        for method in self.config['preprocessing']['modify']['pipeline']:
            method_f = getattr(Preprocess, method)
            img = method_f(img)

        util.imshow(img, 'tt')

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
    parser.add_argument('-c', '--corpus', required=True,
            help='will be used in preprocessing phase')
    parser.add_argument('-d', '--dataset', required=True,
            help='dataset in corpus')
    args = parser.parse_args()

    # load program configuration
    with open('config/preprocessing.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    preprocessor = Preprocessor(args=args, config=config)
    preprocessor.load_data()
    preprocessor.preprocessing()
