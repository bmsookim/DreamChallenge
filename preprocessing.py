from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# default packages
import glob

# installed packages
import yaml

# implemented packages
import util
from Preprocess import alignment
from Preprocess import matcher

logger = util.build_logger()

class Preprocessor(object):
    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.config = kwargs['config']

        # verify argument
        if args.corpus not in config['data'] and args.corpus != 'all':
            logger.error('Invalid data corpusr: {0}'.format(args.corpus))

    def load_data(self):
        args = self.args

        # all corpus
        if args.corpus == 'all':
            corpus_dirs = [config['data'][corpus]
                    for corpus in config['data'].keys()
                    if  corpus != 'pilot']
        # specific corpus
        else:
            corpus_dirs = [config['data'][args.corpus]]


        self.dcm_filePaths = dict()
        for corpus_dir in corpus_dirs:
            logger.info('load dcm files in {0}'.format(corpus_dir))
            self.dcm_filePaths[corpus_dir] = list()

            for dcm_filePath in glob.glob('/'.join([corpus_dir, '*.dcm'])):
                self.dcm_filePaths[corpus_dir].append(dcm_filePath)

    """
    Preprocessing Main Pipeline
        |- alignment
            |- trimming
            |- feature extraction
            |- feature matching
            |- alignment
        |- rio extraction
    """
    def preprocessing(self):
        # dcm_preprocessing
        for corpus_dir in self.dcm_filePaths.keys():
            for dcm_filePath in self.dcm_filePaths[corpus_dir]:
                self.__alignment__(dcm_filePath)
                self.__roi_extraction__(dcm_filePath)


    def __alignment__(self, dcm_filePath):
        dcm = util.load_dcm(dcm_filePath)
        pxl = dcm.pixel_array
        img = cv2.convertScaleAbs(arr, alpha(255.0/arr.max(axis=1).max(axis=0)))

        m = matcher.flann(img, method='surf')
        a = alignment.simple(
                features = m[0],
                matches  = m[1],
                masks    = m[2])

    def __roi_extraction__(self, img):
        # TODO:not implemented, yet
        pass

if __name__ == '__main__':
    import ConfigParser
    import argparse

    # program argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', required=True,
            help='will be used in preprocessing phase')
    args = parser.parse_args()

    # load program configuration
    with open('config/preprocessing.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    preprocessor = Preprocessor(args=args, config=config)
    preprocessor.load_data()
    preprocessor.preprocessing()
