from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# default packages
import glob
import sys
import multiprocessing

from timeit import default_timer as timer

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
        if args.dataset not in config['data'][args.corpus]:
            logger.error('Invalid data corpusr: {0}'.format(args.corpus))
            sys.exit(-1)

        # TODO
        """
        if self.args.process_cnt > util.get_cpu_cnt():
            logger.error('-p , --process_num over the number of pysical process')
            sys.exit(-1)
        """

        self.data_dir = self.config['data'][args.corpus][args.dataset]

        self.proc_cnt = util.get_cpu_cnt() -1
        if self.proc_cnt < 1 :
            self.proc_cnt = util.get_cpu_cnt()

        #self.proc_cnt = 1

    def load_data(self):
        args = self.args

        logger.info('load dcm file list in {0}'.format(self.data_dir))

        self.dcm_file_paths = list()
        for dcm_file_path in glob.glob('/'.join([self.data_dir, '*.dcm'])):
            self.dcm_file_paths.append(dcm_file_path)

    """
    Preprocessing Main Pipeline (end-point)
    """
    def preprocessing(self):
        target_dir = '/'.join([
            self.config['resultDir'],
            self.args.corpus,self.args.dataset])

        # find patient pair
        proc_feeds = self.build_patient_data()

        procs = list()
        for proc_num in range(self.proc_cnt):
            proc_feed = proc_feeds[proc_num]

            proc = multiprocessing.Process(target=self.preprocessing_single_proc,
                    args=(proc_feed,target_dir, proc_num))

            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    # TODO:below function only works in dreamCh dcm format
    def build_patient_data(self):
        logger.info('build patient dicom dict')

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


        logger.debug('split patient_dict start')
        if self.proc_cnt == 1:
            result = [patient_dict]
        else:
            # TODO:round? floor?
            feed_size = int(len(patient_dict) / self.proc_cnt) + 1
            result = util.split_dict(patient_dict, feed_size)
        logger.debug('split patient_dict finish')

        """
        for i in range(len(result)):
            print(type(result), type(result[i]),len(result[i]))
        """

        return result

    def preprocessing_single_proc(self, p_dict, target_dir, proc_num=0):
        logger.info('Proc{proc_num} start'.format(proc_num = proc_num))
        start = timer()

        for p_id in p_dict.keys():
            logger.debug('Proc{proc_num} : Preprocessing... {p_id} --> {target_dir}'.format(
                proc_num = proc_num,
                p_id = p_id,
                target_dir = target_dir))

            dicom_dict = p_dict[p_id]
            target_dir = '/'.join([target_dir, p_id])
            for view in dicom_dict.keys():
                for direction in dicom_dict[view].keys():
                    dcm = dicom.read_file(p_dict[p_id][view][direction])
                    self.preprocessing_dcm(dcm, target_dir, proc_num)

        logger.info('Proc{proc_num} Finish : size[{p_size}]\telapsed[{elapsed_time}]'.format(
            proc_num = proc_num,
            p_size = len(p_dict),
            elapsed_time = timer() - start
            ))

    def preprocessing_dcm(self, dcm, target_dir, proc_num=0):
        logger.debug('convert dicom to cv2')
        img = Preprocess.dcm2cvimg(dcm, proc_num=proc_num)
        (d, v) = dcm.SeriesDescription.split(' ', 1)

        # execute pipeline methods by configuration
        for method in self.config['preprocessing']['modify']['pipeline']:
            if method == 'flip' and  d == 'R': continue
            logger.debug(method)
            method_f = getattr(Preprocess, method)
            img = method_f(img)

        # save preprocessed dcm image
        path = '/'.join([target_dir, v, d + '.png'])
        util.mkdir(path)

        logger.debug('write image')
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
    parser.add_argument('-p', '--process_cnt',
            type=int,
            required=False,
            help='how many use processor in multiprocessing')

    args = parser.parse_args()

    # load program configuration
    with open('config/preprocessing.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    preprocessor = Preprocessor(args=args, config=config)
    preprocessor.load_data()
    preprocessor.preprocessing()
