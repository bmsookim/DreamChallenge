from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# default packages
import glob
import sys
import multiprocessing
import csv

from timeit import default_timer as timer

# installed packages
import yaml
import dicom

# implemented packages
import util

import Preprocessor
from   Preprocessor import alignment
from   Preprocessor import matcher


logger = util.build_logger()

class App(object):
    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.config = kwargs['config']

        # verify argument
        if args.dataset not in config['data'][args.corpus]:
            logger.error('Invalid data corpusr: {0}'.format(args.corpus))
            sys.exit(-1)

        self.data_dir = self.config['data'][args.corpus][args.dataset]

        self.__assign_proc_cnt()
        self.display_setups()

    def __assign_proc_cnt(self):
        machine_proc_cnt = util.get_cpu_cnt()
        args_proc_cnt    = self.args.processor

        # no assigned args.processor argument
        #  use machine_proc_cnt - 1 (min=1)
        if self.args.processor is None:
            self.proc_cnt = machine_proc_cnt - 1
            if self.proc_cnt < 1:
                self.proc_cnt = 1
        else:
            self.proc_cnt = args_proc_cnt
            if self.proc_cnt > machine_proc_cnt or self.proc_cnt < 1:
                logger.error('Invalid used proc_cnt: {0}'.format(self.args.processor))
                sys.exit(-1)

    """
    Build Image Data from dicom files or metadata file
    """
    def build_image_data(self):
        logger.info('load dcm file list in {0}'.format(self.data_dir))

        if not self.args.metadata:
            return self.__build_image_data_from_dicom()
        else:
            return self.__build_image_data_from_metadata()

    def __build_image_data_from_dicom(self):
        logger.error('This method is deprecated: {0}'.format('__build_image_data_from_dicom'))
        sys.exit(-1)
        logger.info('build patient dicom dict')

        """
        dcm_file_paths = list()
        for dcm_file_path in glob.glob('/'.join([self.data_dir, '*.dcm'])):
            dcm_file_paths.append(dcm_file_path)

        # pid: patient id
        # d  : direction [L | R]
        # v  : view [CC | MLO | ..]
        patient_dict = dict()
        for dcm_file_path in dcm_file_paths:
            dcm = dicom.read_file(dcm_file_path)
            pid = dcm.PatientID
            (l, v) = dcm.SeriesDescription.split(' ', 1)

            if pid not in patient_dict:
                patient_dict[pid] = dict()
            if v not in patient_dict[pid]:
                patient_dict[pid][v] = dict()

            patient_dict[pid][v][l] = dcm_file_path

        return patient_dict
        """

    def __build_image_data_from_metadata(self):
        logger.info('build image data from metadata')

        metadata_dir = self.args.metadata
        cross_walk_file_path = '/'.join([metadata_dir, self.config['data']['meta']['images_crosswalk']])

        p_dict = dict()
        with open(cross_walk_file_path, 'rt') as f:
            next(f, None) # skip first line
            walker = csv.reader(f, delimiter='\t')
            for row in walker:
                (p_id, exam_idx, img_idx, v, l, fname, cancer) = row

                if (p_id, exam_idx) not in p_dict:
                    p_dict[(p_id, exam_idx)] = dict()
                if v not in p_dict[(p_id, exam_idx)]:
                    p_dict[(p_id, exam_idx)][v] = dict()

                p_dict[(p_id, exam_idx)][v][l] = {
                        'img_idx': img_idx,
                        'fname' : fname,
                        'cancer': cancer
                }

            return p_dict

    """
    Preprocessing Main Pipeline (end-point)
    """
    @profile
    def preprocessing(self):
        target_dir = '/'.join([
            self.config['resultDir'],
            self.args.corpus,self.args.dataset])
        util.mkdir(target_dir)
        util.mkdir('./tmp/')

        # find patient pair
        patient_dict = self.build_image_data()
        logger.debug('split patient_dict start')
        if self.proc_cnt == 1:
            proc_feeds = [patient_dict]
        else:
            # TODO:round? floor?
            feed_size = int(len(patient_dict) / self.proc_cnt / 10) + 1
            proc_feeds = util.split_dict(patient_dict, feed_size)
        logger.debug('split patient_dict finish')

        procs = list()
        for proc_num in range(self.proc_cnt):
            proc_feed = proc_feeds[proc_num]

            proc = multiprocessing.Process(target=self.preprocessing_single_proc,
                    args=(proc_feed,self.data_dir, target_dir, proc_num))

            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()
        """
        self.preprocessing_single_proc(proc_feeds[0],self.data_dir, target_dir, 0)
        """

    def preprocessing_single_proc(self, p_dict, source_dir, target_dir, proc_num=0):
        logger.info('Proc{proc_num} start'.format(proc_num = proc_num))
        start = timer()

        for (p_id, exam_idx) in p_dict.keys():
            k = (p_id, exam_idx)

            logger.debug('Proc{proc_num} : Preprocessing... {p_id} --> {target_dir}'.format(
                proc_num = proc_num,
                p_id = p_id,
                target_dir = target_dir))

            dicom_dict = p_dict[k]
            img_target_dir = '/'.join([target_dir, p_id, exam_idx])
            for v in dicom_dict.keys():
                for l in dicom_dict[v].keys():
                    info = dicom_dict[v][l]
                    dcm = dicom.read_file('/'.join([source_dir,dicom_dict[v][l]['fname']]))
                    self.preprocessing_dcm(dcm, img_target_dir, proc_num)

        logger.info('Proc{proc_num} Finish : size[{p_size}]\telapsed[{elapsed_time}]'.format(
            proc_num = proc_num,
            p_size = len(p_dict),
            elapsed_time = timer() - start
            ))

    @profile
    def preprocessing_dcm(self, dcm, target_dir, proc_num=0):
        logger.debug('convert dicom to cv2')
        img = Preprocessor.dcm2cvimg(dcm, proc_num=proc_num)
        (d, v) = dcm.SeriesDescription.split(' ', 1)

        # execute pipeline methods by configuration
        for method in self.config['preprocessing']['modify']['pipeline']:
            if method == 'flip' and  d == 'R': continue
            logger.debug(method)
            method_f = getattr(Preprocessor, method)
            img = method_f(img)

        # save preprocessed dcm image
        path = '/'.join([target_dir, v, d + '.png'])
        util.mkdir(path)

        logger.debug('write image')
        Preprocessor.write_img(path, img)


    def alignment_both(self, l_img, r_img):
        # feature extraction & matching
        features, matches, masks = matcher.flann(l_img, r_img)

        # adjust image
        img1, img2 = alignment.adjust(features, matches, masks)
        return img2

    def extract_roi(self, img):
        # from hwejin
        pass

    def display_setups(self):
        # TODO:
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
    parser.add_argument('-m', '--metadata',
            required=False,
            help='directory path including images_crosswalk.csv')
    parser.add_argument('-p', '--processor',
            type=int,
            required=False,
            help='how many use processores')

    args = parser.parse_args()

    # load program configuration
    with open('config/preprocessing.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    preprocessor = App(args=args, config=config)
    preprocessor.preprocessing()
