from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# default packages
import glob
import sys
import multiprocessing
import csv
import shutil

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
            next(f, None)
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
    def preprocessing(self):
        target_dir = '/'.join([
            self.config['resultDir'],
            self.args.corpus,self.args.dataset])
        tmp_dir    = '/'.join([
            self.config['resultDir'],
            'tmp'])

        try:
            shutil.rmtree(target_dir)
        except:
            pass

        util.mkdir(target_dir)
        util.mkdir(tmp_dir)

        # find patient pair
        patient_dict = self.build_image_data()

        logger.debug('split patient_dict start')
        if self.proc_cnt == 1:
            proc_feeds = [patient_dict]
        else:
            # TODO:round? floor?
            feed_size = int(len(patient_dict) / self.proc_cnt) + 1
            proc_feeds = util.split_dict(patient_dict, feed_size)
        logger.debug('split patient_dict finish')

        procs = list()
        for proc_num in range(self.proc_cnt):
            proc_feed = proc_feeds[proc_num]

            proc = multiprocessing.Process(target=self.preprocessing_single_proc,
                    args=(proc_feed,self.data_dir, target_dir, tmp_dir, proc_num))

            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        # merge tmp merge data
        metadata_f = open('/'.join([target_dir, 'metadata.tsv']), 'w')
        for metadata_tmp in glob.glob('/'.join([tmp_dir, 'metadata_*.tsv'])):
            metadata_tmp_f = open(metadata_tmp, 'r')
            metadata_f.write(metadata_tmp_f.read())
        metadata_f.close()

    def preprocessing_single_proc(self, p_dict, source_dir, target_dir, tmp_dir, proc_num=0):
        logger.info('Proc{proc_num} start'.format(proc_num = proc_num))
        start = timer()

        meta_f = open('/'.join([tmp_dir, 'metadata_' + str(proc_num) + '.tsv']),'w')
        cnt=0
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
                    filename = info['fname']
                    dcm = dicom.read_file('/'.join([source_dir, info['fname']]))

                    self.preprocessing_dcm(dcm, (v,l), img_target_dir, proc_num)

                    meta_f.write('\t'.join([p_id, exam_idx, v, l, info['cancer']]))
                    meta_f.write('\n')
                    cnt+=1

        logger.info('Proc{proc_num} Finish : size[{p_size}]\telapsed[{elapsed_time}]'.format(
            proc_num = proc_num,
            p_size = len(p_dict),
            elapsed_time = timer() - start
            ))
        print("{0} - {1}".format(proc_num, cnt))
        meta_f.close()

    def preprocessing_dcm(self, dcm, (v,l), target_dir, proc_num=0):
        img = Preprocessor.dcm2cvimg(dcm, proc_num=proc_num)

        # execute pipeline methods by configuration
        for method in self.config['preprocessing']['modify']['pipeline']:
            if method == 'flip' and  l == 'R': continue
            logger.debug('start: {method}'.format(method=method))
            img = getattr(Preprocessor, method)(img)
            logger.debug('end  : {method}'.format(method=method))

        # save preprocessed dcm image
        logger.debug('start: {method}'.format(method='write'))
        img_dir = '/'.join([target_dir, l])
        util.mkdir(img_dir)
        Preprocessor.write_img('/'.join([img_dir, v + '.png']), img)
        logger.debug('end  : {method}'.format(method='write'))

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
    parser.add_argument('-v', '--valid',
            type=int,
            required=False,
            help='force to create valid set from training set')

    args = parser.parse_args()

    # load program configuration
    with open('config/preprocessing.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    preprocessor = App(args=args, config=config)
    preprocessor.preprocessing()