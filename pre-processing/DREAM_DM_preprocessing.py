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
        #self.data_dir = self.config['data'][args.dataset]

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
    Build Image and Exam  Data from dicom files or metadata file
    """
    def build_metadata(self):
        logger.info('load dcm file list in {0}'.format(self.data_dir))

        return (self.__build_image_data_from_metadata(),
                self.__build_exams_data_from_metadata())


    def __build_image_data_from_metadata(self):
        config_metadata = self.config['data'][self.args.corpus]['metadata']
        cross_walk_file_path = '/'.join([
            config_metadata['dir'],
            config_metadata['images_crosswalk']
        ])

        s_dict = dict()
        with open(cross_walk_file_path, 'rt') as f:
            walker = csv.DictReader(f, delimiter='\t')
            for row in walker:
                k = (row['subjectId'], row['examIndex'])

                if k not in s_dict:
                    s_dict[k] = dict()
                if row['view'] not in s_dict[k]:
                    s_dict[k][row['view']] = dict()

                s_dict[k][row['view']][row['laterality']] = {
                    'img_idx': row['imageIndex'],
                    'fname'  : row['filename']
                }

        return s_dict

    def __build_exams_data_from_metadata(self):
        config_metadata = self.config['data'][self.args.corpus]['metadata']
        exams_file_path = '/'.join([
            config_metadata['dir'],
            config_metadata['exams_metadata']
        ])

        e_dict = dict()
        with open(exams_file_path, 'rt') as f:
            walker = csv.DictReader(f, delimiter='\t')
            for row in walker:
                k = (row['subjectId'], row['examIndex'])

                e_dict[k] = row

        return e_dict

    """
    Preprocessing Main Pipeline (end-point)
    """
    def preprocessing(self):

        # subject_dict / exams_dict
        s_dict, self.e_dict = self.build_metadata()
        logger.info('The size of data: {0}'.format(len(s_dict)))

        if self.args.valid == 1:
            s_dict_train, s_dict_valid = self.split_train_valid(s_dict)
            logger.info('preprocessing start : {0}'.format(self.args.dataset))
            self.preprocessing_dataset(s_dict_train, self.args.dataset)
            logger.info('preprocessing start : {0}'.format('val'))
            self.preprocessing_dataset(s_dict_valid, 'val')
        else:
            logger.info('preprocessing start : {0}'.format(self.args.dataset))
            self.preprocessing_dataset(s_dict, self.args.dataset)

    def preprocessing_dataset(self, s_dict, dataset_name):
        tmp_dir    = '/'.join([
            self.config['resultDir'],
            'tmp'
        ])
        target_dir = '/'.join([
            self.config['resultDir'],
            self.args.corpus,
            dataset_name
        ])
        try:    shutil.rmtree(target_dir)
        except: pass

        util.mkdir(target_dir)
        util.mkdir(tmp_dir)


        logger.debug('split subject_dict start')
        if self.proc_cnt == 1:
            proc_feeds = [s_dict]
        else:
            # TODO:round? floor?
            feed_size = int(len(s_dict) / self.proc_cnt) + 1
            proc_feeds = util.split_dict(s_dict, feed_size)
        logger.debug('split subject_dict finish')

        procs = list()
        for proc_num in range(self.proc_cnt):
            proc_feed = proc_feeds[proc_num]

            proc = multiprocessing.Process(target=self.preprocessing_single_proc,
                    args=(proc_feed, self.data_dir, target_dir, tmp_dir, proc_num))

            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.merge_metadata(target_dir, tmp_dir)

    def split_train_valid(self, s_dict):
        s_size = len(s_dict)
        train_size = int(s_size * 0.8)

        train_valid = util.split_dict(s_dict, train_size)

        return train_valid[0],train_valid[1]

    def preprocessing_single_proc(self, s_dict, source_dir, target_dir, tmp_dir, proc_num=0):
        logger.info('Proc{proc_num} start'.format(proc_num = proc_num))
        start = timer()

        meta_f = open('/'.join([tmp_dir, 'metadata_' + str(proc_num) + '.tsv']),'w')
        cnt = 0
        for (s_id, exam_idx) in s_dict.keys():
            k = (s_id, exam_idx)

            logger.debug('Proc{proc_num} : Preprocessing... {s_id} --> {target_dir}'.format(
                proc_num = proc_num,
                s_id = s_id,
                target_dir = target_dir))

            dicom_dict = s_dict[k]
            for v in dicom_dict.keys():
                for l in dicom_dict[v].keys():
                    info = dicom_dict[v][l]
                    filename = info['fname']
                    dcm = dicom.read_file('/'.join([source_dir, info['fname']]))

                    # run image preprocessing and save result
                    img = self.preprocessing_dcm(dcm, l, proc_num)
                    exams = self.e_dict[k]

                    self.write_img(img, exams, {
                        's_id': s_id,
                        'exam_idx': exam_idx,
                        'v': v,
                        'l': l
                        }, target_dir)

                    meta_f.write('\t'.join([s_id, exam_idx, v, l, exams['cancer' + l]]))
                    meta_f.write('\n')
            cnt+=1

        logger.info('Proc{proc_num} Finish : size[{p_size}]\telapsed[{elapsed_time}]'.format(
            proc_num = proc_num,
            p_size = len(s_dict),
            elapsed_time = timer() - start
            ))
        meta_f.close()

    def preprocessing_dcm(self, dcm, l, proc_num=0):
        logger.debug('start: {method}'.format(method='handle dcm'))
        img = Preprocessor.dcm2cvimg(dcm, proc_num)

        # execute pipeline methods by configuration
        for method in self.config['preprocessing']['modify']['pipeline']:
            if method == 'flip' and  l == 'R': continue
            logger.debug('start: {method}'.format(method=method))
            img = getattr(Preprocessor, method)(img)
            logger.debug('end  : {method}'.format(method=method))

        return img

    def write_img(self, img, exams, meta, target_dir):
        if self.args.form == 'class':
            cancer_label = exams['cancer' + meta['l']]
            img_dir = '/'.join([target_dir, cancer_label])
            util.mkdir(img_dir)
            Preprocessor.write_img('/'.join([img_dir,
                '_'.join([meta['s_id'], meta['exam_idx'], meta['v'], meta['l']])  + '.png']), img)
        elif self.args.form == 'robust':
            img_dir = '/'.join([target_dir, meta['s_id'], meta['exam_idx'], meta['l']])
            util.mkdir(img_dir)
            Preprocessor.write_img('/'.join([img_dir, v + '.png']), img)
        else:
            logger.error('invalid form: {form}'.format(form=self.args.form))
            sys.exit(-1)

    def merge_metadata(self, target_dir, tmp_dir):
        img_cnt = 0

        # merge tmp merge data
        metadata_f = open('/'.join([target_dir, 'metadata.tsv']), 'w')
        for metadata_tmp in glob.glob('/'.join([tmp_dir, 'metadata_*.tsv'])):
            metadata_tmp_f = open(metadata_tmp, 'r')
            metadata_f.write(metadata_tmp_f.read())
            img_cnt += 1

        return img_cnt

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
    parser.add_argument('-p', '--processor',
            type=int,
            required=False,
            help='how many use processores')
    parser.add_argument('-f', '--form',
            required=True,
            help='class, robust')
    parser.add_argument('-v', '--valid',
            type=int,
            required=False,
            help='generate validset from training')

    args = parser.parse_args()

    # load program configuration
    with open('config/preprocessing.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    preprocessor = App(args=args, config=config)
    preprocessor.preprocessing()
