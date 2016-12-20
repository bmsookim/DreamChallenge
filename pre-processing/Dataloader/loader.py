import csv

import _init_paths
import util

logger = util.build_logger()

def load(corpus, data_dir, tmp_dir, config_metadata, sampler=None):
    logger.info('load dcm file list in {0}'.format(data_dir))

    # get file paths
    cross_walk_file_path = '/'.join([
        config_metadata['dir'],
        config_metadata['images_crosswalk']
    ])

    # build exams_dict if exams_metadata exist, otherwise return None
    if 'exams_metadata' in config_metadata:
        exams_file_path  = '/'.join([
            config_metadata['dir'],
            config_metadata['exams_metadata']
        ])
        try:
            exams_dict = build_exams_dict(exams_file_path)
        except:
            exams_dict = None
    else:
        exams_dict = None

    # sampling
    target_path = tmp_dir + '/images_crosswalk.tsv'
    sampler(cross_walk_file_path, target_path, exams_dict)

    # build data walker from sampled image data
    image_walker= build_image_walker(target_path)

    return image_walker, exams_dict

def build_exams_dict(exams_file_path):
    if exams_file_path == None:
        return None

    e_dict = dict()

    f = open(exams_file_path, 'rt')
    walker = csv.DictReader(f, delimiter='\t')
    for row in walker:
        k = (row['subjectId'], row['examIndex'])

        e_dict[k] = row
    f.close()

    return e_dict

def build_image_walker(cross_walk_file_path):
    s_dict = dict()
    f = open(cross_walk_file_path, 'rt')
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

        if 'cancer' in row:
            s_dict[k][row['view']][row['laterality']]['cancer'] = row['cancer']

    f.close()

    return s_dict

