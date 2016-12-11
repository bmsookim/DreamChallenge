import csv

import _init_paths
import util

logger = util.build_logger()

def load(data_dir, config):
    logger.info('load dcm file list in {0}'.format(data_dir))

    image_walker= build_image_walker(config)
    exams_dict  = build_exams_dict  (config)

    return image_walker, exams_dict

def build_image_walker(config_metadata):
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

            if 'cancer' in row:
                s_dict[k][row['view']][row['laterality']]['cancer'] = row['cancer']

    return s_dict


def build_exams_dict(config_metadata):
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
