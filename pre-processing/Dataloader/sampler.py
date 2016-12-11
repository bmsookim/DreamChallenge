import csv
import sys

import shutil

def default(src_path, target_path, exams_dict):
    shutil.copyfile(src_path, target_path)

def undersampling(src_path, target_path, exams_dict):
    target_f= open(target_path, 'w')
    target_f.write('\t'.join([
        'subjectId','examIndex','imageIndex',
        'view','laterality','filename']) + '\n')

    src_f   = open(src_path, 'r')
    walker  = csv.DictReader(src_f, delimiter='\t')

    for row in walker:
        s_id = row['subjectId']
        e_id = row['examIndex']
        laterality = row['laterality']

        exam = exams_dict[(s_id, e_id)]

        if exam['cancerL'] == '1' or \
        exam['cancerR'] == '1' or \
        exam['invL']    == '1' or \
        exam['invR']    == '1':
            is_cancer = True
        else:
            is_cancer = False

        if is_cancer:
            target_f.write('\t'.join([
                s_id, e_id, row['imageIndex'],
                row['view'], row['laterality'],
                row['filename']
                ])
            )
            target_f.write('\n')

    target_f.close()
    src_f.close()
