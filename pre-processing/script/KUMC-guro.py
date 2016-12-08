import csv
import glob
import dicom
import progressbar
import sys
from os.path import basename

def find_dicom_normal(u_id):
    dicom_path_regex = '/'.join([
            'Normal',
            u_id[:4] + '-' + u_id[4:],
            '*.dcm'
    ])
    dcm_paths = glob.glob(dicom_path_regex)

    return dcm_paths

def find_dicom_cancer(u_id, img_cnt):
    global err_f
    uid_num = int(u_id[-4:])
    start_idx = uid_num - uid_num%100 + 1
    end_idx  = start_idx + 99

    if img_cnt == '':
        dicom_path_regex = '/'.join([
                'Cancer',
                'image-4',
                str(start_idx) + '-' + str(end_idx),
                u_id,
                '*.dcm'
        ])
        dcm_paths = glob.glob(dicom_path_regex)

    elif img_cnt == '2':

        dicom_path_regex = '/'.join([
                'Cancer',
                'image-2',
                u_id ,
                '*.dcm'
        ])
        dcm_paths = glob.glob(dicom_path_regex)

    if len(dcm_paths) == 0:
        err_f.write(dicom_path_regex + '\n')
    return dcm_paths

def gen_meta(row, dicom_paths):
    global err_f
    metas = list()

    imageIdx = 1
    for dicom_path in dicom_paths:
        dcm = dicom.read_file(dicom_path)

        view = dcm.ViewPosition
        laterality = dcm.ImageLaterality

        try:
            if row['laterality'][0] ==  laterality:
                cancer = '1'
            else:
                cancer = '0'

            metas.append('\t'.join([
                row['subjectId'],
                '1',
                str(imageIdx),
                view,
                laterality,
                dicom_path,
                cancer
            ]))
            imageIdx += 1
        except:
            err_f.write(dicom_path)
            err_f.write('\n')
    return metas

""" Cancer"""
meta_f = open('metadata/cancer.meta.tsv', 'w')
meta_f.write('subjectId\texamIndex\timageIndex\tview\tlaterality\tfilename\tcancer')
meta_f.write('\n')

err_f = open('metadata/cancer.err.txt', 'w')

f = open('cancer.csv')
walker = csv.DictReader(f)

bar = progressbar.ProgressBar(maxval = 891, widgets=[
    progressbar.Bar('=', '[', ']'), ' ',
    progressbar.Percentage(), ' ',
    progressbar.ETA()])

cnt = 0
bar.start()
for row in walker:
    u_id = row['uniqueId']
    img_cnt = row['Image']
    dcm_paths = find_dicom_cancer(u_id, img_cnt)

    for meta in gen_meta(row, dcm_paths):
        meta_f.write(meta)
        meta_f.write('\n')
    cnt += 1
    bar.update(cnt)

bar.finish()
f.close()

f.close()
meta_f.close()
err_f.close()

f.close()

""" Normal"""
meta_f = open('metadata/normal.meta.tsv', 'w')
meta_f.write('subjectId\texamIndex\timageIndex\tview\tlaterality\tfilename\tcancer')
meta_f.write('\n')

err_f = open('metadata/normal.err.txt', 'w')

f = open('normal.csv')
walker = csv.DictReader(f)

bar = progressbar.ProgressBar(maxval = 1050, widgets=[
    progressbar.Bar('=', '[', ']'), ' ',
    progressbar.Percentage(), ' ',
    progressbar.ETA()])

cnt = 0
bar.start()
for row in walker:
    u_id = row['uniqueId']
    dcm_paths = find_dicom(u_id)

    for meta in  gen_meta(row, dcm_paths):
        meta_f.write(meta)
        meta_f.write('\n')
    cnt += 1
    bar.update(cnt)
bar.finish()
f.close()

f.close()
meta_f.close()
err_f.close()
