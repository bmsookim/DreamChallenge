import numpy as np
import pandas as pd
import re
import csv
patient_dict = dict()

with open('/preprocessedData/results.txt', 'r') as f:
    for line in f:
        components = re.split(r'\t+', line)
        p = components[0]
        dir_lst = [splits for splits in p.split("/") if splits is not ""]

        patient_id = dir_lst[3]
        exam_num = dir_lst[4]
        laterality = dir_lst[5]
        view = [splits for splits in dir_lst[6].split(".") if splits is not ""][0]

        # if(patient_dict[patient_id] == None):
        if patient_id not in patient_dict:
            patient_dict[patient_id] = dict()
        if laterality not in patient_dict[patient_id]:
            patient_dict[patient_id][laterality] = []
        patient_dict[patient_id][laterality].append(float(components[1].split("\n")[0]))

with open('/output/predictions.tsv', 'w') as csvfile:
    fieldnames = ['subjectId', 'laterality', 'confidence']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames, delimiter='\t')
    writer.writeheader()
    for p_id in patient_dict.keys():
        for l in patient_dict[p_id].keys():
            scores = patient_dict[p_id][l]
            scores = np.array(scores)
            max_score = (scores.max())
            writer.writerow({'subjectId' : p_id, 'laterality' : l, 'confidence' : max_score})
