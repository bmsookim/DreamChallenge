import csv
from collections import defaultdict

crosswalk = '/data/KUMC-guro/metadata/images_crosswalk.all.tsv'
result = '/data/KUMC-guro/metadata/images_crosswalk.undersampling.tsv'

with open(crosswalk, 'r') as cw:
	with open(result, "w") as result_tsv :
		cross = [line.strip().split('\t') for line in cw]

		print(cross[0])
		field_string = ""
		for field in cross[0]:
			field_string += (field + "\t")
		field_string += "\n"
		result_tsv.write(field_string)
		field_string = ""

		patientId_idx = [i for i in range(len(cross[0])) if cross[0][i] == 'subjectId'][0]
		cancer_idx = [i for i in range(len(cross[0])) if cross[0][i] == 'cancer'][0]

		matchingPatientId = ""
		count_cancer = 0
		count_benign = 0
		cancer_lst = []

		for i in range(1, len(cross)):
			if(cross[i][cancer_idx] == "1"):
				# print ("Writing patient " + cross[i][patientId_idx] + " to " + result + "...")
				for field in cross[i]:
					field_string += (field + "\t")
				field_string += '\n'
				result_tsv.write(field_string)
				field_string = ""
				count_cancer += 1
				matchingPatientId = cross[i][patientId_idx]
			elif(cross[i][patientId_idx] == matchingPatientId):
				cancer_lst.append(i)
				print ("Writing patient " + cross[i][patientId_idx] + " to " + result + "...")
				for field in cross[i]:
					field_string += (field + "\t")
				field_string += '\n'
				result_tsv.write(field_string)
				field_string = ""
				count_benign += 1
				matchingPatientId = ""

		for i in range(1, len(cross)):
			if(cross[i][cancer_idx] == "0" and i not in cancer_lst):
				# print ("Writing patient " + cross[i][patientId_idx] + " to " + result + "...")
				for field in cross[i]:
					field_string += (field + "\t")
				field_string += '\n'
				result_tsv.write(field_string)
				field_string = ""
				count_benign += 1

				if count_benign >= count_cancer :
					break

		print count_cancer, count_benign
		result_tsv.close()