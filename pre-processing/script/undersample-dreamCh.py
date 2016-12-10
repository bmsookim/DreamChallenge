import csv
from collections import defaultdict

filename = '/metadata/All/exams_metadata.tsv'
crosswalk = '/metadata/All/images_crosswalk.tsv'
result = "/metadata/images_crosswalk.tsv"
total = 0
cancer = 0

with open(filename, 'r') as tsv:
	with open(crosswalk, 'r') as cw:
		with open(result, "w") as result_tsv :
			each_line = [line.strip().split('\t') for line in tsv]
			cross = [line.strip().split('\t') for line in cw]

			print(cross[0])
			field_string = ""
			for field in cross[0]:
				field_string += (field + "\t")
			field_string += "\n"
			result_tsv.write(field_string)
			field_string = ""

			patientId_idx = [i for i in range(len(cross[0])) if cross[0][i] == 'subjectId'][0]

			matches = [i for i in range(len(each_line[0])) if \
			each_line[0][i] == 'cancerL' or\
			each_line[0][i] == 'cancerR' or\
			each_line[0][i] == 'invL' or\
			each_line[0][i] == 'invR']

			for i in range(1, len(each_line)):
				isCancer = False
				for view in matches :
					total = total + 1
					if(each_line[i][view] == "1"):
						cancer = cancer + 1
						isCancer = True

				if(isCancer):
					print ("Writing patient " + each_line[i][0] + " to " + result + "...")
					for j in range(1, len(cross)):
						if(cross[j][patientId_idx] == each_line[i][0]):
							for field in cross[j]:
								field_string += (field + "\t")
							field_string += '\n'
							result_tsv.write(field_string)
							field_string = ""