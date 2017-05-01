import numpy as np
import sys
import os 
import xml.etree.ElementTree as ET
from sets import Set

try:
	data = sys.argv[1]
	results = sys.argv[2]
except BaseException:
	print("\nError: This script should be run with the following (valid) flags:\n python pre_processing.py data/ results/\n")
	sys.exit(-1)

map_path = os.getcwd() + "/" + data + "/entity_id_to_uuid.txt" 
snv_path = os.getcwd() + "/" + data + "/snv"
clinical_path = os.getcwd() + "/" + data + "/clinical"
results_path = os.getcwd() + "/" + results

# creating a map from the entity id (in our snv data) to our patient uuid (in our clinical id)
entity_id_to_uuid = {}
with open(map_path, 'r') as file:
	for line in file:
		raw = line.replace("\n", "").lstrip().split("\t")
		entity_id = raw[0]
		uuid =  raw[2]
		entity_id_to_uuid[entity_id] = uuid

# Determining the UUID and Gleason scores of our clinical data
uuid_to_gleason = {}
for subdir, dirs, files in os.walk(clinical_path):
	for file in files:
		path = subdir + "/" + file
		isXML = ((file[len(file) - 3:]) == "xml")
		if isXML:
			tree = ET.parse(path)
			root = tree.getroot()
			uuid = None
			gleason = None 
			for neighbor in root.iter():
				if neighbor.tag == "{http://tcga.nci/bcr/xml/shared/2.7}bcr_patient_uuid":
					uuid = neighbor.text.lower()
				if neighbor.tag == "{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}gleason_score":
					gleason = neighbor.text
			uuid_to_gleason[uuid] = int(gleason)

entity_id_to_gleason = {}

# Populating entity id to gleason map 
for entity_id in entity_id_to_uuid.keys():
	entity_id_to_gleason[entity_id] = (uuid_to_gleason[entity_id_to_uuid[entity_id]])

label_list = []

#get sorted list of every entity ID
entity_ids = entity_id_to_gleason.keys()
entity_ids.sort()
entity_ids_to_row = {}

for i in range (0, len(entity_ids)):
	entity_ids_to_row[entity_ids[i]] = i

#column numbers from file
id_column = 32
reference_column = 10
allele_one_column = 11
allele_two_column = 12
start_position_column = 5
end_position_column = 6

feature_set = Set()
data_matrix = []
entity_id_to_features = {}

# gathering feature set now
for subdir, dirs, files in os.walk(snv_path):
	for file in files:
		path = subdir + "/" + file
		isMAF = ((file[len(file) - 3:]) == "maf")
		if isMAF:
			snv_data = np.genfromtxt(path,dtype='str', delimiter='\t')

			for i in range(1, len(snv_data)):
				#skip onto next item
				if snv_data[i][start_position_column] != snv_data[i][end_position_column]:
					continue

				entity_id = snv_data[i][id_column]

				if not(entity_id in entity_id_to_features):
					entity_id_to_features[entity_id] = []

				feature_id = snv_data[i][start_position_column]

				#want to leave this sparse 
				if snv_data[i][reference_column] == snv_data[i][allele_one_column] and snv_data[i][reference_column] == snv_data[i][allele_two_column]:
					continue
				if snv_data[i][reference_column] == '-' or snv_data[i][allele_one_column] == '-' or snv_data[i][allele_two_column] == '-':
					continue
				elif snv_data[i][reference_column] != snv_data[i][allele_one_column] and snv_data[i][reference_column] == snv_data[i][allele_two_column]:
					next_feature = [feature_id, 1]
				elif snv_data[i][reference_column] == snv_data[i][allele_one_column] and snv_data[i][reference_column] != snv_data[i][allele_two_column]:
					next_feature = [feature_id, 1]
				else:
					next_feature = [feature_id, 2]

				feature_set.add(feature_id)
				entity_id_to_features[entity_id].append(next_feature)

			feature_set = list(feature_set)
			feature_set.sort()
			feature_id_to_column = {}

			for i in range(0, len(feature_set)):
				feature_id_to_column[feature_set[i]] = i

			data_matrix = np.zeros((len(entity_ids), len(feature_set)))

			for entity_id in entity_id_to_features:
				for i in range(0, len(entity_id_to_features[entity_id])):
					feature_vector = entity_id_to_features[entity_id][i]
					feature_id = feature_vector[0]
					feature_value = feature_vector[1]
					row = entity_ids_to_row[entity_id]
					column = feature_id_to_column[feature_id]
					data_matrix[row][column] = feature_value
	

# for entity_id in entity_ids:
# 	gleason = entity_id_to_gleason[entity_id]
# 	if gleason >= 8:
# 		label_list.append(entity_id + ": " + str(1))
# 	else:
# 		label_list.append(entity_id + ": " + str(0))

output = open(results_path + "/train_matrix.csv", 'w')
headers = 'id,'
for i in range (0, len(feature_set)):
	headers += feature_set[i] + ','
headers += '\n'
output.write(headers)


for i in range(0, len(entity_ids)):
	row = entity_ids[i] + ','
	for j in range (0, len(feature_set)):
		row += str(data_matrix[i][j]) + ','
	row += '\n'
	output.write(row)

output.close()

output = open(results_path + "/labels.txt", 'w')
output.write('This is a test\n')
output.close()
print(label_list)	



					
