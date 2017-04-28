import numpy as np
import sys
import os 
import xml.etree.ElementTree as ET

try:
	data = sys.argv[1]
except BaseException:
	print("\nError: This script should be run with the following (valid) flags:\n python pre_processing.py data\n")
	sys.exit(-1)

map_path = os.getcwd() + "/" + data + "entity_id_to_uuid.txt" 
snv_path = os.getcwd() + "/" + data + "/snv"
clinical_path = os.getcwd() + "/" + data + "/clinical"

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
			uuid_to_gleason[uuid] = gleason

entity_id_to_gleason = {}

# Populating entity id to gleason map 
for entity_id in entity_id_to_uuid.keys():
	entity_id_to_gleason[entity_id] = (uuid_to_gleason[entity_id_to_uuid[entity_id]])


					
