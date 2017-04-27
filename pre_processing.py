import numpy as np
import sys
import os 
import xml.etree.ElementTree as ET

try:
	snv_path = sys.argv[1]
	clinical_path = sys.argv[2]
except BaseException:
	print("\nError: This script should be run with the following (valid) flags:\n python pre_processing.py data/snv/ data/clinical/\n")
	sys.exit(-1)

snv_path = os.getcwd() + "/" + snv_path
clinical_path = os.getcwd() + "/" + clinical_path

# Determining the UUID and Gleason scores of our clinical data
for subdir, dirs, files in os.walk(clinical_path):
	for file in files:
		path = subdir + "/" + file
		isXML = ((file[len(file) - 3:]) == "xml")
		if isXML:
			tree = ET.parse(path)
			root = tree.getroot()
			for neighbor in root.iter():
				if neighbor.tag == "{http://tcga.nci/bcr/xml/shared/2.7}bcr_patient_uuid":
					uuid = neighbor.text
				if neighbor.tag == "{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}gleason_score":
					gleason = neighbor.text
					
