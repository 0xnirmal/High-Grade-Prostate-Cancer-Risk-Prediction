import numpy as np
import sys
import os 
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pickle
from sklearn.metrics import accuracy_score

try:
	data = sys.argv[1]
	results = sys.argv[2]
except BaseException:
	print("\nError: This script should be run with the following (valid) flags:\n python test.py results/ results/\n")
	sys.exit(-1)

data_path = os.getcwd() + "/" + data + "/test_data_matrix.csv" 
labels_path = os.getcwd() + "/" + data + "/test_labels.csv"

pca_path = os.getcwd() + "/" + data + "/pca_object"
pca = pickle.load(open(pca_path, "rb" ) ) 
nn_path = os.getcwd() + "/" + data + "/nn_object_relu_125_100"
nn = pickle.load(open(nn_path, "rb" ) ) 
poly_svm_path = os.getcwd() + "/" + data + "/svm_object_degree_4"
poly_svm = pickle.load(open(poly_svm_path, "rb" ) ) 
log_reg_path = os.getcwd() + "/" + data + "/logreg_object_l1"
log_reg = pickle.load(open(log_reg_path, "rb" ) ) 

raw_data_matrix = np.genfromtxt(data_path, skip_header=1, delimiter=',')
labels = np.genfromtxt(labels_path, delimiter=',', usecols=1)

print("Normalizing Data...")
means = np.mean(raw_data_matrix, axis=0)
stds = np.std(raw_data_matrix, axis=0, ddof=1)	
std_data_matrix = np.zeros(raw_data_matrix.shape)
for i in range(raw_data_matrix.shape[0]):
	for j in range(raw_data_matrix.shape[1]):
		if stds[j] != 0:
			std_data_matrix[i, j] = (raw_data_matrix[i, j] - means[j]) / stds[j]
		else:
			std_data_matrix[i, j] = 0

raw_data_matrix = std_data_matrix
reduced_dim = pca.transform(raw_data_matrix)
print("Accuracy on Training Set:")

label_predict = nn.predict(reduced_dim)
print("Neural Network (125, 100) and Relu Activation Function")
print(accuracy_score(labels, label_predict))

label_predict = poly_svm.predict(reduced_dim)
print("Poly SVM, degree=4, c=1")
print(accuracy_score(labels, label_predict))

label_predict = log_reg.predict(reduced_dim)
print("Logistic Regression w/ L1 Penalty")
print(accuracy_score(labels, label_predict))


