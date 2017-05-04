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

try:
	data = sys.argv[1]
	results = sys.argv[2]
except BaseException:
	output.write("\nError: This script should be run with the following (valid) flags:\n python train.py results/ results/\n")
	sys.exit(-1)

data_path = os.getcwd() + "/" + data + "/train_data_matrix.csv" 
labels_path = os.getcwd() + "/" + data + "/train_labels.csv"
results_path = os.getcwd() + "/" + results
output = open(results_path + "/train_output.txt", 'w')

raw_data_matrix = np.genfromtxt(data_path, skip_header=1, delimiter=',')
labels = np.genfromtxt(labels_path, delimiter=',', usecols=1)

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

pca = PCA(n_components=125)
reduced_dim = pca.fit_transform(raw_data_matrix)
output.write("Variance Explained by each PC:\n")
for i in range(len(pca.explained_variance_ratio_)):
	output.write(str(pca.explained_variance_ratio_[i]))
	output.write("\n")
output.write("\n")
pickle.dump(pca, open(results_path + "pca_object", "wb" ) )

output.write("Performance on Validation Set (Average, Max): \n")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Linear SVM w/ C = 1\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = svm.SVC(kernel='poly', C=1, degree=2)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Poly SVM, degree=2, w/ C = 1\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")
clf.fit(reduced_dim, labels)
pickle.dump(clf, open(results_path + "svm_object_degree_2", "wb" ) )

clf = svm.SVC(kernel='poly', C=1, degree=4)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Poly SVM, degree=4 w/ C = 1\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = svm.SVC(kernel='poly', C=1, degree=6)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Poly SVM, degree=6 w/ C = 1\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = svm.SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("RBF SVM, w/ C = 1\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

output.write("Accuracy on Validation Set:")
clf = svm.SVC(kernel='poly', degree=2, C=1)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Poly SVM (d=2) w/ C = 1\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = svm.SVC(kernel='poly', degree=2, C=2)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Poly SVM (d=2) w/ C = 2\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = svm.SVC(kernel='poly', degree=2, C=5)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Poly SVM (d=2) w/ C = 5\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = svm.SVC(kernel='poly', degree=2, C=10)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Poly SVM (d=2) w/ C = 10\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

model = linear_model.LogisticRegression(C=1e86, penalty="l1", fit_intercept=True) 
scores = cross_val_score(model, reduced_dim, labels, cv=5)
output.write("Logistic Regression w/ L1 Penalty\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")
model.fit(reduced_dim, labels)
pickle.dump(model, open(results_path + "logreg_object_l1", "wb" ) )

model = linear_model.LogisticRegression(C=1e86, penalty="l2", fit_intercept=True) 
scores = cross_val_score(model, reduced_dim, labels, cv=5)
output.write("Logistic Regression w/ L2 Penalty\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = MLPClassifier(hidden_layer_sizes=(125, 100), random_state=1, activation='logistic', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Neural Network (125, 100) w/ Logistic Activation Function\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = MLPClassifier(hidden_layer_sizes=(125, 125, 125, 125), random_state=1, activation='logistic', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Neural Network (125, 125, 125, 125) w/ Logistic Activation Function\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

clf = MLPClassifier(hidden_layer_sizes=(125, 100), random_state=1, activation='relu', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Neural Network (125, 100) w/ Relu Activation Function\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")
clf.fit(reduced_dim, labels)
pickle.dump(clf, open(results_path + "nn_object_relu_125_100", "wb" ) )

clf = MLPClassifier(hidden_layer_sizes=(125, 125, 125, 125), random_state=1, activation='relu', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
output.write("Neural Network (125, 125, 125, 125) w/ Relu Activation Function\n")
output.write(str(sum(scores)/len(scores)) + ", " + str(max(scores)) + "\n")

output.close()

