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
	print("\nError: This script should be run with the following (valid) flags:\n python train.py results/ results/\n")
	sys.exit(-1)

data_path = os.getcwd() + "/" + data + "/train_data_matrix.csv" 
labels_path = os.getcwd() + "/" + data + "/train_labels.csv"
results_path = os.getcwd() + "/" + results

raw_data_matrix = np.genfromtxt(data_path, skip_header=1, delimiter=',')
labels = np.genfromtxt(labels_path, delimiter=',', usecols=1)

print("Normalizing Data...\n\n\n")
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
print("Variance Explained by each PC:")
print(pca.explained_variance_ratio_)
print("\n\n\n")
pickle.dump(pca, open(results_path + "pca_object", "wb" ) )

print("Accuracy on Validation Set:")
clf = MLPClassifier(hidden_layer_sizes=(125, 100), random_state=1, activation='logistic', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 100) and Logistic Activation Function")
print(sum(scores)/len(scores))

clf = MLPClassifier(hidden_layer_sizes=(125, 125, 125, 125), random_state=1, activation='logistic', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 125, 125, 125) and Logistic Activation Function")
print(sum(scores)/len(scores))

clf = MLPClassifier(hidden_layer_sizes=(125, 100), random_state=1, activation='relu', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 100) and Relu Activation Function")
print(sum(scores)/len(scores))
clf.fit(reduced_dim, labels)
pickle.dump(clf, open(results_path + "nn_object_relu_125_100", "wb" ) )


clf = MLPClassifier(hidden_layer_sizes=(125, 125, 125, 125), random_state=1, activation='relu', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 125, 125, 125) and Relu Activation Function")
print(sum(scores)/len(scores))

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Linear SVM w/ C = 1")
print(sum(scores)/len(scores))

clf = svm.SVC(kernel='poly', C=1, degree=2)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Poly SVM, degree=2, w/ C = 1")
print(sum(scores)/len(scores))
clf.fit(reduced_dim, labels)
pickle.dump(clf, open(results_path + "svm_object_degree_2", "wb" ) )

clf = svm.SVC(kernel='poly', C=1, degree=4)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Poly SVM, degree=4 w/ C = 1")
print(sum(scores)/len(scores))
clf.fit(reduced_dim, labels)
pickle.dump(clf, open(results_path + "svm_object_degree_4", "wb" ) )

clf = svm.SVC(kernel='poly', C=1, degree=6)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Poly SVM, degree=6 w/ C = 1")
print(sum(scores)/len(scores))

clf = svm.SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("RBF SVM, w/ C = 1")
print(sum(scores)/len(scores))

model = linear_model.LogisticRegression(C=1e86, penalty="l1", fit_intercept=True) 
scores = cross_val_score(model, reduced_dim, labels, cv=5)
print("Logistic Regression w/ L1 Penalty")
print(sum(scores)/len(scores))
model.fit(reduced_dim, labels)
pickle.dump(model, open(results_path + "logreg_object_l1", "wb" ) )

model = linear_model.LogisticRegression(C=1e86, penalty="l2", fit_intercept=True) 
scores = cross_val_score(model, reduced_dim, labels, cv=5)
print("Logistic Regression w/ L2 Penalty")
print(sum(scores)/len(scores))
