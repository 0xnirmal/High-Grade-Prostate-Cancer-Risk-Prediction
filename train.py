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

pca = PCA(n_components=125)
reduced_dim = pca.fit_transform(raw_data_matrix)
print("Variance Explained by each PC:")
print(pca.explained_variance_ratio_)
pickle.dump(pca, open(results_path + "pca_object", "wb" ) )

print("Accuracy on Validation Set:")
clf = MLPClassifier(hidden_layer_sizes=(125, 100), random_state=1, activation='logistic', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 100) and Logistic Activation Function")
print(max(scores))

clf = MLPClassifier(hidden_layer_sizes=(125, 125, 125, 125), random_state=1, activation='logistic', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 125, 125, 125) and Logistic Activation Function")
print(max(scores))

clf = MLPClassifier(hidden_layer_sizes=(125, 100), random_state=1, activation='relu', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 100) and Relu Activation Function")
print(max(scores))
clf.fit(reduced_dim, labels)
pickle.dump(clf, open(results_path + "nn_object_relu_125_100", "wb" ) )


clf = MLPClassifier(hidden_layer_sizes=(125, 125, 125, 125), random_state=1, activation='relu', max_iter=10000)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("Neural Network (125, 125, 125, 125) and Relu Activation Function")
print(max(scores))

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("RBF Linear w/ C = 1")
print(max(scores))

clf = svm.SVC(kernel='poly', C=1, degree=2)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("RBF poly, degree=2, w/ C = 1")
print(max(scores))

clf = svm.SVC(kernel='poly', C=1, degree=4)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("RBF poly, degree=4 w/ C = 1")
print(max(scores))
clf.fit(reduced_dim, labels)
pickle.dump(clf, open(results_path + "svm_object_degree_4", "wb" ) )

clf = svm.SVC(kernel='poly', C=1, degree=6)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("RBF poly, degree=6 w/ C = 1")
print(max(scores))

clf = svm.SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, reduced_dim, labels, cv=5)
print("RBF Kernel w/ C = 1")
print(max(scores))

model = linear_model.LogisticRegression(C=1e86, penalty="l1", fit_intercept=True) 
scores = cross_val_score(model, reduced_dim, labels, cv=5)
print("Logistic Regression w/ L1 Penalty")
print(max(scores))
model.fit(reduced_dim, labels)
pickle.dump(model, open(results_path + "logreg_object_l1", "wb" ) )

model = linear_model.LogisticRegression(C=1e86, penalty="l2", fit_intercept=True) 
scores = cross_val_score(model, reduced_dim, labels, cv=5)
print("Logistic Regression w/ L2 Penalty")
print(max(scores))

# clf = LinearDiscriminantAnalysis()
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("LDA w/ no DR")
# print(max(scores))

# clf = LinearDiscriminantAnalysis(n_components=10000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("LDA w/ DR=10000")
# print(max(scores))

# clf = LinearDiscriminantAnalysis(n_components=1000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("LDA w/ DR=1000")
# print(max(scores))

# clf = LinearDiscriminantAnalysis(n_components=100)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("LDA w/ DR=100")
# print(max(scores))

# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("RBF Linear w/ C = 1")
# print(max(scores))

# clf = svm.SVC(kernel='rbf', C=1)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("RBF Kernel w/ C = 1")
# print(max(scores))

# model = linear_model.LogisticRegression(C=1e86, penalty="l1", fit_intercept=True) 
# scores = cross_val_score(model, raw_data_matrix, labels, cv=5)
# print("Logistic Regression w/ L1 Penalty")
# print(max(scores))

# model = linear_model.LogisticRegression(C=1e86, penalty="l2", fit_intercept=True) 
# scores = cross_val_score(model, raw_data_matrix, labels, cv=5)
# print("Logistic Regression w/ L2 Penalty")
# print(max(scores))

# clf = MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), random_state=1, activation='logistic', max_iter=10000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("Neural Network (50, 50, 50, 50) and Logistic Activation Function")
# print(max(scores))

# clf = MLPClassifier(hidden_layer_sizes=(1000, 1000), random_state=1, activation='logistic', max_iter=10000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("Neural Network (1000, 1000) and Logistic Activation Function")
# print(max(scores))

# clf = MLPClassifier(hidden_layer_sizes=(10000, 5000, 1000, 100), random_state=1, activation='logistic', max_iter=10000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("Neural Network (10000, 5000, 1000, 100) and Logistic Activation Function")
# print(max(scores))

# clf = MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), random_state=1, activation='relu', max_iter=10000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("Neural Network (50, 50, 50, 50) and Relu Activation Function")
# print(max(scores))

# clf = MLPClassifier(hidden_layer_sizes=(1000, 1000), random_state=1, activation='relu', max_iter=10000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("Neural Network (1000, 1000) and Relu Activation Function")
# print(max(scores))

# clf = MLPClassifier(hidden_layer_sizes=(10000, 5000, 1000, 100), random_state=1, activation='relu', max_iter=10000)
# scores = cross_val_score(clf, raw_data_matrix, labels, cv=5)
# print("Neural Network (10000, 5000, 1000, 100) and Relu Activation Function")
# print(max(scores))
