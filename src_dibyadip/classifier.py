from __future__ import absolute_import
import pandas as pd
import numpy as np
import imutils
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt



# fix random seed for reproducibility
def neural_net_classifier(dataset):

	seed = 7
	
	np.random.seed(seed)

	labels= dataset[:,-1]
#	print(labels.shape)
	p = np.unique(labels)

	output_dim = len(p)

	no_of_hidden_neurons = 8

	feature = dataset[::-1]

	no_of_samples,input_dimension = feature.shape

	
	model = Sequential()
	model.add(Dense(no_of_hidden_neurons, input_dim=input_dimension, activation='relu'))
	model.add(Dense(output_dim, activation='softmax'))
	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(feature, labels, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

	scores = model.evaluate(feature, labels)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	
	

	'''estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

	results = cross_val_score(estimator, feature, labels, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))'''


def SVM(dataset):

	#a = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
	seed = 7
	np.random.seed(seed)
	
	labels= dataset[:,-1]
	#print(labels.shape)

	p = np.unique(labels)
	output_dim = len(p)

	#feature = dataset[:,:-1]	
	feature = np.delete(dataset, -1, 1)

	#print(feature.shape)
	#print(dataset)

	no_of_samples, input_dimension = feature.shape
	print(no_of_samples, input_dimension)
	
	X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size = 0.3)
 	
	#print(X_train.shape, y_train.shape, X_test.shape , y_test.shape)	

	# training a linear SVM classifier
	svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
	svm_predictions = svm_model_linear.predict(X_test)	

	#print(np.equal(svm_predictions, y_test))
	#model accuracy for X_test
	train_accuracy = svm_model_linear.score(X_train, y_train)
	test_accuracy = svm_model_linear.score(X_test, y_test)
 
	# creating a confusion matrix
	cm = confusion_matrix(y_test, svm_predictions)
	cm = cm.astype(np.float32)

	#normalising the confusion matrix
	for i in range(len(cm)):
		s = sum(cm[i])
		for j in range(len(cm[i])):
			cm[i][j]/=s;


	print(cm)
	print("Training Accuracy: ", train_accuracy*100, end='%\n')
	print("Testing Accuracy: ", test_accuracy*100, end='%\n')



def desicion_tree_classifier(dataset):

	seed = 7
	np.random.seed(seed)

	labels= dataset[:,-1]

	p = np.unique(labels)

	output_dim = len(p)

	feature = np.delete(dataset, -1, 1)

	no_of_samples,input_dimension = feature.shape
	
	X_train, X_test, y_train, y_test = train_test_split(feature, labels, random_state = 0)

	dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
	dtree_predictions = dtree_model.predict(X_test)

	#calculating the accuracy of the model
	train_accuracy = dtree_model.score(X_train, y_train)
	test_accuracy = dtree_model.score(X_test, y_test)
 	

	# creating a confusion matrix
	cm = confusion_matrix(y_test, dtree_predictions)
	cm = cm.astype(np.float32)

	#normalising the confusion matrix
	for i in range(len(cm)):
		s = sum(cm[i])
		for j in range(len(cm[i])):
			cm[i][j]/=s;


	print(cm)	
	print("Training Accuracy: ", train_accuracy*100, end='%\n')
	print("Testing Accuracy: ", test_accuracy*100, end='%\n')
	


if __name__=="__main__":

	#dataset = np.array([[0,0,1,0],[0,1,1,0],[1,0,1,1],[1,1,1,1],[1,1,1,1],[1,1,0,0],[0,1,0,1],[1,0,0,1],[1,0,1,0],[1,1,0,1]])
	df = pd.read_csv('feature_matrix.csv',sep=',',header=None)
	dataset = df.values
	neural_net_classifier(dataset)
	#dataset =np.load("feature_matrix.dat")
