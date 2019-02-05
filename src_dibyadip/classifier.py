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
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def one_hot(labels, output_dim):
	labels = list(labels)
	
	vector = np.zeros((len(labels), output_dim), dtype=int)
	for i in range(len(labels)):
		#print(i, int(labels[i])-1)
		vector[i][int(labels[i])-1]=1

	return vector


def MLP(dataset):
	seed = 7
	epoch = 100

	np.random.seed(seed)

	labels= dataset[:,-1]

	p = np.unique(labels)

	output_dim = len(p)
	labels = one_hot(labels, output_dim)

	feature = np.delete(dataset, -1, 1)	

	no_of_samples, input_dimension = feature.shape
	print(no_of_samples, input_dimension)
	
	X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size = 0.3)
	#print(X_train.shape, y_train.shape)

	tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

	net = input_data(shape=[None, input_dimension], name='input')
	
	net = tflearn.fully_connected(net, 1024, activation='relu')
	net = tflearn.dropout(net, 0.5)

	net = tflearn.fully_connected(net, 1024, activation='relu')
	net = tflearn.dropout(net, 0.5)

	net = tflearn.fully_connected(net, output_dim, activation='softmax')
	net = regression(net, optimizer='adam', learning_rate=0.01, 
					loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(net)
	
	#model.summary()
	tr = (input_dimension*1024 + 1) + (1024*1024 + 1) + (1024*6 + 1) 
	print("Trainable parameters:", tr)

	model.fit({'input' : X_train}, {'targets' : y_train}, n_epoch=epoch, 
				validation_set=({'input' : X_test}, {'targets' : y_test}),
						show_metric=True, run_id='DCNet')




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
	#print(X_train[0])
 
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
