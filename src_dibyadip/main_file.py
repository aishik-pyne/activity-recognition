from __future__ import division
import os
import pandas as pd
import numpy as np 
import cv2
import imutils
import scipy
import math
import sys
from decimal import Decimal
from skimage import exposure
from skimage import feature
from shapely.geometry import Point
from shapely.geometry import Polygon
from PIL import Image
from classifier import SVM, desicion_tree_classifier, MLP
from dopt import dense_flow

if __name__=="__main__":
	process = sys.argv[1]

	if(process=="process"):
		Flag = False
		feature_vector = np.array([])
		count = -1
		rootDir = '/media/dibyadip/DC/Project Work/SKS/activity_recognition/KTH'
		#rootDir = './Weizman/'

		for dirName, subdirList, fileList in os.walk(rootDir):

			str = ""
			count = count+1 
			count1 = 0

			is_first = True
			for fname in fileList:

				count1 = count1 + 1
				str = dirName+'/'+fname
				b = dense_flow(str)
				#b.append(count)
				count_final = np.array([count],dtype='int')

				b = np.concatenate((b,count_final),axis=0)	

				#print(b)
	  
				if Flag==False and not feature_vector:
					feature_vector = np.array([b])
					Flag=True
					
				else:
					if(b.shape[0]==feature_vector.shape[1]):
						feature_vector = np.vstack((feature_vector,b))
					
				print(feature_vector.shape)
				is_first=False

		print(feature_vector.shape)
		np.save("./data/feature_vector_KTH_model.npy", feature_vector)
	else:
		feature_vector = np.load('./data/feature_vector_KTH_9parts.npy')

	#print(feature_vector.shape)
	SVM(feature_vector)
	#desicion_tree_classifier(feature_vector)
	#MLP(feature_vector)
	#feature_vector.dump("feature_matrix.dat")
	#mat2 = numpy.load("my_matrix.dat")
