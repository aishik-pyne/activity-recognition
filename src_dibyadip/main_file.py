from __future__ import division
from decimal import Decimal
from skimage import exposure
from skimage import feature
from shapely.geometry import Point
from shapely.geometry import Polygon
from trackbody import trackbody, HOG, HOG_temporal
from classifier import SVM, neural_net_classifier, desicion_tree_classifier
from PIL import Image
import pandas as pd
import numpy as np 
import cv2
import imutils
import scipy
import math
import os
import sys

if __name__=="__main__":
	process = sys.argv[1]

	if(process=="process"):
		Flag = False
		feature_vector = np.array([])
		count = -1
		#the root directory for the dataset is to be written as 'location/dataset_name'
		rootDir = ''

		for dirName, subdirList, fileList in os.walk(rootDir):
			vid=""
			count+=1 
			count1 = 0

			is_first = True
			for fname in fileList:
				count1 = count1 + 1
				vid = dirName+'/'+fname

				b = trackbody(vid)
				h = HOG_temporal(vid)
				b = np.concatenate((b, h), axis=0)
				
				count_final = np.array([count],dtype='int')

				b = np.concatenate((b,count_final),axis=0)	

				if Flag==False and not feature_vector:
					feature_vector = np.array([b])
					Flag=True
					
				else:
					feature_vector = np.vstack((feature_vector,b))
					
				is_first=False

				print(feature_vector.shape)
				#if count1==2:
				#	break
				#feature_vector = np.matrix(feature_vector)

		np.save('./data/feature_vector_HMDB_Hog_temp.npy', feature_vector)

	#print(feature_vector)
	else:
		feature_vector = np.load('./data/feature_vector_HMDB_Hog_temp.npy')

	SVM(feature_vector)
	#desicion_tree_classifier(feature_vector)
