import numpy as np
import os
import sys

def mag_check(histogram):
	if(len(histogram)==0):
		histogram = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	
	histogram = np.array(histogram)
	return histogram

def dir_check(histogram):
	if(len(histogram)==0):
		histogram = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	
	histogram = np.array(histogram)
	return histogram
