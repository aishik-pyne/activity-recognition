from __future__ import division
from decimal import Decimal
from skimage import exposure
from skimage import feature
from shapely.geometry import Point
from shapely.geometry import Polygon
from PIL import Image
import numpy as np 
import cv2
import imutils
import scipy
import math
import os

def angle(x1,y1,x2,y2):
    if x2==x1:
        return 90
    rad = math.atan(float(y2-y1)/float(x2-x1))
    deg = math.degrees(rad)
    return deg

def mag(x1,y1,x2,y2):

    sq = math.sqrt((y2-y1)**2+(x2-x1)**2)
    return sq