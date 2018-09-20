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


def find_parts(x1,y1,x2,y2,x3,y3,w,h):

    y_lower = int(y3+h/3)
    y_middle = int(y3+2*h/3)

    x_max,y_max,x_min,y_min,x_move,y_move = x1,y1,x2,y2,x3,y3

    #print(y_lower,y1,y2)
    h_move = int(h/3)
    h1_move = int(2*h/3)
    w_move = w
    c1_x = 0
    c1_y = 0
    c2_x = 0
    c2_y = 0
    c3_x = 0
    c3_y = 0
    if x1==-9000:
        
        polygon = Polygon([(x3, y_lower), (x3+w, y_lower), (x3+w, y3),(x3,y3)])
        points = polygon.centroid
        c1_x = points.x
        c1_y = points.y

    elif y_lower<=y2 and y_lower<=y1:
        
        polygon = Polygon([(x3, y_lower), (x3+w, y_lower), (x3+w, y3),(x3,y3)])
        points = polygon.centroid
        c1_x = points.x
        c1_y = points.y

    elif y_lower>=y2 and y_lower<=y1:

        polygon = Polygon([(x2, y_lower), (x1, y_lower), (x1, y2),(x3+w,y2),(x3+w,y3),(x3,y3),(x3,y2),(x2,y2)])
        points = polygon.centroid
        c1_x = points.x
        c1_y = points.y

    else:
        polygon = Polygon([(x3,y3+h_move),(x3,y_max),(x_min,y_max),(x_min,y_min),(x3,y_min),(x3,y3),(x3+w_move,y3),(x3+w_move,y_min),(x_max,y_min),(x_max,y_max),(x3+w_move,y_max),(x3+w_move,y3+h_move)])
        points = polygon.centroid
        c1_x = points.x
        c1_y = points.y

    if x1==-9000:
        
        polygon = Polygon([(x3, y_middle), (x3+w, y_middle), (x3+w, y_lower),(x3,y_lower)])
        points = polygon.centroid
        c2_x = points.x
        c2_y = points.y

    elif y_middle<=y2 and y_middle<=y1:
        
        polygon = Polygon([(x3, y_middle), (x3+w, y_middle), (x3+w,y_lower),(x3,y_lower)])
        points = polygon.centroid
        c2_x = points.x
        c2_y = points.y

    elif y_middle>=y2 and y_middle<=y1:

        polygon = Polygon([(x2, y_middle), (x1, y_middle), (x1, y2),(x3+w,y2),(x3+w,y_lower),(x3,y_lower),(x3,y2),(x2,y2)])
        points = polygon.centroid
        c2_x = points.x
        c2_y = points.y

    else:

        polygon = Polygon([(x3,y_middle),(x3,y_max),(x_min,y_max),(x_min,y_min),(x3,y_min),(x3,y_lower),(x3+w_move,y_lower),(x3+w_move,y_min),(x_max,y_min),(x_max,y_max),(x3+w_move,y_max),(x3+w_move,y_middle)])
        points = polygon.centroid
        c2_x = points.x
        c2_y = points.y


    if x1==-9000:
        
        polygon = Polygon([(x3, y3+h), (x3+w, y3+h), (x3+w, y_middle),(x3,y_middle)])
        points = polygon.centroid
        c3_x = points.x
        c3_y = points.y

    elif y3+h<=y2 and y3+h<=y1:
        
        polygon = Polygon([(x3, y3+h), (x3+w, y3+h), (x3+w,y_middle),(x3,y_middle)])
        points = polygon.centroid
        c3_x = points.x
        c3_y = points.y

    elif y3+h>=y2 and y3+h<=y1:

        polygon = Polygon([(x2, y3+h), (x1, y3+h), (x1, y2),(x3+w,y2),(x3+w,y_middle),(x3,y_middle),(x3,y2),(x2,y2)])
        points = polygon.centroid
        c3_x = points.x
        c3_y = points.y

    else:

        polygon = Polygon([(x3,y3+h),(x3,y_max),(x_min,y_max),(x_min,y_min),(x3,y_min),(x3,y_middle),(x3+w_move,y_middle),(x3+w_move,y_min),(x_max,y_min),(x_max,y_max),(x3+w_move,y_max),(x3+w_move,y3+h)])
        points = polygon.centroid
        c3_x = points.x
        c3_y = points.y

    return (c1_x,c1_y,c2_x,c2_y,c3_x,c3_y)