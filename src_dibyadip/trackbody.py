from __future__ import division
from decimal import Decimal
from skimage import exposure
from skimage import feature
from shapely.geometry import Point
from shapely.geometry import Polygon
from PIL import Image
from finding_features import mag,angle
from processing import find_parts
import numpy as np 
import cv2
import imutils
import scipy
import math
import os


def HOG(fm):
    cap = cv2.VideoCapture(fm)
    hist = []

    while True:
        ret,frame1 = cap.read()

        if frame1 is None:
            break

        image = frame1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #w, h = gray.shape
        #image = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_CUBIC)
        #print(w, h)
        #winSize = (w, h)
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)   
        winStride = (8,8)
        padding = (8,8)
        locations = ((10,20),)
        hist.append(hog.compute(image,winStride,padding,locations))
        #print(hist)
        #print(hist.shape)  
        if cv2.waitKey(30)==27 & 0xff:
            break

    hog_final = hist[0]
    for i in range(1, len(hist)):
        np.add(hog_final, hist[i])

    #for i in hog_final:
    #    print(i)
    cv2.destroyAllWindows()
    cap.release()
    return hog_final.flatten()




def trackbody(fm):
    body_cascade = cv2.CascadeClassifier('cascadG.xml')

    #reading the video file
    cap = cv2.VideoCapture(fm)

    ret,frame1 = cap.read()
    ret,frame2 = cap.read()
    frame3 = frame1

    count = 0
    magnitude = []
    direction = []
    while True:
        frame3 = frame2

        
        d = cv2.absdiff(frame1,frame2)
        #cv2.imshow('d', d)

        grey = cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('grey', grey)
        
        #used for the full body tracking
        grey1 = cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grey,(5,5),0)
        #cv2.imshow('blur', blur)

        ret,th = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
        #cv2.imshow('th', th)


        dilated = cv2.dilate(th,np.ones((3,3),np.uint8),iterations = 2)
        #cv2.imshow('dilated', dilated)

        eroded = cv2.erode(dilated,np.ones((3,3),np.uint8),iterations = 2)
        #cv2.imshow('eroded', eroded)

        _,c,h = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # for finding the max moving contour
        test = 9000
        test1 = -9000
        x_min = test
        x_max = test1
        y_min = test
        y_max = test1
        for cnts in c:
            (x,y,w,h) = cv2.boundingRect(cnts)
            if x<=x_min:
                x_min = x
            if x+w>=x_max:
                x_max = x+w
            if y<=y_min:
                y_min = y
            if y+h>y_max:
                y_max = y+h

        # for drawing a rectangle over the moving part
        if x_min!=9000 and x_max!=-9000:
            cv2.rectangle(frame1,(x_min,y_max),(x_max,y_min),(0,0,255),2)

        #for drawing a rectangle over the entire body
        body = body_cascade.detectMultiScale(grey1,1.05,3,)

        x_move = 0
        y_move = 0
        h_move = 0
        w_move = 0

        for (x,y,w,h) in body:
            x_move,y_move,h_move,w_move = x,y,h,w
            vrx = np.array(([x_move,y_move+h_move],[x_move,y_max],[x_min,y_max],[x_min,y_min],[x_move,y_min],[x_move,y_move],[x_move+w_move,y_move],[x_move+w_move,y_min],[x_max,y_min],[x_max,y_max],[x_move+w_move,y_max],[x_move+w_move,y_move+h_move]), np.int32)
            vrx = vrx.reshape((-1,1,2))
           # cv2.polylines(frame1, [vrx], True, (0,255,255),2)
            cv2.rectangle(frame1,(x_move,y_move),(x_move+w_move,y_move+h_move),(255,0,0),2)

        if x_move!=0:
            
            c1_x,c1_y,c2_x,c2_y,c3_x,c3_y = find_parts(x_max,y_max,x_min,y_min,x_move,y_move,w_move,h_move)

        else:

            c1_x,c1_y,c2_x,c2_y,c3_x,c3_y = find_parts(9001,9001,9000,9000,x_move,y_move,w_move,h_move)

        if count==0:
            
            prev1_x,prev1_y,prev2_x,prev2_y,prev3_x,prev3_y = c1_x,c1_y,c2_x,c2_y,c3_x,c3_y

    
        else:
            mg1 = mag(prev1_x,prev1_y,c1_x,c1_y)
            mg2 = mag(prev2_x,prev2_y,c2_x,c2_y)
            mg3 = mag(prev3_x,prev3_y,c3_x,c3_y)

            dg1 = angle(prev1_x,prev1_y,c1_x,c1_y)
            dg2 = angle(prev2_x,prev2_y,c2_x,c2_y)
            dg3 = angle(prev3_x,prev3_y,c3_x,c3_y)

            magnitude.append(mg1)
            magnitude.append(mg2)
            magnitude.append(mg3)
            direction.append(dg1)
            direction.append(dg2)
            direction.append(dg3)

            prev1_x,prev1_y,prev2_x,prev2_y,prev3_x,prev3_y = c1_x,c1_y,c2_x,c2_y,c3_x,c3_y

        
        #cv2.imshow("original",frame2)
        #cv2.imshow("frame with moving and full",frame1)
        
        
        if cv2.waitKey(30)==27 & 0xff:
            break
        frame1 = frame2
        ret,frame2 = cap.read()
        
        if frame2 is None:
            break
        count = count + 1

    deg = range(0,181,10)
    bins = deg
    hist,bin_edges = np.histogram(direction,bins)
    hist = hist/len(direction)
    max_mag = max(magnitude)
    min_mag = min(magnitude)

    diff = max_mag - min_mag

    #magnitude = (magnitude-np.mean(magnitude))/np.std(magnitude)
    #magnitude = ((x-min_mag)/diff for x in magnitude)

    magnit = np.arange(0.0,1.1,0.1)

    hist1,bin_edges1 = np.histogram(magnitude,magnit)

    hist1 = hist1/len(magnitude)

    #print(hist)

    #print(hist1)

    histogram = np.concatenate((np.array(hist),np.array(hist1)),axis=0)
    histogram = histogram/len(magnitude)

    #print(histogram)

    #print(len(histogram))

    cv2.destroyAllWindows()
    cap.release()
    return histogram



def HOG_temporal(fm):
    cap = cv2.VideoCapture(fm)
    hist = []

    ret,frame1 = cap.read()
    ret,frame2 = cap.read()
    while True:
        ret,frame3 = cap.read()

        if frame3 is None:
            break

        hist_temp = []
        image1 = frame1
        image2 = frame2
        image3 = frame3
        '''gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)'''

        #w, h = gray1.shape
        #image = cv2.resize(gray1, (64, 64), interpolation=cv2.INTER_CUBIC)
        #print(w, h)
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)   
        winStride = (8,8)
        padding = (8,8)
        locations = ((10,20),)
        hist_temp.append(hog.compute(image1,winStride,padding,locations))

        #w, h = gray2.shape
        #image = cv2.resize(gray2, (64, 64), interpolation=cv2.INTER_CUBIC)
        #print(w, h)        
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)   
        winStride = (8,8)
        padding = (8,8)
        locations = ((10,20),)        
        hist_temp.append(hog.compute(image2,winStride,padding,locations))

        #w, h = gray3.shape
        #image = cv2.resize(gray3, (64, 64), interpolation=cv2.INTER_CUBIC)
        #print(w, h)
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels)   
        winStride = (8,8)
        padding = (8,8)
        locations = ((10,20),)
        hist_temp.append(hog.compute(image3,winStride,padding,locations))
        #print((np.array(hist_temp)).shape)
        hist.append(np.array(hist_temp))
        #print((np.array(hist)).shape)
        #print(hist_temp[0])


        frame1 = frame2
        frame2 = frame3
        if cv2.waitKey(30)==27 & 0xff:
            break

    #print(len(hist))
    hog_final = hist[0]
    #print(hog_final.shape)
    for i in range(1, len(hist)):
        np.add(hog_final, hist[i])

    #for i in hog_final:
    #    print(i)
    cv2.destroyAllWindows()
    cap.release()
    return hog_final.flatten()