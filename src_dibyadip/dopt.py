import numpy as np
import cv2
from sklearn.cluster import KMeans
from check_hist import mag_check, dir_check


body_cascade = cv2.CascadeClassifier('cascadG.xml')

#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = np.ones((5,5),np.uint8)

def dense_flow(fm):
    count = 0
    x = y = w = h = 0
    magnitude_histogram = []
    direction_histogram = []
    
    magnitude_histogram1 = []
    direction_histogram1 = []

    magnitude_histogram2 = []
    direction_histogram2 = []

    magnitude_histogram3 = []
    direction_histogram3 = []

    magnitude_histogram4 = []
    direction_histogram4 = []  

    magnitude_histogram5 = []
    direction_histogram5 = []

    magnitude_histogram6 = []
    direction_histogram6 = []

    magnitude_histogram7 = []
    direction_histogram7 = []

    magnitude_histogram8 = []
    direction_histogram8 = []

    magnitude_histogram9 = []
    direction_histogram9 = []


    cap = cv2.VideoCapture(fm)

    # Take the first frame and convert it to gray
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create the HSV color image
    hsvImg = np.zeros_like(frame)
    hsvImg[..., 1] = 255

    # Play until the user decides to stop
    while True:
        # Save the previous frame data
        previousGray = gray
        # Get the next frame
        ret , frame = cap.read()
    
        if ret:
            fgmask = fgbg.apply(frame)#background-subtraction
            seg_mask = cv2.medianBlur(fgmask, 5)#median-blur
            seg_mask = cv2.dilate(seg_mask, kernel, iterations = 1)#dilation
            #cv2.imshow('filtered mask', seg_mask)

            #drawing countours and bounding-box
            #for drawing a rectangle over the entire body
            body = body_cascade.detectMultiScale(gray, 1.05, 3)
            if(len(body)!=0):
                for (x_t,y_t,w_t,h_t) in body:  
                    x, y, w, h = x_t, y_t, w_t, h_t
            #print(body)
            #Convert the frame to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #exception-handling
            if((x, y, w, h) == (0 ,0, 0, 0)):
                continue

            # Calculate the dense optical flow
            flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
            # Obtain the flow magnitude and direction angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = cv2.bitwise_and(mag, mag, mask = seg_mask)
            ang = cv2.bitwise_and(ang, ang, mask = seg_mask)

            ang=((ang*180)/(np.pi/2))%180#scaling
            #magnitude.append(mag)
            #direction.append(ang)
            k=1
            if(w%3==0):
                k=0
            c_x1 = x+(w//3)+k
            c_x2 = (x+2*(w//3))+k
            
            k=1
            if(h%3==0):
                k=0
            c_y1 = y+(h//3)+k
            c_y2 = (y+2*(h//3))+k            

            flag1=flag2=flag3=flag4=0
            if(x-5>=0):
                x-=5
                flag1=1
                if(x+w+10<ang.shape[1]):
                    w+=10
                    flag2=1                    
            if(y-5>=0):
                y-=5
                flag3=1
                if(y+h+10<ang.shape[0]):
                    h+=10
                    flag4=1

            #print(x, w, x+w, c_x1, c_x2)
            #print(y, h, y+h, c_y1, c_y2)

            roi_mag1 = mag[y:c_y1, x:c_x1]
            roi_mag2 = mag[y:c_y1, c_x1:c_x2]
            roi_mag3 = mag[y:c_y1, c_x2:x+w]
            roi_mag4 = mag[c_y1:c_y2, x:c_x1]
            roi_mag5 = mag[c_y1:c_y2, c_x1:c_x2]
            roi_mag6 = mag[c_y1:c_y2, c_x2:x+w]
            roi_mag7 = mag[c_y2:y+h, x:c_x1]
            roi_mag8 = mag[c_y2:y+h, c_x1:c_x2]
            roi_mag9 = mag[c_y2:y+h, c_x2:x+w]

            roi_dir1 = ang[y:c_y1, x:c_x1]
            roi_dir2 = ang[y:c_y1, c_x1:c_x2]
            roi_dir3 = ang[y:c_y1, c_x2:x+w]
            roi_dir4 = ang[c_y1:c_y2, x:c_x1]
            roi_dir5 = ang[c_y1:c_y2, c_x1:c_x2]
            roi_dir6 = ang[c_y1:c_y2, c_x2:x+w]
            roi_dir7 = ang[c_y2:y+h, x:c_x1]
            roi_dir8 = ang[c_y2:y+h, c_x1:c_x2]
            roi_dir9 = ang[c_y2:y+h, c_x2:x+w]

            #print(roi_mag1.shape, roi_mag2.shape, roi_mag3.shape, roi_mag4.shape)
            #print(roi_dir1.shape, roi_dir2.shape, roi_dir3.shape, roi_dir4.shape)
            
            magnitude = np.array(mag).flatten()
            direction = np.array(ang).flatten()

            magnitude1 = np.array(roi_mag1).flatten()
            direction1 = np.array(roi_dir1).flatten()

            magnitude2 = np.array(roi_mag2).flatten()
            direction2 = np.array(roi_dir2).flatten()

            magnitude3 = np.array(roi_mag3).flatten()
            direction3 = np.array(roi_dir3).flatten()

            magnitude4 = np.array(roi_mag4).flatten()
            direction4 = np.array(roi_dir4).flatten()

            magnitude5 = np.array(roi_mag5).flatten()
            direction5 = np.array(roi_dir5).flatten()

            magnitude6 = np.array(roi_mag6).flatten()
            direction6 = np.array(roi_dir6).flatten()

            magnitude7 = np.array(roi_mag7).flatten()
            direction7 = np.array(roi_dir7).flatten()

            magnitude8 = np.array(roi_mag8).flatten()
            direction8 = np.array(roi_dir8).flatten()

            magnitude9 = np.array(roi_mag9).flatten()
            direction9 = np.array(roi_dir9).flatten()

            #histogram for full image
            #---------------------------------------------------------#
            mx = max(magnitude)
            #mn = min(magnitude)
            if(mx!=0):
                magnitude = magnitude/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude, bins)
                #hist = hist/len(direction)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram.append(hist)            

            #direction-histogram
            mx = max(direction)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction, bins)
                #hist = hist/len(direction)
                #print(bin_edges)
                #print(hist)
                direction_histogram.append(hist)

            #---------------------------------------------------------#
            #magnitude-histogram for upper-left 
            mx = max(magnitude1)
            #mn = min(magnitude1)
            if(mx!=0):
                magnitude1 = magnitude1/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude1, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram1.append(hist)            

            #direction-histogram for upper-left 
            mx = max(direction1)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction1, bins)                
                #print(bin_edges)
                #print(hist)
                direction_histogram1.append(hist)

            #---------------------------------------------------------#
            #magnitude-histogram for upper-middle 
            mx = max(magnitude2)
            #mn = min(magnitude2)
            if(mx!=0):
                magnitude2 = magnitude2/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude2, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram2.append(hist)            

            #direction-histogram for upper-middle 
            mx = max(direction2)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction2, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram2.append(hist)

            #---------------------------------------------------------#
            #magnitude-histogram for upper-right 
            mx = max(magnitude3)
            #mn = min(magnitude3)
            if(mx!=0):
                magnitude3 = magnitude3/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude3, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram3.append(hist)            

            #direction-histogram for upper-right
            mx = max(direction3)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction3, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram3.append(hist)

            #---------------------------------------------------------#
            #magnitude-histogram for middle-left
            mx = max(magnitude4)
            #mn = min(magnitude4)
            if(mx!=0):
                magnitude4 = magnitude4/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude4, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram4.append(hist)            

            #direction-histogram for middle-left 
            mx = max(direction4)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction4, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram4.append(hist)

            #---------------------------------------------------------#
            #magnitude-histogram for middle-middle
            mx = max(magnitude5)
            #mn = min(magnitude5)
            if(mx!=0):
                magni5ude4 = magnitude4/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude5, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram5.append(hist)            

            #direction-histogram for middle-middle 
            mx = max(direction5)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction5, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram5.append(hist)
                
            #---------------------------------------------------------#
            #magnitude-histogram for middle-right
            mx = max(magnitude6)
            #mn = min(magnitude6)
            if(mx!=0):
                magni6ude4 = magnitude4/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude6, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram6.append(hist)            

            #direction-histogram for middle-right 
            mx = max(direction6)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction6, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram6.append(hist)
                
            #---------------------------------------------------------#
            #magnitude-histogram for lower-left
            mx = max(magnitude7)
            #mn = min(magnitude7)
            if(mx!=0):
                magnitude7 = magnitude7/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude7, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram7.append(hist)            

            #direction-histogram for lower-left 
            mx = max(direction7)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction7, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram7.append(hist)
                
            #---------------------------------------------------------#
            #magnitude-histogram for lower-middle
            mx = max(magnitude8)
            #mn = min(magnitude8)
            if(mx!=0):
                magnitude8 = magnitude8/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude8, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram8.append(hist)            

            #direction-histogram for lower-middle 
            mx = max(direction8)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction8, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram8.append(hist)
                
            #---------------------------------------------------------#
            #magnitude-histogram for lower-right
            mx = max(magnitude9)
            #mn = min(magnitude9)
            if(mx!=0):
                magnitude9 = magnitude9/mx
                bins = np.arange(0.0,1.1,0.1)
                #print(bins)
                hist, bin_edges = np.histogram(magnitude9, bins)
                #print(bin_edges)
                #print(hist)
                magnitude_histogram9.append(hist)            

            #direction-histogram for lower-right 
            mx = max(direction9)
            if(mx!=0):
                bins = range(0,181,10)
                hist, bin_edges = np.histogram(direction9, bins)
                #print(bin_edges)
                #print(hist)
                direction_histogram9.append(hist)
            #---------------------------------------------------------#

            #Update the color image
            hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
            hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
            
            #drawing the bounding box
            cv2.rectangle(rgbImg, (x,y), (c_x1,c_y1), (255,0,0), 2)
            cv2.rectangle(rgbImg, (c_x1,y), (c_x2,c_y1), (0,255,0), 2)
            cv2.rectangle(rgbImg, (c_x2,y), (x+w,c_y1), (0,0,255), 2)
            cv2.rectangle(rgbImg, (x,c_y1), (c_x1,c_y2), (255,255,0), 2)
            cv2.rectangle(rgbImg, (c_x1,c_y1), (c_x2,c_y2), (0,255,255), 2)
            cv2.rectangle(rgbImg, (c_x2,c_y1), (x+w,c_y2), (255,0,255), 2)
            cv2.rectangle(rgbImg, (x,c_y2), (c_x1,y+h), (128,255,0), 2)
            cv2.rectangle(rgbImg, (c_x1,c_y2), (c_x2,y+h), (255,128,0), 2)
            cv2.rectangle(rgbImg, (c_x2,c_y2), (x+w,y+h), (0,255,128), 2)
            
            #Display the resulting frame
            cv2.imshow('dense optical flow', np.hstack((frame, rgbImg)))

            if(flag1==1):
                x+=5
                if(flag2==1):
                    w-=10
            if(flag3==1):
                y+=5
                if(flag4==1):
                    h-=10

            k = cv2.waitKey(30) & 0xff        
            if k == 27:
                break
        
        else:
            break


    magnitude_histogram = np.array(magnitude_histogram)
    direction_histogram = np.array(direction_histogram)


    magnitude_histogram1 = mag_check(magnitude_histogram1)
    magnitude_histogram2 = mag_check(magnitude_histogram2)
    magnitude_histogram3 = mag_check(magnitude_histogram3)
    magnitude_histogram4 = mag_check(magnitude_histogram4)
    magnitude_histogram5 = mag_check(magnitude_histogram5)
    magnitude_histogram6 = mag_check(magnitude_histogram6)
    magnitude_histogram7 = mag_check(magnitude_histogram7)
    magnitude_histogram8 = mag_check(magnitude_histogram8)
    magnitude_histogram9 = mag_check(magnitude_histogram9)
    
    '''print(magnitude_histogram1.shape)
    print(magnitude_histogram2.shape)
    print(magnitude_histogram3.shape)
    print(magnitude_histogram4.shape)
    print(magnitude_histogram5.shape)
    print(magnitude_histogram6.shape)
    print(magnitude_histogram7.shape)
    print(magnitude_histogram8.shape)
    print(magnitude_histogram9.shape)'''

    direction_histogram1 = dir_check(direction_histogram1)
    direction_histogram2 = dir_check(direction_histogram2)
    direction_histogram3 = dir_check(direction_histogram3)
    direction_histogram4 = dir_check(direction_histogram4)    
    direction_histogram5 = dir_check(direction_histogram5)
    direction_histogram6 = dir_check(direction_histogram6)
    direction_histogram7 = dir_check(direction_histogram7)
    direction_histogram8 = dir_check(direction_histogram8)
    direction_histogram9 = dir_check(direction_histogram9)

    #print(direction_histogram1.shape)

    #---------------------------------------------------------#
    mag_avg_hist = np.mean(magnitude_histogram, axis=0)
    dir_avg_hist = np.mean(direction_histogram, axis=0)

    mag_avg_hist1 = np.mean(magnitude_histogram1, axis=0)
    dir_avg_hist1 = np.mean(direction_histogram1, axis=0)

    mag_avg_hist2 = np.mean(magnitude_histogram2, axis=0)
    dir_avg_hist2 = np.mean(direction_histogram2, axis=0)

    mag_avg_hist3 = np.mean(magnitude_histogram3, axis=0)
    dir_avg_hist3 = np.mean(direction_histogram3, axis=0)

    mag_avg_hist4 = np.mean(magnitude_histogram4, axis=0)
    dir_avg_hist4 = np.mean(direction_histogram4, axis=0)

    mag_avg_hist5 = np.mean(magnitude_histogram5, axis=0)
    dir_avg_hist5 = np.mean(direction_histogram5, axis=0)

    mag_avg_hist6 = np.mean(magnitude_histogram6, axis=0)
    dir_avg_hist6 = np.mean(direction_histogram6, axis=0)

    mag_avg_hist7 = np.mean(magnitude_histogram7, axis=0)
    dir_avg_hist7 = np.mean(direction_histogram7, axis=0)

    mag_avg_hist8 = np.mean(magnitude_histogram8, axis=0)
    dir_avg_hist8 = np.mean(direction_histogram8, axis=0)

    mag_avg_hist9 = np.mean(magnitude_histogram9, axis=0)
    dir_avg_hist9 = np.mean(direction_histogram9, axis=0)

    #---------------------------------------------------------#
    mag_std_hist = np.std(magnitude_histogram, axis=0)
    dir_std_hist = np.std(direction_histogram, axis=0)

    mag_std_hist1 = np.std(magnitude_histogram1, axis=0)
    dir_std_hist1 = np.std(direction_histogram1, axis=0)

    mag_std_hist2 = np.std(magnitude_histogram2, axis=0)
    dir_std_hist2 = np.std(direction_histogram2, axis=0)

    mag_std_hist3 = np.std(magnitude_histogram3, axis=0)
    dir_std_hist3 = np.std(direction_histogram3, axis=0)

    mag_std_hist4 = np.std(magnitude_histogram4, axis=0)
    dir_std_hist4 = np.std(direction_histogram4, axis=0)

    mag_std_hist5 = np.std(magnitude_histogram5, axis=0)
    dir_std_hist5 = np.std(direction_histogram5, axis=0)

    mag_std_hist6 = np.std(magnitude_histogram6, axis=0)
    dir_std_hist6 = np.std(direction_histogram6, axis=0)

    mag_std_hist7 = np.std(magnitude_histogram7, axis=0)
    dir_std_hist7 = np.std(direction_histogram7, axis=0)

    mag_std_hist8 = np.std(magnitude_histogram8, axis=0)
    dir_std_hist8 = np.std(direction_histogram8, axis=0)

    mag_std_hist9 = np.std(magnitude_histogram9, axis=0)
    dir_std_hist9 = np.std(direction_histogram9, axis=0)

    #---------------------------------------------------------#

    #print(mag_avg_hist1.shape)
    #print(mag_std_hist1.shape)
    #print(dir_avg_hist1.shape)
    #print(dir_std_hist1.shape)
    
    histogram = mag_avg_hist
    histogram = np.hstack((histogram, mag_std_hist))
    histogram = np.hstack((histogram, dir_avg_hist))
    histogram = np.hstack((histogram, dir_std_hist))
    histogram = np.hstack((histogram, mag_avg_hist1))
    histogram = np.hstack((histogram, mag_std_hist1))
    histogram = np.hstack((histogram, dir_avg_hist1))
    histogram = np.hstack((histogram, dir_std_hist1))
    histogram = np.hstack((histogram, mag_avg_hist2))
    histogram = np.hstack((histogram, mag_std_hist2))
    histogram = np.hstack((histogram, dir_avg_hist2))
    histogram = np.hstack((histogram, dir_std_hist2))
    histogram = np.hstack((histogram, mag_avg_hist3))
    histogram = np.hstack((histogram, mag_std_hist3))
    histogram = np.hstack((histogram, dir_avg_hist3))
    histogram = np.hstack((histogram, dir_std_hist3))
    histogram = np.hstack((histogram, mag_avg_hist4))
    histogram = np.hstack((histogram, mag_std_hist4))
    histogram = np.hstack((histogram, dir_avg_hist4))
    histogram = np.hstack((histogram, dir_std_hist4))
    histogram = np.hstack((histogram, mag_avg_hist5))
    histogram = np.hstack((histogram, mag_std_hist5))
    histogram = np.hstack((histogram, dir_avg_hist5))
    histogram = np.hstack((histogram, dir_std_hist5))
    histogram = np.hstack((histogram, mag_avg_hist6))
    histogram = np.hstack((histogram, mag_std_hist6))
    histogram = np.hstack((histogram, dir_avg_hist6))
    histogram = np.hstack((histogram, dir_std_hist6))
    histogram = np.hstack((histogram, mag_avg_hist7))
    histogram = np.hstack((histogram, mag_std_hist7))
    histogram = np.hstack((histogram, dir_avg_hist7))
    histogram = np.hstack((histogram, dir_std_hist7))
    histogram = np.hstack((histogram, mag_avg_hist8))
    histogram = np.hstack((histogram, mag_std_hist8))
    histogram = np.hstack((histogram, dir_avg_hist8))
    histogram = np.hstack((histogram, dir_std_hist8))
    histogram = np.hstack((histogram, mag_avg_hist9))
    histogram = np.hstack((histogram, mag_std_hist9))
    histogram = np.hstack((histogram, dir_avg_hist9))
    histogram = np.hstack((histogram, dir_std_hist9))
    
    #print(histogram.shape)

    cv2.destroyAllWindows()
    cap.release()
    return histogram
