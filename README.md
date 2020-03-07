# opencvfinding-lanes

Link to youtube video
https://youtu.be/sFSAZh_v9F8

This repository is for sharing the python anaconda jypiter notebook for the code which finds the lanes.

Below are the steps

1. Download the test4_small.mp4 file to your download directory
2. download the jupyter notebookmto your download directory
3. make sure you have below setup
- anaconda installation
- jupyter notebook installation
- numpy version 1.18
- opencv version 4.2.0
- matplotlib version 3.1.3
4. open the jupyter code in notebook
5. update the 1st line of the below code snippet with your download directory
cd C:\Users\santo\Downloads --> change to your download directory
5. run the code


Below is the code base

************************************************************************************************************

##With DashCam video
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 280,20)
    return canny

def region_of_interest(image):
    height = 550 
    polygons = np.array([[(350, height), (1000, height), (630, 375)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

def  average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    try:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2),1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
                right_fit.append((0.82578829, -149.48243243)) #147.48243243
            else:
                right_fit.append((slope, intercept))
                left_fit.append((-7.04968944e-01, 8.31574534e+02))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average) 
        right_line = make_coordinates(image, right_fit_average)

    finally:
        pass
    return np.array([left_line, right_line])
        
def make_coordinates (image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 =image.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
    finally:
         pass
    return np.array([x1, y1, x2, y2])
    
cap = cv2.VideoCapture('test4_small.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    try:
        lines = cv2.HoughLinesP(cropped_image,3,np.pi/180, 80, np.array([]), minLineLength=10, maxLineGap=3)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame,.75, line_image, 1, 1)
    finally:
        pass

    cv2.namedWindow("preview - Final Line Detection")
    cv2.imshow("preview - Final Line Detection", combo_image)
    if cv2.waitKey(20) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

************************************************************************************************************


