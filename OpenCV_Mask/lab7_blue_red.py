import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    lower_red = np.array([-20, 100, 100])
    upper_red = np.array([13, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_red = cv.inRange(hsv, lower_red, upper_red)
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    mask = cv.bitwise_or(mask_red, mask_blue)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if(len(contours)) != 0:
        for contour in contours:
            if cv.contourArea(contour) > 500:
                max_contour = max(contours, key = cv.contourArea)
                x, y, w, h = cv.boundingRect(max_contour)
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:                             # escape key is 27;  space is 32, etc
        break
cv.destroyAllWindows()
