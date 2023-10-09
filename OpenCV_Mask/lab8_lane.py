import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread("C:/Users/avi65/Downloads/road_image_3.jpeg")


def grey(image):
  #convert to grayscale
    image = np.asarray(image)
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


  #Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv.GaussianBlur(image, (5, 5), 0)


  #outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv.Canny(image,50,150)
    return edges


def region(image):
    height, width = image.shape
    #isolate the gradients that correspond to the lane lines
    triangle = np.array([
                       [(100, height), (475, 325), (width, height)]
                       ])
    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    mask = cv.fillPoly(mask, triangle, 255)
    mask = cv.bitwise_and(image, mask)
    return mask


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            cv.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def average(image, lines):
    left = []
    right = []


    if lines is not None:
      for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
            
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])


def make_points(image, average):
    print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


copy = np.copy(img)
edges = cv.Canny(copy,50,150)
isolated = region(edges)
plt.imshow(edges)
plt.show()
plt.imshow(isolated)
plt.show()
cv.waitKey(0)


#DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array, 
lines = cv.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average(copy, lines)
black_lines = display_lines(copy, averaged_lines)
#taking wighted sum of original image and lane lines image
lanes = cv.addWeighted(copy, 0.8, black_lines, 1, 1)
plt.imshow(lanes)
plt.show()
cv.waitKey(0)
