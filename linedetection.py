import cv2 as cv
import cv2
import numpy as np 
#import matplotlib.pyplot as plt 

## takes canny edges of the road 
def take_canny(road):
    # gray image of the road
    gray_road = cv.cvtColor(road, cv.COLOR_BGR2GRAY)
    # Gaussian blur for the noise
    blur_road = cv.GaussianBlur(gray_road, (5,5), cv.COLOR_RGB2GRAY)
    # Canny Edges of the road 
    canny_road = cv.Canny(blur_road, 100, 200)

    return canny_road


## makes coordinates
def coordinates(road, line_parameters):
    # defines slope & intercept as line_parameters
    slope, intercept = line_parameters
    # y1 is all numpy vectors with 0 x
    y1 = road.shape[0]
    # y2
    y2 = int(y1 * (3/5))
    # x1
    x1 = int((y1 - intercept) / slope)
    # x2
    x2 = int((y2 - intercept) / slope)
    # stores x1, y1, x2, y2 in an array 
    return np.array([x1, y1, x2, y2])


## display lines
def display_lines(road, lines):
    line_road = np.zeros_like(road)
    # if the approximated line isn't None, for each x1, y1, x2, y2 of lines, draw line
    sum = 0
    if lines is not None:
        for line in lines:
            # draws a line cv.line(img, first point, second point, color, thickness)
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_road, (x1, y1), (x2, y2), (248, 210, 21), 3)
            sum += (x1 + x2)/2
    adjust = (sum/len(lines) - 600)/600
    #average is the middle of all the lines - used for finding direction

    return [line_road, adjust]


## specifies the Region of Interest in the road
def ROI(road):
    height = road.shape[0]
    # matlab coordinates of road image MUST BE AN INTEGER
    start_x = 200
    end_x = 1100
    midpoint_x = int((start_x + end_x) / 2)
    depth = -170
    # creates the triangle
    polygons = np.array([
    [(start_x, height), (end_x, height), (midpoint_x, depth)]
    ])
    mask = np.zeros_like(road)
    cv.fillPoly(mask, polygons, 255)
    # takes AND of the bits of road and mask image
    masked_road = cv.bitwise_and(road, mask)

    return masked_road



def find_road(image):
    # Road image
    road = cv.imread(image)
    # Returns an array copy of the road
    road_np = np.copy(road)
    # calls take_canny to get canny_road
    canny = take_canny(road)
    # calls ROI to get the cropped_road image
    cropped_road = ROI(canny)
    # Hough Space for slope prediction
    lines = cv.HoughLinesP(cropped_road, rho = 2, theta = np.pi/180, threshold=120, minLineLength = 150, maxLineGap = 80)
    # displays approximated lines
    line_road, adjust = display_lines(road_np, lines)
    # combination of road and the line prediction image
    # cv.addWeighted(1st img, transparency of the 1st img, 2nd img)
    return [cv.addWeighted(road_np, 0.4, line_road, 1, 1), adjust]


##The following lines can be uncommented for testing purposes
##View detected lines and adjust value:
##combine_road, adjust = find_road("Photos/road1.jpg")
##cv.imshow('road', combine_road)
##print(adjust)
##cv.waitKey(0)