import cv2 as cv
import numpy as np 

road = cv.imread("Photos/road1.jpg")

# takes canny edges of the road 
def Canny(road):
    # Gaussian blur for the noise
    blur_road = cv.GaussianBlur(road, (3,3), cv.BORDER_DEFAULT) 
    canny_road = cv.Canny(blur_road, 125, 175)

    return canny_road

