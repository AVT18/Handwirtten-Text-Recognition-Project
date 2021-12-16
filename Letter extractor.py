# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:33:02 2021

@author: vinay
"""

import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#image pre-processing functions
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

#tesseract engine inititation
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img=cv2.imread("Sample dataset.png")
converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im=Image.fromarray(converted)
letters=[]
h, w, c = img.shape

#character segmentation
boxes = pytesseract.image_to_boxes(img,config='--psm 3') 
data=pytesseract.image_to_data(img,config='--psm 3')
for b in boxes.splitlines():
    b = b.split(' ')
    t=im.crop((int(b[1]),h-int(b[4]),int(b[3]),h-int(b[2])))
    s="test_"+str(b[0])+".png"
    t.show()
    try:
        t.save(s)
    except:
        print("Error")    
    letters.append(t)

