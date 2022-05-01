# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 01:37:24 2022

@author: vinay
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import imutils


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

#CHARACTER SEGMENTATION STEP
def letter_extractor(image):
#tesseract engine inititation
    pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    img=cv2.imread(image)
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im=Image.fromarray(converted)
    letters=[]
    h, w, c = img.shape
    
    #character segmentation
    sp=0
    i=0
    st=pytesseract.image_to_string(img)
    boxes = pytesseract.image_to_boxes(img,config='--psm 3') 
    data=pytesseract.image_to_data(img,config='--psm 3')
    for b in boxes.splitlines():
        b = b.split(' ')
        t=im.crop((int(b[1]),h-int(b[4]),int(b[3]),h-int(b[2])))
        s="E:\\Stuff\\Projects\\TargetFolder\\target_"+str(i)+".png"
        i=i+1
        #t.show()
        t.save(s)
        
    return st

def image_to_feature_vector(image, size=(32, 32)):
    
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()

#k-NN CLASSIFICATION FOR CHARACTERS
imagePaths = list(paths.list_images("E:\Stuff\Projects\Dataset"))

#print(imagePaths)
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    try:
        image=cv2.imread(imagePath)
        label = imagePath[-5]
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
        rawImages.append(pixels)
        features.append(hist)
        labels.append(label)
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))
    except Exception as e:
        print(str(e))

#print(labels)
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.1, random_state=42)

model = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
model.fit(trainRI, trainRL)


#ASSEMBLING DETECTED CHARACTERS TO FORM TEXT
targetImage="target.png"
tst=letter_extractor(targetImage)
print(tst)
targetPath=list(paths.list_images("E:\Stuff\Projects\TargetFolder"))
rawTImage=[]
Tfeatures=[]
for (i, targetPathPath) in enumerate(targetPath):
    image = cv2.imread(targetPathPath)
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)
    rawTImage.append(pixels)
    Tfeatures.append(hist)
Tfeatures=np.array(Tfeatures)
rawTImage=np.array(rawTImage)
res=model.predict(rawTImage)
for i in range(len(res)):
    if tst[i]==' ':
        print(" ")
    else:
        print(res[i])