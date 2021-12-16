# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 01:37:26 2021

@author: vinay
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

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

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=1,
# 	help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# 	help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())

imagePaths = list(paths.list_images("E:\Stuff\Projects\Dataset"))

#print(imagePaths)
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	
	image = cv2.imread(imagePath)
	label = imagePath[-5]
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

#print(labels)
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.1, random_state=42)

model = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
model.fit(trainFeat, trainLabels)

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.9, random_state=42)
out=[]

print(testLabels)
print(model.predict(testFeat))
print("Accuracy = "+str(model.score(testFeat,testLabels)))



