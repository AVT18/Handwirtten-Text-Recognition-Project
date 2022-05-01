# Handwirtten-Text-Recognition-Project
1. Purpose
The main purpose of this project is to develop a system to digitize text present in handwritten documents. For this, we will using a k-NN classifier model and train it with datasets of handwritten alphabet images. Classification will be done on a character level, and to break down images into individual characters we have designed a program which implements Tesseract engine to recognise text and perform character segmentation on the image to produce classification-compatible character images.
As computerization progresses in the contemporary scenario, the need for digitization of handwritten documents follows.
	 
			Figure 1: Important details still exist in handwritten form




OCR (Optical Character Recognition) is a technique that identifies, locates and extracts texts, printed or handwritten, from images and converts them into digitally recognisable text. It involves performing a sequence of smaller processes on the input image which are:
•	Image Pre processing
•	Text Detection
•	Character Segmentation
•	Character Classification
•	Text Assembly
We shall implement OCR in our project using the help of Tesseract OCR engine, by using its python-wrapper i.e. pytesseract. 
As for Character Recognition, we shall use our own kNN(k- nearest neighbours) model, trained with self-generated data sets, for classification of the characters segmented using Tesseract.

2. Functional Description:

 
			Figure 2: Flowchart of plan of action
      ![image](https://user-images.githubusercontent.com/64633535/166134405-c12be7be-324e-4d21-b3fe-d5b4ce867d4e.png)


2.1. Pre-processing
	Before subjecting an image to processing directly into the Tesseract Engine, it is pre-processed by performing pixel manipulative operations like anti-noise, grayscaling, shearing , thresholding, contrast stretching, erosion, canny edge detection, skew correction and template matching.
	Varying combinations of these operations will produce images with different feature spaces. In easy words, we need experiment with multiple combinations manually to find the one that yields highest accuracy.

2.2. Text Recognition
	In the tesseract engine, word finding is done by organizing text lines into boxes (rectangles) line by line and feeding it to a newly developed Long Short-Term Memory (LSTM) model. An attempt to recognise each word is made which, if satisfactory, is passed to the adaptive classifier as training data.


![image](https://user-images.githubusercontent.com/64633535/166134409-b4ae6af0-9581-46b0-b2b3-a8a23117d405.png)




2.3. Character Segmentation
	Once a text piece has been recognised and located, the Tesseract engine isolates individual characters of each word. Since it uses a “convert-to-boxes” method to separate text which can also be applied to segregate isolated characters and extract cropped individual images. Now we have a set of character images present the input image.

2.4. Character Recognition
	For recognising the characters present in the images so obtained, we will create a k-NN (k Nearest Neighbours) model to classify the segmented images into alphabets, digits and special characters.
	The model will be trained using authentic and natural data, i.e. samples of handwriting of real people.

2.5. Recombine Text
	The identified characters from the previous process will be stored in an array which will be emptied in FIFO manner to achieve the reassembled text as output. This recombined text maybe displayed as a text output on the console or returned as text file.
	
	Various formatting functionalities may be made available for these outputs as the project progresses. These may include classic text editor features like line spacing, padding, margins etc.
