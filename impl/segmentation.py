import numpy as np
import cv2
import time
import copy

def process(video, display = 0):
	cap = cv2.VideoCapture(video)
	fgbg = cv2.createBackgroundSubtractorMOG2(history = 1000, varThreshold = 256, detectShadows = True)
	
	fgbg.setShadowValue(0)
	fgbg.setComplexityReductionThreshold(0.3)
	while(1):
		ret, frame = cap.read()
		
		if(ret == False):
			break;
			
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		img = fgbg.apply(frame)

		img1 = copy.deepcopy(img)
		img1, contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		if(display == True):
			cv2.drawContours(frame, contours, -1, (255,0,0), 3)
			cv2.imshow('frame', frame)
			cv2.imshow('frame1', img)
			cv2.imshow('frame2', img1)
	
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break

	
	cap.release()
	cv2.destroyAllWindows()