import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
	_ , frame = cap.read()	
	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

	gblur = cv2.GaussianBlur(gray , (5,5) , 0)

	th1  = cv2.adaptiveThreshold(gblur , 255 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 11 , 2)
	#th3 = cv2.adaptiveThreshold(gblur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)	
	
	# Edges
	edges = cv2.Canny(gblur , 20 ,150 , apertureSize=3)
	
	# Hough Line P 
	lines = cv2.HoughLinesP(edges , 1 , np.pi/180 , 100 ,minLineLength=100 , maxLineGap=10)
	
	if lines is not None:
		for line in lines:
			x1 , y1 , x2 , y2 = line[0]
			cv2.line(frame , (x1 , y1) , (x2,y2) , (0,255,0) , 2)
			cv2.line(th1 , (x1 , y1) , (x2,y2) , (0,255,0) , 2)
	else:
		pass

	# Contours
	contours , hierarchy = cv2.findContours(th1, cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)

	cv2.imshow('frame' , frame) 
	cv2.imshow('edges' , edges)
	cv2.imshow('th1' , th1)
	
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
