from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


#Locates and extracts the sudoku puzzle board from the input image
#Debug is A optional boolean indicating whether to show intermediate steps so you can better visualize what is happening under the hood of our computer vision pipeline.

def find_puzzle(image , debug=False):
    # Convert the image into grayscale and blur it 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray , (7,7) , 3)
    
    # Apply adaptive thresh and invert the thres map
    thresh = cv2.adaptiveThreshold(blurred , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY , 11 ,2)
    thresh  = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)
    
    # Find contours in  the thresh image and sort them by size in descending order
    cnts = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts , key = cv2.contourArea , reverse =True)

    # Initialize a contour that corresponds to the puccle outline
    puzzlecnt = None

    # Looping over contours
    for i in cnts:
        # Determine the perimeter of the contour
        peri = cv2.arcLength(i , True)
        # Approximating the contour
        approx = cv2.approxPolyDP(i , 0.02 * peri , True)
        # if our approximated contour has four points which means four vertices, then we can assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzlecnt = approx
            break

    	# if the puzzle contour is empty then our script could not find the outline of the sudoku puzzle so raise an error
    if puzzlecnt is None:
        raise Exception(("Could not find sudoku puzzle outline. "
	"Try debugging your thresholding and contour steps."))

    # Check to see if we are visualizing the outline of the detected
    # sudoku puzzle
    if debug:
    # draw the contour of the puzzle on the image and then display it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # apply a four point perspective transform to both the original image and grayscale image to obtain a top-down bird's eye view of the puzzle
    puzzle = four_point_transform(image ,puzzlecnt.reshape(4,2))
    warped = four_point_transform(gray ,puzzlecnt.reshape(4,2))

    # CHecking
    if debug:
        cv2.imshow("Puzzle Transform" , puzzle)
        cv2.waitKey(0)
    return(puzzle,warped)

# Next , we examine each of the individual cells in a sudoku board, detect if there is a digit in the cell, and if so, extract the digit.

def extract_digit(cell , debug=False):
    # Applying automatic thresh to cell
    thresh = cv2.threshold(cell , 0 , 255 , cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # clearing any connected borders that touch the border of cell
    thresh = clear_border(thresh)

    # checking
    if debug:
        cv2.imshow("Cell thresh" , thresh)
        cv2.waitKey(0)
    
    # finding contours in the thresh cell
    cnts = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
    # otherwise,
    # find the largest contour in the cell and create a mask for the contour
    c = max(cnts , key=cv2.contourArea)
    mask = np.zeros(thresh.shape , dtype = "uint8")
    cv2.drawContours(mask , [c] , -1 , 255, -1)

# Now after finding the contour , we'll isolate digit

    # computing the percentage of masked pixels 
    (h , w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w*h)

    # if less than 3% mask is filled than we are looking at noise and and ignore the contour
    if percentFilled < 0.03:
        return None

    # Apply mask to thresh cell
    digit = cv2.bitwise_and(thresh , thresh , mask=mask)

    # check
    if debug:
        cv2.imshow("Digit" , digit)
        cv2.waitKey(0)
    return digit
