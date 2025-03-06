import cv2
import numpy as np

def calculate_edge_lengths(approx):
    edge_lengths = []
    for i in range(len(approx)):
        x1, y1 = approx[i][0]
        x2, y2 = approx[(i + 1) % len(approx)][0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        edge_lengths.append(length)
    return edge_lengths

max_area = 0
counter = 0
pIndex = -1

# Read the image
image = cv2.imread('./input/image_3.jpeg')
blurred = cv2.GaussianBlur(image, (3, 3), 0)
image = cv2.resize(blurred, (0,0), fx=0.5, fy=0.5) 
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()

# Detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = list(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True))
    
    if len(approx) == 5:
        edge_lengths = calculate_edge_lengths(approx)
        avg_length = np.mean(edge_lengths)
        std_dev = np.std(edge_lengths)
        
        # Check if the standard deviation is within a threshold (e.g., 10% of the average length)
        if std_dev < 0.05 * avg_length:
            area = cv2.contourArea(cnt)
            if max_area <= area:
                max_area = area
                pIndex = counter

    counter += 1

image_copy = image.copy()
if pIndex != -1:
    image_copy = cv2.drawContours(image_copy, contours, pIndex, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

# See the results
cv2.imshow('Simple approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('contours_simple_image1.jpg', image_copy)
cv2.destroyAllWindows()