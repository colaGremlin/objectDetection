import cv2
import numpy as np

max_area = 5000
counter = 0
pentIndex = 0

# Read the image
image = cv2.imread('./input/image_4.jpeg')
blurred = cv2.GaussianBlur(image, (5, 5), 0)
image = cv2.resize(blurred, (0,0), fx=0.5, fy=0.5) 
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()

# Detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

    if (len(approx) == 5):
        area = cv2.contourArea(cnt)
        if (max_area <= area):
            max_area = area
            pentIndex = counter

    counter += 1

# Draw the contour of the pentagon
image_copy = image.copy()
cv2.drawContours(image_copy, contours, pentIndex, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

# Get the bounding box of the pentagon
x, y, w, h = cv2.boundingRect(contours[pentIndex])

# Draw the bounding box around the pentagon
cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Calculate the center of the bounding box
center_x = x + w // 2
center_y = y + h // 2

# Draw the center of the bounding box
cv2.circle(image_copy, (center_x, center_y), 2, (0, 255, 0), -1)  # Draw a small circle at the center
cv2.putText(image_copy, f"Center: ({center_x}, {center_y})", (center_x - 50, center_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the result
cv2.imshow('Pentagon with Bounding Box and Center', image_copy)
cv2.waitKey(0)
cv2.imwrite('pentagon_with_bbox_and_center.jpg', image_copy)
cv2.destroyAllWindows()

# Print the center coordinates for precision landing
print(f"Center coordinates of the bounding box: ({center_x}, {center_y})")