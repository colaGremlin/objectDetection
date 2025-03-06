import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Reset variables for each frame
        max_area = 5000
        counter = 0
        pentIndex = -1  # Initialize to -1 (invalid index)

        # Capture frame-by-frame
        ret, image = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Flip the frame horizontally to correct mirroring
        image = cv2.flip(image, 1)  # 1 means horizontal flip, 0 means vertical flip, -1 means both

        # Resize the frame (optional)
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        # Convert to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Threshold the image
        ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest pentagon
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            if len(approx) == 5:  # Check if the contour is a pentagon
                area = cv2.contourArea(cnt)
                if max_area <= area:
                    max_area = area
                    pentIndex = counter  # Update pentIndex to the current contour index

            counter += 1

        # Draw the contour of the pentagon (if detected)
        image_copy = image.copy()
        if pentIndex != -1:  # Check if a pentagon was detected
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

            # Print the center coordinates for precision landing
            print(f"Center coordinates of the bounding box: ({center_x}, {center_y})")

        # Display the result
        cv2.imshow('Pentagon with Bounding Box and Center', image_copy)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
