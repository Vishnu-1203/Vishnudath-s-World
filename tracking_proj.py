import cv2
import numpy as np

def detect_red_color(frame, x, y, w, h):
    # Extract the region of interest (ROI) from the frame using the rectangle coordinates
    roi = frame[y:y+h, x:x+w]
    
    # Convert the ROI to the HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define the range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Combine the two masks to get the final mask for red color detection
    mask = mask1 + mask2
    
    # Count the number of non-zero pixels in the mask
    num_red_pixels = cv2.countNonZero(mask)
    
    # Check if there are any red pixels in the ROI
    if num_red_pixels > 0:
        return True
    else:
        return False


# Initialize object detection
from object_detection import ObjectDetection
od = ObjectDetection()

# Initialize the video capture object
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame using the object detection model
    center_points_cur_frame = []
    (class_ids, scores, boxes) = od.detect(frame)

    # Loop through the detected boxes and draw rectangles on the frame
    for box in boxes:
        (x, y, w, h) = box
        
        # Check if there is any red color present in the rectangle
        is_red_color_present = detect_red_color(frame, x, y, w, h)
        
        # If red color is present, draw a red border around the rectangle
        if is_red_color_present:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            red_object=True
            print("red object found")
    # Show the frame on the screen
    cv2.imshow("Frame", frame)
    
    # Exit the loop if the user presses the 'Esc' key
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
