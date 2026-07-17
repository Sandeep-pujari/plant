"""
Debug stem features of IMG_0334.JPEG.
"""
import cv2
import numpy as np

path = "dataset/train/male/IMG_0334.JPEG"
img = cv2.imread(path)
if img is None:
    print("Cannot read image")
else:
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    
    violet_mask = cv2.inRange(hsv, np.array([120, 50, 40]), np.array([170, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    maroon_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([15, 200, 120]))
    
    combined_mask = cv2.bitwise_or(violet_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, maroon_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        stem_area = cv2.contourArea(largest)
        _, _, cw, ch = cv2.boundingRect(largest)
        stem_aspect = ch / max(cw, 1)
        print(f"IMG_0334.JPEG: stem_area={stem_area:.2f}, stem_aspect={stem_aspect:.2f}, width={cw}, height={ch}")
    else:
        print("No contours found")
