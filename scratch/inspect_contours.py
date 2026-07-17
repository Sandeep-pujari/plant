"""
Inspect all contours found in the color mask for IMG_0334.JPEG.
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
    print(f"Total contours found: {len(contours)}")
    
    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, c in enumerate(contours[:5]):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / max(w, 1)
        print(f"Contour {i}: area={area:.2f}, x={x}, y={y}, w={w}, h={h}, aspect={aspect:.2f}")
