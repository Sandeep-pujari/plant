"""Check ALL training images for stem_area to verify threshold safety."""
import cv2
import numpy as np
import os

def get_stem_area(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    
    # Combined pigment mask
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
        return cv2.contourArea(largest)
    return 0

dataset_dir = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"

for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset_dir, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        areas = []
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            a = get_stem_area(fpath)
            if a is not None:
                areas.append(a)
        if areas:
            above_250 = [a for a in areas if a >= 250]
            print(f"{split}/{cls}: count={len(areas)}, max_area={max(areas):.0f}, above_250={len(above_250)}/{len(areas)}")
