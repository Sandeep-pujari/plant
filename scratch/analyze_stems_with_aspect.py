"""
Analyze stem contour areas with aspect ratio >= 1.5 constraint.
"""
import cv2
import numpy as np
import os

dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"

def get_vertical_stem_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
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
    
    vertical_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:  # ignore tiny noise
            continue
        _, _, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(cw, 1)
        if aspect >= 1.5:
            vertical_contours.append((area, aspect))
            
    if vertical_contours:
        # Return the one with the largest area
        largest = max(vertical_contours, key=lambda x: x[0])
        return largest
    return (0.0, 0.0)

# Analyze male images in train
print("=== MALE TRAIN ===")
male_vals = []
for fname in os.listdir(os.path.join(dataset, "train", "male")):
    fpath = os.path.join(dataset, "train", "male", fname)
    r = get_vertical_stem_features(fpath)
    if r:
        area, aspect = r
        if area > 0:
            male_vals.append((fname, area, aspect))

male_vals = sorted(male_vals, key=lambda x: x[1], reverse=True)
for fname, area, aspect in male_vals[:10]:
    print(f"  {fname}: area={area:.2f}, aspect={aspect:.2f}")
print(f"Male min area: {min(x[1] for x in male_vals):.2f}")

# Analyze hybrid images in train
print("\n=== HYBRID TRAIN ===")
hybrid_vals = []
for fname in os.listdir(os.path.join(dataset, "train", "hybrid")):
    fpath = os.path.join(dataset, "train", "hybrid", fname)
    r = get_vertical_stem_features(fpath)
    if r:
        area, aspect = r
        if area > 0:
            hybrid_vals.append((fname, area, aspect))

hybrid_vals = sorted(hybrid_vals, key=lambda x: x[1], reverse=True)
for fname, area, aspect in hybrid_vals[:15]:
    print(f"  {fname}: area={area:.2f}, aspect={aspect:.2f}")
if hybrid_vals:
    print(f"Hybrid max area: {max(x[1] for x in hybrid_vals):.2f}")
