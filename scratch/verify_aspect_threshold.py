"""Verify: stem_area >= 265 AND stem_aspect >= 1.5 threshold across ALL data."""
import cv2
import numpy as np
import os

def get_stem_data(path):
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
    combined = cv2.bitwise_or(violet_mask, red_mask)
    combined = cv2.bitwise_or(combined, maroon_mask)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        aspect = h / max(w, 1)
        return area, h, w, aspect
    return 0, 0, 0, 0

base = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"

for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(base, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        triggered = []
        total = 0
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            result = get_stem_data(fpath)
            if result is None:
                continue
            total += 1
            area, h, w, aspect = result
            # New condition: area >= 265 AND aspect >= 1.5
            if area >= 265 and aspect >= 1.5:
                triggered.append((fname, area, h, w, aspect))
        
        if triggered:
            print(f"\n{split}/{cls}: {len(triggered)}/{total} would trigger override")
            for fname, area, h, w, aspect in triggered:
                print(f"  {fname}: area={area:.0f}, h={h}, w={w}, aspect={aspect:.2f}")
        else:
            print(f"{split}/{cls}: 0/{total} would trigger override OK")

# Check the user's male plant
male_img = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772\media__1783867042180.jpg"
if os.path.exists(male_img):
    area, h, w, aspect = get_stem_data(male_img)
    triggered = area >= 265 and aspect >= 1.5
    print(f"\nUser's male plant: area={area:.0f}, h={h}, w={w}, aspect={aspect:.2f}, triggers={triggered}")
