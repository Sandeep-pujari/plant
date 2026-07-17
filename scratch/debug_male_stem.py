"""
Debug script to analyze stem features of male plant image and all training images
to determine the right thresholds for hybrid->male override.
"""
import cv2
import numpy as np
import os

def get_stem_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    total_pixels = img_224.shape[0] * img_224.shape[1]
    
    # Violet/purple mask
    violet_mask = cv2.inRange(hsv, np.array([120, 50, 40]), np.array([170, 255, 255]))
    violet_pct = (cv2.countNonZero(violet_mask) / total_pixels) * 100.0
    
    # Red/maroon mask  
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_pct = (cv2.countNonZero(red_mask) / total_pixels) * 100.0
    
    # Maroon mask
    maroon_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([15, 200, 120]))
    maroon_pct = (cv2.countNonZero(maroon_mask) / total_pixels) * 100.0
    
    # Combined pigment mask for contour detection
    combined_mask = cv2.bitwise_or(violet_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, maroon_mask)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stem_area = 0
    stem_h = 0
    stem_w = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        stem_area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        stem_h = h
        stem_w = w
    
    # Green percentage
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = (cv2.countNonZero(green_mask) / total_pixels) * 100.0
    
    # Dark percentage (background)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
    dark_pct = (cv2.countNonZero(dark_mask) / total_pixels) * 100.0
    
    return {
        'violet_pct': violet_pct,
        'red_pct': red_pct,
        'maroon_pct': maroon_pct,
        'stem_area': stem_area,
        'stem_h': stem_h,
        'stem_w': stem_w,
        'green_pct': green_pct,
        'dark_pct': dark_pct
    }

# Analyze the user's male plant image
user_img = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\IMG_0334.JPEG"
if os.path.exists(user_img):
    print("=== USER'S MALE PLANT IMAGE ===")
    feats = get_stem_features(user_img)
    if feats:
        for k, v in feats.items():
            print(f"  {k}: {v:.2f}")
else:
    print(f"Image not found: {user_img}")

# Analyze training data
dataset_dir = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset\train"
for cls in ['female', 'hybrid', 'male']:
    cls_dir = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    print(f"\n=== TRAINING CLASS: {cls.upper()} ===")
    areas = []
    heights = []
    for fname in os.listdir(cls_dir)[:30]:  # Sample 30
        fpath = os.path.join(cls_dir, fname)
        feats = get_stem_features(fpath)
        if feats:
            areas.append(feats['stem_area'])
            heights.append(feats['stem_h'])
    if areas:
        print(f"  stem_area: min={min(areas):.0f}, avg={np.mean(areas):.0f}, max={max(areas):.0f}")
        print(f"  stem_h:    min={min(heights):.0f}, avg={np.mean(heights):.0f}, max={max(heights):.0f}")
        print(f"  All areas: {sorted([int(a) for a in areas])}")
        print(f"  All heights: {sorted([int(h) for h in heights])}")
