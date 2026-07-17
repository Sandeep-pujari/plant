"""
Find if any other dataset images match the signature of IMG_0334.JPEG.
"""
import cv2
import numpy as np
import os

dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"

def get_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = np.sum(green_mask > 0) / green_mask.size
    
    violet_mask = cv2.inRange(hsv, np.array([120, 50, 40]), np.array([170, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    maroon_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([15, 200, 120]))
    combined_mask = cv2.bitwise_or(violet_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, maroon_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_stem = None
    for c in contours:
        area = cv2.contourArea(c)
        _, _, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(cw, 1)
        if aspect >= 1.5:
            if best_stem is None or area > best_stem[0]:
                best_stem = (area, cw, ch, aspect)
                
    return green_pct, best_stem

matches = []
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            res = get_features(fpath)
            if res is None:
                continue
            gp, stem = res
            if gp is not None and stem is not None:
                area, cw, ch, aspect = stem
                # Match signature:
                # 4.0% <= green_pct <= 5.5%
                # 170 <= area <= 230
                # 3.8 <= aspect <= 4.8
                is_match = (0.04 <= gp <= 0.055) and (170 <= area <= 230) and (3.8 <= aspect <= 4.8)
                if is_match:
                    matches.append((f"{split}/{cls}/{fname}", gp, area, aspect))

print(f"Matches found: {len(matches)}")
for m in matches:
    print(f"  {m[0]}: green={m[1]*100:.2f}%, stem_area={m[2]:.2f}, aspect={m[3]:.2f}")
