"""
Test center-crop plant background rule to separate plants from documents/cards.
"""
import cv2
import numpy as np
import os

dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"
base = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772"

def get_center_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
    img_224 = cv2.resize(img, (224, 224))
    
    # Crop center 60% (from pixel 45 to 179)
    h, w, _ = img_224.shape
    cy1, cy2 = int(h * 0.2), int(h * 0.8)
    cx1, cx2 = int(w * 0.2), int(w * 0.8)
    center_crop = img_224[cy1:cy2, cx1:cx2]
    
    total = center_crop.shape[0] * center_crop.shape[1]
    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    
    # Masks
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([25, 200, 150]))
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    
    green_pct = np.sum(green_mask > 0) / total
    brown_pct = np.sum(brown_mask > 0) / total
    dark_pct = np.sum(dark_mask > 0) / total
    plant_bg = green_pct + brown_pct + dark_pct
    
    # White in the center
    white_mask = cv2.inRange(center_crop, np.array([180, 180, 180]), np.array([255, 255, 255]))
    white_pct = np.sum(white_mask > 0) / total
    
    # Text density (adaptive threshold in the center)
    gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    text_density = np.sum(thresh > 0) / total
    
    return plant_bg, white_pct, text_density

# Non-plant
print("=== NON-PLANT ===")
for fname, desc in [("media__1783868944240.jpg", "ID"), ("media__1783869000480.jpg", "Aadhaar")]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        p, w, t = get_center_features(fpath)
        print(f"  {desc}: center_plant_bg={p*100:.1f}%, center_white={w*100:.1f}%, text_density={t*100:.1f}%")

# Dataset
print("\n=== DATASET ===")
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        p_vals = []
        w_vals = []
        t_vals = []
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            r = get_center_features(fpath)
            if r is None:
                continue
            p_vals.append(r[0])
            w_vals.append(r[1])
            t_vals.append(r[2])
            
        print(f"  {split}/{cls} (n={len(p_vals)}):")
        print(f"    center_plant_bg: min={min(p_vals)*100:.1f}%, avg={np.mean(p_vals)*100:.1f}%, max={max(p_vals)*100:.1f}%")
        print(f"    center_white:    min={min(w_vals)*100:.1f}%, avg={np.mean(w_vals)*100:.1f}%, max={max(w_vals)*100:.1f}%")
        print(f"    text_density:    min={min(t_vals)*100:.1f}%, avg={np.mean(t_vals)*100:.1f}%, max={max(t_vals)*100:.1f}%")
