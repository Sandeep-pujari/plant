"""Test rule: white_pct > 15% AND plant_bg < 10% for document rejection."""
import cv2
import numpy as np
import os

dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"
base = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772"

def get_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_224 = cv2.resize(img, (224, 224))
    total = img_224.shape[0] * img_224.shape[1]
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    
    white_mask = cv2.inRange(img_224, np.array([180, 180, 180]), np.array([255, 255, 255]))
    white_pct = np.sum(white_mask > 0) / total
    
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([25, 200, 150]))
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    plant_bg = (np.sum(green_mask > 0) + np.sum(brown_mask > 0) + np.sum(dark_mask > 0)) / total
    
    avg_sat = np.mean(hsv[:,:,1])
    
    return white_pct, plant_bg, avg_sat

# Non-plant
print("=== NON-PLANT ===")
for fname, desc in [("media__1783868944240.jpg", "ID"), ("media__1783869000480.jpg", "Aadhaar")]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        w, p, s = get_features(fpath)
        reject = w > 0.15 and p < 0.12
        print(f"  {desc}: white={w*100:.1f}%, plant_bg+dark={p*100:.1f}%, sat={s:.1f} -> {'REJECT' if reject else 'PASS'}")

# Dataset
print("\n=== DATASET ===")
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        fp = 0
        total = 0
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            r = get_features(fpath)
            if r is None:
                continue
            total += 1
            w, p, s = r
            reject = w > 0.15 and p < 0.12
            if reject:
                fp += 1
                print(f"  FALSE REJECT: {split}/{cls}/{fname}: white={w*100:.1f}%, plant_bg+dark={p*100:.1f}%, sat={s:.1f}")
        if fp == 0:
            print(f"  {split}/{cls}: 0/{total} OK")
