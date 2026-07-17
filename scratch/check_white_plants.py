"""Find which plant images have high white_pct to check if combined rules work."""
import cv2
import numpy as np
import os

dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"

print("=== PLANT IMAGES WITH white_pct > 10% (threshold 180,180,180) ===")
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img_224 = cv2.resize(img, (224, 224))
            total = img_224.shape[0] * img_224.shape[1]
            white_mask = cv2.inRange(img_224, np.array([180, 180, 180]), np.array([255, 255, 255]))
            white_pct = np.sum(white_mask > 0) / total
            
            hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
            low_sat_mask = hsv[:,:,1] < 40
            low_sat_pct = np.sum(low_sat_mask) / total
            
            green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
            brown_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([25, 200, 150]))
            plant_bg = (np.sum(green_mask > 0) + np.sum(brown_mask > 0)) / total
            
            avg_sat = np.mean(hsv[:,:,1])
            
            if white_pct > 0.10:
                print(f"  {split}/{cls}/{fname}: white={white_pct*100:.1f}%, low_sat={low_sat_pct*100:.1f}%, plant_bg={plant_bg*100:.1f}%, avg_sat={avg_sat:.1f}")

# Also check: what if we use white_pct > 15% AND low_sat_pct > 50%?
print("\n=== Combined: white_pct > 15% AND low_sat_pct > 50% ===")
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        triggered = 0
        total = 0
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            total += 1
            img_224 = cv2.resize(img, (224, 224))
            t = img_224.shape[0] * img_224.shape[1]
            white_mask = cv2.inRange(img_224, np.array([180, 180, 180]), np.array([255, 255, 255]))
            white_pct = np.sum(white_mask > 0) / t
            hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
            low_sat_pct = np.sum(hsv[:,:,1] < 40) / t
            if white_pct > 0.15 and low_sat_pct > 0.50:
                triggered += 1
                print(f"  FALSE REJECT: {split}/{cls}/{fname}")
        if triggered == 0:
            print(f"  {split}/{cls}: 0/{total} OK")
