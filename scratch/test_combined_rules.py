"""
Optimize and test combined rules for document / non-plant rejection.
"""
import cv2
import numpy as np
import os

dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"
base = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772"

def evaluate_rules(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    total = img_224.shape[0] * img_224.shape[1]
    
    # 1. Green percentage
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = np.sum(green_mask > 0) / total
    
    # 2. Brown percentage
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([25, 200, 150]))
    brown_pct = np.sum(brown_mask > 0) / total
    
    # 3. White percentage (R, G, B > 180)
    white_mask = cv2.inRange(img_224, np.array([180, 180, 180]), np.array([255, 255, 255]))
    white_pct = np.sum(white_mask > 0) / total
    
    # 4. Average saturation
    avg_sat = np.mean(hsv[:,:,1])
    
    # 5. Text / Line density
    gray = cv2.cvtColor(img_224, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    num_lines = 0 if lines is None else len(lines)
    
    plant_bg = green_pct + brown_pct
    
    # Proposed Rejection Logic:
    # A document has high white, low saturation, low plant background, and high line/text count.
    # Rule A: plant_bg is too low relative to white
    is_doc_ratio = (white_pct > 0.12) and (plant_bg < 0.30 * white_pct)
    # Rule B: very low saturation document in a darker room
    is_doc_dark = (white_pct > 0.05) and (avg_sat < 35.0) and (plant_bg < 0.12) and (num_lines > 20)
    
    reject = is_doc_ratio or is_doc_dark
    
    return {
        'green': green_pct,
        'brown': brown_pct,
        'plant_bg': plant_bg,
        'white': white_pct,
        'sat': avg_sat,
        'lines': num_lines,
        'is_doc_ratio': is_doc_ratio,
        'is_doc_dark': is_doc_dark,
        'reject': reject
    }

# Non-plant
print("=== NON-PLANT ===")
for fname, desc in [("media__1783868944240.jpg", "ID"), ("media__1783869000480.jpg", "Aadhaar")]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        r = evaluate_rules(fpath)
        print(f"  {desc}: plant_bg={r['plant_bg']*100:.1f}%, white={r['white']*100:.1f}%, sat={r['sat']:.1f}, lines={r['lines']}")
        print(f"    is_doc_ratio={r['is_doc_ratio']}, is_doc_dark={r['is_doc_dark']} -> REJECT={r['reject']}")

# Dataset
print("\n=== DATASET ===")
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        reject_count = 0
        total = 0
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            r = evaluate_rules(fpath)
            if r is None:
                continue
            total += 1
            if r['reject']:
                reject_count += 1
                print(f"  FALSE REJECT: {split}/{cls}/{fname}: plant_bg={r['plant_bg']*100:.1f}%, white={r['white']*100:.1f}%, sat={r['sat']:.1f}, lines={r['lines']}")
        
        print(f"  {split}/{cls}: rejected={reject_count}/{total}")
