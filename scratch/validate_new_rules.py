"""
Validate proposed new rules against ALL training + test images to ensure zero false rejections.
Rules:
1. green contour must have connected area >= 0.5% of image (max_green_contour_ratio >= 0.005)
2. white_pct must be < 10%
3. Plant must have meaningful green (green_pct >= 0.003) -- already exists
"""
import cv2
import numpy as np
import os

def check_rules(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    total = img_224.shape[0] * img_224.shape[1]
    
    # Green mask and largest contour
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = np.sum(green_mask > 0) / total
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_green_area = 0
    if green_contours:
        max_green_area = max(cv2.contourArea(c) for c in green_contours) / total
    
    # White percentage
    white_mask = cv2.inRange(img_224, np.array([200, 200, 200]), np.array([255, 255, 255]))
    white_pct = np.sum(white_mask > 0) / total
    
    # Straight line count (documents have many)
    gray = cv2.cvtColor(img_224, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    num_lines = 0 if lines is None else len(lines)
    
    # Avg saturation
    avg_sat = np.mean(hsv[:,:,1])
    
    # Rule: Reject if green contour is too small (< 0.5% of image)
    # AND the model would likely be uncertain
    rule1_fail = max_green_area < 0.005  # No significant connected green plant structure
    rule2_fail = white_pct > 0.10  # Too much white (document-like)
    
    # Combined: reject if no meaningful plant structure detected
    # Must have at least one significant green contour (connected plant leaves/stem)
    reject = rule1_fail
    
    return {
        'green_pct': green_pct,
        'max_green_area': max_green_area,
        'white_pct': white_pct,
        'num_lines': num_lines,
        'avg_sat': avg_sat,
        'rule1_fail': rule1_fail,
        'rule2_fail': rule2_fail,
        'reject': reject
    }

# Check non-plant images
base = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772"
print("=== NON-PLANT IMAGES ===")
for fname, desc in [("media__1783868944240.jpg", "ID Card"), ("media__1783869000480.jpg", "Aadhaar")]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        r = check_rules(fpath)
        status = "REJECT" if r['reject'] else "PASS"
        print(f"  {desc}: green_area={r['max_green_area']:.4f}, white={r['white_pct']:.4f}, lines={r['num_lines']} -> {status}")

# Check ALL training+test images
dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"
print("\n=== DATASET IMAGES ===")
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        false_rejects = []
        total = 0
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            r = check_rules(fpath)
            if r is None:
                continue
            total += 1
            if r['reject']:
                false_rejects.append((fname, r['max_green_area'], r['white_pct']))
        
        if false_rejects:
            print(f"\n  {split}/{cls}: {len(false_rejects)}/{total} FALSE REJECTS!")
            for fname, ga, wp in false_rejects:
                print(f"    {fname}: green_area={ga:.4f}, white={wp:.4f}")
        else:
            print(f"  {split}/{cls}: 0/{total} false rejects - OK")
