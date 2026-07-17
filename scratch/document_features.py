"""
Find the best combination of features to separate documents from plant images.
Key insight: Documents have text (high edge density + many straight lines + low saturation)
while plants have organic textures on soil/dark backgrounds.
"""
import cv2
import numpy as np
import os

def get_document_score(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_224, cv2.COLOR_BGR2GRAY)
    total = img_224.shape[0] * img_224.shape[1]
    
    # Feature 1: Average saturation (documents are desaturated, plants on soil are saturated)
    avg_sat = np.mean(hsv[:,:,1])
    
    # Feature 2: White + near-white percentage
    white_mask = cv2.inRange(img_224, np.array([180, 180, 180]), np.array([255, 255, 255]))
    white_pct = np.sum(white_mask > 0) / total
    
    # Feature 3: Horizontal edge dominance (text has horizontal lines)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h_energy = np.mean(np.abs(sobely))
    v_energy = np.mean(np.abs(sobelx))
    
    # Feature 4: Straight line count
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    num_lines = 0 if lines is None else len(lines)
    
    # Feature 5: Green + Brown coverage (plant background)
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = np.sum(green_mask > 0) / total
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([25, 200, 150]))
    brown_pct = np.sum(brown_mask > 0) / total
    plant_bg = green_pct + brown_pct
    
    # Feature 6: Low saturation pixel percentage (sat < 40)  
    low_sat_mask = hsv[:,:,1] < 40
    low_sat_pct = np.sum(low_sat_mask) / total
    
    # Feature 7: Rectangular shape detection
    contours_all, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_count = 0
    for c in contours_all:
        if cv2.contourArea(c) > 100:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                rect_count += 1
    
    return {
        'avg_sat': avg_sat,
        'white_pct': white_pct,
        'low_sat_pct': low_sat_pct,
        'num_lines': num_lines,
        'plant_bg': plant_bg,
        'rect_count': rect_count
    }

# Non-plant images
base = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772"
print("=== NON-PLANT IMAGES ===")
for fname, desc in [("media__1783868944240.jpg", "ID Card"), ("media__1783869000480.jpg", "Aadhaar")]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        r = get_document_score(fpath)
        print(f"  {desc}: sat={r['avg_sat']:.1f}, white={r['white_pct']*100:.1f}%, low_sat={r['low_sat_pct']*100:.1f}%, lines={r['num_lines']}, plant_bg={r['plant_bg']*100:.1f}%, rects={r['rect_count']}")

# All dataset images
dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"
print("\n=== ALL DATASET IMAGES ===")
all_results = {}
for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            r = get_document_score(fpath)
            if r:
                key = f"{split}/{cls}"
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(r)

for key in sorted(all_results.keys()):
    results = all_results[key]
    sats = [r['avg_sat'] for r in results]
    whites = [r['white_pct'] for r in results]
    low_sats = [r['low_sat_pct'] for r in results]
    plant_bgs = [r['plant_bg'] for r in results]
    
    print(f"  {key} (n={len(results)}):")
    print(f"    avg_sat: min={min(sats):.1f}, avg={np.mean(sats):.1f}, max={max(sats):.1f}")
    print(f"    white_pct: min={min(whites)*100:.1f}%, avg={np.mean(whites)*100:.1f}%, max={max(whites)*100:.1f}%")
    print(f"    low_sat: min={min(low_sats)*100:.1f}%, avg={np.mean(low_sats)*100:.1f}%, max={max(low_sats)*100:.1f}%")
    print(f"    plant_bg: min={min(plant_bgs)*100:.1f}%, avg={np.mean(plant_bgs)*100:.1f}%, max={max(plant_bgs)*100:.1f}%")
