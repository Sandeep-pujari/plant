"""Find the optimal stem_area threshold that separates hybrid from male with zero false positives."""
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
    
    # Combined pigment mask
    violet_mask = cv2.inRange(hsv, np.array([120, 50, 40]), np.array([170, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    maroon_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([15, 200, 120]))
    combined_mask = cv2.bitwise_or(violet_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, maroon_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stem_area = 0
    stem_h = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        stem_area = cv2.contourArea(largest)
        _, _, _, h = cv2.boundingRect(largest)
        stem_h = h
    
    return stem_area, stem_h

dataset_dir = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"

# Collect ALL areas and heights for hybrid and female (non-male)
hybrid_areas = []
female_areas = []
male_areas = []

for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset_dir, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            result = get_stem_features(fpath)
            if result:
                area, h = result
                if cls == 'hybrid':
                    hybrid_areas.append((area, h, fname))
                elif cls == 'female':
                    female_areas.append((area, h, fname))
                else:
                    male_areas.append((area, h, fname))

print("=== HYBRID (all) ===")
hybrid_areas.sort(key=lambda x: x[0], reverse=True)
for a, h, f in hybrid_areas[:10]:
    print(f"  area={a:.0f}, h={h}, file={f}")

print("\n=== FEMALE (top 10 by area) ===")
female_areas.sort(key=lambda x: x[0], reverse=True)
for a, h, f in female_areas[:10]:
    print(f"  area={a:.0f}, h={h}, file={f}")

print("\n=== MALE (bottom 10 by area) ===")
male_areas.sort(key=lambda x: x[0])
for a, h, f in male_areas[:10]:
    print(f"  area={a:.0f}, h={h}, file={f}")

# Check threshold: stem_area >= 270 (hybrid max is 262)
print(f"\n=== THRESHOLD ANALYSIS ===")
for threshold in [250, 260, 265, 270, 280, 300]:
    hybrid_fp = sum(1 for a, h, f in hybrid_areas if a >= threshold)
    female_fp = sum(1 for a, h, f in female_areas if a >= threshold)
    male_tp = sum(1 for a, h, f in male_areas if a >= threshold)
    print(f"  stem_area >= {threshold}: hybrid_fp={hybrid_fp}, female_fp={female_fp}, male_tp={male_tp}/{len(male_areas)}")

# User's image
user_img = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\IMG_0334.JPEG"
if os.path.exists(user_img):
    a, h = get_stem_features(user_img)
    print(f"\n=== USER'S MALE PLANT: area={a:.0f}, h={h} ===")
