"""Debug IMG_0509 hybrid vs the male plant that was being mispredicted."""
import cv2
import numpy as np
import os

def get_all_features(path, label=""):
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read: {path}")
        return
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    total_pixels = img_224.shape[0] * img_224.shape[1]
    
    # Violet
    violet_mask = cv2.inRange(hsv, np.array([120, 50, 40]), np.array([170, 255, 255]))
    violet_pct = (cv2.countNonZero(violet_mask) / total_pixels) * 100.0
    
    # Red
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_pct = (cv2.countNonZero(red_mask) / total_pixels) * 100.0
    
    # Maroon
    maroon_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([15, 200, 120]))
    maroon_pct = (cv2.countNonZero(maroon_mask) / total_pixels) * 100.0
    
    # Combined
    combined_mask = cv2.bitwise_or(violet_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, maroon_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stem_area = 0
    stem_h = 0
    stem_w = 0
    stem_aspect = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        stem_area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        stem_h = h
        stem_w = w
        stem_aspect = h / max(w, 1)
    
    # Green percentage
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = (cv2.countNonZero(green_mask) / total_pixels) * 100.0
    
    # Dark percentage
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
    dark_pct = (cv2.countNonZero(dark_mask) / total_pixels) * 100.0
    
    # Brown/earth tone
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([25, 200, 150]))
    brown_pct = (cv2.countNonZero(brown_mask) / total_pixels) * 100.0
    
    print(f"\n=== {label} ===")
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"  violet_pct: {violet_pct:.2f}%")
    print(f"  red_pct: {red_pct:.2f}%")
    print(f"  maroon_pct: {maroon_pct:.2f}%")
    print(f"  stem_area: {stem_area:.0f}")
    print(f"  stem_h: {stem_h}")
    print(f"  stem_w: {stem_w}")
    print(f"  stem_aspect (h/w): {stem_aspect:.2f}")
    print(f"  green_pct: {green_pct:.2f}%")
    print(f"  dark_pct: {dark_pct:.2f}%")
    print(f"  brown_pct: {brown_pct:.2f}%")
    
    return {
        'stem_area': stem_area, 'stem_h': stem_h, 'stem_w': stem_w,
        'stem_aspect': stem_aspect, 'violet_pct': violet_pct,
        'red_pct': red_pct, 'maroon_pct': maroon_pct,
        'green_pct': green_pct, 'dark_pct': dark_pct, 'brown_pct': brown_pct
    }

base = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday"

# The problematic hybrid image
get_all_features(os.path.join(base, "dataset", "train", "hybrid", "IMG_0509.jpg"), "HYBRID IMG_0509 (wrongly predicted as male)")

# The male plant that was previously mispredicted as hybrid  
# Check brain directory for the user's male image from previous conversation
male_img = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772\media__1783867042180.jpg"
if os.path.exists(male_img):
    get_all_features(male_img, "MALE PLANT (previously mispredicted as hybrid)")

# Also check all hybrid training images for stem_area >= 265
print("\n\n=== ALL HYBRID IMAGES WITH stem_area >= 200 ===")
hybrid_dir = os.path.join(base, "dataset", "train", "hybrid")
for fname in sorted(os.listdir(hybrid_dir)):
    fpath = os.path.join(hybrid_dir, fname)
    img = cv2.imread(fpath)
    if img is None:
        continue
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    violet_mask = cv2.inRange(hsv, np.array([120, 50, 40]), np.array([170, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    maroon_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([15, 200, 120]))
    combined_mask = cv2.bitwise_or(violet_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, maroon_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        if area >= 200:
            total = img_224.shape[0] * img_224.shape[1]
            green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
            green_pct = (cv2.countNonZero(green_mask) / total) * 100.0
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
            dark_pct = (cv2.countNonZero(dark_mask) / total) * 100.0
            print(f"  {fname}: area={area:.0f}, h={h}, w={w}, aspect={h/max(w,1):.2f}, green={green_pct:.1f}%, dark={dark_pct:.1f}%")
