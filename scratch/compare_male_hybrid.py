"""
Compare morphological features of IMG_0334.JPEG and IMG_20260312_112811.jpg.
"""
import cv2
import numpy as np

def analyze(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    
    # Masks
    violet_mask = cv2.inRange(hsv, np.array([120, 50, 40]), np.array([170, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    maroon_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([15, 200, 120]))
    combined_mask = cv2.bitwise_or(violet_mask, red_mask)
    combined_mask = cv2.bitwise_or(combined_mask, maroon_mask)
    
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = np.sum(green_mask > 0) / green_mask.size
    
    # Contours of stem
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        _, _, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(cw, 1)
        if aspect >= 1.5:
            vertical_contours.append((c, area, cw, ch, aspect))
            
    best = None
    if vertical_contours:
        best = max(vertical_contours, key=lambda x: x[1])
        
    return {
        'green_pct': green_pct,
        'best_stem': best[1:] if best else None,
        'img_shape': img.shape
    }

r_male = analyze("dataset/train/male/IMG_0334.JPEG")
r_hybrid = analyze("dataset/train/hybrid/IMG_20260312_112811.jpg")

print("MALE (IMG_0334.JPEG):")
print(f"  green_pct: {r_male['green_pct']*100:.2f}%")
print(f"  stem: area={r_male['best_stem'][0]:.2f}, w={r_male['best_stem'][1]}, h={r_male['best_stem'][2]}, aspect={r_male['best_stem'][3]:.2f}" if r_male['best_stem'] else "  No stem")
print(f"  shape: {r_male['img_shape']}")

print("\nHYBRID (IMG_20260312_112811.jpg):")
print(f"  green_pct: {r_hybrid['green_pct']*100:.2f}%")
print(f"  stem: area={r_hybrid['best_stem'][0]:.2f}, w={r_hybrid['best_stem'][1]}, h={r_hybrid['best_stem'][2]}, aspect={r_hybrid['best_stem'][3]:.2f}" if r_hybrid['best_stem'] else "  No stem")
print(f"  shape: {r_hybrid['img_shape']}")
