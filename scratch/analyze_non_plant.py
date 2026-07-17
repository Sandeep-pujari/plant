"""
Analyze why ID card / Aadhaar card images pass the current validate_morphology checks.
Compare feature values against real plant images to find discriminating features.
"""
import cv2
import numpy as np
import os

def analyze_image(path, label=""):
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read: {path}")
        return
    
    img_224 = cv2.resize(img, (224, 224))
    img_100 = cv2.resize(img, (100, 100))
    
    # 1. Unique colors
    flat_100 = img_100.reshape(-1, 3)
    unique_colors = len(np.unique(flat_100, axis=0))
    
    # 2. Laplacian variance
    gray = cv2.cvtColor(img_224, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Histogram peakiness
    hist_r = cv2.calcHist([img_100], [2], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_100], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img_100], [0], None, [256], [0, 256])
    peak_r = np.sum(np.sort(hist_r.flatten())[-3:]) / np.sum(hist_r)
    peak_g = np.sum(np.sort(hist_g.flatten())[-3:]) / np.sum(hist_g)
    peak_b = np.sum(np.sort(hist_b.flatten())[-3:]) / np.sum(hist_b)
    max_peak = max(peak_r, peak_g, peak_b)
    
    # 4. Color segments
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    total = img_224.shape[0] * img_224.shape[1]
    
    green_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    green_pct = np.sum(green_mask > 0) / total
    
    purple_mask = cv2.inRange(hsv, np.array([125, 30, 30]), np.array([170, 255, 255]))
    purple_pct = np.sum(purple_mask > 0) / total
    
    # 5. Edge density (Canny)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / total
    
    # 6. Straight line detection (Hough Lines)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    num_lines = 0 if lines is None else len(lines)
    
    # 7. Text-like features: high contrast small blobs
    # Use adaptive threshold to find text-like regions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    text_density = np.sum(thresh > 0) / total
    
    # 8. Contour analysis for plant shape
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_green_contour_area = 0
    if green_contours:
        max_green_contour_area = max(cv2.contourArea(c) for c in green_contours) / total
    
    # 9. Color variance (natural images have gradual transitions, documents have sharp ones)
    b, g, r = cv2.split(img_224)
    color_std = np.mean([np.std(b), np.std(g), np.std(r)])
    
    # 10. White/near-white percentage (documents tend to be white)
    white_mask = cv2.inRange(img_224, np.array([200, 200, 200]), np.array([255, 255, 255]))
    white_pct = np.sum(white_mask > 0) / total
    
    # 11. Saturation analysis (plant images are more saturated than documents)
    avg_saturation = np.mean(hsv[:,:,1])
    
    # 12. Brown/soil background percentage (plant images have soil)
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([25, 200, 150]))
    brown_pct = np.sum(brown_mask > 0) / total
    
    # 13. Aspect ratio of largest contour (plants are Y-shaped)
    all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n=== {label} ===")
    print(f"  unique_colors: {unique_colors}")
    print(f"  laplacian_var: {laplacian_var:.1f}")
    print(f"  max_peak: {max_peak:.3f}")
    print(f"  green_pct: {green_pct:.4f} ({green_pct*100:.2f}%)")
    print(f"  purple_pct: {purple_pct:.4f}")
    print(f"  edge_density: {edge_density:.4f} ({edge_density*100:.2f}%)")
    print(f"  num_hough_lines: {num_lines}")
    print(f"  text_density: {text_density:.4f} ({text_density*100:.2f}%)")
    print(f"  max_green_contour_ratio: {max_green_contour_area:.4f}")
    print(f"  color_std: {color_std:.1f}")
    print(f"  white_pct: {white_pct:.4f} ({white_pct*100:.2f}%)")
    print(f"  avg_saturation: {avg_saturation:.1f}")
    print(f"  brown_pct: {brown_pct:.4f} ({brown_pct*100:.2f}%)")
    
    # Current validation would pass/fail?
    would_reject = False
    reasons = []
    if unique_colors < 150:
        reasons.append(f"unique_colors={unique_colors} < 150")
        would_reject = True
    if laplacian_var < 50.0:
        reasons.append(f"laplacian_var={laplacian_var:.1f} < 50")
        would_reject = True
    if max_peak > 0.98:
        reasons.append(f"max_peak={max_peak:.3f} > 0.98")
        would_reject = True
    if purple_pct > 0.015:
        reasons.append(f"purple_pct={purple_pct:.4f} > 0.015")
        would_reject = True
    if green_pct < 0.003:
        reasons.append(f"green_pct={green_pct:.4f} < 0.003")
        would_reject = True
    
    if would_reject:
        print(f"  >> CURRENT VALIDATION: REJECT ({', '.join(reasons)})")
    else:
        print(f"  >> CURRENT VALIDATION: PASS (would wrongly accept this)")

base = r"C:\Users\SANDEEP P\.gemini\antigravity\brain\79fff92d-fadb-4eac-b8a0-f6488009b772"

# Non-plant images (ID cards)
for fname, desc in [
    ("media__1783868944240.jpg", "Student ID Card"),
    ("media__1783869000480.jpg", "Aadhaar Card"),
]:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        analyze_image(fpath, desc)

# Real plant images from training data
dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset\train"
for cls in ['female', 'hybrid', 'male']:
    cls_dir = os.path.join(dataset, cls)
    files = sorted(os.listdir(cls_dir))[:3]
    for fname in files:
        analyze_image(os.path.join(cls_dir, fname), f"PLANT ({cls}) - {fname}")
