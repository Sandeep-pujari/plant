"""
Find and inspect dataset images with low saturation (< 35) or high white percentage (> 10%).
"""
import cv2
import numpy as np
import os

dataset = r"c:\Users\SANDEEP P\Downloads\Ganapati\updated\1st with feamle trit\9thday\9thday\9thday\dataset"

def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_224 = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_224, cv2.COLOR_BGR2HSV)
    total = img_224.shape[0] * img_224.shape[1]
    
    avg_sat = np.mean(hsv[:,:,1])
    white_mask = cv2.inRange(img_224, np.array([180, 180, 180]), np.array([255, 255, 255]))
    white_pct = np.sum(white_mask > 0) / total
    
    gray = cv2.cvtColor(img_224, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    num_lines = 0 if lines is None else len(lines)
    
    return avg_sat, white_pct, num_lines

for split in ['train', 'test']:
    for cls in ['female', 'hybrid', 'male']:
        cls_dir = os.path.join(dataset, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            r = analyze_image(fpath)
            if r is None:
                continue
            sat, white, lines = r
            if sat < 35.0 or white > 0.10:
                print(f"{split}/{cls}/{fname}: sat={sat:.1f}, white={white*100:.1f}%, lines={lines}")
