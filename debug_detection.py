# debug_detection.py - Visual comparison of Stage 1 vs Stage 2 detection

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def preprocess_image(img_path):
    """Stage 1 preprocessing (CLAHE + sharpening)"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def main():
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = 'best.pt'
    TEST_IMAGE_PATH = 'data/demo_images/img203.jpg'  # ← UPDATE THIS
    
    # Verify files exist
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return
    
    print("🔍 Loading model and test image...")
    model = YOLO(MODEL_PATH)
    
    # Stage 1: Preprocessed detection
    print("📸 Running Stage 1 detection (preprocessed image)...")
    preprocessed = preprocess_image(TEST_IMAGE_PATH)
    cv2.imwrite("preprocessed_debug.jpg", preprocessed)
    results1 = model("preprocessed_debug.jpg")
    results1[0].save(filename="stage1_detection.jpg")
    print(f"✅ Stage 1 detected {len(results1[0].boxes)} QRs")
    
    # Stage 2: Original detection (with lower confidence)
    print("📸 Running Stage 2 detection (original image, conf=0.25)...")
    results2 = model(TEST_IMAGE_PATH, conf=0.25)
    results2[0].save(filename="stage2_detection.jpg")
    print(f"✅ Stage 2 detected {len(results2[0].boxes)} QRs")
    
    print("\n✅ Done! Check these files:")
    print("- stage1_detection.jpg (preprocessed)")
    print("- stage2_detection.jpg (original)")
    print("- preprocessed_debug.jpg (preprocessed image)")

if __name__ == "__main__":
    main()