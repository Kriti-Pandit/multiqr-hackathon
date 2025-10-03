import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import re

def preprocess_for_detection(img_path):
    """Lighter preprocessing that won't destroy QR patterns"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Only CLAHE (remove sharpening - it destroys QR patterns)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img

def classify_qr_type(value):
    """Classify QR based on real pharmaceutical patterns"""
    if not value:
        return "undecoded"
    
    v = value.upper()
    if any(kw in v for kw in ["BATCH", "LOT", "BATCH NO", "LOT NO", "BATCH#", "LOT#"]):
        return "batch"
    if v.startswith(("B", "L")) and len(value) >= 5 and any(c.isdigit() for c in value):
        return "batch"
    if any(kw in v for kw in ["EXP", "EXPIRY", "EXPIRE", "EXPY", "VALID UNTIL"]):
        return "expiry"
    if re.search(r'(20[2-9][0-9])(0[1-9]|1[0-2])', v):
        return "expiry"
    if re.search(r'(0[1-9]|1[0-2])/(20[2-9][0-9])', v):
        return "expiry"
    if re.search(r'\d{2}-\d{2}-\d{4}', v):
        return "expiry"
    if any(kw in v for kw in ["MRP", "PRICE", "RS", "₹", "RUPEES", "COST"]):
        return "price"
    if re.search(r'RS\s*\d+\.?\d*', v, re.IGNORECASE):
        return "price"
    if any(kw in v for kw in ["MFR", "MANUFACTURER", "MFG", "MADE BY", "SIG", "PRODUCED BY"]):
        return "manufacturer"
    if (len(value) >= 8 and len(value) <= 20 and 
        value.isalnum() and 
        any(c.isalpha() for c in value) and 
        any(c.isdigit() for c in value)):
        return "serial"
    if (len(value) >= 6 and 
        any(c.isupper() for c in value) and 
        any(c.islower() for c in value) and 
        any(c.isdigit() for c in value)):
        return "product_code"
    return "unknown"

def decode_qr_optimized(original_img, bbox):
    """Optimized QR decoding with multiple enhancement strategies"""
    x1, y1, x2, y2 = map(int, bbox)
    pad = 30
    h, w = original_img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    crop = original_img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    
    min_size = 400
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        scale = max(min_size / crop.shape[0], min_size / crop.shape[1])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    attempts = []
    attempts.append(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    attempts.append(clahe.apply(gray))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(binary)
    attempts.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    kernel = np.ones((2,2), np.uint8)
    attempts.append(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel))
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    attempts.append(cv2.filter2D(gray, -1, sharpen_kernel))
    
    qr_decoder = cv2.QRCodeDetector()
    for attempt in attempts:
        try:
            data, points, _ = qr_decoder.detectAndDecode(attempt)
            if data and points is not None and len(data.strip()) > 0:
                return data.strip()
        except:
            continue
    return ""

def main(input_dir, output_detection, output_bonus=None):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model 'best.pt' not found in root directory.")
    
    model = YOLO(model_path)
    test_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        raise ValueError(f"No images found in {input_dir}")
    test_images.sort()
    
    detection_results = []
    bonus_results = []
    
    for img_name in test_images:
        img_path = os.path.join(input_dir, img_name)
        image_id = Path(img_name).stem
        
        try:
            original_img = cv2.imread(img_path)
            if original_img is None:
                detection_results.append({"image_id": image_id, "qrs": []})
                if output_bonus:
                    bonus_results.append({"image_id": image_id, "qrs": []})
                continue
            
            # 🔥 SINGLE-LINE FIX FOR TILTED QRs: 
            # Use Test-Time Augmentation (TTA) + lower confidence threshold
            preds = model(img_path, conf=0.2, augment=True, verbose=False)
            boxes = preds[0].boxes.xyxy.cpu().numpy()
            
            qrs_detection = []
            qrs_bonus = []
            for box in boxes:
                x1, y1, x2, y2 = box
                bbox = [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]
                qrs_detection.append({"bbox": bbox})
                
                if output_bonus:
                    value = decode_qr_optimized(original_img, bbox)
                    qtype = classify_qr_type(value)
                    qrs_bonus.append({
                        "bbox": bbox,
                        "value": value,
                        "type": qtype
                    })
            
            detection_results.append({"image_id": image_id, "qrs": qrs_detection})
            if output_bonus:
                bonus_results.append({"image_id": image_id, "qrs": qrs_bonus})
                
        except Exception as e:
            detection_results.append({"image_id": image_id, "qrs": []})
            if output_bonus:
                bonus_results.append({"image_id": image_id, "qrs": []})
    
    os.makedirs(os.path.dirname(output_detection), exist_ok=True)
    with open(output_detection, 'w') as f:
        json.dump(detection_results, f, indent=2)
    
    if output_bonus:
        os.makedirs(os.path.dirname(output_bonus), exist_ok=True)
        with open(output_bonus, 'w') as f:
            json.dump(bonus_results, f, indent=2)
    
    print(f"✅ Detection results saved to: {output_detection}")
    if output_bonus:
        print(f"✅ Bonus results saved to: {output_bonus}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1Pharmacy Hackathon: Multi-QR Detection')
    parser.add_argument('--input', required=True, help='Input directory containing test images')
    parser.add_argument('--output', required=True, help='Output JSON for Stage 1 (detection)')
    parser.add_argument('--output_bonus', help='Output JSON for Stage 2 (bonus decoding + classification)')
    args = parser.parse_args()
    
    main(args.input, args.output, args.output_bonus)