

# Multi-QR Code Recognition for Medicine Packs

🏆 **Solution for 1Pharmacy Hackathon**

The solution achieves near-perfect detection of multiple QR codes on medicine packs, even under challenging conditions like tilt, blur, partial occlusion, and varying lighting. With **99.1% mAP50**, **99.9% precision**, and **98.8% recall**, our model reliably identifies all QR codes while maintaining exceptional speed and efficiency.

---

### 📊 Performance Summary

| Metric         | Score                  |
| -------------- | ---------------------- |
| **mAP50** | **0.991** |
| **Precision** | **0.999** |
| **Recall** | **0.988** |
| **mAP50-95** | **0.798** |
| **Inference Speed** | **~3.5ms/image (285+ FPS)** |
| **Model Size** | **6.2 MB** |
| **Real-World Test** | **12/12 QRs detected** on diverse medicine packs |

---

### 🧠 Model Selection & Experimentation

Trained and evaluated three different approaches to solve the multi-QR detection challenge:

1.  **Standard YOLOv8n (Selected for Submission)**
    -   **mAP50:** 0.991
    -   **Precision:** 0.999 | **Recall:** 0.988
    -   **Inference Speed:** 3.5ms/image (~285 FPS)
    -   **Model Size:** 6.2 MB
    -   **Training:** 50 epochs, no early stopping

2.  **YOLOv8n with Enhanced Augmentations**
    -   **mAP50:** 0.990
    -   **Precision:** 0.941 | **Recall:** 0.995
    -   **Inference Speed:** 56.7ms/image (~18 FPS)
    -   **Key Issue:** Lower precision indicates more false positives, which is unacceptable for real-world medicine pack scanning.

3.  **YOLOv8n-OBB (Oriented Bounding Boxes)**
    -   **mAP50:** 0.989
    -   **mAP50-95:** 0.901 (best for tight boxes)
    -   **Inference Speed:** ~15ms/image (~67 FPS)
    -   **Key Issue:** Slower inference with minimal accuracy gain for our use case.

---

### 🏆 Why Choose Standard YOLOv8n

I selected the standard YOLOv8n model for our final submission because:

-   **Highest Overall Accuracy:** Achieved the best mAP50 (0.991) with near-perfect precision (0.999).
-   **Real-World Reliability:** Extremely low false positive rate ensures trustworthy detection on medicine packs.
-   **Speed & Efficiency:** Fastest inference (3.5ms/image) qualifies for extra credit on efficiency.
-   **Robustness:** Handles tilted, blurred, and partially covered QR codes effectively without needing OBB complexity.
-   **Production Ready:** Lightweight model (6.2 MB) suitable for deployment on edge devices in pharmacy settings.


---
### 🎯 Bonus Task: QR Decoding & Classification

For the bonus challenge, I implemented a **robust decoding pipeline** that achieves **67% decoding success rate** on real medicine pack QR codes.

#### 🔍 Decoding Strategy
- **Pure OpenCV implementation** (no external APIs)
- **Multi-stage enhancement**: CLAHE, adaptive thresholding, morphological operations
- **Aggressive resizing** (400px minimum) for tiny QRs
- **Generous padding** (30px) to capture full QR boundaries
#### 🏷️ Classification Accuracy
Successfully classified decoded QRs into:
- **`batch`**: Manufacturing batch/lot numbers
- **`price`**: MRP/price information (Indian context)
- **`expiry`**: Expiry dates
- **`manufacturer`**: Manufacturer codes
- **`serial`**: Unique serial numbers

---

### 🚀 Setup Instructions

**Environment**

-   **Python:** 3.8+
-   **GPU:** Recommended (but CPU works)
-   **Dependencies:** Listed in `requirements.txt`

**Installation**

```bash
git clone https://github.com/Kriti-Pandit/multiqr-hackathon.git
cd multiqr-hackathon
pip install -r requirements.txt
```

---

### 🏃 Training (Optional)

```bash
python train.py --data_path /path/to/QR_Dataset
```
*Note: Pre-trained model `best.pt` is included in this repository.*

---

### 🔍 Inference (Required for Evaluation)

**Stage 1:** Detection Only

```bash
python infer.py --input data/demo_images --output outputs/submission_detection_stage1.json
```
**Stage 2:** Detection + Bonus Decoding
```bash
python infer.py --input data/demo_images --output outputs/submission_detection_stage2.json --output_bonus outputs/submission_decoding_2.json
```

---

### 📁 Repository Structure

```
multiqr-hackathon/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── train.py                  # Training script (optional)
├── infer.py                  # Main inference script
├── evaluate.py               # Validation script (optional)
├── best.pt                   # Pre-trained model weights
├── data/
│   └── demo_images/          # Sample test images
└── outputs/                  # Generated submission files
```

---

### 📝 Usage Notes

**Input Requirements**

-   Input directory should contain `.jpg`, `.jpeg`, or `.png` files.
-   No annotations required for inference.
-   Model automatically handles preprocessing (CLAHE + sharpening).

**Output Format**

The output JSON follows the exact format required by the hackathon:

```json
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
```

**Model Details**

-   **Architecture:** YOLOv8n
-   **Input Size:** 640x640
-   **Preprocessing:** CLAHE contrast enhancement + mild sharpening
-   **Post-processing:** Non-maximum suppression (NMS) with confidence threshold 0.5

---

