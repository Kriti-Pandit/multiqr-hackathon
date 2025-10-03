import argparse
import json
import os
from ultralytics import YOLO

def load_ground_truth(gt_json):
    """Load ground truth annotations"""
    with open(gt_json, 'r') as f:
        return json.load(f)

def main(gt_json, pred_json, model_path='best.pt'):
    """Evaluate model performance against ground truth"""
    # Load ground truth
    if not os.path.exists(gt_json):
        print(f"⚠️ Ground truth file not found: {gt_json}")
        print("Running model validation instead...")
        model = YOLO(model_path)
        try:
            metrics = model.val(data='data.yaml')
            print(f"📊 Validation mAP50: {metrics.box.map50:.3f}")
            print(f"📊 Validation mAP50-95: {metrics.box.map:.3f}")
        except:
            print("❌ Could not run validation (data.yaml not found)")
        return
    
    # Load predictions
    with open(pred_json, 'r') as f:
        predictions = json.load(f)
    
    gt_data = load_ground_truth(gt_json)
    
    print(f"📊 Ground truth images: {len(gt_data)}")
    print(f"📊 Predicted images: {len(predictions)}")
    
    # Simple stats
    total_gt_qrs = sum(len(item['qrs']) for item in gt_data)
    total_pred_qrs = sum(len(item['qrs']) for item in predictions)
    
    print(f"📊 Total ground truth QRs: {total_gt_qrs}")
    print(f"📊 Total predicted QRs: {total_pred_qrs}")
    
    # Load model for additional metrics
    try:
        model = YOLO(model_path)
        if os.path.exists('data.yaml'):
            metrics = model.val(data='data.yaml')
            print(f"📊 Model validation mAP50: {metrics.box.map50:.3f}")
    except Exception as e:
        print(f"⚠️ Could not run model validation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--gt', help='Ground truth JSON file')
    parser.add_argument('--pred', help='Prediction JSON file')
    parser.add_argument('--model', default='best.pt', help='Model path')
    args = parser.parse_args()
    
    main(args.gt, args.pred, args.model)