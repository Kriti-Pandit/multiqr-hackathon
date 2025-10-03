
import argparse
import os
import shutil
import random
import yaml
from ultralytics import YOLO

def main(data_path):
    """Train YOLOv8 model on QR dataset"""
    # Paths
    train_images_dir = os.path.join(data_path, "train_images")
    labels_dir = os.path.join(data_path, "labels")
    
    # Validate paths
    assert os.path.exists(train_images_dir), f"Train images not found: {train_images_dir}"
    assert os.path.exists(labels_dir), f"Labels not found: {labels_dir}"
    
    # Sync labels to train_images directory
    print("🔄 Syncing labels to train_images...")
    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            src = os.path.join(labels_dir, file)
            dst = os.path.join(train_images_dir, file)
            shutil.copy(src, dst)
    
    # Get images with annotations
    image_files = []
    for f in os.listdir(train_images_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            txt_path = os.path.join(train_images_dir, f.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(txt_path):
                image_files.append(f)
    
    print(f"📊 Found {len(image_files)} images with annotations")
    
    # Create train/val split (80/20)
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Save file lists
    with open('train.txt', 'w') as f:
        for img in train_files:
            f.write(f"{train_images_dir}/{img}\n")
    with open('val.txt', 'w') as f:
        for img in val_files:
            f.write(f"{train_images_dir}/{img}\n")
    
    # Create data.yaml
    data_config = {
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': 1,
        'names': ['qr']
    }
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"📊 Train: {len(train_files)} | Val: {len(val_files)} images")
    
    # Train model
    print("🚀 Starting training...")
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        name='qr_yolov8_final',
        exist_ok=True
    )
    print("✅ Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train QR detection model')
    parser.add_argument('--data_path', required=True, 
                       help='Path to QR_Dataset directory (containing train_images/ and labels/)')
    args = parser.parse_args()
    main(args.data_path)