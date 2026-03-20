#!/usr/bin/env python3
"""
YOLOv8 Training Module for Traffic Analysis
Trains a custom YOLOv8 model to detect pedestrians, vehicles, and ambulances
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def train_model(
    data_config: str = "configs/data.yaml",
    model_size: str = "yolov8n.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "traffic_train"
):
    """
    Train YOLOv8 model on custom traffic dataset
    
    Args:
        data_config: Path to dataset configuration file
        model_size: Pre-trained model to start from (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        img_size: Input image size for training
        batch_size: Batch size for training
        device: Device to use for training (auto, cpu, 0, 1, etc.)
        project: Project name for saving results
        name: Experiment name
    """
    
    # Validate data configuration file exists
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Dataset configuration file not found: {data_config}")
    
    print(f"🚀 Starting training with {model_size}")
    print(f"📊 Dataset config: {data_config}")
    print(f"⏰ Epochs: {epochs}")
    print(f"🖼️ Image size: {img_size}x{img_size}")
    print(f"💾 Batch size: {batch_size}")
    print(f"🔧 Device: {device}")
    
    # Load pre-trained YOLOv8 model
    model = YOLO(model_size)
    
    # Train the model
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,  # Overwrite existing experiment
        patience=50,    # Early stopping
        save_period=10, # Save checkpoint every 10 epochs
        val=True,       # Validate during training
        plots=True,     # Generate training plots
        verbose=True    # Show training progress
    )
    
    # Print training results
    print("\n✅ Training completed successfully!")
    print(f"📁 Results saved to: {results.save_dir}")
    print(f"🏆 Best model saved to: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
    print(f"📈 Final mAP50: {results.results_dict['metrics/mAP50-0.5']:.4f}")
    print(f"📈 Final mAP50-95: {results.results_dict['metrics/mAP50-95']:.4f}")
    
    return results

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for traffic analysis")
    
    parser.add_argument(
        "--data", 
        type=str, 
        default="configs/data.yaml",
        help="Path to dataset configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Pre-trained model size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--img-size", 
        type=int, 
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, 0, 1, etc.)"
    )
    
    args = parser.parse_args()
    
    try:
        results = train_model(
            data_config=args.data,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device
        )
        return 0
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
