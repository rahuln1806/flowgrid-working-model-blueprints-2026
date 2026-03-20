#!/usr/bin/env python3
"""
YOLOv8 Training Script for Roboflow Traffic Dataset
Trains a custom YOLOv8 model using the downloaded Roboflow dataset
"""

import os
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

def train_roboflow_model(
    data_config: str = "configs/data.yaml",
    model_size: str = "yolov8n.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "roboflow_traffic"
):
    """
    Train YOLOv8 model on Roboflow traffic dataset
    
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
    
    # Read and display dataset info
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        print("📋 Dataset Configuration:")
        print(f"   Train: {config.get('train', 'N/A')}")
        print(f"   Val: {config.get('val', 'N/A')}")
        print(f"   Classes: {config.get('names', [])}")
        print(f"   Number of classes: {config.get('nc', 0)}")
        
    except Exception as e:
        print(f"⚠️ Could not read dataset config: {e}")
    
    print(f"\n🚀 Starting training with {model_size}")
    print(f"⏰ Epochs: {epochs}")
    print(f"🖼️ Image size: {img_size}x{img_size}")
    print(f"💾 Batch size: {batch_size}")
    print(f"🔧 Device: {device}")
    
    # Check if dataset directories exist
    try:
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        train_dir = config.get('train')
        val_dir = config.get('val')
        
        if train_dir and not os.path.exists(train_dir):
            print(f"⚠️ Training directory not found: {train_dir}")
            print("   Make sure you have run the setup_dataset.py script first")
        
        if val_dir and not os.path.exists(val_dir):
            print(f"⚠️ Validation directory not found: {val_dir}")
            print("   Make sure you have run the setup_dataset.py script first")
        
        # Count training images
        if train_dir and os.path.exists(train_dir):
            train_images = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"📊 Training images found: {train_images}")
        
        # Count validation images
        if val_dir and os.path.exists(val_dir):
            val_images = len([f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"📊 Validation images found: {val_images}")
    
    except Exception as e:
        print(f"⚠️ Could not verify dataset directories: {e}")
    
    # Load pre-trained YOLOv8 model
    print(f"\n🔄 Loading pre-trained model: {model_size}")
    model = YOLO(model_size)
    
    # Train the model
    print(f"\n🏃‍♂️ Starting training...")
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
    
    if hasattr(results, 'results_dict') and results.results_dict:
        print(f"📈 Final mAP50: {results.results_dict.get('metrics/mAP50-0.5', 'N/A')}")
        print(f"📈 Final mAP50-95: {results.results_dict.get('metrics/mAP50-95', 'N/A')}")
    
    # Test the trained model
    print(f"\n🧪 Testing trained model...")
    try:
        # Load the best model
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        test_model = YOLO(best_model_path)
        
        # Run validation
        val_results = test_model.val(data=data_config)
        
        if hasattr(val_results, 'results_dict') and val_results.results_dict:
            print(f"📊 Validation Results:")
            print(f"   mAP50: {val_results.results_dict.get('metrics/mAP50-0.5', 'N/A')}")
            print(f"   mAP50-95: {val_results.results_dict.get('metrics/mAP50-95', 'N/A')}")
            print(f"   Precision: {val_results.results_dict.get('metrics/precision(B)', 'N/A')}")
            print(f"   Recall: {val_results.results_dict.get('metrics/recall(B)', 'N/A')}")
        
    except Exception as e:
        print(f"⚠️ Could not test trained model: {e}")
    
    return results

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on Roboflow traffic dataset")
    
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
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/detect",
        help="Project name for saving results"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="roboflow_traffic",
        help="Experiment name"
    )
    
    args = parser.parse_args()
    
    try:
        results = train_roboflow_model(
            data_config=args.data,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device,
            project=args.project,
            name=args.name
        )
        return 0
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
