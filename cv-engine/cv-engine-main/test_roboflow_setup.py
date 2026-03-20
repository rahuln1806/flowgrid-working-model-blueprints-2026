#!/usr/bin/env python3
"""
Test script to verify Roboflow dataset setup and configuration
"""

import os
import yaml
import cv2
from pathlib import Path

def test_dataset_setup():
    """Test if dataset is properly set up"""
    print("🧪 Testing Roboflow Dataset Setup")
    print("=" * 50)
    
    # Test 1: Check if dataset directory exists
    dataset_dir = "traffic-9"
    if os.path.exists(dataset_dir):
        print(f"✅ Dataset directory exists: {dataset_dir}")
    else:
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    # Test 2: Check train and validation directories
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "valid")
    
    if os.path.exists(train_dir):
        train_images = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✅ Train directory exists: {train_images} images")
    else:
        print(f"❌ Train directory not found: {train_dir}")
        return False
    
    if os.path.exists(val_dir):
        val_images = len([f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✅ Validation directory exists: {val_images} images")
    else:
        print(f"❌ Validation directory not found: {val_dir}")
        return False
    
    # Test 3: Check data.yaml configuration
    data_config_path = "configs/data.yaml"
    if os.path.exists(data_config_path):
        try:
            with open(data_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Data config exists: {data_config_path}")
            print(f"   Classes: {config.get('names', [])}")
            print(f"   Number of classes: {config.get('nc', 0)}")
            print(f"   Train path: {config.get('train', 'N/A')}")
            print(f"   Val path: {config.get('val', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error reading data config: {e}")
            return False
    else:
        print(f"❌ Data config not found: {data_config_path}")
        return False
    
    # Test 4: Check class mapping
    mapping_config_path = "configs/class_mapping.yaml"
    if os.path.exists(mapping_config_path):
        try:
            with open(mapping_config_path, 'r') as f:
                mapping = yaml.safe_load(f)
            
            print(f"✅ Class mapping exists: {mapping_config_path}")
            print(f"   Dataset classes: {mapping.get('dataset_classes', {})}")
            print(f"   Traffic classes: {mapping.get('traffic_classes', {})}")
            print(f"   Mapping: {mapping.get('mapping', {})}")
            
        except Exception as e:
            print(f"❌ Error reading class mapping: {e}")
            return False
    else:
        print(f"❌ Class mapping not found: {mapping_config_path}")
        return False
    
    # Test 5: Check if sample images can be loaded
    print(f"\n🖼️ Testing image loading...")
    
    # Test training image
    train_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if train_files:
        sample_train_img = os.path.join(train_dir, train_files[0])
        try:
            img = cv2.imread(sample_train_img)
            if img is not None:
                print(f"✅ Can load training image: {train_files[0]} ({img.shape})")
            else:
                print(f"❌ Cannot load training image: {train_files[0]}")
                return False
        except Exception as e:
            print(f"❌ Error loading training image: {e}")
            return False
    
    # Test validation image
    val_files = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if val_files:
        sample_val_img = os.path.join(val_dir, val_files[0])
        try:
            img = cv2.imread(sample_val_img)
            if img is not None:
                print(f"✅ Can load validation image: {val_files[0]} ({img.shape})")
            else:
                print(f"❌ Cannot load validation image: {val_files[0]}")
                return False
        except Exception as e:
            print(f"❌ Error loading validation image: {e}")
            return False
    
    return True

def test_model_loading():
    """Test if model can be loaded"""
    print(f"\n🤖 Testing Model Loading")
    print("=" * 50)
    
    # Test pretrained model loading
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("✅ Can load pretrained YOLOv8n model")
    except Exception as e:
        print(f"❌ Cannot load pretrained model: {e}")
        return False
    
    # Test trained model if it exists
    trained_model_path = "runs/detect/roboflow_traffic/weights/best.pt"
    if os.path.exists(trained_model_path):
        try:
            model = YOLO(trained_model_path)
            print(f"✅ Can load trained model: {trained_model_path}")
        except Exception as e:
            print(f"❌ Cannot load trained model: {e}")
    else:
        print(f"⚠️ Trained model not found (expected after training): {trained_model_path}")
    
    return True

def test_inference_setup():
    """Test inference setup"""
    print(f"\n🔍 Testing Inference Setup")
    print("=" * 50)
    
    # Test if inference script exists
    inference_script = "inference/detect_roboflow.py"
    if os.path.exists(inference_script):
        print(f"✅ Inference script exists: {inference_script}")
    else:
        print(f"❌ Inference script not found: {inference_script}")
        return False
    
    # Test import of detection class
    try:
        import sys
        sys.path.append('.')
        from inference.detect_roboflow import RoboflowTrafficDetector
        print("✅ Can import RoboflowTrafficDetector class")
    except Exception as e:
        print(f"❌ Cannot import RoboflowTrafficDetector: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Roboflow CV Engine Setup Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Dataset Setup", test_dataset_setup),
        ("Model Loading", test_model_loading),
        ("Inference Setup", test_inference_setup)
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Tests...")
        try:
            if not test_func():
                all_tests_passed = False
                print(f"❌ {test_name} tests failed")
            else:
                print(f"✅ {test_name} tests passed")
        except Exception as e:
            print(f"❌ {test_name} tests failed with error: {e}")
            all_tests_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! Your Roboflow CV Engine is ready to use.")
        print("\n📋 Next steps:")
        print("1. Train the model: python training/train_roboflow.py")
        print("2. Run detection: python inference/detect_roboflow.py")
        print("3. Test with webcam: python inference/detect_roboflow.py --source 0")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you ran: python setup_dataset.py")
        print("2. Check that all directories exist and contain images")
        print("3. Verify configuration files are correct")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
