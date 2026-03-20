#!/usr/bin/env python3
"""
Roboflow Dataset Setup Script
Downloads and configures the traffic dataset from Roboflow
"""

import os
import shutil
import yaml
from pathlib import Path
import sys

def install_roboflow():
    """Install roboflow package"""
    print("📦 Installing Roboflow package...")
    try:
        import roboflow
        print("✅ Roboflow already installed")
        return True
    except ImportError:
        print("🔄 Installing roboflow...")
        os.system(f"{sys.executable} -m pip install roboflow")
        try:
            import roboflow
            print("✅ Roboflow installed successfully")
            return True
        except ImportError:
            print("❌ Failed to install Roboflow")
            return False

def download_dataset():
    """Download dataset from Roboflow"""
    print("📥 Downloading dataset from Roboflow...")
    
    try:
        from roboflow import Roboflow
        
        # Initialize Roboflow with your API key
        rf = Roboflow(api_key="v2vOzSYMN9tQtvyAnXmf")
        
        # Access the project
        project = rf.workspace("traffic-twinv").project("traffic-3bhpc")
        
        # Get version 9
        version = project.version(9)
        
        # Download dataset in YOLOv8 format
        print("⬇️ Downloading traffic dataset (this may take a while)...")
        dataset = version.download("yolov8")
        
        print(f"✅ Dataset downloaded to: {dataset.location}")
        return dataset.location
        
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return None

def analyze_dataset_classes(dataset_path):
    """Analyze the classes in the downloaded dataset"""
    print("🔍 Analyzing dataset classes...")
    
    # Look for data.yaml file
    data_yaml_path = None
    for root, dirs, files in os.walk(dataset_path):
        if 'data.yaml' in files:
            data_yaml_path = os.path.join(root, 'data.yaml')
            break
    
    if not data_yaml_path:
        print("⚠️ No data.yaml found in dataset")
        return None
    
    try:
        with open(data_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        print(f"📋 Dataset classes: {dataset_config.get('names', [])}")
        print(f"🔢 Number of classes: {dataset_config.get('nc', 0)}")
        
        return dataset_config
        
    except Exception as e:
        print(f"❌ Failed to analyze dataset: {e}")
        return None

def update_project_config(dataset_config, dataset_path):
    """Update the project data.yaml configuration"""
    print("⚙️ Updating project configuration...")
    
    # Find train and val directories
    train_dir = None
    val_dir = None
    
    for root, dirs, files in os.walk(dataset_path):
        if 'train' in dirs:
            train_dir = os.path.join(root, 'train')
        if 'valid' in dirs:  # Roboflow uses 'valid' instead of 'val'
            val_dir = os.path.join(root, 'valid')
    
    if not train_dir or not val_dir:
        print("❌ Could not find train/validation directories")
        return False
    
    # Update project data.yaml
    project_config_path = "configs/data.yaml"
    
    try:
        # Read current config
        with open(project_config_path, 'r') as f:
            current_config = yaml.safe_load(f)
        
        # Update paths to use Roboflow dataset
        current_config['train'] = os.path.relpath(train_dir, '.').replace('\\', '/')
        current_config['val'] = os.path.relpath(val_dir, '.').replace('\\', '/')
        
        # Update classes from dataset
        if dataset_config:
            current_config['nc'] = dataset_config.get('nc', current_config['nc'])
            current_config['names'] = dataset_config.get('names', current_config['names'])
        
        # Write updated config
        with open(project_config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
        
        print(f"✅ Updated {project_config_path}")
        print(f"   Train: {current_config['train']}")
        print(f"   Val: {current_config['val']}")
        print(f"   Classes: {current_config['names']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update project config: {e}")
        return False

def create_class_mapping(dataset_config):
    """Create class mapping for traffic analysis"""
    print("🗺️ Creating class mapping...")
    
    if not dataset_config:
        print("⚠️ No dataset config available")
        return
    
    dataset_classes = dataset_config.get('names', [])
    
    # Map dataset classes to our traffic classes
    traffic_classes = ["pedestrian", "vehicle", "ambulance"]
    class_mapping = {}
    
    for i, class_name in enumerate(dataset_classes):
        class_name_lower = class_name.lower()
        
        if 'person' in class_name_lower or 'pedestrian' in class_name_lower:
            class_mapping[i] = 0  # pedestrian
        elif 'car' in class_name_lower or 'vehicle' in class_name_lower or 'truck' in class_name_lower or 'bus' in class_name_lower or 'motorcycle' in class_name_lower:
            class_mapping[i] = 1  # vehicle
        elif 'ambulance' in class_name_lower or 'emergency' in class_name_lower:
            class_mapping[i] = 2  # ambulance
        else:
            # Map other classes to vehicle by default
            class_mapping[i] = 1
    
    # Save mapping to file
    mapping_file = "configs/class_mapping.yaml"
    try:
        with open(mapping_file, 'w') as f:
            yaml.dump({
                'dataset_classes': dict(enumerate(dataset_classes)),
                'traffic_classes': dict(enumerate(traffic_classes)),
                'mapping': class_mapping
            }, f, default_flow_style=False)
        
        print(f"✅ Class mapping saved to {mapping_file}")
        print(f"   Mapping: {class_mapping}")
        
    except Exception as e:
        print(f"❌ Failed to save class mapping: {e}")

def update_utils_for_mapping():
    """Update detection utils to use class mapping"""
    print("🔧 Updating detection utilities...")
    
    try:
        utils_file = "utils/detection_utils.py"
        
        # Read current utils file
        with open(utils_file, 'r') as f:
            content = f.read()
        
        # Check if mapping function already exists
        if 'def map_dataset_to_traffic_classes' in content:
            print("✅ Detection utils already updated")
            return
        
        # Add mapping function to the file
        mapping_function = '''
    @staticmethod
    def map_dataset_to_traffic_classes(dataset_class_id: int, class_mapping: dict) -> int:
        """
        Map dataset class IDs to traffic analysis classes using class mapping
        
        Args:
            dataset_class_id: Dataset class ID
            class_mapping: Class mapping dictionary
            
        Returns:
            Traffic class ID (0=pedestrian, 1=vehicle, 2=ambulance)
        """
        return class_mapping.get(dataset_class_id, 1)  # Default to vehicle
'''
        
        # Find the end of the class and add the function
        class_end = content.rfind('    @staticmethod')
        if class_end == -1:
            class_end = content.rfind('def ')
        
        if class_end != -1:
            # Insert before the last @staticmethod or function
            insert_pos = content.rfind('\n    @staticmethod', 0, class_end)
            if insert_pos == -1:
                insert_pos = content.rfind('\n    def ', 0, class_end)
            
            if insert_pos != -1:
                updated_content = content[:insert_pos] + mapping_function + '\n' + content[insert_pos:]
                
                # Write updated file
                with open(utils_file, 'w') as f:
                    f.write(updated_content)
                
                print("✅ Detection utilities updated")
            else:
                print("⚠️ Could not find insertion point in detection utils")
        else:
            print("⚠️ Could not update detection utilities")
            
    except Exception as e:
        print(f"❌ Failed to update detection utilities: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up Roboflow dataset for Traffic Analysis CV Engine")
    print("=" * 60)
    
    # Step 1: Install Roboflow
    if not install_roboflow():
        return False
    
    # Step 2: Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        return False
    
    # Step 3: Analyze dataset
    dataset_config = analyze_dataset_classes(dataset_path)
    
    # Step 4: Update project configuration
    if not update_project_config(dataset_config, dataset_path):
        return False
    
    # Step 5: Create class mapping
    create_class_mapping(dataset_config)
    
    # Step 6: Update utilities
    update_utils_for_mapping()
    
    print("\n" + "=" * 60)
    print("✅ Dataset setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review the updated configs/data.yaml file")
    print("2. Check configs/class_mapping.yaml for class assignments")
    print("3. Run training: python training/train_enhanced.py")
    print("4. Test detection: python inference/detect_enhanced.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Dataset setup failed. Please check the error messages above.")
        exit(1)
    else:
        exit(0)
