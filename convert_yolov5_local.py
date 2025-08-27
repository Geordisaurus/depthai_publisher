#!/usr/bin/env python3
"""
YOLOv5-specific model conversion script for DepthAI blob format
Compatible with original YOLOv5 repository
"""

import os
import sys
import subprocess
from pathlib import Path
import torch

def check_yolov5_dependencies():
    """Check if YOLOv5 dependencies are installed"""
    try:
        import torch
        import torchvision
        print("✓ PyTorch and Torchvision found")
        
        # Check if we have YOLOv5 models module
        sys.path.append('.')  # Add current directory to path
        try:
            from models.common import DetectMultiBackend
            from utils.general import check_requirements
            print("✓ YOLOv5 modules found")
            return True
        except ImportError as e:
            print(f"✗ YOLOv5 modules not found: {e}")
            print("Make sure you're running this script from the YOLOv5 repository directory")
            return False
            
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def convert_with_export_script(pt_path, img_size=416):
    """Use YOLOv5's built-in export.py script"""
    try:
        # Check if export.py exists
        if not os.path.exists('export.py'):
            print("✗ export.py not found. Make sure you're in the YOLOv5 repository directory")
            return None
        
        print(f"Converting {pt_path} to ONNX using YOLOv5's export.py...")
        
        cmd = [
            'python', 'export.py',
            '--weights', str(pt_path),
            '--img', str(img_size),
            '--batch', '1',
            '--device', 'cpu',
            '--include', 'onnx',
            '--simplify'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Find the generated ONNX file
            pt_path_obj = Path(pt_path)
            onnx_path = pt_path_obj.with_suffix('.onnx')
            
            if onnx_path.exists():
                print(f"✓ ONNX conversion successful: {onnx_path}")
                return str(onnx_path)
            else:
                print("✗ ONNX file not found after conversion")
                return None
        else:
            print(f"✗ Export script failed:")
            print(result.stderr)
            return None
            
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return None

def convert_with_blobconverter(onnx_path, output_dir=None):
    """Convert ONNX to blob using blobconverter"""
    try:
        import blobconverter
        
        if output_dir is None:
            output_dir = Path(onnx_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting {onnx_path} to blob...")
        
        blob_path = blobconverter.from_onnx(
            model_path=str(onnx_path),
            data_type="FP16",
            shaves=6,
            use_cache=False,
            output_dir=str(output_dir)
        )
        
        print(f"✓ Blob conversion successful: {blob_path}")
        return blob_path
        
    except ImportError:
        print("✗ blobconverter not installed. Install with: pip install blobconverter")
        print("Alternatively, upload the ONNX file to: https://blobconverter.luxonis.com/")
        return None
    except Exception as e:
        print(f"✗ Blob conversion failed: {e}")
        print("You can try uploading the ONNX file to: https://blobconverter.luxonis.com/")
        return None

def find_class_names():
    """Try to find class names from various sources"""
    class_names = []
    
    # Look for data.yaml files
    yaml_files = [
        'data.yaml',
        'runs/train/exp/data.yaml',
        'runs/train/exp/opt.yaml'
    ]
    
    for yaml_file in yaml_files:
        if os.path.exists(yaml_file):
            try:
                import yaml
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if 'names' in data:
                    class_names = data['names']
                    print(f"✓ Found class names in {yaml_file}: {class_names}")
                    return class_names
                elif 'nc' in data:
                    nc = data['nc']
                    print(f"Found {nc} classes in {yaml_file}")
                    
            except Exception as e:
                print(f"Error reading {yaml_file}: {e}")
    
    # If no YAML found, check if this is the fire/smoke model from the notebook
    print("No class names found in YAML files.")
    print("If this is the fire/smoke detection model from the notebook, classes are likely: ['fire', 'smoke']")
    
    return class_names

def main():
    """Main conversion function"""
    print("=== YOLOv5 to DepthAI Blob Converter ===")
    
    # Configuration
    pt_file = "runs/train/exp/weights/best.pt"
    output_dir = "converted_models"
    
    print(f"Model file: {pt_file}")
    print(f"Output directory: {output_dir}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check if we're in the right directory
    if not os.path.exists('export.py'):
        print("✗ export.py not found.")
        print("Please run this script from the YOLOv5 repository directory")
        print("Current directory should contain: export.py, models/, utils/, etc.")
        return
    
    # Check if model exists
    if not os.path.exists(pt_file):
        print(f"✗ Model file not found: {pt_file}")
        print("Please make sure the model path is correct")
        return
    
    # Check dependencies
    if not check_yolov5_dependencies():
        print("Please install YOLOv5 dependencies:")
        print("pip install -r requirements.txt")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Convert to ONNX using YOLOv5's export script
    onnx_path = convert_with_export_script(pt_file)
    if not onnx_path:
        return
    
    # Step 2: Convert to blob
    blob_path = convert_with_blobconverter(onnx_path, output_dir)
    if blob_path:
        print(f"✓ Blob file created: {blob_path}")
    else:
        print(f"ONNX file available for manual upload: {onnx_path}")
        print("Upload it to: https://blobconverter.luxonis.com/")
    
    # Step 3: Find class information
    class_names = find_class_names()
    if class_names:
        # Save class names to a text file
        class_file = os.path.join(output_dir, "class_names.txt")
        with open(class_file, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        print(f"Class names saved to: {class_file}")
    
    print("\n=== Conversion Complete! ===")
    if blob_path:
        print(f"Your model is ready to use: {blob_path}")
    print(f"ONNX file: {onnx_path}")
    print("\nUpdate your detection script with the correct paths and class names!")

if __name__ == "__main__":
    main()
