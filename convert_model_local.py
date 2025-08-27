#!/usr/bin/env python3
"""
Local model conversion script for YoloV5 to DepthAI blob format
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import ultralytics
        print("✓ PyTorch and Ultralytics found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def convert_pt_to_onnx(pt_path, output_dir=None, img_size=416):
    """Convert PyTorch .pt model to ONNX format"""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Model file not found: {pt_path}")
    
    if output_dir is None:
        output_dir = pt_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {pt_path} to ONNX...")
    
    try:
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(str(pt_path))
        
        # Export to ONNX
        onnx_path = model.export(
            format='onnx', 
            imgsz=img_size,
            simplify=True,
            opset=11
        )
        
        print(f"✓ ONNX conversion successful: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"✗ ONNX conversion failed: {e}")
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
        return None
    except Exception as e:
        print(f"✗ Blob conversion failed: {e}")
        return None

def get_class_names_from_yaml(yaml_path):
    """Extract class names from YAML file"""
    import yaml
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'names' in data:
            return data['names']
        else:
            print("Warning: No 'names' field found in YAML")
            return []
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return []

def main():
    """Main conversion function"""
    # Configuration
    model_dir = "/Users/geordihills/Downloads/content/yolov5"
    pt_file = "runs/train/exp/weights/best.pt"
    yaml_file = "runs/train/exp/hyp.yaml"  # or data.yaml if available
    
    # Full paths
    pt_path = os.path.join(model_dir, pt_file)
    yaml_path = os.path.join(model_dir, yaml_file)
    output_dir = os.path.join(model_dir, "converted_models")
    
    print("=== YoloV5 to DepthAI Blob Converter ===")
    print(f"Model path: {pt_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if model exists
    if not os.path.exists(pt_path):
        print(f"✗ Model file not found: {pt_path}")
        print("Please update the model_dir and pt_file variables in this script")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies:")
        print("pip install torch torchvision ultralytics blobconverter")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Convert to ONNX
    onnx_path = convert_pt_to_onnx(pt_path, output_dir)
    if not onnx_path:
        return
    
    # Step 2: Convert to blob
    blob_path = convert_with_blobconverter(onnx_path, output_dir)
    if not blob_path:
        print("Blob conversion failed. You can try uploading the ONNX file to:")
        print("https://blobconverter.luxonis.com/")
        return
    
    # Step 3: Get class information
    if os.path.exists(yaml_path):
        class_names = get_class_names_from_yaml(yaml_path)
        if class_names:
            print(f"\nClass names found: {class_names}")
            
            # Save class names to a text file
            class_file = os.path.join(output_dir, "class_names.txt")
            with open(class_file, 'w') as f:
                for name in class_names:
                    f.write(f"{name}\n")
            print(f"Class names saved to: {class_file}")
    
    print("\n=== Conversion Complete! ===")
    print(f"ONNX file: {onnx_path}")
    print(f"Blob file: {blob_path}")
    print(f"\nYou can now use the blob file with DepthAI!")

if __name__ == "__main__":
    main()
