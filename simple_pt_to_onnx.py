#!/usr/bin/env python3
"""
Simple PyTorch to ONNX converter for YOLOv5 models
Works without the full YOLOv5 repository
"""

import torch
import os
from pathlib import Path

def convert_pt_to_onnx_simple(pt_path, img_size=416):
    """Convert PyTorch model to ONNX using torch.onnx.export"""
    
    print(f"Loading model from: {pt_path}")
    
    try:
        # Load the model
        device = torch.device('cpu')
        model = torch.load(pt_path, map_location=device)
        
        # If model is a dictionary (training checkpoint), get the model
        if isinstance(model, dict):
            model = model['model'].float().eval()
        else:
            model = model.float().eval()
        
        print(f"Model loaded successfully")
        print(f"Model type: {type(model)}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
        
        # Output path
        pt_path_obj = Path(pt_path)
        onnx_path = pt_path_obj.with_suffix('.onnx')
        
        print(f"Converting to ONNX: {onnx_path}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✓ ONNX conversion successful: {onnx_path}")
        return str(onnx_path)
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        print("This might require the full YOLOv5 repository for complex models")
        return None

def convert_to_blob(onnx_path):
    """Convert ONNX to blob using blobconverter"""
    try:
        import blobconverter
        
        output_dir = Path(onnx_path).parent / "converted_models"
        output_dir.mkdir(exist_ok=True)
        
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
        print(f"Or upload {onnx_path} to: https://blobconverter.luxonis.com/")
        return None
    except Exception as e:
        print(f"✗ Blob conversion failed: {e}")
        print(f"You can upload {onnx_path} to: https://blobconverter.luxonis.com/")
        return None

def main():
    """Main function"""
    # Update this path to your model
    pt_path = "/Users/geordihills/Downloads/content/yolov5/runs/train/exp/weights/best.pt"
    
    print("=== Simple PyTorch to ONNX Converter ===")
    print(f"Model path: {pt_path}")
    
    if not os.path.exists(pt_path):
        print(f"✗ Model not found: {pt_path}")
        print("Please update the pt_path variable in this script")
        return
    
    # Convert to ONNX
    onnx_path = convert_pt_to_onnx_simple(pt_path)
    if not onnx_path:
        return
    
    # Convert to blob
    blob_path = convert_to_blob(onnx_path)
    
    print("\n=== Conversion Results ===")
    print(f"ONNX file: {onnx_path}")
    if blob_path:
        print(f"Blob file: {blob_path}")
        print("✓ Ready to use with DepthAI!")
    else:
        print("Upload the ONNX file manually to get the blob")
    
    # Print class info for fire/smoke detection
    print("\nFor fire/smoke detection, your classes are likely:")
    print("CLASS_NAMES = ['fire', 'smoke']")

if __name__ == "__main__":
    main()
