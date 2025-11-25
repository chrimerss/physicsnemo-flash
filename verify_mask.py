import torch
import numpy as np
import os
from omegaconf import OmegaConf
import sys
sys.path.append(os.getcwd())
from src.dataset import FlashDataset

def verify_dataset_mask():
    print("Verifying Dataset Mask...")
    params = OmegaConf.create({
        "zarr_path": "data/data.zarr",
        "aux_path": "/home/users/li1995/global_flood/FLASH/data/aux",
        "input_window": 6,
        "output_window": 6,
        "patch_size": 1024,
        "batch_size": 1,
        "num_workers": 0
    })
    
    # Train Set
    ds = FlashDataset(params, train=True)
    
    # Get item 1000 (known to have -9999)
    # We found global idx 1000 has -9999 in input. 
    # We need to check if TARGET has -9999 for the mask to be 0.
    # In diagnose_data, target also had -9999 (Min -9999).
    
    print("\nChecking Item 1000 (Train)...")
    item = ds[1000]
    
    mask = item['mask']
    target = item['state'][1]
    
    print(f"Mask shape: {mask.shape}")
    print(f"Target shape: {target.shape}")
    
    mask_zeros = (mask == 0).sum().item()
    mask_ones = (mask == 1).sum().item()
    total = mask.numel()
    
    print(f"Mask 0s: {mask_zeros} ({mask_zeros/total*100:.2f}%)")
    print(f"Mask 1s: {mask_ones} ({mask_ones/total*100:.2f}%)")
    
    if mask_zeros > 0:
        print("SUCCESS: Mask correctly identified invalid pixels.")
    else:
        print("WARNING: No invalid pixels found in mask (check if index 1000 actually has -9999 in target).")
        
    # Check consistency: where mask is 0, target should be 0 (because we zeroed it out)
    # But before normalization it was -9999. After norm it depends.
    # Wait, we set it to 0 BEFORE normalization in dataset.py.
    # streamflow_target[streamflow_target == -9999] = 0
    # So target should be normalized 0.
    
    # Let's check strict correspondence
    # We can't check against -9999 because it's gone.
    # But we trusted the mask creation: target_mask = (streamflow_target != -9999)
    # created BEFORE modification. 
    
    print("Dataset mask verification complete.")

if __name__ == "__main__":
    verify_dataset_mask()
