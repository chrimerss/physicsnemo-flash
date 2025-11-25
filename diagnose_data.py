import zarr
import numpy as np
import os
import torch
from omegaconf import OmegaConf
import sys

# Add src to path if needed
sys.path.append(os.getcwd())

from src.dataset import FlashDataset

def check_dataset_debug():
    print("\nChecking Dataset with Debug Prints...")
    # Mock params
    params = OmegaConf.create({
        "zarr_path": "data/data.zarr",
        "aux_path": "/home/users/li1995/global_flood/FLASH/data/aux",
        "input_window": 6,
        "output_window": 6,
        "patch_size": 1024,
        "batch_size": 1,
        "num_workers": 0
    })
    
    try:
        # Train set check (Global 1000)
        print("\n--- Training Set Check (Global 1000) ---")
        dataset_train = FlashDataset(params, train=True)
        # Global 1000 -> idx 1000
        _ = dataset_train[1000]
        
        # Val set check (Global 160000)
        print("\n--- Validation Set Check (Global 160000) ---")
        dataset_val = FlashDataset(params, train=False)
        
        # Global 160000. Val starts at 157680.
        # Idx = 160000 - 157680 = 2320
        val_idx = 160000 - 157680
        if val_idx < len(dataset_val):
            _ = dataset_val[val_idx]
        else:
            print(f"Index {val_idx} out of bounds for val set (len {len(dataset_val)})")

    except Exception as e:
        print(f"Error in Dataset check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_dataset_debug()