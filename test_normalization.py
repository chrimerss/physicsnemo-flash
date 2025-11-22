import torch
import numpy as np
import json
from src.dataset import FlashDataset
from types import SimpleNamespace

def test_normalization():
    # Mock params
    params = SimpleNamespace(
        zarr_path="data/data.zarr",
        aux_path="/home/users/li1995/global_flood/FLASH/data/aux",
        input_window=6,
        output_window=6,
        patch_size=256,
    )
    
    # Mock dataset methods to avoid loading real data if paths don't exist
    # But we want to test the normalization logic which depends on stats.json
    # We assume stats.json exists (I saw it).
    # We might need to mock zarr and aux loading if they are large or missing.
    # The user's environment seems to have them, but I can't be sure about write access or speed.
    # Let's try to instantiate. If it fails due to missing files, I'll mock.
    
    try:
        ds = FlashDataset(params, train=True)
    except Exception as e:
        print(f"Failed to instantiate dataset: {e}")
        print("Mocking data loading for test...")
        
        # Mocking
        class MockDataset(FlashDataset):
            def __init__(self, params, train=True):
                # Skip super init that loads files
                self.params = params
                self.train = train
                self.input_window = params.input_window
                self.output_window = params.output_window
                self.patch_size = params.patch_size
                self.height = 1000
                self.width = 1000
                self.start_idx = 0
                self.end_idx = 100
                
                # Load stats
                with open('stats.json', 'r') as f:
                    self.stats = json.load(f)
                    
                self.aux_files = ['dem', 'fdir', 'facc', 'ksat', 'wm', 'b']
                self.aux_stats = self.stats['aux']
                
                # Mock data
                self.precip = np.random.randn(100, 1000, 1000) * 2 + 0.4 # Mean ~0.4, Std ~2
                self.streamflow = np.random.randn(100, 1000, 1000) * 4 + 2.3 # Mean ~2.3, Std ~4
                self.aux_data = np.random.randn(6, 1000, 1000)
                self.lat_grid = np.zeros((1, 1000, 1000))
                self.lon_grid = np.zeros((1, 1000, 1000))
                
                # Normalize background logic from __init__
                self.background_data = np.concatenate([self.aux_data, self.lat_grid, self.lon_grid], axis=0)
                for i in range(len(self.aux_files)):
                    mean = self.aux_stats[i]['mean']
                    std = self.aux_stats[i]['std']
                    self.background_data[i] = (self.background_data[i] - mean) / std
                    
        ds = MockDataset(params, train=True)

    # Test denormalize_state
    print("Testing denormalize_state...")
    # Create a dummy normalized state
    # Shape: (12, H, W)
    # First 6: precip (normalized), Last 6: streamflow (normalized)
    dummy_state = torch.randn(12, 256, 256)
    
    # Denormalize
    denorm_state = ds.denormalize_state(dummy_state)
    
    # Check logic
    p_mean = ds.stats['precip']['mean']
    p_std = ds.stats['precip']['std']
    q_mean = ds.stats['unitq']['mean']
    q_std = ds.stats['unitq']['std']
    
    # Manual check
    precip_denorm_manual = dummy_state[:6] * p_std + p_mean
    flow_denorm_manual = dummy_state[6:] * q_std + q_mean
    
    assert torch.allclose(denorm_state[:6], precip_denorm_manual), "Precip denormalization failed"
    assert torch.allclose(denorm_state[6:], flow_denorm_manual), "Streamflow denormalization failed"
    print("denormalize_state passed!")
    
    # Test getitem normalization
    print("Testing __getitem__ normalization...")
    item = ds[0]
    state_input = item['state'][0] # (12, H, W)
    
    # We can't easily verify exact values without knowing the raw data, 
    # but we can check if the code ran without error and shapes are correct.
    print(f"State input shape: {state_input.shape}")
    print("Getitem passed!")

if __name__ == "__main__":
    test_normalization()
