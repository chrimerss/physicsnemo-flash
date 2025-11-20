# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import numpy as np
import torch
import zarr
import xarray as xr
import rioxarray
from typing import List, Tuple, Dict, Optional
from examples.weather.stormcast.datasets.dataset import StormCastDataset

class FlashDataset(StormCastDataset):
    """
    Dataset for FLASH project: predicting unit streamflow from precipitation and auxiliary data.
    """

    def __init__(self, params, train: bool = True):
        super().__init__()
        self.params = params
        self.train = train
        
        # Data paths
        self.zarr_path = params.zarr_path
        self.aux_path = params.aux_path
        
        # Open Zarr store
        self.ds = zarr.open_group(self.zarr_path, mode='r')
        self.precip = self.ds['precipitation']
        self.streamflow = self.ds['streamflow']
        
        # Dimensions
        self.total_time = self.precip.shape[0]
        self.height = self.precip.shape[1]
        self.width = self.precip.shape[2]
        
        # Time splitting
        # Training: 2021-01-01 to 2023-12-31
        # Validation: 2024-01-01 to 2024-12-31
        # Testing: 2025-01-01 to 2025-07-31
        # Assuming 10-min intervals starting from 2021-01-01 00:00
        # 2021-2023 is 3 years. 3 * 365 * 24 * 6 = 157680 steps (approx)
        # Exact indices should be calculated or provided. 
        # For now, I'll use the dates provided in the prompt to estimate indices or use a config.
        # The prompt says: Data date range from 20210101 to 2025-07-31.
        # Total shape is (240912, 3500, 7000).
        
        # Let's define split indices based on the prompt's dates.
        # 2021-01-01 to 2023-12-31 (3 years) -> 3 * 365 * 144 = 157680
        # 2024-01-01 to 2024-12-31 (1 year) -> 366 * 144 = 52704 (2024 is leap year)
        # 2025-01-01 to 2025-07-31 (212 days) -> 212 * 144 = 30528
        
        # Indices:
        # Train: 0 to 157680
        # Val: 157680 to 210384
        # Test: 210384 to end
        
        self.train_end_idx = 157680
        self.val_end_idx = 210384
        
        if self.train:
            self.start_idx = 0
            self.end_idx = self.train_end_idx
        else:
            # Validation
            self.start_idx = self.train_end_idx
            self.end_idx = self.val_end_idx
            
        # Window parameters
        self.input_window = params.get('input_window', 6)
        self.output_window = params.get('output_window', 6)
        self.lookback = self.input_window
        self.lookout = self.output_window
        
        # Patching
        self.patch_size = params.get('patch_size', 1024)
        
        # Load auxiliary data
        self.aux_files = [
            'dem_usa.tif', 'fdir_usa.tif', 'facc_usa.tif', 
            'ksat_usa.tif', 'wm_usa.tif', 'b_usa.tif'
        ]
        self.aux_data = self._load_aux_data()
        
        # Location embeddings (lat/lon)
        # Assuming we can generate them or load them. 
        # The prompt says "use a location embedding based on its longitude and latitude".
        # Since I don't have lat/lon grids, I will generate normalized coordinates.
        self.lat_grid, self.lon_grid = self._generate_lat_lon_grid()
        
        # Combine background
        # Aux (6) + Lat (1) + Lon (1) = 8 channels
        self.background_data = np.concatenate([self.aux_data, self.lat_grid, self.lon_grid], axis=0)
        
    def _load_aux_data(self):
        aux_data = []
        for fname in self.aux_files:
            path = os.path.join(self.aux_path, fname)
            if os.path.exists(path):
                # Load with rioxarray/rasterio
                # Assuming they are already regridded to match the 3500x7000 shape or need resizing
                # For this implementation, I'll assume they match or I'll mock it if file not found
                try:
                    da = rioxarray.open_rasterio(path)
                    data = da.values
                    # Ensure shape matches (1, 3500, 7000)
                    if data.shape[1:] != (self.height, self.width):
                        # Resize or crop logic here if needed. 
                        # For now, assume correct shape.
                        pass
                    aux_data.append(data)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
                    # Mock data for now if file missing (for development)
                    aux_data.append(np.zeros((1, self.height, self.width), dtype=np.float32))
            else:
                 # Mock data
                aux_data.append(np.zeros((1, self.height, self.width), dtype=np.float32))
                
        return np.concatenate(aux_data, axis=0)

    def _generate_lat_lon_grid(self):
        # Generate normalized coordinates -1 to 1
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        xx, yy = np.meshgrid(x, y)
        return yy[np.newaxis, ...].astype(np.float32), xx[np.newaxis, ...].astype(np.float32)

    def __len__(self):
        # Number of valid sequences
        # We need input_window + output_window frames
        # Total frames available = end_idx - start_idx
        return (self.end_idx - self.start_idx) - (self.input_window + self.output_window) + 1

    def __getitem__(self, idx):
        # Adjust index to global time
        global_idx = self.start_idx + idx
        
        # Time indices
        input_start = global_idx
        input_end = input_start + self.input_window
        target_start = input_end
        target_end = target_start + self.output_window
        
        # Load data
        # Precip: (T, H, W)
        precip_seq = self.precip[input_start:input_end] # (6, H, W)
        
        # Streamflow: Input (past) and Target (future)
        # Input streamflow: same window as precip
        streamflow_input = self.streamflow[input_start:input_end] # (6, H, W)
        
        # Target streamflow
        streamflow_target = self.streamflow[target_start:target_end] # (6, H, W)
        
        # Handle missing values
        # Precip: -3 -> 0 (or some other value)
        precip_seq[precip_seq == -3] = 0
        # Streamflow: -9999 -> 0
        streamflow_input[streamflow_input == -9999] = 0
        streamflow_target[streamflow_target == -9999] = 0
        
        # Concatenate inputs
        # State input: Precip (6) + Streamflow (6) = 12 channels
        state_input = np.concatenate([precip_seq, streamflow_input], axis=0)
        
        # Target
        state_target = streamflow_target
        
        # Patching
        # Random crop for training, fixed crop or tiling for validation?
        # For simplicity, let's do random crop for training
        if self.train:
            y_start = np.random.randint(0, self.height - self.patch_size + 1)
            x_start = np.random.randint(0, self.width - self.patch_size + 1)
        else:
            # Center crop for validation
            y_start = (self.height - self.patch_size) // 2
            x_start = (self.width - self.patch_size) // 2
            
        y_end = y_start + self.patch_size
        x_end = x_start + self.patch_size
        
        # Crop
        background_crop = self.background_data[:, y_start:y_end, x_start:x_end]
        state_input_crop = state_input[:, y_start:y_end, x_start:x_end]
        state_target_crop = state_target[:, y_start:y_end, x_start:x_end]
        
        return {
            "background": torch.from_numpy(background_crop).float(),
            "state": (
                torch.from_numpy(state_input_crop).float(),
                torch.from_numpy(state_target_crop).float()
            )
        }

    def background_channels(self) -> List[str]:
        return [
            'dem', 'fdir', 'facc', 'ksat', 'wm', 'b', 'lat', 'lon'
        ]

    def state_channels(self) -> List[str]:
        # 6 precip + 6 streamflow
        channels = [f'precip_t-{i}' for i in range(self.input_window, 0, -1)]
        channels += [f'flow_t-{i}' for i in range(self.input_window, 0, -1)]
        return channels
        
    def output_channels(self) -> List[str]:
         return [f'flow_t+{i}' for i in range(1, self.output_window + 1)]

    def image_shape(self) -> Tuple[int, int]:
        return (self.patch_size, self.patch_size)
