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
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from physicsnemo.models import Module
from src.dataset import FlashDataset
from src.vis import create_video
import wandb

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run inference and generate visualizations"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    # Determine if regression or diffusion
    # For inference, we usually want the full pipeline if diffusion
    # But let's assume we are testing the model specified in config
    
    model_path = cfg.training.initial_weights
    if not model_path:
        print("No model weights provided in training.initial_weights!")
        return

    print(f"Loading model from {model_path}")
    net = Module.from_checkpoint(model_path)
    net = net.eval().to(device)
    
    # Load dataset (Validation or Test)
    dataset = FlashDataset(cfg.dataset, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Run inference on a few samples
    print("Running inference...")
    
    # Setup wandb for logging results
    wandb.init(project="FLASH_Inference", name="inference_run")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5: break # Limit to 5 samples
            
            background = batch['background'].to(device)
            state = batch['state']
            state_input = state[0].to(device)
            state_target = state[1].to(device)
            
            # Construct condition (Simplified, match training logic)
            # Assuming regression for now or simple forward
            # If diffusion, need sampler.
            
            # For this script, let's assume simple forward pass of the loaded model
            # If it's regression:
            condition = torch.cat([state_input, background], dim=1)
            pred = net(condition)
            
            # If diffusion, we need the regression output first...
            # This inference script is a placeholder for the full pipeline.
            
            # Visualize
            # Pred: (1, 6, H, W)
            # Target: (1, 6, H, W)
            
            # Create video for streamflow
            # Assuming channels are streamflow t+1 to t+6
            
            pred_np = pred[0].cpu().numpy()
            target_np = state_target[0].cpu().numpy()
            
            video = create_video(pred_np, target_np, field_name="Streamflow")
            wandb.log({f"sample_{i}": video})
            
    print("Inference done.")

if __name__ == "__main__":
    main()
