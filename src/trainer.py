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

import hydra
import os
import time
import numpy as np
from omegaconf import OmegaConf
import torch
import psutil
import wandb
from physicsnemo.models import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import EDMLoss, EDMLossLogUniform
from physicsnemo.utils.diffusion import InfiniteSampler
from physicsnemo.launch.utils import save_checkpoint, load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from torch.nn.utils import clip_grad_norm_

# Import from local src
from src.dataset import FlashDataset, worker_init
from src.vis import validation_plot, create_video

# Import from stormcast utils if possible, otherwise we need to copy them
# Assuming we can import from examples.weather.stormcast.utils
# If not, we will need to copy utils.nn
try:
    from utils.nn import (
        diffusion_model_forward,
        regression_loss_fn,
        get_preconditioned_architecture,
        build_network_condition_and_target,
        unpack_batch,
    )
except ImportError:
    # Fallback or error. For now assume it works or I will need to copy it.
    # To make it work, I might need to add the project root to python path.
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
    from utils.nn import (
        diffusion_model_forward,
        regression_loss_fn,
        get_preconditioned_architecture,
        build_network_condition_and_target,
        unpack_batch,
    )

logger = PythonLogger("train")

def training_loop(cfg):
    # Initialize.
    start_time = time.time()
    dist = DistributedManager()
    device = dist.device
    logger0 = RankZeroLoggingWrapper(logger, dist)

    # Shorthand for config items
    batch_size = cfg.dataset.batch_size
    # Assuming batch_size is total batch size or per gpu?
    # In stormcast it was cfg.training.batch_size. 
    # In my config I put it in dataset. Let's use cfg.dataset.batch_size as per GPU for simplicity or handle it.
    # Stormcast logic:
    # batch_size = cfg.training.batch_size
    # local_batch_size = batch_size // dist.world_size
    
    # Let's assume cfg.dataset.batch_size is PER GPU
    local_batch_size = cfg.dataset.batch_size
    batch_size = local_batch_size * dist.world_size
    num_accumulation_rounds = 1 # Simplify for now

    log_to_wandb = cfg.training.log_to_wandb

    loss_type = cfg.training.loss
    # In my config loss is a string "regression" or "edm"
    # In stormcast it was a dict or string?
    # Stormcast: loss.type
    # My config: loss: "regression"
    if isinstance(loss_type, str):
        loss_type_name = loss_type
    else:
        loss_type_name = loss_type.type

    if loss_type_name == "regression":
        net_name = "regression"
    elif loss_type_name == "edm":
        net_name = "diffusion"

    # Conditions
    # I need to define these in config or hardcode for FLASH
    # FLASH: 
    # Phase 1 (Regression): Input: State(t), Background. Output: State(t+1)
    # Phase 2 (Diffusion): Input: State(t), Background, RegressionOutput(t+1). Output: Residual(t+1)
    
    # Stormcast uses condition_list from config.
    # I should update my config to include these lists or define them here.
    # Let's update config later, for now default them.
    if net_name == "regression":
        condition_list = ["state", "background"]
    else:
        condition_list = ["state", "background", "regression"]

    # Seed
    np.random.seed((cfg.training.seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(cfg.training.seed)
    
    total_train_steps = cfg.training.total_train_steps

    # Load dataset.
    logger0.info("Loading dataset...")
    dataset_train = FlashDataset(cfg.dataset, train=True)
    dataset_valid = FlashDataset(cfg.dataset, train=False)

    background_channels = dataset_train.background_channels()
    state_channels = dataset_train.state_channels()
    lead_time_steps = 0 # FlashDataset doesn't use lead time embeddings yet

    sampler = InfiniteSampler(
        dataset=dataset_train,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    valid_sampler = InfiniteSampler(
        dataset=dataset_valid,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        num_workers=cfg.dataset.num_workers,
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=False,
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=local_batch_size,
        num_workers=cfg.dataset.num_workers,
        sampler=valid_sampler,
        drop_last=True,
        pin_memory=False,
    )

    dataset_iterator = iter(data_loader)
    valid_dataset_iterator = iter(valid_data_loader)

    # load pretrained regression net if training diffusion
    if "regression" in condition_list:
        if cfg.model.regression_weights:
            regression_net = Module.from_checkpoint(cfg.model.regression_weights)
            regression_net = regression_net.eval().requires_grad_(False).to(device)
        else:
            regression_net = None
            if net_name == "diffusion":
                logger0.warning("Training diffusion but no regression weights provided!")
    else:
        regression_net = None

    invariant_tensor = None # Not using invariants for now

    # Construct network
    logger0.info("Constructing network...")
    # Calculate channels
    # State: 12 (6 precip + 6 flow)
    # Background: 8
    # Regression: 6 (output flow)
    
    num_condition_channels = 0
    if "state" in condition_list:
        num_condition_channels += len(state_channels)
    if "background" in condition_list:
        num_condition_channels += len(background_channels)
    if "regression" in condition_list:
        num_condition_channels += 6 # Output channels
        
    logger0.info(f"num_condition_channels {num_condition_channels}")

    # Using StormCastUNet
    # I need to make sure get_preconditioned_architecture works or instantiate directly
    # StormCastUNet signature:
    # in_channels, out_channels, ...
    # But get_preconditioned_architecture wraps it.
    # Let's instantiate directly from config target if possible, or use the util.
    # The config has _target_: physicsnemo.models.stormcast.StormCastUNet
    
    # If using get_preconditioned_architecture, it expects specific args.
    # Let's try to instantiate via hydra if possible, or adapt.
    # For simplicity, I'll use hydra.utils.instantiate(cfg.net) but I need to set channels.
    
    # Update cfg.net params
    cfg.model.net.img_in_channels = num_condition_channels
    # Out channels is 6 (streamflow)
    
    # Convert to primitives to avoid ListConfig serialization issues in checkpointing
    net_cfg = OmegaConf.to_container(cfg.model.net, resolve=True)
    print(f"DEBUG: net_cfg type: {type(net_cfg)}")
    print(f"DEBUG: img_resolution type: {type(net_cfg.get('img_resolution'))}")
    net = hydra.utils.instantiate(net_cfg)
    
    # Sanitize _args in case physicsnemo captured something weird or default args are ListConfig
    if hasattr(net, "_args") and isinstance(net._args, dict):
        print("DEBUG: Sanitizing net._args...")
        try:
            # Round-trip through OmegaConf to ensure all ListConfigs are converted to lists
            # We wrap in DictConfig first to handle nested structures easily
            temp_conf = OmegaConf.create(net._args)
            net._args = OmegaConf.to_container(temp_conf, resolve=True)
            print("DEBUG: net._args sanitized.")
        except Exception as e:
            print(f"DEBUG: Failed to sanitize net._args: {e}")
    net.train().requires_grad_(True).to(device)

    # Setup loss function.
    logger0.info("Setting up loss function...")

    if loss_type_name == "regression":
        loss_fn = regression_loss_fn
    elif loss_type_name == "edm":
        # EDM Loss
        loss_fn = EDMLoss(sigma_data=0.5) # Simplify

    # Setup optimizer
    logger0.info("Setting up optimizer...")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)
    
    # Resume training
    ckpt_path = os.path.join(cfg.training.rundir, f"checkpoints_{net_name}")
    total_steps = load_checkpoint(
        path=ckpt_path,
        models=net,
        optimizer=optimizer,
        epoch=None
    )

    # Train loop
    logger0.info(f"Starting training...")
    done = total_steps >= total_train_steps
    
    while not done:
        optimizer.zero_grad()
        
        batch = next(dataset_iterator)
        # unpack_batch expects dict with 'background', 'state', 'lead_time_label'
        # My dataset returns this dict.
        background = batch['background'].to(device)
        state = batch['state']
        state_input = state[0].to(device)
        state_target = state[1].to(device)
        lead_time_label = None
        
        # Build condition
        # Condition is concatenation of inputs
        # Order: State, Background, Regression
        # Note: StormCastUNet expects specific channel order matching training?
        # Usually it concatenates along dim 1.
        
        cond_tensors = []
        if "state" in condition_list:
            cond_tensors.append(state_input)
        if "background" in condition_list:
            cond_tensors.append(background)
        if "regression" in condition_list:
             # Run regression net
             with torch.no_grad():
                 # Regression net input: State + Background
                 # Assuming regression net was trained with [state, background]
                 reg_input = torch.cat([state_input, background], dim=1)
                 reg_out = regression_net(reg_input)
                 cond_tensors.append(reg_out)
        else:
            reg_out = None
            
        condition = torch.cat(cond_tensors, dim=1)
        
        # Target
        if net_name == "regression":
            target = state_target
        else:
            # Diffusion target is residual
            target = state_target - reg_out
            
        # Forward & Loss
        # Forward & Loss
        # Both regression_loss_fn and EDMLoss now support the same signature
        # loss_fn(net, images, condition, ...)
        loss = loss_fn(net, target, condition).mean()
            
        loss.backward()
        optimizer.step()
        
        # Logging
        if total_steps % 100 == 0 and dist.rank == 0:
            logger0.info(f"Step {total_steps}: Loss {loss.item()}")
            if log_to_wandb:
                wandb.log({"loss": loss.item()}, step=total_steps)
                
        # Validation
        if total_steps % cfg.training.validation_freq == 0:
            # Run validation
            logger0.info(f"Running validation at step {total_steps}...")
            net.eval()
            with torch.no_grad():
                try:
                    val_batch = next(valid_dataset_iterator)
                except StopIteration:
                    valid_dataset_iterator = iter(valid_data_loader)
                    val_batch = next(valid_dataset_iterator)
                
                val_bg = val_batch['background'].to(device)
                val_state = val_batch['state']
                val_state_input = val_state[0].to(device)
                val_state_target = val_state[1].to(device)
                
                # Build condition
                val_cond_tensors = []
                if "state" in condition_list:
                    val_cond_tensors.append(val_state_input)
                if "background" in condition_list:
                    val_cond_tensors.append(val_bg)
                
                # Regression part for validation
                if "regression" in condition_list:
                     val_reg_input = torch.cat([val_state_input, val_bg], dim=1)
                     val_reg_out = regression_net(val_reg_input)
                     val_cond_tensors.append(val_reg_out)
                else:
                    val_reg_out = None
                    
                val_condition = torch.cat(val_cond_tensors, dim=1)
                
                # Forward
                val_pred = net(val_condition)
                
                # Compute Loss

                val_loss = loss_fn(net, val_state_target, val_condition).mean()
                
                logger0.info(f"Validation Loss: {val_loss.item()}")
                if log_to_wandb and dist.rank == 0:
                    wandb.log({"val_loss": val_loss.item()}, step=total_steps)
                    
                    # Log images/video
                    # We want to log denormalized streamflow
                    # Model output is normalized streamflow (for regression)
                    # For diffusion, it's residual, so we need to add regression output back
                    
                    if net_name == "regression":
                        pred_flow = val_pred
                        target_flow = val_state_target
                    else:
                        # Diffusion prediction is residual
                        # To get full flow: pred_flow = reg_out + pred_residual
                        # But wait, diffusion model predicts score or noise usually?
                        # If using EDMLoss, the net output depends on parameterization.
                        # Assuming standard regression-like output for now or skip if complex.
                        # Let's focus on regression as per current context.
                        pred_flow = val_pred # Placeholder if diffusion logic differs
                        target_flow = val_target
                        
                    # Denormalize
                    # We need to construct full state to use denormalize_state
                    # State: Precip (6) + Streamflow (6)
                    # We have input precip in val_state_input[:, :6]
                    
                    # Construct predicted state
                    # (B, 12, H, W)
                    pred_state = torch.cat([val_state_input[:, :6], pred_flow], dim=1)
                    target_state = torch.cat([val_state_input[:, :6], target_flow], dim=1)
                    
                    # Denormalize
                    pred_state_denorm = dataset_train.denormalize_state(pred_state)
                    target_state_denorm = dataset_train.denormalize_state(target_state)
                    
                    # Extract streamflow (last 6 channels)
                    # (B, 6, H, W)
                    pred_flow_denorm = pred_state_denorm[:, 6:]
                    target_flow_denorm = target_state_denorm[:, 6:]
                    
                    # Take first sample in batch
                    # (6, H, W)
                    pred_seq = pred_flow_denorm[0].cpu().numpy()
                    target_seq = target_flow_denorm[0].cpu().numpy()
                    
                    # Log video
                    # Range [0, 4]
                    video = create_video(pred_seq, target_seq, field_name="streamflow", vmin=0, vmax=4)
                    wandb.log({"val_video": video}, step=total_steps)

            net.train()
            
        # Checkpoint
        if total_steps % cfg.training.checkpoint_freq == 0 and dist.rank == 0:
            save_checkpoint(
                path=ckpt_path,
                models=net,
                optimizer=optimizer,
                epoch=total_steps
            )
            
        total_steps += 1
        done = total_steps >= total_train_steps

    logger0.info("Training finished.")
