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

import matplotlib.pyplot as plt
import numpy as np
import wandb
import io
from PIL import Image

def validation_plot(generated, truth, field_name):
    """
    Create a side-by-side comparison plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Generated
    im0 = axes[0].imshow(generated, cmap='viridis')
    axes[0].set_title(f"Generated {field_name}")
    plt.colorbar(im0, ax=axes[0])
    
    # Plot Truth
    im1 = axes[1].imshow(truth, cmap='viridis')
    axes[1].set_title(f"Truth {field_name}")
    plt.colorbar(im1, ax=axes[1])
    
    # Plot Difference
    diff = generated - truth
    im2 = axes[2].imshow(diff, cmap='coolwarm')
    axes[2].set_title(f"Difference {field_name}")
    plt.colorbar(im2, ax=axes[2])
    
    return fig

def create_video(predictions, targets, field_name="streamflow"):
    """
    Create a video from a sequence of predictions and targets.
    predictions: (T, H, W)
    targets: (T, H, W)
    """
    frames = []
    T = predictions.shape[0]
    
    for t in range(T):
        fig = validation_plot(predictions[t], targets[t], f"{field_name} t={t}")
        
        # Convert plot to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        plt.close(fig)
        
    # Frames is list of (H, W, C)
    # Wandb expects (T, C, H, W) for video? Or (T, H, W, C)?
    # wandb.Video expects numpy array or path.
    # Format: (time, channels, height, width)
    
    video_data = np.array(frames) # (T, H, W, C)
    video_data = np.transpose(video_data, (0, 3, 1, 2)) # (T, C, H, W)
    
    return wandb.Video(video_data, fps=4, format="gif")
