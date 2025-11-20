# PhysicsNemo FLASH Surrogate Model

This repository contains the implementation of a surrogate model for unit streamflow using the PhysicsNemo framework and StormCast architecture (UNet + Diffusion).

## Overview

The model predicts unit streamflow based on precipitation and auxiliary static features (DEM, Flow Direction, etc.). It uses a two-stage training process:
1.  **Phase 1 (Deterministic Regression)**: A UNet predicts the mean future streamflow.
2.  **Phase 2 (Stochastic Diffusion)**: A Flow-Matching DDPM generates residuals to capture fine-scale details, conditioned on the Phase 1 prediction.

## Directory Structure

```
.
├── config/                 # Hydra configuration files
│   ├── config.yaml         # Main config
│   ├── model/              # Model architectures
│   └── training/           # Training hyperparameters
├── src/                    # Source code
│   ├── dataset.py          # Data loading (FlashDataset)
│   ├── trainer.py          # Training loop
│   └── vis.py              # Visualization utilities
├── train.py                # Training script
├── inference.py            # Inference script
├── test_dataset.py         # Unit tests
└── GEMINI.md               # Original requirements
```

## Prerequisites

-   Python 3.10+
-   NVIDIA PhysicsNemo
-   PyTorch
-   Zarr, Xarray, RioXarray
-   WandB (for logging)

## Data Preparation

Ensure your data is located as specified in `config/config.yaml`:
-   Zarr Data: `data/data.zarr` (Precipitation and Streamflow)
-   Auxiliary Data: `/home/users/li1995/global_flood/FLASH/data/aux` (GeoTiffs)

## Training

### Phase 1: Regression (UNet)

Train the deterministic UNet model:

```bash
python train.py training.loss="regression"
```

This will save checkpoints to `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints_regression/`.

### Phase 2: Diffusion (EDM)

Train the diffusion model. You must provide the path to the best checkpoint from Phase 1:

```bash
python train.py training.loss="edm" model.regression_weights="/path/to/phase1/checkpoint.mdlus"
```

## Inference

Run inference using a trained model:

```bash
python inference.py training.initial_weights="/path/to/checkpoint.mdlus"
```

This will generate predictions and log visualization videos to WandB.

## Testing

Run unit tests to verify data loading:

```bash
python test_dataset.py
```

## Running with Singularity

If you are using Singularity, you can run the training scripts as follows. Assumes you have the `physicsnemo` image (e.g., `physicsnemo.sif`).

You need to bind the directories containing your code and data.

```bash
# Interactive shell
singularity shell --nv -B /path/to/physicsnemo-flash:/workspace -B /home/users/li1995/global_flood/FLASH/data:/data physicsnemo.sif

# Inside the container
cd /workspace
python train.py training.loss="regression"
```

Or run directly:

```bash
singularity exec --nv \
  -B $(pwd):/workspace \
  -B /home/users/li1995/global_flood/FLASH/data:/data \
  physicsnemo.sif \
  bash -c "cd /workspace && python train.py training.loss='regression'"
```

**Note:** Ensure that the paths in `config/config.yaml` match the mount points inside the container. For example, if you mount data to `/data`, update `dataset.zarr_path` and `dataset.aux_path` accordingly.

## Distributed Training

To run on multiple GPUs (e.g., all available GPUs on a node), use `torchrun`.

**Inside Singularity:**

```bash
# Example: Run on 4 GPUs
torchrun --nproc_per_node=4 train.py training.loss="regression"
```

**Or directly with Singularity exec:**

```bash
singularity exec --nv \
  -B $(pwd):/workspace \
  -B /home/users/li1995/global_flood/FLASH/data:/data \
  physicsnemo.sif \
  bash -c "cd /workspace && torchrun --nproc_per_node=\$(nvidia-smi -L | wc -l) train.py training.loss='regression'"
```
