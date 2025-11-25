
## Context
I want to build a surrogate model (diffusion-based model) to mimic a conventional hydrologic model that transforms precipitation into distributed streamflow. I have downloaded required data for training, which includes:

1. CREST UNITSTREAMFLOW (target) at 10 min/1km resolution (grib file), dimension is (6, 3500,7000). The first dimension is time which is from 00:00, to 00:10, to 00:20, to 00:30, to 00:40, to 00:50 which has 10 minute frequency. Data date range from 20210101 to 2025-07-31

2. MRMS precipitation (input) at 1 h / 1km resolution (grib file), dimension is (6, 3500,7000). The first dimension is time which is from 00:00, to 00:10, to 00:20, to 00:30, to 00:40, to 00:50 which has 10 minute frequency.

3. Static features such as Digital Elevation Model (dem_usa.tif), Flow Direction (fdir_usa.tif), Flow Accumulation (facc_usa.tif), Hydraulic conductivity (ksat_usa.tif), Maximum Soil Water Holding Capacity (wm_usa.tif), Soil infiltration speed (b_usa.tif).

Data Locations:

1. for unit streamflow and precipitation: data/data.zarr
    - precipitation: <Array file://data/data.zarr/precipitation shape=(240912, 3500, 7000) dtype=float32>
    - unit streamflow: <Array file://data/data.zarr/streamflow shape=(240912, 3500, 7000) dtype=float32>
2. for auxillary data (inputl Geotiff): /home/users/li1995/global_flood/FLASH/data/aux

no data value for precip: -3
no data value for CREST: -9999.

For the deep learning model framework, I want to use Nvidia's physicsnemo https://docs.nvidia.com/physicsnemo/latest/overview.html. For the structure, use a flow matching DDPM and UNet as backbone. The input dimension is (6+6+7, 3500, 7000) but can patchify into small patches (1024x1024) and use a location embedding based on its longitude and latitude.

The paper describing the structure of StormCast is in: https://arxiv.org/pdf/2408.10958. The example code to train such system is in https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/stormcast. Essentially, they proposed a two-stage training, where the first stage is a UNet to mimic the mean state of the system, and the second stage is a DDPM to capture the fine details of the system.

Of course. Here are the specific inputs and targets for each of the two training steps, as described in the paper.

### Phase 1: Deterministic Regression

This phase trains a U-Net model (`Fθ`) to predict the average, most likely future state.

*   **Inputs:**
    *   **`Mt`**: The high-resolution mesoscale state at the current time `t`. This includes 99 variables like wind, temperature, and humidity at various vertical levels.
    *   **`St`**: The coarse-resolution synoptic state at the current time `t`, which provides the large-scale weather context.

*   **Target:**
    *   **`Mt+1`**: The actual, ground-truth high-resolution mesoscale state at the next hour, `t+1`.

The model is trained to minimize the difference between its prediction and the true `Mt+1`, effectively learning the conditional mean `E[Mt+1 | Mt, St]`. The output of this trained model is denoted as `μt+1`.

### Phase 2: Stochastic Diffusion

This phase trains a diffusion model (`Dϕ`) to generate the fine-scale, unpredictable details that the first model missed.

*   **Inputs (for conditioning):** The diffusion model's generation is conditioned on a concatenated set of variables:
    *   **`μt+1`**: The deterministic mean forecast that was the *output* of the Phase 1 model.
    *   **`Mt`**: The mesoscale state at the current time `t`.
    *   **`St`**: The synoptic-scale state at the current time `t`.

*   **Target (the data distribution it learns to model):**
    *   **`rt+1`**: The residual, which is calculated beforehand. As defined in Equation 3 on page 3, the residual is the difference between the ground truth and the Phase 1 prediction: `rt+1 = Mt+1 - μt+1`.



Training data: 2021-01-01 to 2023-12-31
Validation data: 2024-01-01 to 2024-12-31
Testing data: 2025-01-01 to 2025-07-31

Load input data (19, 1024, 1024) with a sliding time window. For instance, precipitation at 00:10, 00:20, 00:30, 00:40, 00:50, 01:00 (6 in total), unit streamflow at 00:10, 00:20, 00:30, 00:40, 00:50, 01:00 (6 in total), and auxillary data (7 in total)to predict unit streamflow at 01:10, 01:20, 01:30, 01:40, ; 00:20, 00:30,00:40,00:50,01:00,01:10 to predict 01:20, 01:30, 01:40.

Please use the lookback window and lookout window as a parameter to be tuned in the future (meaning that the time dimension is a hyperparameter, both default to 6).

Target dimension: (6,1024,1024) which is the unit streamflow for the next half hour, but keep the time dimension a hyperparameter and I need to tune what is the maximum forecast window.

Data logging: use wandb and plot results as video (see src/vis.py).

NOTE: if you want to test any implementation, please use singularity with command: singularity exec --nv -B "$(pwd)":/workspace -B "$(pwd)/data":/data /home/users/li1995/global_flood/physicsnemo-flash/physicsnemo_25.11.sif bash -c "torchrun train.py training.loss=regression"

## Instruction

- Write clear, concise code with meaningful variable names.
- Prefer small, single-purpose functions with short docstrings.
- Always use PEP 8 style for Python; add type hints where practical.
- Add a brief comment for any complex logic.
- Avoid unnecessary abstractions or patterns not needed for scientific analysis.
- Use context managers for I/O operations.
- Handle missing or invalid data with clear error messages, but skip advanced error handling.
- Don’t include code not directly related to research goals.
- after testing, remove any test codes
- Avoid using try except structure because that will slow the debug process
- My python pakcage version is in environment.yml so make sure they are compatible with the code you are writing.
- Please inspect file structure before writing code and don't presume file naming etc.