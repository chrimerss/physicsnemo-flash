import zarr
import numpy as np
import os
import json

def check_training_distribution():
    print("Checking Training Data Distribution...")
    zarr_path = "data/data.zarr"
    if not os.path.exists(zarr_path):
        print(f"Zarr path not found: {zarr_path}")
        return

    try:
        ds = zarr.open_group(zarr_path, mode='r')
        streamflow = ds['streamflow']
        
        # Load stats
        with open('stats.json', 'r') as f:
            stats = json.load(f)
            q_mean_stats = stats['unitq']['mean']
            q_std_stats = stats['unitq']['std']
            
        print(f"Stats.json -> Mean: {q_mean_stats}, Std: {q_std_stats}")
        
        # Train range: 0 to 157680
        train_end = 157680
        num_samples = 50
        indices = np.linspace(0, train_end - 1, num_samples, dtype=int)
        
        accum_mean = []
        accum_std = []
        accum_zeros = [] # masked -9999
        accum_real_zeros = [] # original 0s
        
        print(f"Sampling {num_samples} frames from training set...")
        
        all_values = []
        
        for idx in indices:
            frame = streamflow[idx]
            
            # Count -9999
            mask_nodata = (frame == -9999)
            count_nodata = np.sum(mask_nodata)
            pct_nodata = (count_nodata / frame.size) * 100
            
            # Count 0s (original)
            mask_zeros = (frame == 0)
            count_zeros = np.sum(mask_zeros)
            pct_zeros = (count_zeros / frame.size) * 100
            
            # Apply mask (mimic dataset)
            frame_masked = frame.copy()
            frame_masked[mask_nodata] = 0
            
            # Collect stats
            accum_zeros.append(pct_nodata)
            accum_real_zeros.append(pct_zeros)
            
            # We can't keep all pixels in memory, but we can keep subsample
            # Subsample 1% of pixels
            subsample = np.random.choice(frame_masked.flatten(), size=frame.size // 100, replace=False)
            all_values.append(subsample)
            
        # Concatenate all subsamples
        all_values_np = np.concatenate(all_values)
        
        calculated_mean = np.mean(all_values_np)
        calculated_std = np.std(all_values_np)
        
        print("\n--- Analysis Results ---")
        print(f"Calculated Mean (after masking -9999 -> 0): {calculated_mean}")
        print(f"Calculated Std  (after masking -9999 -> 0): {calculated_std}")
        print(f"Stats.json Mean: {q_mean_stats}")
        print(f"Stats.json Std:  {q_std_stats}")
        
        print(f"\nAvg % of -9999 (NoData) per frame: {np.mean(accum_zeros):.2f}%")
        print(f"Avg % of 0 (Real Zero) per frame:  {np.mean(accum_real_zeros):.2f}%")
        
        diff_mean = abs(calculated_mean - q_mean_stats)
        diff_std = abs(calculated_std - q_std_stats)
        
        print(f"\nDifference Mean: {diff_mean}")
        print(f"Difference Std: {diff_std}")
        
        if diff_mean > 0.5 or diff_std > 0.5:
            print("\nWARNING: Significant discrepancy between calculated stats and stats.json!")
            print("This could explain weird loss. normalization might be shifting data incorrectly.")
        else:
            print("\nStats seem consistent.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_training_distribution()
