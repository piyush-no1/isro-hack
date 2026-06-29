import os
import glob
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import torch.nn.functional as F

def radiance_to_brightness_temp(dataset):
    """Converts raw satellite radiance observations into absolute Kelvin scales with NaN sanitization."""
    rad = dataset['Rad'].values.copy()
    
    # 🚀 CRITICAL SANITIZATION FLOOR: Convert raw unindexed space NaN flags to a clean zero baseline
    rad[np.isnan(rad)] = 0.0
    rad = np.maximum(rad, 0.001)
    
    fk1 = float(dataset['planck_fk1'].values)
    fk2 = float(dataset['planck_fk2'].values)
    bc1 = float(dataset['planck_bc1'].values)
    bc2 = float(dataset['planck_bc2'].values)
    
    return (fk2 / np.log((fk1 / rad) + 1.0) - bc1) / bc2

def normalize_bt(bt_array):
    """Maps Kelvin dimensions strictly into a standard bounded meteorological scale [0, 1]."""
    bt_array = np.clip(bt_array, 180.0, 320.0)
    return (bt_array - 180.0) / (320.0 - 180.0)

def repair_dead_sensor_lines(matrix):
    """
    Scans the macro full-disk array for horizontal dead lines cutting through the planet
    and reconstructs them seamlessly using contextual linear interpolations.
    """
    repaired = matrix.copy()
    H, W = repaired.shape
    
    # Target center columns to explicitly evaluate planetary data, ignoring pure outer space rows
    center_strip = repaired[:, W // 3 : (2 * W) // 3]
    row_variance = np.var(center_strip, axis=1)
    
    # Identify rows where the planet data drops to flat absolute zero variance (dead lines)
    dead_row_indices = np.where((row_variance == 0.0) & (np.mean(center_strip, axis=1) > 0.0))[0]
    
    if len(dead_row_indices) > 0:
        for r in dead_row_indices:
            if 0 < r < H - 1:
                repaired[r, :] = (repaired[r-1, :] + repaired[r+1, :]) / 2.0
    return repaired

class GOES19GlobalDataset(Dataset):
    def __init__(self, data_dir, target_size=1356, cache_dir="processed_tensors_cache"):
        """
        Hyper-Optimized SSD-Backed Virtual Data Array Layer.
        Compiles high-resolution arrays to disk binary vectors at a memory-safe 
        1356x1356 resolution step to completely avoid VRAM OOM crashes.
        """
        self.target_size = target_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        search_path = os.path.join(data_dir, "**/*.nc")
        all_files = sorted(glob.glob(search_path, recursive=True))
        
        valid_triplets = []
        for i in range(len(all_files) - 2):
            f0, f1, f2 = all_files[i], all_files[i+1], all_files[i+2]
            try:
                t0 = datetime.strptime(f0.split('_s')[1][:11], "%Y%j%H%M")
                t1 = datetime.strptime(f1.split('_s')[1][:11], "%Y%j%H%M")
                t2 = datetime.strptime(f2.split('_s')[1][:11], "%Y%j%H%M")
                if (t1 - t0) == timedelta(minutes=10) and (t2 - t1) == timedelta(minutes=10):
                    valid_triplets.append((f0, f1, f2))
            except Exception:
                pass

        self.total_sequences = len(valid_triplets)
        if self.total_sequences == 0:
            raise FileNotFoundError(f"Could not parse any valid chronological triplets inside '{data_dir}'.")
            
        # Check if disk compiler pass has already built the local tracking manifest
        self.cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "sequence_*.pt")))
        
        if len(self.cache_files) == self.total_sequences:
            print(f"--> [LINKED] Found {len(self.cache_files)} precompiled {target_size}x{target_size} global tensors inside disk storage structures.")
            return

        print(f"--> Found {self.total_sequences} valid full-disk sequences. Compiling binary matrix records to SSD...")
        
        for idx, (f0, f1, f2) in enumerate(valid_triplets):
            out_path = os.path.join(self.cache_dir, f"sequence_{idx:04d}.pt")
            print(f"   📥 Processing and writing record block [{idx+1}/{self.total_sequences}] at {target_size} resolution...", end="\r")
            
            # Skip if this specific block was compiled successfully in a previous incomplete run
            if os.path.exists(out_path):
                continue
                
            ds0 = xr.open_dataset(f0).load()
            ds1 = xr.open_dataset(f1).load()
            ds2 = xr.open_dataset(f2).load()
            
            bt0 = repair_dead_sensor_lines(radiance_to_brightness_temp(ds0))
            bt1 = repair_dead_sensor_lines(radiance_to_brightness_temp(ds1))
            bt2 = repair_dead_sensor_lines(radiance_to_brightness_temp(ds2))
            
            bt0[bt0 < 100.0] = 180.0
            bt1[bt1 < 100.0] = 180.0
            bt2[bt2 < 100.0] = 180.0
            
            norm0 = normalize_bt(bt0)
            norm1 = normalize_bt(bt1)
            norm2 = normalize_bt(bt2)
            
            ds0.close(); ds1.close(); ds2.close()
            
            t0_tensor = torch.tensor(norm0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            t1_tensor = torch.tensor(norm1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            t2_tensor = torch.tensor(norm2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Force target dimension configurations with align_corners enabled to maintain precision parity
            t0_scaled = F.interpolate(t0_tensor, size=(self.target_size, self.target_size), mode='bilinear', align_corners=True).squeeze(0)
            t1_scaled = F.interpolate(t1_tensor, size=(self.target_size, self.target_size), mode='bilinear', align_corners=True).squeeze(0)
            t2_scaled = F.interpolate(t2_tensor, size=(self.target_size, self.target_size), mode='bilinear', align_corners=True).squeeze(0)
            
            roi_mask = (t1_scaled > 0.0).to(torch.float32)
            
            # Save vector structure cleanly to hard disk binary allocations
            torch.save((t0_scaled, t1_scaled, t2_scaled, roi_mask), out_path)
            
        self.cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "sequence_*.pt")))
        print(f"\n✅ Disk compilation phase complete. All elements securely allocated to virtual memory maps.")

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        # Pull single element from high-speed local disk storage dynamically
        return torch.load(self.cache_files[idx])