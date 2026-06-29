import os
import glob
import torch
import xarray as xr
import numpy as np
import matplotlib
# Force headless rendering to prevent terminal freezes from GUI components
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import torch.amp as amp

# Import network modules safely from your structured local paths
from networks.slomo import SuperSlomo

def radiance_to_brightness_temp(dataset):
    """Converts raw satellite radiance observations into absolute Kelvin scales with NaN sanitization."""
    rad = dataset['Rad'].values.copy()
    
    # 🚀 CRITICAL SANITIZATION FLOOR: Clean raw space NaN flags before evaluating logs
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
    """Scans the macro full-disk array for horizontal dead lines and reconstructs them."""
    repaired = matrix.copy()
    H, W = repaired.shape
    center_strip = repaired[:, W // 3 : (2 * W) // 3]
    row_variance = np.var(center_strip, axis=1)
    dead_row_indices = np.where((row_variance == 0.0) & (np.mean(center_strip, axis=1) > 0.0))[0]
    
    if len(dead_row_indices) > 0:
        for r in dead_row_indices:
            if 0 < r < H - 1:
                repaired[r, :] = (repaired[r-1, :] + repaired[r+1, :]) / 2.0
    return repaired

def evaluate_entire_dataset_global_pass(checkpoint_name, model_size=1356, native_size=2712):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Target Compute Hardware Engine Bound: [{device.type.upper()}]")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 1. Directory Setup Configuration
    EVAL_DIR, OUTPUT_VIS_DIR = "noaa_goes19_eval_data", "eval_stitched_outputs"
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
    # 2. Extract and match all file sequences from the raw evaluation folder paths
    search_path = os.path.join(EVAL_DIR, "**/*.nc")
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
        except Exception: pass

    total_sequences = len(valid_triplets)
    if total_sequences == 0: raise FileNotFoundError(f"Could not parse triplets in '{EVAL_DIR}'.")
    
    # 3. Model Weight Reconstruction
    model = SuperSlomo(n_channels=1).to(device)
    checkpoint = torch.load(checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    global_mse, global_psnr, global_ssim = [], [], []

    # 4. Evaluation Loop
    for idx, (file_t, file_t10_gt, file_t20) in enumerate(valid_triplets):
        print(f"\n🎒 [Sequence {idx+1}/{total_sequences}] Executing Macro Tensor Interpolation...")
        
        ds_t, ds_t10, ds_t20 = xr.open_dataset(file_t).load(), xr.open_dataset(file_t10_gt).load(), xr.open_dataset(file_t20).load()
        full_t, full_t10_gt, full_t20 = repair_dead_sensor_lines(radiance_to_brightness_temp(ds_t)), repair_dead_sensor_lines(radiance_to_brightness_temp(ds_t10)), repair_dead_sensor_lines(radiance_to_brightness_temp(ds_t20))
        norm_t, norm_t10_gt, norm_t20 = normalize_bt(full_t), normalize_bt(full_t10_gt), normalize_bt(full_t20)
        ds_t.close(); ds_t10.close(); ds_t20.close()
        
        tensor_t = torch.tensor(norm_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_t10_gt = torch.tensor(norm_t10_gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_t20 = torch.tensor(norm_t20, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        t_scaled = F.interpolate(tensor_t, size=(model_size, model_size), mode='bilinear', align_corners=True).to(device)
        t20_scaled = F.interpolate(tensor_t20, size=(model_size, model_size), mode='bilinear', align_corners=True).to(device)
        
        with torch.no_grad():
            with amp.autocast('cuda'):
                output_dict = model(t_scaled, t20_scaled, 0.5)
                synthesized_t10_scaled = next(v for v in output_dict.values() if torch.is_tensor(v) and v.shape[-2:] == (model_size, model_size))
        
        pred_native = F.interpolate(synthesized_t10_scaled, size=(native_size, native_size), mode='bicubic', align_corners=True).cpu().squeeze().numpy()
        t10_gt_native = F.interpolate(tensor_t10_gt, size=(native_size, native_size), mode='bilinear', align_corners=True).squeeze().numpy()

        roi_mask = (t10_gt_native > 0.0).astype(np.float32)
        full_prediction = (roi_mask * pred_native) + ((1.0 - roi_mask) * t10_gt_native)

        active = np.where(roi_mask > 0.5)
        mse_val = np.mean((t10_gt_native[active] - full_prediction[active]) ** 2)
        psnr_val = 20 * np.log10(1.0 / np.sqrt(mse_val)) if mse_val > 0 else 100.0
        ssim_val = ssim(t10_gt_native, full_prediction, data_range=1.0)
        global_mse.append(mse_val); global_psnr.append(psnr_val); global_ssim.append(ssim_val)
        print(f"   📈 Native Metrics -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        
        # 5. HORIZONTAL DASHBOARD (High Contrast Black Labels)
        if idx < 3:
            t_native = F.interpolate(tensor_t, size=(native_size, native_size), mode='bilinear', align_corners=True).squeeze().numpy()
            t20_native = F.interpolate(tensor_t20, size=(native_size, native_size), mode='bilinear', align_corners=True).squeeze().numpy()
            fig, axes = plt.subplots(1, 4, figsize=(28, 7))
            images = [t_native, full_prediction, t10_gt_native, t20_native]
            titles = ["Input Frame (T)", "Model Interpolation (T+10)", "Ground Truth (T+10)", "Input Frame (T+20)"]
            for i in range(4):
                axes[i].imshow(images[i], cmap='inferno')
                axes[i].set_title(titles[i], fontsize=16, fontweight='bold', color='black')
                axes[i].axis('off')
            fig.suptitle(f"Sequence {idx+1} | Native Metrics @ {native_size}x{native_size}\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}", fontsize=24, fontweight='bold', color='black', y=1.15)
            plt.subplots_adjust(top=0.85)
            plt.savefig(os.path.join(OUTPUT_VIS_DIR, f"dashboard_horizontal_seq_{idx+1:02d}.png"), dpi=200, bbox_inches='tight')
            plt.close()
            
        del t_scaled, t20_scaled, output_dict
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n================ FINAL EVALUATION SUMMARY ================")
    print(f" Total Evaluated Sequences: {len(global_mse)}")
    print(f" Global Average PSNR      : {np.mean(global_psnr):.4f} dB")
    print(f" Global Average SSIM      : {np.mean(global_ssim):.4f}")
    with open(os.path.join(OUTPUT_VIS_DIR, "evaluation_summary.txt"), "w") as f:
        f.write(f"PSNR: {np.mean(global_psnr):.4f}\nSSIM: {np.mean(global_ssim):.4f}\n")
    print("==========================================================\n")

if __name__ == "__main__":
    evaluate_entire_dataset_global_pass("checkpoints/global_model_epoch_27.pt")