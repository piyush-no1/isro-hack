import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.amp as amp
import torch.nn.functional as F
import sys

# Import your updated global SSD dataset engine and architecture modules
from goes_dataset import GOES19GlobalDataset
from networks.slomo import SuperSlomo

class MaskedStructuralLoss(nn.Module):
    def __init__(self, alpha=0.84):
        """
        Hybrid optimization loss engine. Combines L1 absolute scaling metric math 
        with localized Structural Similarity (SSIM) bounds.
        """
        super(MaskedStructuralLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, gt, mask, step_info=""):
        # Enforce masked context immediately
        masked_pred = pred * mask
        masked_gt = gt * mask
        
        # 1. Compute baseline pixel scaling validation metrics
        l1_loss = self.l1(masked_pred, masked_gt)
        
        # 2. Check if we have valid active planet pixels in this batch step
        num_active_pixels = torch.sum(mask > 0.5)
        if num_active_pixels == 0:
            return l1_loss
        
        # 3. Compute Structural Topology preservation matrices
        ux = F.avg_pool2d(masked_pred, 11, stride=1, padding=5)
        uy = F.avg_pool2d(masked_gt, 11, stride=1, padding=5)
        
        ux2 = F.avg_pool2d(masked_pred ** 2, 11, stride=1, padding=5)
        uy2 = F.avg_pool2d(masked_gt ** 2, 11, stride=1, padding=5)
        uxy = F.avg_pool2d(masked_pred * masked_gt, 11, stride=1, padding=5)
        
        vx = torch.clamp(ux2 - ux ** 2, min=0.0)
        vy = torch.clamp(uy2 - uy ** 2, min=0.0)
        vxy = uxy - ux * uy
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        denom = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2) + 1e-8
        num = (2 * ux * uy + c1) * (2 * vxy + c2)
        ssim_map = num / denom
        
        # --- FORENSIC LOSS CHECKING ---
        if torch.isnan(ssim_map).any() or torch.isinf(ssim_map).any():
            print(f"\n\n🚨 [LOSS CRASH] NaN/Inf detected inside SSIM map calculation at {step_info}!")
            print(f"   -> Min/Max Denominator: {denom.min().item():.8f} / {denom.max().item():.8f}")
            print(f"   -> Min/Max Numerator: {num.min().item():.8f} / {num.max().item():.8f}")
            print(f"   -> Active pixel subset count: {num_active_pixels.item()}")
            sys.exit(1)

        active_subset = ssim_map[mask > 0.5]
        if active_subset.numel() == 0 or torch.isnan(active_subset).any():
            print(f"\n\n🚨 [LOSS CRASH] Active subset mask selection failed or returned NaN at {step_info}!")
            sys.exit(1)

        ssim_loss = 1.0 - torch.mean(active_subset)
        total_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        
        if torch.isnan(total_loss):
            print(f"\n\n🚨 [LOSS CRASH] Hybrid blending generated NaN total loss value! SSIM loss: {ssim_loss.item()}, L1 loss: {l1_loss.item()}")
            sys.exit(1)
            
        return total_loss

def train_global_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Initializing High-Performance Training Engine on: [{device.type.upper()}]")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("⚡ cuDNN Auto-Tuner Benchmarking Layer: [ENABLED]")

    DATA_DIR = "noaa_goes19_data"
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    BATCH_SIZE = 1  
    LEARNING_RATE = 1e-4
    EPOCHS = 40  # Run up to 40 epochs
    PATIENCE = 5  # Early stopping threshold

    print("--> Initializing SSD-Backed Global Data Matrix Loader...")
    full_dataset = GOES19GlobalDataset(data_dir=DATA_DIR, target_size=1356)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"--> Data Stream Sliced: {len(train_loader)} Train Steps | {len(val_loader)} Validation Steps per Epoch.")

    model = SuperSlomo(n_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_hybrid = MaskedStructuralLoss(alpha=0.84)
    scaler = amp.GradScaler('cuda')

    # --- CHECKPOINT RESTORATION ENGINE ---
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0

    print("🔍 Scanning for pre-existing checkpoints...")
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("global_model_epoch_") and f.endswith(".pt")]
    if checkpoints:
        # Extract epoch numbers and find the latest checkpoint
        epochs_found = [int(f.split("_")[-1].split(".")[0]) for f in checkpoints]
        latest_epoch = max(epochs_found)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"global_model_epoch_{latest_epoch}.pt")
        
        print(f"🔄 Found checkpoint from epoch {latest_epoch}. Restoring state...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ State loaded successfully. Resuming from Epoch {start_epoch}...")
    else:
        print("🆕 No checkpoints detected. Starting a clean training pipeline.")

    # --- MAIN OPTIMIZATION LOOP ---
    for epoch in range(start_epoch, EPOCHS + 1):
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        print(f"\n🚀 [Epoch {epoch}/{EPOCHS}] Commencing Optimization Pass...")
        
        for step, (frame_t, frame_t10_gt, frame_t20, roi_mask) in enumerate(train_loader):
            step_label = f"Epoch {epoch}, Step {step+1}"
            
            if torch.isnan(frame_t).any() or torch.isnan(frame_t20).any() or torch.isnan(frame_t10_gt).any():
                print(f"\n\n🚨 [DATA EXCEPTION] NaN found inside raw tensors at {step_label}!")
                sys.exit(1)

            frame_t = frame_t.to(device, non_blocking=True)
            frame_t10_gt = frame_t10_gt.to(device, non_blocking=True)
            frame_t20 = frame_t20.to(device, non_blocking=True)
            roi_mask = roi_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast('cuda'):
                output_dict = model(frame_t, frame_t20, 0.5)
                
                synthesized_t10 = None
                for key, value in output_dict.items():
                    if torch.is_tensor(value) and value.shape == frame_t10_gt.shape:
                        synthesized_t10 = value
                        break
                        
                if synthesized_t10 is None:
                    raise KeyError("Could not extract synthesized target tensor match inside output dictionary.")

                if torch.isnan(synthesized_t10).any():
                    print(f"\n\n🚨 [MODEL EXCEPTION] network forward pass outputted NaN at {step_label}!")
                    sys.exit(1)

                loss = criterion_hybrid(synthesized_t10, frame_t10_gt, roi_mask, step_info=step_label)

            if torch.isnan(loss):
                print(f"\n\n🚨 [SCALER EXCEPTION] Loss evaluated to NaN before backwards step at {step_label}!")
                sys.exit(1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            print(f"   ⚡ Step [{step+1}/{len(train_loader)}] | Step Hybrid Loss: {loss.item():.6f}", end="\r")

        epoch_train_loss = running_train_loss / len(train_loader)
        print(f"\n✨ Epoch [{epoch}/{EPOCHS}] Completed | Average Train Hybrid Loss: {epoch_train_loss:.6f}")

        # 2. Validation Phase (Required for Early Stopping evaluation)
        model.eval()
        running_val_loss = 0.0
        print(f"🧪 Evaluating Validation Set Matrix...")
        
        with torch.no_grad():
            for step, (frame_t, frame_t10_gt, frame_t20, roi_mask) in enumerate(val_loader):
                frame_t = frame_t.to(device, non_blocking=True)
                frame_t10_gt = frame_t10_gt.to(device, non_blocking=True)
                frame_t20 = frame_t20.to(device, non_blocking=True)
                roi_mask = roi_mask.to(device, non_blocking=True)
                
                with amp.autocast('cuda'):
                    output_dict = model(frame_t, frame_t20, 0.5)
                    synthesized_t10 = None
                    for key, value in output_dict.items():
                        if torch.is_tensor(value) and value.shape == frame_t10_gt.shape:
                            synthesized_t10 = value
                            break
                    
                    if synthesized_t10 is not None:
                        val_loss = criterion_hybrid(synthesized_t10, frame_t10_gt, roi_mask, step_info=f"Val Step {step+1}")
                        running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"📊 Validation Hybrid Loss: {epoch_val_loss:.6f}")

        # 3. Save Checkpoint State Matrix
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"global_model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss
        }, checkpoint_path)
        print(f"   💾 Checkpoint securely archived to: {checkpoint_path}")

        # 4. Early Stopping Evaluation Engine
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            print(f"   📈 Validation loss improved. Resetting patience counter.")
        else:
            patience_counter += 1
            print(f"   ⚠️ Validation loss did not improve. Patience: [{patience_counter}/{PATIENCE}]")
            if patience_counter >= PATIENCE:
                print(f"\n🛑 [EARLY STOPPING] Validation loss stagnated for {PATIENCE} epochs consecutively. Terminating run execution safely.")
                break

        # Clean down loop garbage
        del frame_t, frame_t10_gt, frame_t20, roi_mask, output_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🏁 Process finished!")

if __name__ == "__main__":
    train_global_model()