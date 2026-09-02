"""
Fusion Model Training Script (16-bit, Mixed Precision AMP)
Combines RCAN super-resolution + Attention U-Net segmentation with hybrid loss.

Usage:
    1. Modify the paths in the `if __name__ == "__main__"` section below.
    2. Run: python train.py

Dependencies:
    - torch, torchvision
    - tifffile, numpy, tqdm
    - RCAN2D.py and AUNet.py (must be in the same directory or importable)
"""

import os
import glob
import random
import tifffile
import numpy as np
from tqdm import tqdm
from math import exp

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.functional as TF

from RCAN2D import *
from AUNet import *


# ============================================================================
# Data Loading
# ============================================================================

def AppendLoad16(Path, Transtorch=True):
    """
    Load 16-bit TIFF images from a glob path and normalize to [0, 1].

    Args:
        Path: glob pattern (e.g., r"G:\data\GT\*.tif")
        Transtorch: if True, return as torch.Tensor with shape (N, 1, H, W)

    Returns:
        numpy array or torch tensor of loaded images
    """
    all_imgs_path = glob.glob(Path)
    images = []
    for i, img_path in tqdm(enumerate(all_imgs_path), total=len(all_imgs_path),
                            desc="Loading TIFFs"):
        img = tifffile.imread(img_path)
        img = np.array(img)
        img = img / 65535.0
        img = img.astype(np.float16)
        images.append(img)
    images = np.array(images)
    if Transtorch:
        images = torch.tensor(images, dtype=torch.float16).unsqueeze(1)
    return images


# ============================================================================
# Loss Functions
# ============================================================================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True,
         full=False, val_range=None):
    """
    Compute Structural Similarity Index (SSIM).
    Value range can differ from 255; common ranges are 1 (sigmoid) and 2 (tanh).
    """
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.005 * L) ** 2
    C2 = (0.01 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    """SSIM module with reusable window."""
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1 / img1.max() * 255, img2 / img2.max() * 255,
                    window=window.cuda(), window_size=self.window_size,
                    size_average=self.size_average)


class PearsonCorrelation(nn.Module):
    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def forward(self, x, y):
        x_mean = torch.mean(x, dim=0)
        y_mean = torch.mean(y, dim=0)
        x_norm = x - x_mean
        y_norm = y - y_mean
        diff_sq = torch.sum((x_norm) ** 2)
        correlation = torch.sum((y_norm) * (x_norm)) / (diff_sq + 1e-8)
        return correlation


class BlurryLoss(nn.Module):
    def __init__(self):
        super(BlurryLoss, self).__init__()

    def forward(self, x, y, PSF):
        imgD = F.conv2d(x, PSF, padding=PSF.shape[2] // 2)
        imgB = y
        criterion = PearsonCorrelation()
        d1, d2, d3, d4 = imgD.size()
        new_shape = (1, d2, d3, d4 * d1)
        imgD = imgD.reshape(new_shape)
        imgB = imgB.reshape(new_shape)
        differ = criterion(imgD, imgB)
        return 1 - differ


class BlurryLossMSE(nn.Module):
    def __init__(self):
        super(BlurryLossMSE, self).__init__()

    def forward(self, x, y, PSF):
        imgD = F.conv2d(x, PSF, padding=PSF.shape[2] // 2)
        imgB = y
        criterion = nn.MSELoss()
        differ = criterion(imgD, imgB)
        return differ


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation, addresses class imbalance.
    pred: (B, n_classes, H, W); target: (B, H, W) integer labels.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Ensure target shape is (B, H, W), remove channel dim if present
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.long()
        # Softmax on predictions
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        # One-hot encoding and permute to (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        # Compute Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def hybrid_loss(seg_out, seg_label, rcan_out, rcan_gt,
                seg_loss_weight=1.0, rec_loss_weight=0.2):
    """
    Hybrid loss function.

    Args:
        seg_out: segmentation output (B, n_classes, H, W)
        seg_label: segmentation label (B, H, W)
        rcan_out: RCAN restoration output (B, 1, H, W)
        rcan_gt: restoration GT image (B, 1, H, W)
        seg_loss_weight: segmentation loss weight (primary, set to 1)
        rec_loss_weight: restoration loss weight (auxiliary, e.g. 0.1 / 0.05)

    Returns:
        total_loss, seg_loss, rec_loss (for monitoring)
    """
    # Segmentation loss: Dice + CrossEntropy (complementary)
    dice_loss = DiceLoss()(seg_out, seg_label)
    ce_loss = F.cross_entropy(seg_out, seg_label)
    total_seg_loss = dice_loss + ce_loss
    # Restoration loss: L1 (more robust than MSE against extreme pixels)
    rec_loss = F.l1_loss(rcan_out, rcan_gt)
    # Weighted sum
    total_loss = seg_loss_weight * total_seg_loss + rec_loss_weight * rec_loss
    return total_loss, total_seg_loss, rec_loss


# ============================================================================
# Fusion Model (RCAN + Attention U-Net)
# ============================================================================

class FusionModel(nn.Module):
    """
    Two-branch fusion model:
        RCAN (super-resolution restoration) -> Attention U-Net (segmentation)
    """
    def __init__(self, rcan_pretrained_path, num_seg_classes=2, rcan_scale=2):
        super().__init__()
        self.rcan = RCAN2D(input_channels=1)
        self.rcan.load_state_dict(torch.load(rcan_pretrained_path, map_location='cpu'))
        print(f"[Info] Successfully loaded RCAN pretrained weights: {rcan_pretrained_path}")
        self.attention_unet = AttentionUNet(
            n_channels=1,
            n_classes=num_seg_classes,
            bilinear=False
        )

    def forward(self, x):
        rcan_out = self.rcan(x)          # (B, 1, H*scale, W*scale)
        seg_out = self.attention_unet(rcan_out)  # (B, n_classes, H*scale, W*scale)
        return seg_out, rcan_out


# ============================================================================
# Custom Augmentation Dataset
# ============================================================================

class EnhancedDataset(Dataset):
    """
    Custom dataset with on-the-fly augmentation:
        1. Brightness adjustment
        2. Gamma correction
        3. Base value offset
        4. Background addition (Gaussian-blurred overlay)
    No scaling transform is applied.

    Args:
        X: noisy/low-res image tensor (N, 1, H, W)
        y: restoration GT tensor (N, 1, H*scale, W*scale)
        label: segmentation label tensor (N, H*scale, W*scale)
    """
    def __init__(self, X, y, label):
        self.X = X.float()
        self.y = y.float()
        self.label = label.long()
        # Pre-create Gaussian kernel (sigma=4) for background augmentation
        self.gaussian_kernel = self._create_gaussian_kernel(sigma=4)

    def _create_gaussian_kernel(self, sigma=4, kernel_size=None):
        """Create a 2D Gaussian blur kernel for background augmentation."""
        if kernel_size is None:
            kernel_size = int(6 * sigma + 1)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        x = torch.arange(kernel_size).float() - kernel_size // 2
        x_grid, y_grid = torch.meshgrid(x, x, indexing="ij")
        gaussian = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        return gaussian.unsqueeze(0).unsqueeze(0)

    def adjust_brightness(self, img, factor):
        """Adjust brightness. img shape: (1, H, W)."""
        return TF.adjust_brightness(img, factor)

    def adjust_gamma(self, img, gamma):
        """Gamma correction. img shape: (1, H, W), pixel range [0, 1]."""
        return TF.adjust_gamma(img, gamma)

    def add_base_value(self, img, value):
        """Add a grayscale base value, clamped to [0, 1]."""
        img = img + value
        return torch.clamp(img, 0.0, 1.0)

    def add_background(self, img):
        """
        Add background augmentation: Gaussian-blurred overlay,
        then renormalize to original max intensity.
        """
        orig_max = img.max()
        if orig_max < 1e-6:
            orig_max = 1.0
        padding = self.gaussian_kernel.shape[-1] // 2
        blurred = F.conv2d(img, self.gaussian_kernel, padding=padding)
        bg_weight = random.uniform(0.1, 0.3)
        img_with_bg = img + bg_weight * blurred
        img_with_bg = img_with_bg * (orig_max / img_with_bg.max())
        return img_with_bg

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        label = self.label[idx]

        # 1. Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            X = self.adjust_brightness(X, brightness_factor)
            y = self.adjust_brightness(y, brightness_factor)

        # 2. Gamma correction
        if random.random() < 0.5:
            gamma = random.uniform(0.6, 1.2)
            X = self.adjust_gamma(X, gamma)
            y = self.adjust_gamma(y, gamma)

        # 3. Base value offset
        if random.random() < 0.5:
            base_value = random.uniform(0.05, 0.2)
            X = self.add_base_value(X, base_value)
            y = self.add_base_value(y, base_value)

        # 4. Background addition (only on noisy input)
        if random.random() < 0.2:
            X = self.add_background(X)

        return X, y, label

    def __len__(self):
        return len(self.X)


# ============================================================================
# Training Function (16-bit Mixed Precision)
# ============================================================================

def train16(
        X_train, y_train, label_train,
        X_val, y_val, label_val,
        rcan_pretrained_path,
        rcan_scale=2,
        save_dir="./fusion_model_ckpt",
        epochs=50,
        batch_size=8,
        unet_lr=1e-4,
        rec_loss_weight=0.1,
        device='cuda'
):
    """
    Train the fusion model with 16-bit mixed precision (AMP).

    Args:
        X_train, y_train, label_train: training data tensors
        X_val, y_val, label_val: validation data tensors
        rcan_pretrained_path: path to pretrained RCAN weights (.pth)
        rcan_scale: RCAN upscaling factor
        save_dir: directory to save model checkpoints
        epochs: number of training epochs
        batch_size: batch size
        unet_lr: learning rate for U-Net (RCAN uses 1/10 of this)
        rec_loss_weight: weight for the auxiliary restoration loss
        device: 'cuda' or 'cpu'
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val_compound_loss = float('inf')
    best_val_seg_loss = float('inf')

    train_size = X_train.shape[0]
    val_size = X_val.shape[0]

    print(f"\n[Data] Training samples: {train_size}, Validation samples: {val_size}")
    print(f"[Mode] Joint training with 16-bit mixed precision (AMP)")
    print(f"[Config] epochs={epochs}, batch_size={batch_size}, "
          f"UNet-LR={unet_lr}, RCAN-LR={unet_lr / 10}, "
          f"rec_loss_weight={rec_loss_weight}")

    # Build datasets and loaders
    train_dataset = EnhancedDataset(X_train, y_train, label_train)
    val_dataset = EnhancedDataset(X_val, y_val, label_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    model = FusionModel(rcan_pretrained_path, rcan_scale).to(device)
    rcan_params = list(model.rcan.parameters())
    unet_params = list(model.attention_unet.parameters())
    optimizer = optim.Adam([
        {'params': rcan_params, 'lr': unet_lr / 10},
        {'params': unet_params, 'lr': unet_lr}
    ], betas=(0.9, 0.999), weight_decay=1e-5)

    # Gradient scaler for 16-bit training (prevents gradient underflow)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(1, epochs + 1):
        # ---------------------- Training Phase ----------------------
        model.train()
        train_compound_loss, train_seg_loss, train_rec_loss = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]")

        for degraded, rec_gt, seg_label in pbar:
            degraded = degraded.to(device)
            rec_gt = rec_gt.to(device)
            seg_label = seg_label.to(device)

            if seg_label.dim() == 4 and seg_label.shape[1] == 1:
                seg_label = seg_label.squeeze(1)
            seg_label = seg_label.long()

            optimizer.zero_grad()
            with autocast():
                seg_out, rcan_out = model(degraded)
                dice_loss = DiceLoss()(seg_out, seg_label)
                ce_loss = F.cross_entropy(seg_out, seg_label)
                seg_loss = dice_loss + ce_loss
                rec_loss = F.l1_loss(rcan_out, rec_gt)
                # Compound loss (detach logic preserved)
                compound_loss = seg_loss.detach() + rec_loss_weight * rec_loss

            # Backward pass with gradient scaling
            scaler.scale(seg_loss).backward(retain_graph=True)
            scaler.scale(compound_loss).backward()
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_compound_loss += compound_loss.item() * degraded.size(0)
            train_seg_loss += seg_loss.item() * degraded.size(0)
            train_rec_loss += rec_loss.item() * degraded.size(0)
            pbar.set_postfix({
                'Comp Loss': f"{compound_loss.item():.4f}",
                'Seg Loss': f"{seg_loss.item():.4f}"
            })

        train_compound_loss /= train_size
        train_seg_loss /= train_size
        train_rec_loss /= train_size

        # ---------------------- Validation Phase ----------------------
        model.eval()
        val_compound_loss, val_seg_loss, val_rec_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs} [Val]  ")
            for degraded, rec_gt, seg_label in pbar:
                degraded = degraded.to(device)
                rec_gt = rec_gt.to(device)
                seg_label = seg_label.to(device)

                if seg_label.dim() == 4 and seg_label.shape[1] == 1:
                    seg_label = seg_label.squeeze(1)
                seg_label = seg_label.long()

                with autocast():
                    seg_out, rcan_out = model(degraded)
                    dice_loss = DiceLoss()(seg_out, seg_label)
                    ce_loss = F.cross_entropy(seg_out, seg_label)
                    seg_loss = dice_loss + ce_loss
                    rec_loss = F.l1_loss(rcan_out, rec_gt)
                    compound_loss = seg_loss + rec_loss_weight * rec_loss

                val_compound_loss += compound_loss.item() * degraded.size(0)
                val_seg_loss += seg_loss.item() * degraded.size(0)
                val_rec_loss += rec_loss.item() * degraded.size(0)
                pbar.set_postfix({'Val Comp Loss': f"{compound_loss.item():.4f}"})

        val_compound_loss /= val_size
        val_seg_loss /= val_size
        val_rec_loss /= val_size
        scheduler.step(val_compound_loss)

        # ---------------------- Logging + Checkpointing ----------------------
        print(f"\n[Epoch {epoch:02d}] Loss summary:")
        print(f"  Train - Compound: {train_compound_loss:.4f} | "
              f"Seg: {train_seg_loss:.4f} | Rec: {train_rec_loss:.4f}")
        print(f"  Val   - Compound: {val_compound_loss:.4f} | "
              f"Seg: {val_seg_loss:.4f} | Rec: {val_rec_loss:.4f}")

        if val_compound_loss < best_val_compound_loss:
            best_val_compound_loss = val_compound_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_model_compound_loss.pth"))
            print(f"  [Best] Saved best compound-loss model: {best_val_compound_loss:.4f}")

        if val_seg_loss < best_val_seg_loss:
            best_val_seg_loss = val_seg_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_model_seg_loss.pth"))
            print(f"  [Best] Saved best seg-loss model: {best_val_seg_loss:.4f}")

        torch.save(model.state_dict(),
                   os.path.join(save_dir, "last_fusion_model.pth"))
        print("-" * 100)

    print(f"\n[Done] Training finished. Models saved to: {save_dir}")
    return model


# ============================================================================
# Main Entry Point — modify paths here and run
# ============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example: Actin dataset configuration
    # Modify the paths below to match your data layout.
    # Expected folder structure:
    #   head_dir/
    #     GT/*.tif        (high-quality restoration ground truth)
    #     Noisy/*.tif     (degraded / low-resolution input)
    #     Label/*.tif     (segmentation labels, integer masks)
    #     ValGT/*.tif
    #     ValNoisy/*.tif
    #     ValLabel/*.tif
    # ------------------------------------------------------------------

    # --- Training data ---
    head_dir = r"G:\VSCell\BioSR\MT\Training3\128data\Training"
    Training_GT_path   = head_dir + r"\GT\*.tif"
    Training_Raw_path  = head_dir + r"\Noisy\*.tif"
    Training_Label_path = head_dir + r"\Label\*.tif"

    # --- Validation data ---
    val_dir = r"G:\VSCell\BioSR\MT\Training3\128data\Training"
    Val_GT_path    = val_dir + r"\ValGT\*.tif"
    Val_Raw_path   = val_dir + r"\ValNoisy\*.tif"
    Val_Label_path = val_dir + r"\ValLabel\*.tif"

    # --- Pretrained RCAN weights ---
    rcan_pretrained_path = r"G:\VSCell\BioSR\MT\ori\logfile4\best_model.pth"

    # --- Output directory ---
    save_dir = head_dir + r"\fusion_model_ckpt_MTAug"

    # ==================================================================
    # Load data
    # ==================================================================
    print("[Step 1/3] Loading training data...")
    X_train     = AppendLoad16(Training_Raw_path)
    y_train     = AppendLoad16(Training_GT_path)
    label_train = AppendLoad16(Training_Label_path)

    print("[Step 2/3] Loading validation data...")
    X_val     = AppendLoad16(Val_Raw_path)
    y_val     = AppendLoad16(Val_GT_path)
    label_val = AppendLoad16(Val_Label_path)

    # Prepare segmentation labels: remove channel dim and convert to long
    label_train = label_train.squeeze(1)
    label_val   = label_val.squeeze(1)
    label_train = torch.tensor(label_train, dtype=torch.long)
    label_val   = torch.tensor(label_val, dtype=torch.long)
    label_train = label_train.squeeze()
    label_val   = label_val.squeeze()

    # Print data shapes
    print("X_train.shape    :", X_train.shape)
    print("y_train.shape    :", y_train.shape)
    print("X_val.shape      :", X_val.shape)
    print("y_val.shape      :", y_val.shape)
    print("label_train.shape:", label_train.shape)
    print("label_val.shape  :", label_val.shape)

    # ==================================================================
    # Start training
    # ==================================================================
    print("[Step 3/3] Starting training...")
    train16(
        X_train, y_train, label_train,
        X_val, y_val, label_val,
        rcan_pretrained_path,
        rcan_scale=2,
        save_dir=save_dir,
        epochs=20,
        batch_size=1,
        unet_lr=1e-4,
        rec_loss_weight=0.1,
        device='cuda'
    )
