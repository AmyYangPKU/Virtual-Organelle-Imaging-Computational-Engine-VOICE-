"""
Fusion Model Inference Script
Supports two modes:
    1. Tiling / stitching inference (for large images, with weighted overlap blending)
    2. Full-size inference (for images that fit in GPU memory)

Usage:
    1. Modify the paths and settings in the CONFIG section below.
    2. Run: python test.py

Dependencies:
    - train.py (FusionModel and other imports must be available)
    - AppendLoad.py (or ensure glob, numpy, etc. are imported)
    - torch, tifffile, numpy, tqdm
"""

from train import *
import torch
import tifffile
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
import glob


# ============================================================================
# CONFIG — modify these paths and settings before running
# ============================================================================

# --- Data paths ---
head_dir = r'G:\VSCell\BioSR\Factin\Training\128data\Training'
Raw_path = head_dir + r'\ValNoisy'          # directory containing input TIFFs

# --- Model paths ---
rcan_pretrained_path = r'G:\VSCell\BioSR\MT\Training_\128data_Res\Training\logfile\RCAN510\best_model.pth'
model_path = r'G:\VSCell\FusionCode\fusion_model_ckpt_FactinAug\best_model_seg_loss.pth'

# --- Model settings ---
rcan_scale = 2          # RCAN upscaling factor
num_classes = 2         # number of segmentation classes
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Inference mode: 'tiling' or 'full' ---
#   'tiling' : sliding-window with weighted stitching (recommended for large images)
#   'full'   : feed the entire image to the model at once
inference_mode = 'tiling'

# --- Tiling settings (only used when inference_mode='tiling') ---
tile_size = 128         # tile size for inference (input resolution)
overlap = 0             # overlap between adjacent tiles (in pixels)

# --- Output suffixes ---
model_label = 'FusionModel'
end_str = '.tif'
end_str_seg = '_seg.tif'


# ============================================================================
# Helper Functions
# ============================================================================

def percentile_normalize(arr, lower_p=2, upper_p=100):
    """Percentile-based normalization to [0, 1]."""
    lower = np.percentile(arr, lower_p, axis=None, keepdims=True)
    upper = np.percentile(arr, upper_p, axis=None, keepdims=True)
    normalized_arr = (arr - lower) / (upper - lower + 1e-8)
    return normalized_arr


def get_weight_mask(size, overlap_size):
    """
    Generate a weight mask for smooth stitching.
    Edge weights linearly decay toward the boundary.

    Args:
        size: tile size (square)
        overlap_size: overlap region width in pixels

    Returns:
        2D numpy array of shape (size, size) with weights in [0, 1]
    """
    mask = np.ones((size, size), dtype=np.float32)
    if overlap_size <= 0:
        return mask
    # Create linear ramp
    ramp = np.linspace(0, 1, overlap_size).astype(np.float32)
    # Apply to all four borders
    mask[:overlap_size, :] *= ramp[:, None]        # top
    mask[-overlap_size:, :] *= ramp[::-1, None]    # bottom
    mask[:, :overlap_size] *= ramp[None, :]        # left
    mask[:, -overlap_size:] *= ramp[None, ::-1]    # right
    return mask


# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load the fusion model with pretrained RCAN and trained weights."""
    model = FusionModel(rcan_pretrained_path, rcan_scale).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Info] Loaded model from: {model_path}")
    print(f"[Info] Device: {device}")
    return model


# ============================================================================
# Tiling / Stitching Inference
# ============================================================================

def inference_tiling(model):
    """
    Sliding-window inference with weighted overlap blending.
    Suitable for large images that do not fit in GPU memory.
    """
    # Create output directory
    save_path = os.path.join(head_dir, 'output', model_label)
    os.makedirs(save_path, exist_ok=True)

    # Collect image list
    img_list = [f for f in os.listdir(Raw_path) if f.endswith(end_str)]
    if not img_list:
        print(f"[Warning] No TIFF files found in: {Raw_path}")
        return

    stride = tile_size - overlap

    # Pre-generate weight masks for input and output resolutions
    mask_in = get_weight_mask(tile_size, overlap)
    mask_out = F.interpolate(
        torch.from_numpy(mask_in).unsqueeze(0).unsqueeze(0),
        size=(tile_size * rcan_scale, tile_size * rcan_scale),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()

    pbar = tqdm(img_list, desc="Tiling Inference")
    for img_name in pbar:
        # 1. Read and preprocess
        img_path = os.path.join(Raw_path, img_name)
        I = tifffile.imread(img_path).astype(np.float32)
        I /= (I.max() + 1e-8)   # global normalization to avoid inter-tile brightness mismatch
        h, w = I.shape

        # 2. Initialize large-image buffers
        # Output size is rcan_scale times the input size
        out_h, out_w = h * rcan_scale, w * rcan_scale
        t_tile = tile_size * rcan_scale
        rcan_accum = np.zeros((out_h, out_w), dtype=np.float32)
        seg_accum = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
        weight_accum = np.zeros((out_h, out_w), dtype=np.float32)

        # 3. Sliding-window inference
        # Boundary handling: if the last tile does not fit, shift it to the edge
        for y in range(0, h - overlap, stride):
            for x in range(0, w - overlap, stride):
                y_s = min(y, h - tile_size)
                x_s = min(x, w - tile_size)

                # Crop and convert to tensor
                crop = I[y_s:y_s + tile_size, x_s:x_s + tile_size]
                im_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    seg_out, rcan_out = model(im_t)
                    # seg_prob: (num_classes, t_tile, t_tile)
                    # rcan_res: (t_tile, t_tile)
                    seg_prob = F.softmax(seg_out, dim=1).squeeze().cpu().numpy()
                    rcan_res = rcan_out.squeeze().cpu().numpy()

                # Map to large-image coordinates
                ty, tx = y_s * rcan_scale, x_s * rcan_scale

                # Accumulate results with weighting
                rcan_accum[ty:ty + t_tile, tx:tx + t_tile] += rcan_res * mask_out
                seg_accum[:, ty:ty + t_tile, tx:tx + t_tile] += seg_prob * mask_out
                weight_accum[ty:ty + t_tile, tx:tx + t_tile] += mask_out

                # Stop at boundary to avoid duplicate processing
                if x_s + tile_size >= w:
                    break
            if y_s + tile_size >= h:
                break

        # 4. Fuse and post-process
        # Normalize by accumulated weights
        rcan_final = rcan_accum / (weight_accum + 1e-8)
        seg_final_prob = seg_accum / (weight_accum + 1e-8)
        seg_final = np.argmax(seg_final_prob, axis=0)

        # Normalize and convert to 16-bit
        rcan_final = (rcan_final - rcan_final.min()) / (rcan_final.max() - rcan_final.min() + 1e-8)
        rcan_final = (rcan_final * 65535).astype(np.uint16)
        seg_final = (seg_final * 65535).astype(np.uint16)

        # 5. Save results
        save_name_rcan = img_name.replace(end_str, f'_{model_label}{end_str}')
        save_name_seg = img_name.replace(end_str, f'_{model_label}{end_str_seg}')
        tifffile.imwrite(os.path.join(save_path, save_name_rcan), rcan_final)
        tifffile.imwrite(os.path.join(save_path, save_name_seg), seg_final)

    pbar.close()
    print(f"[Done] Tiling inference complete. Results saved to: {save_path}")


# ============================================================================
# Full-Size Inference
# ============================================================================

def inference_full(model):
    """
    Full-image inference (no tiling).
    Suitable for small images that fit in GPU memory.
    Expects numerically named files (1.tif, 2.tif, ...).
    """
    # Create output directory
    save_path0 = os.path.join(head_dir, 'output')
    if not os.path.exists(save_path0):
        os.mkdir(save_path0)
    save_path = os.path.join(head_dir, 'output', model_label)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Collect image list (numerically named)
    all_imgs_path = glob.glob(os.path.join(Raw_path, '*.tif'))
    num = len(all_imgs_path)
    if num == 0:
        print(f"[Warning] No TIFF files found in: {Raw_path}")
        return

    pbar = tqdm(total=num, desc="Full-size Inference")
    for i in range(num):
        numstr = str(i + 1)
        I = tifffile.imread(os.path.join(Raw_path, numstr + end_str))
        I = np.array(I)
        I = I / (I.max() + 1e-8)
        # Optional: percentile normalization (commented out by default)
        # I = percentile_normalize(I, lower_p=0.2, upper_p=98)
        I[I < 0] = 0
        I[I > 1] = 1

        im = torch.tensor(I, dtype=torch.float32)
        im = im.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            seg_out, rcan_out = model(im)
            seg_pred = F.softmax(seg_out, dim=1).argmax(dim=1).squeeze().cpu().numpy()
            rcan_out = rcan_out.squeeze().cpu().numpy()

            # Post-process RCAN output
            rcan_out = rcan_out - rcan_out.min()
            rcan_out = rcan_out / (rcan_out.max() + 1e-8)
            rcan_out[rcan_out < 0] = 0
            rcan_out = (2 ** 16 - 1) * rcan_out
            rcan_out = rcan_out.astype(np.uint16)

            # Post-process segmentation output
            seg_pred = 65535 * seg_pred
            seg_pred = seg_pred.astype(np.uint16)

            # Save
            tifffile.imwrite(os.path.join(save_path, numstr + model_label + end_str), rcan_out)
            tifffile.imwrite(os.path.join(save_path, numstr + model_label + end_str_seg), seg_pred)

        pbar.update(1)

    pbar.close()
    print(f"[Done] Full-size inference complete. Results saved to: {save_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"[Config] Inference mode: {inference_mode}")
    print(f"[Config] Input directory: {Raw_path}")
    print(f"[Config] RCAN scale: {rcan_scale}, Num classes: {num_classes}")

    # Load model
    model = load_model()

    # Run inference
    if inference_mode == 'tiling':
        inference_tiling(model)
    elif inference_mode == 'full':
        inference_full(model)
    else:
        print(f"[Error] Unknown inference_mode: '{inference_mode}'. "
              f"Use 'tiling' or 'full'.")
