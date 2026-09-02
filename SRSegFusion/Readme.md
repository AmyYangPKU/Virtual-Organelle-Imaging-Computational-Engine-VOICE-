# VOICE FusionSR-Seg: Joint Super-Resolution and Segmentation for Fluorescence Microscopy

A deep learning framework that jointly performs **image restoration (super-resolution)** and **binary segmentation** for fluorescence microscopy images. The model consists of a pretrained RCAN super-resolution branch followed by an Attention U-Net segmentation branch, trained with a hybrid loss that combines segmentation loss (Dice + Cross-Entropy) and restoration loss (L1).

---

## Overview

This project addresses the task of restoring low-resolution / noisy fluorescence microscopy images while simultaneously producing a binary segmentation map of organelle structures. The pipeline has three stages:

1. **Data Generation** — Produce paired low-resolution (Noisy), high-resolution (GT), and segmentation label (Label) image patches from raw emitter localization maps.
2. **Pretrained Super-Resolution** — An RCAN model is first pretrained on the Noisy→GT pairs (pretrained weights required as input).
3. **Joint Training & Inference** — The fusion model (RCAN + Attention U-Net) is trained end-to-end, and can be used for inference on new images.


## Repository Structure

```
.
├── train.py                         # Joint model training script (16-bit AMP)
├── test.py                          # Inference script (tiling / full-image modes)
├── generate_segmentation_maps.m     # MATLAB: generate binary segmentation maps from emitter maps
├── prepare_training_data.m          # MATLAB: crop & augment paired Noisy/GT/Label patches
├── utils.m                          # MATLAB utility functions (readMTiffn / writeMTiffnOriginal / Blur3D)
├── RCAN2D.py                        # RCAN super-resolution network (required)
├── AUNet.py                         # Attention U-Net segmentation network (required)
├── FusionModel.py                   # (optional) standalone fusion model definition
├── loss.py                          # (optional) standalone loss functions
├── AppendLoad16.py                  # (optional) standalone 16-bit TIFF loader
└── README.md
```

> **Note:** `RCAN2D.py` and `AUNet.py` must be present in the working directory or importable. The MATLAB scripts require the utility functions from `utils.m` (`readMTiffn`, `writeMTiffnOriginal`, `Blur3D`). See [MATLAB Utility Functions](#matlab-utility-functions) for usage details.


### MATLAB (data preparation)
- MATLAB R2018b or later
- Image Processing Toolbox (for morphological dilation, optional)

## Quick Start

### Option 1: Use the provided dataset

Download the preprocessed dataset and extract it. The dataset should contain the following structure:

```
dataset_root/
├── GT/          1.tif, 2.tif, ...    (high-resolution restoration GT, 16-bit)
├── Noisy/       1.tif, 2.tif, ...    (low-resolution / noisy input, 16-bit)
├── Label/       1.tif, 2.tif, ...    (binary segmentation labels, 16-bit: 0 or 65535)
├── ValGT/       1.tif, 2.tif, ...
├── ValNoisy/    1.tif, 2.tif, ...
└── ValLabel/    1.tif, 2.tif, ...
```

Then skip to the [Training](#training) section.

---

### Option 2: Prepare your own custom dataset

If you want to generate training data from raw emitter localization maps, follow these steps.

#### Step 1: Generate high/low-resolution image pairs

Use the **organelle structure high/low-resolution image pair generation software** to produce:
- **GT images** — high-resolution restored images
- **Noisy images** — degraded / low-resolution images at the specified noise level
- **Emitter maps** — emitter localization maps (used to generate segmentation labels)

Place these in a working directory, e.g.:
```
raw_data/
├── GT/           1.tif, 2.tif, ..., N.tif
├── Noisy/        1.tif, 2.tif, ..., N.tif
└── emitter/      1.tif, 2.tif, ..., N.tif
```

#### Step 2: Generate segmentation maps

Run `generate_segmentation_maps.m` in MATLAB to convert emitter maps into binary segmentation labels.

Open the script and configure the parameters at the top:

```matlab
% --- I/O directories ---
out_dir  = 'raw_data\';     % output root (Label/ and ValLabel/ created here)
raw_dir  = 'raw_data\emitter\';  % input emitter map directory

% --- Dataset split ---
img_num    = 100;           % total emitter maps
train_num  = 80;            % first N -> train, rest -> validation

% --- Segmentation mode ---
%   'gaussian' : Gaussian blur + threshold (smooth connected regions)
%   'direct'   : direct threshold + optional dilation (sharp boundaries)
mode       = 'gaussian';

% --- Gaussian mode parameters ---
sigma              = 0.8;   % Gaussian kernel std (pixels)
threshold_gaussian = 0.01;  % binarization threshold after blur

% --- Direct mode parameters ---
threshold_direct   = 0.02;  % binarization threshold

% --- Optional dilation ---
enable_dilation  = false;   % set true to dilate binary masks
dilation_size    = 1;       % structuring element size
```

Then run:
```matlab
generate_segmentation_maps
```

This produces:
```
raw_data/
├── Label/       1.tif, 2.tif, ...    (training segmentation labels)
└── ValLabel/    1.tif, 2.tif, ...    (validation segmentation labels)
```

#### Step 3: Crop and augment patches

Run `prepare_training_data.m` in MATLAB to randomly crop and augment the three paired image types (Noisy, GT, Label) into training patches.

Configure the parameters at the top:

```matlab
% --- Input directories ---
headpath        = 'raw_data\';
train_raw_dir   = 'Noisy\';
train_gt_dir    = 'GT\';
train_label_dir = 'Label\';
val_raw_dir     = 'ValNoisy\';      % or 'Noisy\' if no separate val set
val_gt_dir      = 'ValGT\';
val_label_dir   = 'ValLabel\';

% --- Cropping ---
aimingsize     = 128;              % crop size at Noisy resolution
totalwant_num  = 15000;            % target number of training patches
train_frame    = 720;              % number of training source images
val_frame      = 180;              % number of validation source images

% --- Quality filtering ---
max_fraction   = 0.05;             % keep patch if max(GT) > 0.05 * image_max
mean_fraction  = 0.01;             % keep patch if mean(GT) > 0.01 * image_max

% --- Augmentation ---
enable_rotation = true;            % random 0/90/180/270 deg rotation
use_norm        = 0;               % 1 = per-patch min-max norm, 0 = keep range
```

Then run:
```matlab
prepare_training_data
```

This produces the final dataset structure:
```
128data/Training/
├── GT/          1.tif, 2.tif, ...
├── Noisy/       1.tif, 2.tif, ...
├── Label/       1.tif, 2.tif, ...
├── ValGT/       1.tif, 2.tif, ...
├── ValNoisy/    1.tif, 2.tif, ...
└── ValLabel/    1.tif, 2.tif, ...
```

---

## Training

### 1. Prepare pretrained RCAN weights

The fusion model requires a **pretrained RCAN super-resolution model** (trained on Noisy→GT pairs). Specify the path to the `.pth` checkpoint.

### 2. Configure training parameters

Open `train.py` and modify the `if __name__ == "__main__"` section:

```python
# --- Training data ---
head_dir = r"path/to/128data/Training"
Training_GT_path    = head_dir + r"\GT\*.tif"
Training_Raw_path   = head_dir + r"\Noisy\*.tif"
Training_Label_path = head_dir + r"\Label\*.tif"

# --- Validation data ---
val_dir = r"path/to/128data/Training"
Val_GT_path    = val_dir + r"\ValGT\*.tif"
Val_Raw_path   = val_dir + r"\ValNoisy\*.tif"
Val_Label_path = val_dir + r"\ValLabel\*.tif"

# --- Pretrained RCAN weights ---
rcan_pretrained_path = r"path/to/pretrained_rcan/best_model.pth"

# --- Output ---
save_dir = head_dir + r"\fusion_model_ckpt"

# --- Hyperparameters ---
epochs          = 20
batch_size      = 1
unet_lr         = 1e-4       # U-Net learning rate (RCAN uses 1/10 of this)
rec_loss_weight = 0.1        # weight for auxiliary L1 restoration loss
device          = 'cuda'     # or 'cpu'
```

### 3. Run training

```bash
python train.py
```

Training uses **16-bit mixed precision (AMP)** for memory efficiency. The following checkpoints are saved to `save_dir`:

| File | Description |
|---|---|
| `best_model_compound_loss.pth` | Best validation compound loss (seg + weighted rec) |
| `best_model_seg_loss.pth` | Best validation segmentation loss |
| `last_fusion_model.pth` | Final epoch checkpoint |

### Training loss

The total loss is:
```
L_total = L_seg + rec_loss_weight * L_rec
```
where:
- `L_seg = DiceLoss + CrossEntropyLoss` (primary segmentation objective)
- `L_rec = L1Loss(rcan_output, GT)` (auxiliary restoration objective)

A gradient-detach strategy is used so that the RCAN branch receives gradients from both losses while the U-Net branch is optimized primarily for segmentation.

---

## Inference

Open `test.py` and configure the parameters at the top:

```python
# --- Data paths ---
head_dir = r"path/to/data"
Raw_path = head_dir + r"\ValNoisy"     # directory of input TIFFs

# --- Model paths ---
rcan_pretrained_path = r"path/to/pretrained_rcan/best_model.pth"
model_path = r"path/to/fusion_model_ckpt/best_model_seg_loss.pth"

# --- Settings ---
rcan_scale     = 2          # RCAN upscaling factor
num_classes    = 2          # number of segmentation classes
device         = 'cuda'

# --- Inference mode ---
#   'tiling' : sliding-window with weighted stitching (for large images)
#   'full'   : feed entire image at once (for small images)
inference_mode = 'tiling'

# --- Tiling settings (only for 'tiling' mode) ---
tile_size  = 128
overlap    = 0
```

Run:
```bash
python test.py
```

### Output

For each input image `X.tif`, two outputs are saved to `head_dir/output/FusionModel/`:

| File | Description |
|---|---|
| `X_FusionModel.tif` | RCAN super-resolution / restoration result (16-bit) |
| `X_FusionModel_seg.tif` | Binary segmentation map (16-bit: 0 or 65535) |

### Tiling vs. Full mode

- **`tiling`** (recommended): Slides a `tile_size × tile_size` window over the image, runs inference on each tile, and blends overlaps with a linear ramp weight mask. Suitable for large images that do not fit in GPU memory.
- **`full`**: Feeds the entire image to the model in one pass. Suitable for small images. Expects numerically named files (`1.tif`, `2.tif`, ...).

---

## Data Format

All images are **16-bit grayscale TIFFs**:

| Image Type | Range | Description |
|---|---|---|
| Noisy (input) | [0, 65535] | Low-resolution / degraded fluorescence image |
| GT (restoration) | [0, 65535] | High-resolution ground truth |
| Label (segmentation) | {0, 65535} | Binary mask: 0 = background, 65535 = foreground |

During training, images are normalized to [0, 1] by dividing by 65535.


---

## License

This project is released under the MIT License.


