%% =========================================================================
%  Generate cropped and augmented training/validation patches from
%  paired widefield (Noisy), ground-truth (GT), and segmentation (Label) images.
%
%  Pipeline per image:
%    1. Read paired Noisy / GT / Label TIFF stacks
%    2. Random crop a patch of size `aimingsize` (Noisy) with proportional
%       crops for GT and Label (handled by RandomCropTriple)
%    3. Quality filtering: keep patches with sufficient signal (max and mean
%       of the GT patch above thresholds)
%    4. Random rotation augmentation (0 / 90 / 180 / 270 degrees)
%    5. Save as 16-bit TIFFs
%
%  Dependencies:
%    - readMTiffn.m        : read 16-bit multi-page TIFF
%    - writeMTiffnOriginal.m : write 16-bit TIFF
%    - (both must be on the MATLAB path)
%
%  Usage:
%    1. Modify parameters in the CONFIG section below.
%    2. Run: prepare_training_data
% =========================================================================

clear; clc; close all;

%% =========================================================================
% CONFIG — adjust these parameters before running
% =========================================================================

% --- Paths ---
% Add custom toolbox directories (modify or remove if not needed)
addpath('D:\Supplementary Files for BioSR\Supplementary Files for BioSR\IO_MRC_MATLAB');
addpath('D:\BasicOperation');

% Input directories (relative to headpath)
headpath       = '';               % prefix for input directories
train_raw_dir  = 'Noisy\';         % training noisy input subfolder
train_gt_dir   = 'GT\';            % training ground-truth subfolder
train_label_dir= 'Label\';         % training segmentation label subfolder
val_raw_dir    = 'ValNoisy\';      % validation noisy input subfolder
val_gt_dir     = 'ValGT\';         % validation ground-truth subfolder
val_label_dir  = 'ValLabel\';      % validation segmentation label subfolder

% --- Cropping ---
aimingsize     = 128;              % crop size for the Noisy (low-res) image
totalwant_num  = 15000;            % target total number of training patches
train_frame    = 720;              % number of training source images
val_frame      = 180;              % number of validation source images

% --- Quality filtering ---
% A crop is kept only if BOTH conditions are satisfied:
%   max(GT_patch)  > max_fraction * max(full_GT_image)
%   mean(GT_patch) > mean_fraction * max(full_GT_image)
max_fraction   = 0.05;             % threshold for patch maximum intensity
mean_fraction  = 0.01;             % threshold for patch mean intensity

% --- Augmentation ---
enable_rotation = true;            % enable random 0/90/180/270 degree rotation

% --- Normalization ---
use_norm       = 0;                % 1 = per-patch min-max normalization to [0, 65535]; 0 = keep original range

%% =========================================================================
% DERIVED PARAMETERS & OUTPUT SETUP
% =========================================================================

% Output root directory
out_root = ['.\', num2str(aimingsize), 'data\Training\'];

% Create output directories
out_dirs = {
    'GT\', 'Noisy\', 'Label\', ...
    'ValGT\', 'ValNoisy\', 'ValLabel\'
};
for k = 1:length(out_dirs)
    if ~exist([out_root, out_dirs{k}], 'dir')
        mkdir([out_root, out_dirs{k}]);
    end
end

% Number of crops per source image
crop_num = fix(totalwant_num / train_frame) + 1;

fprintf('============================================================\n');
fprintf(' Training Data Preparation\n');
fprintf('============================================================\n');
fprintf(' Crop size         : %d x %d (Noisy resolution)\n', aimingsize, aimingsize);
fprintf(' Training images   : %d\n', train_frame);
fprintf(' Validation images : %d\n', val_frame);
fprintf(' Crops per image   : %d\n', crop_num);
fprintf(' Target train crops: %d\n', totalwant_num);
fprintf(' Quality filter    : max > %.2f * img_max, mean > %.2f * img_max\n', ...
    max_fraction, mean_fraction);
fprintf(' Rotation aug      : %s\n', iif(enable_rotation, 'enabled', 'disabled'));
fprintf(' Per-patch norm    : %s\n', iif(use_norm, 'enabled', 'disabled'));
fprintf(' Output directory  : %s\n', out_root);
fprintf('============================================================\n');

%% =========================================================================
% TRAINING SET
% =========================================================================

fprintf('\n--- Generating training patches ---\n');

GTpath    = [out_root, 'GT\'];
Noisypath = [out_root, 'Noisy\'];
Labelpath = [out_root, 'Label\'];

count = 1;
for i = 1:train_frame
    fprintf('  Train image %d / %d ...\n', i, train_frame);

    % Read paired images
    I1 = readMTiffn([headpath, train_raw_dir,  num2str(i), '.tif'], 16);  % Noisy
    I2 = readMTiffn([headpath, train_gt_dir,   num2str(i), '.tif'], 16);  % GT
    I3 = readMTiffn([headpath, train_label_dir,num2str(i), '.tif'], 16);  % Label
    I1 = double(I1);
    I2 = double(I2);
    I3 = double(I3);

    img_max = max(I2(:));

    for cropnum = 1:crop_num
        % Random synchronized crop (Noisy at aimingsize, GT/Label proportional)
        [out_img_raw, out_img_GT, out_img_label] = ...
            RandomCropTriple(I1, I2, I3, aimingsize);

        % Quality filtering: skip patches with too little signal
        if max(out_img_GT(:)) <= max_fraction * img_max
            continue;
        end
        if mean(out_img_GT(:)) <= mean_fraction * img_max
            continue;
        end

        % Optional per-patch min-max normalization
        if use_norm == 1
            out_img_raw   = normalize_patch(out_img_raw);
            out_img_GT    = normalize_patch(out_img_GT);
            out_img_label = normalize_patch(out_img_label);
        end

        % Random rotation augmentation (0 / 90 / 180 / 270 degrees)
        if enable_rotation
            Rotater = randi([1, 4]);
            switch Rotater
                case 1  % 0 degrees (original)
                    % no change
                case 2  % 90 degrees (transpose)
                    out_img_raw   = out_img_raw';
                    out_img_GT    = out_img_GT';
                    out_img_label = out_img_label';
                case 3  % 180 degrees (flip)
                    out_img_raw   = flip(out_img_raw);
                    out_img_GT    = flip(out_img_GT);
                    out_img_label = flip(out_img_label);
                case 4  % 270 degrees (transpose + flip)
                    out_img_raw   = flip(out_img_raw');
                    out_img_GT    = flip(out_img_GT');
                    out_img_label = flip(out_img_label');
            end
        end

        % Save patches
        writeMTiffnOriginal(out_img_GT,    [GTpath,    num2str(count), '.tif'], 16);
        writeMTiffnOriginal(out_img_raw,   [Noisypath, num2str(count), '.tif'], 16);
        writeMTiffnOriginal(out_img_label, [Labelpath, num2str(count), '.tif'], 16);
        count = count + 1;
    end
end

fprintf('  Training patches generated: %d\n', count - 1);

%% =========================================================================
% VALIDATION SET
% =========================================================================

fprintf('\n--- Generating validation patches ---\n');

GTpath    = [out_root, 'ValGT\'];
Noisypath = [out_root, 'ValNoisy\'];
Labelpath = [out_root, 'ValLabel\'];

count = 1;
for i = 1:val_frame
    fprintf('  Val image %d / %d ...\n', i, val_frame);

    % Read paired images
    I1 = readMTiffn([headpath, val_raw_dir,  num2str(i), '.tif'], 16);  % Noisy
    I2 = readMTiffn([headpath, val_gt_dir,   num2str(i), '.tif'], 16);  % GT
    I3 = readMTiffn([headpath, val_label_dir,num2str(i), '.tif'], 16);  % Label
    I1 = double(I1);
    I2 = double(I2);
    I3 = double(I3);

    img_max = max(I2(:));

    for cropnum = 1:crop_num
        % Random synchronized crop
        [out_img_raw, out_img_GT, out_img_label] = ...
            RandomCropTriple(I1, I2, I3, aimingsize);

        % Quality filtering
        if max(out_img_GT(:)) <= max_fraction * img_max
            continue;
        end
        if mean(out_img_GT(:)) <= mean_fraction * img_max
            continue;
        end

        % Optional per-patch normalization
        if use_norm == 1
            out_img_raw   = normalize_patch(out_img_raw);
            out_img_GT    = normalize_patch(out_img_GT);
            out_img_label = normalize_patch(out_img_label);
        end

        % Random rotation augmentation
        if enable_rotation
            Rotater = randi([1, 4]);
            switch Rotater
                case 1  % 0 degrees
                    % no change
                case 2  % 90 degrees
                    out_img_raw   = out_img_raw';
                    out_img_GT    = out_img_GT';
                    out_img_label = out_img_label';
                case 3  % 180 degrees
                    out_img_raw   = flip(out_img_raw);
                    out_img_GT    = flip(out_img_GT);
                    out_img_label = flip(out_img_label);
                case 4  % 270 degrees
                    out_img_raw   = flip(out_img_raw');
                    out_img_GT    = flip(out_img_GT');
                    out_img_label = flip(out_img_label');
            end
        end

        % Save patches
        writeMTiffnOriginal(out_img_GT,    [GTpath,    num2str(count), '.tif'], 16);
        writeMTiffnOriginal(out_img_raw,   [Noisypath, num2str(count), '.tif'], 16);
        writeMTiffnOriginal(out_img_label, [Labelpath, num2str(count), '.tif'], 16);
        count = count + 1;
    end
end

fprintf('  Validation patches generated: %d\n', count - 1);

%% =========================================================================
% SUMMARY
% =========================================================================

fprintf('\n============================================================\n');
fprintf(' Data preparation complete!\n');
fprintf(' Output saved to: %s\n', out_root);
fprintf('   GT\\       Noisy\\       Label\\       (training)\n');
fprintf('   ValGT\\    ValNoisy\\    ValLabel\\    (validation)\n');
fprintf('============================================================\n');

%% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function out = normalize_patch(img)
    % Per-patch min-max normalization to [0, 65535] (16-bit range).
    img = double(img);
    out = (img - min(img(:))) / (max(img(:)) - min(img(:))) * 65535;
end


function [cropped_x, cropped_y, cropped_z] = RandomCropTriple(X, Y, Z, w_small)
    %
    % RandomCropTriple  Random synchronized crop of three paired images.
    %
    %   X is cropped at the base resolution (w_small x w_small).
    %   Y and Z are cropped at a proportionally larger resolution
    %   (determined by the size ratio Y/X), at the corresponding spatial
    %   location. Y and Z must have identical spatial dimensions.
    %
    % Inputs:
    %   X       - base-resolution image (e.g. degraded / Noisy)
    %   Y       - high-resolution image 1 (e.g. restoration GT)
    %   Z       - high-resolution image 2 (e.g. segmentation Label, same
    %             size as Y)
    %   w_small - crop side length for X (square)
    %
    % Outputs:
    %   cropped_x - cropped X
    %   cropped_y - cropped Y (proportional location and size)
    %   cropped_z - cropped Z (same crop as Y)
    %

    % Get dimensions
    [rowsX, colsX, ~] = size(X);
    [rowsYZ, colsYZ, ~] = size(Y);

    % Scale factor between high-res (Y/Z) and base-res (X)
    scaler = rowsYZ / rowsX;

    % Random crop origin in X (base resolution), guaranteed within bounds
    x = randi([1, rowsX - w_small]);  % row start
    y = randi([1, colsX - w_small]);  % col start

    % Corresponding crop origin and size for Y/Z (high resolution)
    x2 = round(x * scaler);
    y2 = round(y * scaler);
    w_small2 = round(w_small * scaler);

    % Crop all three
    cropped_x = X(x:x + w_small - 1,  y:y + w_small - 1,  :);
    cropped_y = Y(x2:x2 + w_small2 - 1, y2:y2 + w_small2 - 1, :);
    cropped_z = Z(x2:x2 + w_small2 - 1, y2:y2 + w_small2 - 1, :);
end
