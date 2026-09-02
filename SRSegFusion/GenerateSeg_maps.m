%% =========================================================================
%  generate_segmentation_maps.m
%  Generate binary segmentation maps from emitter localization maps.
%
%  Two modes are supported:
%    'gaussian' : Apply Gaussian blur first, then threshold (smooth boundaries)
%    'direct'   : Direct thresholding, with optional morphological dilation
%
%  Input : Emitter map TIFFs (grayscale, e.g. 1.tif, 2.tif, ...)
%  Output: Binary segmentation label TIFFs (16-bit, 0 or 65535)
%
%  Dependencies:
%    - Blur3D.m : custom 2D/3D convolution function (required for 'gaussian' mode).
%                 If not available, replace with imfilter() or conv2() as noted below.
%    - Image Processing Toolbox (for strel / imdilate when dilation is enabled)
%
%  Usage:
%    1. Modify the parameters in the CONFIG section below.
%    2. Run: generate_segmentation_maps
% =========================================================================

clear; clc; close all;

%% =========================================================================
% CONFIG — adjust these parameters before running
% =========================================================================

% --- I/O directories ---
out_dir  = 'Training3\';     % output root directory (Label/ and ValLabel/ will be created here)
raw_dir  = 'ori\';           % input directory containing emitter map TIFFs

% --- Dataset split ---
img_num    = 100;            % total number of input images
train_num  = 80;             % first N images go to train set, the rest to validation set

% --- Label replication ---
% Each input emitter map is replicated `level` times as labels.
% This matches augmented training inputs where multiple degraded images
% share the same ground-truth segmentation map.
level      = 11;             % number of label copies per input image

% --- Segmentation mode ---
%   'gaussian' : blur with a Gaussian kernel, then threshold (recommended for
%                sparse emitter maps; produces smoother connected regions)
%   'direct'   : threshold directly without blurring (sharper boundaries;
%                optionally followed by morphological dilation)
mode       = 'gaussian';

% --- Gaussian mode parameters (used only when mode == 'gaussian') ---
sigma      = 0.8;            % standard deviation of the Gaussian blur kernel (in pixels)

% --- Direct mode parameters (used only when mode == 'direct') ---
threshold_direct = 0.02;     % intensity threshold for binarization

% --- Shared threshold for gaussian mode ---
threshold_gaussian = 0.01;   % intensity threshold after Gaussian blur

% --- Morphological dilation (applies to both modes, optional) ---
enable_dilation = false;     % set true to dilate the binary mask after thresholding
dilation_size   = 1;         % square structuring element side length (in pixels)

%% =========================================================================
% VALIDATION & SETUP
% =========================================================================

% Validate mode
valid_modes = {'gaussian', 'direct'};
if ~ismember(mode, valid_modes)
    error('Invalid mode: %s. Choose ''gaussian'' or ''direct''.', mode);
end

% Create output directories
if ~exist([out_dir, 'Label\'], 'dir')
    mkdir([out_dir, 'Label\']);
end
if ~exist([out_dir, 'ValLabel\'], 'dir')
    mkdir([out_dir, 'ValLabel\']);
end

% Prepare structuring element if dilation is enabled
if enable_dilation
    se = strel('square', dilation_size);
    fprintf('[Config] Dilation enabled: square structuring element of size %d\n', dilation_size);
else
    se = [];
    fprintf('[Config] Dilation disabled\n');
end

% Print configuration
fprintf('============================================================\n');
fprintf(' Segmentation Map Generation\n');
fprintf('============================================================\n');
fprintf(' Mode              : %s\n', mode);
fprintf(' Input directory   : %s\n', raw_dir);
fprintf(' Output directory  : %s\n', out_dir);
fprintf(' Total images      : %d\n', img_num);
fprintf(' Train split       : %d (train) / %d (val)\n', train_num, img_num - train_num);
fprintf(' Label copies      : %d per image\n', level);
if strcmp(mode, 'gaussian')
    fprintf(' Gaussian sigma    : %.2f\n', sigma);
    fprintf(' Threshold         : %.2f\n', threshold_gaussian);
else
    fprintf(' Threshold         : %.2f\n', threshold_direct);
end
fprintf('============================================================\n');

%% =========================================================================
% MAIN LOOP
% =========================================================================

train_count = 1;   % counter for training label filenames
val_count   = 1;   % counter for validation label filenames

for i = 1:img_num
    fprintf('Processing image %d / %d ...\n', i, img_num);

    % --- Read and normalize input emitter map ---
    abgt = single(imread([raw_dir, num2str(i), '.tif']));
    abgt = abgt / max(abgt(:));   % normalize to [0, 1]
    [d1, d2] = size(abgt);

    % --- Mode-specific processing ---
    switch mode
        case 'gaussian'
            % Generate 2D Gaussian kernel
            [x, y] = meshgrid( ...
                linspace(-(d2-1)/2, (d2-1)/2, d2), ...
                linspace(-(d1-1)/2, (d1-1)/2, d1));
            gauss_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
            gauss_kernel = gauss_kernel / sum(gauss_kernel(:));   % normalize to sum=1

            % Apply Gaussian blur
            % NOTE: Blur3D is a custom convolution function. If unavailable,
            % replace the line below with:
            %   abgt = imfilter(abgt, gauss_kernel, 'same', 'symmetric');
            abgt = Blur3D(abgt, gauss_kernel);

            % Threshold to binary
            abgt(abgt >= threshold_gaussian) = 1;
            abgt(abgt <  threshold_gaussian) = 0;

        case 'direct'
            % Direct thresholding (no blur)
            abgt(abgt >= threshold_direct) = 1;
            abgt(abgt <  threshold_direct) = 0;
    end

    % --- Optional morphological dilation ---
    if enable_dilation
        abgt = imdilate(abgt, se);
    end

    % --- Replicate and save labels ---
    % Each emitter map produces `level` identical label copies,
    % matching the number of augmented input images in training.
    for j = 1:level
        if i <= train_num
            % Training set
            imwrite(uint16(abgt * 65535), ...
                [out_dir, 'Label\', num2str(train_count), '.tif']);
            train_count = train_count + 1;
        else
            % Validation set
            imwrite(uint16(abgt * 65535), ...
                [out_dir, 'ValLabel\', num2str(val_count), '.tif']);
            val_count = val_count + 1;
        end
    end
end

%% =========================================================================
% SUMMARY
% =========================================================================

fprintf('\n============================================================\n');
fprintf(' Generation complete!\n');
fprintf(' Training labels   : %d files saved to %sLabel\\\n', train_count - 1, out_dir);
fprintf(' Validation labels : %d files saved to %sValLabel\\\n', val_count - 1, out_dir);
fprintf('============================================================\n');
