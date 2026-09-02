%% =========================================================================
%  utils.m
%  Collection of utility functions for TIFF I/O and frequency-domain blurring.
%
%  Due to MATLAB's one-public-function-per-file rule, this file uses a
%  dispatcher pattern. Call the utilities via:
%
%      data   = utils('readMTiffn', filepath, bitdepth);
%              utils('writeMTiffnOriginal', stack, filepath, bitdepth);
%      blurred = utils('Blur3D', image, psf);
%
%  If you prefer direct function calls (e.g. readMTiffn(...)), split each
%  local function below into its own .m file (readMTiffn.m,
%  writeMTiffnOriginal.m, Blur3D.m) and add the containing folder to the
%  MATLAB path. This is the standard MATLAB convention and is fully
%  compatible with generate_segmentation_maps.m and prepare_training_data.m.
%
%  Functions:
%    readMTiffn(raw_file, n)       - read multi-page TIFF (8/16/32-bit)
%    writeMTiffnOriginal(f_stack, filename, n) - append multi-page TIFF
%    Blur3D(f, PSF)                - FFT-based convolution (2D or 3D)
% =========================================================================

function varargout = utils(func_name, varargin)
    switch func_name
        case 'readMTiffn'
            [varargout{1:nargout}] = readMTiffn(varargin{:});
        case 'writeMTiffnOriginal'
            writeMTiffnOriginal(varargin{:});
        case 'Blur3D'
            [varargout{1:nargout}] = Blur3D(varargin{:});
        otherwise
            error('utils: unknown function ''%s''. Use ''readMTiffn'', ''writeMTiffnOriginal'', or ''Blur3D''.', func_name);
    end
end


%% =========================================================================
%  readMTiffn  —  Read a multi-page TIFF stack
%
%  Inputs:
%    raw_file : path to the TIFF file
%    n        : bit depth (8, 16, or 32)
%
%  Output:
%    raw_data : H x W x N stack of type uint8/uint16/uint32
% =========================================================================

function raw_data = readMTiffn(raw_file, n)
    info   = imfinfo(raw_file);
    frames = numel(info);

    if n == 8
        raw_data = zeros(info(1).Height, info(1).Width, frames, 'uint8');
        for k = 1:frames
            raw_data(:,:,k) = im2uint8(imread(raw_file, k));
        end
    elseif n == 16
        raw_data = zeros(info(1).Height, info(1).Width, frames, 'uint16');
        for k = 1:frames
            raw_data(:,:,k) = im2uint16(imread(raw_file, k));
        end
    else
        raw_data = zeros(info(1).Height, info(1).Width, frames, 'uint32');
        for k = 1:frames
            raw_data(:,:,k) = im2uint32(imread(raw_file, k));
        end
    end
end


%% =========================================================================
%  writeMTiffnOriginal  —  Write a 3D stack as a multi-page TIFF (append mode)
%
%  Inputs:
%    f_stack  : H x W x N stack (any numeric type; cast internally)
%    filename : output TIFF path
%    n        : bit depth (8, 16, or 32)
%
%  Note: Each slice is independently min-max preserved (no global scaling).
% =========================================================================

function [] = writeMTiffnOriginal(f_stack, filename, n)
    stackfilename = filename;
    [~, ~, z] = size(f_stack);

    for k = 1:z
        X = f_stack(:,:,k);
        X = double(X);
        % m  = max(max(X));   % (reserved for global scaling, not used)
        % mi = min(min(X));

        if n == 8
            imwrite(uint8(X), stackfilename, 'WriteMode', 'append');
        elseif n == 16
            imwrite(uint16(X), stackfilename, 'WriteMode', 'append');
        else
            imwrite(uint32(X), stackfilename, 'WriteMode', 'append');
        end
    end
end


%% =========================================================================
%  Blur3D  —  Frequency-domain convolution via FFT (works for 2D and 3D)
%
%  Inputs:
%    f   : input image (2D HxW or 3D HxWxD)
%    PSF : point-spread function (same dimensionality as f)
%
%  Output:
%    g   : blurred image, fftshift'd so the PSF center is aligned
%
%  Note: Uses fftn / ifftn; the output is fftshift'd to center the kernel.
% =========================================================================

function g = Blur3D(f, PSF)
    Ff   = fftn(f);
    FPSF = fftn(PSF);
    Fg   = Ff .* FPSF;
    g    = ifftn(Fg);
    g    = fftshift(g);
end
