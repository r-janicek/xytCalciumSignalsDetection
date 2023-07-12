% add package to the path
javaaddpath("PureDenoise_.jar")
javaaddpath("ij.jar")
% import
% import PureDenoise-main.lib.ij.*
% import PureDenoise-main.lib.imageware.*
% import PureDenoise-main.src.denoise.Denoising.*
% import PureDenoise-main.src.denoise.Operations.*

% load test image series
% get metadata of image, number of images in tiff file
fp = '../TEST.tif';
img_info = imfinfo(fp);
N_imgs = size(img_info, 1);
% read images
imgData = zeros(img_info(1).Height, img_info(1).Width, N_imgs);
for i=1:N_imgs
    imgData(:,:,i) = imread(fp, i, 'Info', img_info);
end

imgData = imgData(1:35,1:157,:);

% dimensions of image series
nx = img_info(1).Width;
ny = img_info(1).Height;
nz = N_imgs;

Imax = max(imgData,[],"all");
Imin = min(imgData,[],"all");

% minimum size of image
Nmin = 16;
Ext = [0,0];
if nx<=Nmin || ny<=Nmin
    % image size too small
    return
end

% create java image object
original_img = imageware.Builder.create(imgData);

nz = original_img.getSizeZ();
nxe = ceil(nx / Nmin) * Nmin;
nye = ceil(ny / Nmin) * Nmin;

if (nxe ~= nx || nye ~= ny)
    original_img = ...
        denoise.Operations.symextend2D(original_img, nxe, nye, Ext);
    Ext = [(nxe-nx)/2,(nye-ny)/2];
else
    Ext = [0,0];
end

% parameters of denoising

% * Constructor of the class Denoising.
% *
% * @param input         input data to be denoised (3D ImageWare object)
% * @param Alpha         double array containing the estimated detector gain
% *                      for each frame/slice.
% * @param Delta         double array containing the estimated detector of-
% *                      fset for each frame/slice.
% * @param Sigma         double array containing the estimated AWGN standard
% *                      deviation for each frame/slice.
% * @param FRAMEWISE     true->Framewise noise parameters estimation
% *                      false->Global noise parameters estimation.
% * @param CYCLESPIN     number of cycle-spins (CS>0). A high value of CS
% *                      yields a high-quality denoising result, but the
% *                      computation time linearly increases with CS.
% * @param MULTIFRAME    number of adjacent frames/slices to be considered
% *                      for multi-frame/slices denoising. MF>0 must be odd.

AlphaHat = zeros(nz,1);
DeltaHat = zeros(nz,1);
SigmaHat = zeros(nz,1);
CS = 6;
NBFRAME = 7;
FRAMEWISE = false;
% create denoisung object
denoising = denoise.Denoising( ...
    original_img, AlphaHat, DeltaHat, SigmaHat, FRAMEWISE, CS, NBFRAME);
%denoising.setLog(false);
% estimate parameters 
denoising.estimateNoiseParameters();
% denoise
denoising.perform();
denoising.getProgress()
output = denoising.getOutput();
if (nxe ~= nx || nye ~= ny) 
    output = denoise.Operations.crop2D(output, nx, ny, Ext);
end
% get parameters of denoising
alpha=denoising.getAlpha();
delta=denoising.getDelta();
sigma=denoising.getSigma();
% create denoised image
denoised_img = zeros(nx, ny, nz);
for i = 1:nz
    denoised_img(:,:,i) = reshape(output.getSliceDouble(i-1), nx, ny);
end

hf = figure('Position',[100 100 2000 1000]);
t = tiledlayout(2,1);
t.TileSpacing = 'tight';
ax1 = nexttile;
ax1.Title.String = 'original';
ax2 = nexttile;
ax2.Title.String = 'de-noised';
for i=1:nz
    imagesc(ax1, imgData(:,:,i))
    imagesc(ax2, denoised_img(:,:,i))
    pause(1)
end













