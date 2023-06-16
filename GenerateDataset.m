function [inputs,targets,holos] = GenerateDataset(pth,usedDirs,maxImgs,Z,lambda,dx,type)
% Function for generating dataset for UTIRnet training
%   Inputs:
%       pth - path to the directory containing flower recognition dataset 
%           (it should contain 5 sub-directories with images of different 
%           flowers). Can be downloaded at: 
%           https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
%           Alternatively images from other repositories may be applied
%       usedDirs - sub-directories used for dataset generation, e.g.
%           usedDirs = [1,3,5]
%       maxImgs - number of images from each sub-directory used for dataset
%           generation. Total number of used images = maxImgs * length(usedDirs)
%       Z - propagation distance (um)
%       lambda - wavelength (um)
%       dx - camera pixel size (um) 
%       type - object type (amplitude: type = 'amp' or phase: type = 'phs')
%   Outputs
%       inputs - input images for CNN training (with twin-image,
%           backpropagated with angular spectrum method)
%       targets - target images for CNN training (without twin-image)
%       holos - holograms generated for given objects
%       
%       holos(:,:,:,t) = ASpropagate(targets(:,:,:,t),params).^2
%       inputs(:,:,:,t) = ASpropagate(holos(:,:,:,t),params) <- for phs objects
%       inputs(:,:,:,t) = ASpropagate(sqrt(holos(:,:,:,t)),params) <- for amp objects
% 
% Cite as:   
%   M. Rogalski, P. Arcab, L. Stanaszek, V. Micó, C. Zuo and M. Trusiak, 
%   "Physics-driven universal twin-image removal network for digital 
%   in-line holographic microscopy". Submitted 2023 
% 
% Created by:
%   Mikołaj Rogalski,
%   mikolaj.rogalski.dokt@pw.edu.pl
%   Institute of Micromechanics and Photonics,
%   Warsaw University of Technology, 02-525 Warsaw, Poland
%
% Last modified: 01.06.2023


% Auxiliary variables
% Show images during generation:
showIm = 1; % 1 - yes, 0 - no
showEveryNo = 10; % Show every x image


fld = dir(pth); fld(1:2) = [];

if showIm == 1
    f = figure(55);
    f.Units = 'normalized';
    f.OuterPosition = [0,0,1,1];
    t1 = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    colormap gray
    imgsNo = length(usedDirs)*maxImgs;
end

uu = 0;
for ii = usedDirs
    imgsDir = dir(fullfile(pth,fld(ii).name));
    imgsDir(1:2) = [];
    for tt = 1:maxImgs
        uu = uu + 1;
        % load image
        img = double(imread(fullfile(pth,fld(ii).name,imgsDir(tt).name)));
        [in,tar,holo] = GenerateHologram(img,Z,lambda,dx,type);
        
        inputs(:,:,1,uu) = in;
        targets(:,:,1,uu) = tar;
        holos(:,:,1,uu) = holo;
        
        if showIm == 1 && mod(uu,showEveryNo)==0
            figure(55);
            title(t1,[num2str(uu),'/',num2str(imgsNo),' ',fld(ii).name])
            ShowImages(in,tar,holo)
        end
    end
end
end

%% Auxiliary functions:
function ShowImages(in,tar,holo)
mi = min([in(:);tar(:)]);
ma = max([in(:);tar(:)]);
nexttile(1); imagesc(in,[mi,ma]); axis image; axis off; colorbar
title('input')
nexttile(2); imagesc(tar,[mi,ma]); axis image; axis off; colorbar
title('target')
nexttile(3); imagesc(holo); axis image; axis off; colorbar
title('hologram')
colormap gray
pause(0.01)
end

