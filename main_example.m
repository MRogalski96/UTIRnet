% Code showing the exemplary process of generating training data, training
% UTIRnet and reconstructing holograms with UTIRnet without twin-image
% 
% Code is divided into sections, each one designed to present a separate 
% part of the UTIRnet creation and use process. Run this sections
% separately, as some of them may last a long time (expecially network
% training part). "##########" - this sign indicates that the marked line
% may be (or should be) changed to make the network work for an individual
% user.
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

%% Dataset generation
clear; close all; clc

% System parameters ##########################
% Z = 2600; % camera-sample distance (um)  ###########################
Z = 17000; % ###################
lambda = 0.405; % light source wavelength (um) ###############
dx = 2.4; % pixel size in object plane (cam_pix_size / mag) (um) ##########

% Path to the directory containing flower recognition dataset (it should
% contain 5 subdirectories with images of different flowers).
% Can be downloaded at: 
% https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
% Alternatively images from other repositories may be applied
pth = 'D:\...\flowers'; % ###############

% Training dataset
[inTrAmp,tarTrAmp,holosTrAmp] = GenerateDataset(pth,[1,4,5],650,Z,...
    lambda,dx,'amp'); % for CNN_A
[inTrPhs,tarTrPhs,holosTrPhs] = GenerateDataset(pth,[1,4,5],650,Z,...
    lambda,dx,'phs'); % for CNN_P

% Validation dataset
[inValAmp,tarValAmp,holosValAmp] = GenerateDataset(pth,[2,3],50,Z,...
    lambda,dx,'amp'); % for CNN_A
[inValPhs,tarValPhs,holosValPhs] = GenerateDataset(pth,[2,3],50,Z,...
    lambda,dx,'phs'); % for CNN_P

% Optionally you can save created datasets for later use ###############
%% Network training
% Generate dataset as in "Dataset generation" section or load previously 
% generated dataset ################################

% Creating net architecture
lgraph = NetworkArchitecture(70,2,512,1,1);

% Number of input images
imgN = size(inTrAmp,4); 

% Train CNN_A
CNN_A_info = trainingOptions('adam', ...
    'MiniBatchSize',1, ...
    'MaxEpochs',30,...  
    'LearnRateDropPeriod',5,... 
    'LearnRateDropFactor',0.5,...
    'LearnRateSchedule','piecewise',...
    'InitialLearnRate',1e-4, ... 
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'ValidationData',{(inValAmp), (tarValAmp)},...
    'ValidationFrequency', imgN);

[CNN_A,info_A] = trainNetwork(inTrAmp, tarTrAmp, lgraph, CNN_A_info);
CNN_A_info.ValidationData = [];

% Train CNN_P
CNN_P_info = trainingOptions('adam', ...
    'MiniBatchSize',1, ...
    'MaxEpochs',30,...  
    'LearnRateDropPeriod',5,... 
    'LearnRateDropFactor',0.5,...
    'LearnRateSchedule','piecewise',...
    'InitialLearnRate',1e-4, ... 
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'ValidationData',{(inValPhs), (tarValPhs)},...
    'ValidationFrequency', imgN);

[CNN_P,info_P] = trainNetwork(inTrPhs, tarTrPhs, lgraph, CNN_P_info);
CNN_P_info.ValidationData = [];

% System parameters
UTIRnet_info.Z_mm = Z/1000;
UTIRnet_info.lambda_um = lambda;
UTIRnet_info.dx_um = dx;

% Save network
fnm = ['UTIRnet_my_Z-',num2str(Z/1000),'mm_dx-',num2str(dx),'um_lambda-',...
    num2str(lambda*1000),'nm.mat'];
save(fnm,'CNN_A','CNN_P','CNN_A_info','CNN_P_info','UTIRnet_info')

%% Network testing - on validation data
% Train network as in "Network training" section or load previously 
% trained network ################################
% Generate validation dataset as in "Dataset generation" section or load 
% previously generated dataset ################################

% System parameters (ensure that are the same as used for network training)
% Z = 2600; % camera-sample distance (um)  ###########################
Z = 17000; % ###################
lambda = 0.405; % light source wavelength (um) ###############
dx = 2.4; % pixel size in object plane (cam_pix_size / mag) (um) ##########

imNo = 10; % image number ################################
AmpPhs = 1; % 1 - amplitude data, 2 - phase data ##########################

if AmpPhs == 1
    % Amp data
    GT = tarValAmp(:,:,imNo); % ground truth target image
    holo = holosValAmp(:,:,imNo); % hologram
elseif AmpPhs == 2
    % Phs data
    GT = tarValPhs(:,:,imNo); % ground truth target image
    holo = holosValPhs(:,:,imNo); % hologram
end

ps = 256; % pad size
holoP = padarray(holo,[ps,ps],'replicate'); % hologram padding

% reconstruction
[Yout,Yamp,Yphs,Uout] = UTIRnetReconstruction(holoP,...
    CNN_A,CNN_P,Z,lambda,dx,[],0);
% remove padding
Yout = Yout(ps+1:end-ps,ps+1:end-ps);
Uout = Uout(ps+1:end-ps,ps+1:end-ps);
Yamp = Yamp(ps+1:end-ps,ps+1:end-ps);
Yphs = Yphs(ps+1:end-ps,ps+1:end-ps);

% display results
close all
ax = [];
if AmpPhs == 1
    rng = [0,1.1];
    figure; imagesc(abs(Uout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('input AS amplitude (with twin-image)')
    figure; imagesc(abs(Yout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('UTIRnet amplitude reconstruction')
    figure; imagesc(GT,rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Ground truth amplitrude (without twin-image)') 
elseif AmpPhs == 2
    rng = [-pi,pi];
    figure; imagesc(angle(Uout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('input AS phase (with twin-image)')
    figure; imagesc(angle(Yout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('UTIRnet reconstruction')
    figure; imagesc(GT-pi,rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Ground truth (without twin-image)')
end
linkaxes(ax)

%% Network testing - generating and reconstructing synth data
% Train network as in "Network training" section or load previously 
% trained network ################################
% System parameters (ensure that are the same as used for network training)
% Z = 2600; % camera-sample distance (um)  ###########################
Z = 17000; % ###################
lambda = 0.405; % light source wavelength (um) ###############
dx = 2.4; % pixel size in object plane (cam_pix_size / mag) (um) ##########

AmpPhs = 2; % 1 - amplitude data, 2 - phase data ##########################

% Read any image #################################### 
% img = imread('cameraman.tif');
% img = double(imread('coins.png'));
img = double(imread('rice.png'));

if AmpPhs == 1
    [input,GT,holo] = GenerateHologram(img,Z,lambda,dx,'amp');
elseif AmpPhs == 2
    [input,GT,holo] = GenerateHologram(img,Z,lambda,dx,'phs');
end

ps = 256; % pad size
holoP = padarray(holo,[ps,ps],'replicate'); % hologram padding

% reconstruction
[Yout,Yamp,Yphs,Uout] = UTIRnetReconstruction(holoP,...
    CNN_A,CNN_P,Z,lambda,dx,[],0);
% remove padding
Yout = Yout(ps+1:end-ps,ps+1:end-ps);
Uout = Uout(ps+1:end-ps,ps+1:end-ps);
Yamp = Yamp(ps+1:end-ps,ps+1:end-ps);
Yphs = Yphs(ps+1:end-ps,ps+1:end-ps);

% display results
close all
ax = [];
if AmpPhs == 1
    rng = [0,1.1];
    figure; imagesc(abs(Uout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('input AS amplitude (with twin-image)')
    figure; imagesc(abs(Yout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('UTIRnet amplitude reconstruction')
    figure; imagesc(GT,rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Ground truth amplitrude (without twin-image)') 
elseif AmpPhs == 2
    rng = [-pi,pi];
    figure; imagesc(angle(Uout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('input AS phase (with twin-image)')
    figure; imagesc(angle(Yout),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('UTIRnet reconstruction')
    figure; imagesc(GT-pi,rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Ground truth (without twin-image)')
end
linkaxes(ax)
%% Network testing - experimental data
clear; close all; clc

m1 = 1;

% load hologram data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('.\Holograms\CheekCells_17.08mm.mat')
% load('.\Holograms\CustomPhsTest_2.64mm.mat'); m1 = -1;
% load('.\Holograms\GlialCells_15.14mm.mat')
% load('.\Holograms\USAFamp_2.72mm.mat');
% load('.\Holograms\USAFamp_17.09mm.mat');
% load('.\Holograms\USAFampSynthetic_2.72mm.mat');
% load('.\Holograms\USAFampSynthetic_17.09mm.mat');

% load network trained for propper system parameters %%%%%%%%%%%%%%%%%%%%%%
% load('.\Networks\UTIRnet_Z-2.6mm_dx-2.4um_lambda-405nm.mat')
load('.\Networks\UTIRnet_Z-17mm_dx-2.4um_l1ambda-405nm.mat')

% reconstruction
[Yout,Yamp,Yphs,Uout] = UTIRnetReconstruction(double(holo),...
    CNN_A,CNN_P,Z_um,lambda,dx,m1,1);

% displaying
ax = [];
rng = [min(abs(Uout(:))),max(abs(Uout(:)))];
figure; imagesc(abs(Uout),rng); ax = [ax,gca]; colormap gray;
axis image; colorbar; title('AS amplitude (with twin-image)')
figure; imagesc(abs(Yout),rng); ax = [ax,gca]; colormap gray;
axis image; colorbar; title('UTIRnet amplitude reconstruction')
if exist('GS','var')
    figure; imagesc(abs(GS),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Reference GS amplitrude')
    figure; imagesc(abs(GS_CFF),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Referenmce GS+CFF amplitrude')
end
if exist('GroundTruth','var')
    figure; imagesc(GroundTruth,rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Ground truth amplitude')
end
rng = [-pi,pi];
figure; imagesc(angle(Uout),rng); ax = [ax,gca]; colormap gray;
axis image; colorbar; title('AS phase (with twin-image)')
figure; imagesc(angle(Yout),rng); ax = [ax,gca]; colormap gray;
axis image; colorbar; title('UTIRnet reconstruction')
if exist('GS','var')
    figure; imagesc(angle(GS),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Reference GS phase')
    figure; imagesc(angle(GS_CFF),rng); ax = [ax,gca]; colormap gray;
    axis image; colorbar; title('Reference GS+CFF phase')
end
linkaxes(ax)