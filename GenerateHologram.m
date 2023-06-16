function [img_input,img_target,holo] = GenerateHologram(img,Z,lambda,dx,type)
% Function for generating input and target images for UTIRnet network
%   Inputs:
%       img - any image
%       Z - propagation distance (um)
%       lambda - wavelength (um)
%       dx - camera pixel size (um) 
%       type - object type (amplitude: type = 'amp' or phase: type = 'phs')
%   Outputs
%       img_input - input image for CNN training (with twin-image,
%           backpropagated with angular spectrum method)
%       img_target - target image for CNN training (without twin-image)
%       holo - holograsm generated for given object
%       
%       holo = ASpropagate(img_target,params).^2
%       img_input = ASpropagate(holo,params) <- for phs objects
%       img_input = ASpropagate(sqrt(holo),params) <- for amp objects
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
RI = 1; % refractive index of the medium (1 - air)
imS = 512; % image size
padS = imS/2; % pad size (to avoid aliasing)
den = 0; % denoise img with BM3D to avoid noise presence in training data
         % 0 - no, 1- yes. May be downloaded at: 
         % https://webpages.tuni.fi/foi/GCF-BM3D/index.html
xi = 5; % amount (in %) of highest value pixels that are set as object-free
phi = (rand*1.5 + 0.5)*pi; % phase range ([-phi,0])
sigma = 10; % parameter for high-pass filtering in object phase. Sigma may 
            % be iuncreased for larger Z idstances

% Preprocessing
img = mean(double(img),3); % convert to grayscale
img = mat2gray(img); % Normalize image in 0-1 range
if mean2(img)<0.5
    % Invert colors so that there are more background elements than object 
    % elements
    img = 1-img; 
end
img = imresize(img,[imS,imS]); % Resize image to imS x imS
img = img*(1+xi/100); % Set xi% of highest value pixels equal 1
img(img>1) = 1;

if den == 1; img = BM3D_(img,20); end % optional BM3D denoising

% preprocessing for phase objects
if strcmp(type,'phs')
    img = img - imgaussfilt(img,sigma); % high-pass filtering
    img(img>0) = 0;
    img = (mat2gray(img)-1)*phi; % normalizing phase in 'phi' range
    img_target = angle(exp(1i.*img))+pi; % wrap phase and normalize to 0-2pi range 
else
    img_target = img;
end


% pad object to avoid aliasing
imgP = padarray(img,[padS,padS],'replicate');
if strcmp(type,'phs')
    u0 = exp(1i.*imgP);
else
    u0 = imgP;
end

% propagate object to camera plane
u_in = AS_propagate_p(u0,-Z,RI,lambda,dx);
holo = abs(u_in(padS+1:padS+imS,padS+1:padS+imS)).^2;

if strcmp(type,'phs')
    % backpropagate only the intensaity (hologram)
    u_out = AS_propagate_p(abs(u_in).^2,Z,RI,lambda,dx);
    % remove padding
    u_out = u_out(padS+1:padS+imS,padS+1:padS+imS);
    img_input = angle(u_out)+pi;
else
    % backpropagate only the amplitude (sqrt of hologram)
    u_out = AS_propagate_p(abs(u_in),Z,RI,lambda,dx);
    % remove padding
    u_out = u_out(padS+1:padS+imS,padS+1:padS+imS);
    img_input = abs(u_out);
end

end