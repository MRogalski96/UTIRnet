function [Y_out,Y_amp,Y_phs,U_out] = UTIRnetReconstruction(holo,...
    CNN_A,CNN_P,Z,lambda,dx,m1,nrm)
% Function for twin-image removal with the use of neural networks
%   Inputs:
%       holo - input hologram
%       CNN_A - network that filters the amplitude part of the optical field
%       CNN_P - network that filters the phase part of the optical field
%       if CNN_A (or CNN_P) is empty then amplitude (or phase) is not filtered.
%           This will reduce processing time by 2 - consider this when
%           working with purely amplitude (or purely phase) objects
%           if CNN_P == 0, then algorithm will assume that object phase = 0
%       Z - propagation distance (um)
%       lambda - wavelength (um)
%       dx - camera pixel size (um)
%       m1 = 1; % phase sign: 
%            m1 = 1 - object has higher optical thickness than background
%            (default)
%            m1 = -1 - object has lower optical thickness than background
%       nrm - normalize data by dividing it by median hologram value
%           (1-yes, 0-no). For experimental data choose nrm = 1. For 
%           synthetic data if object amplitude is normalized
%           in 0-1 range then nrm = 0 would allow to avoid eventual
%           normalization error (e.g., for very complex objects median
%           hologram value is not equal to median background value).
%   Outputs:
%       Y_out - filtered optical field
%       Y_amp - filtered amplitude with CNN_A
%       Y_phs - filtered phase with CNN_P
%       After calculating Y_amp and Y_phs, optical field is then propagated
%           to hologram plane, updated with the hologram and backpropagated
%           to object plane. Therefore amplitude and phase parts of Y_out
%           are different than Y_amp and Y_phs. In general, Y_out should be
%           physically more correct (object shape should be preserved). On
%           the other hand, Y_amp and Y_phs will be stronger filtered -
%           those results may be less quantitative (especially object
%           details), but there will be less twin-image artefacts in the
%           output.
%       U_out - hologram backpropagated to object plane
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
if nargin < 7 || isempty(m1); m1 = 1; end
if nargin < 8 || isempty(nrm); nrm = 1; end

% Auxiliary variables
RI = 1; % medium refractive index
        
% Propagate hologram to object plane
U_out = AS_propagate_p(holo,Z,RI,lambda,dx); % for phase reconstruction
U_out0 = AS_propagate_p(sqrt(holo),Z,RI,lambda,dx); % for amplitude reconstruction

% Filter amplitude part of U_out with network CNN_A
U_amp = abs(U_out0);
if ~isempty(CNN_A)
    if nrm == 0
        me = 1;
    else
        me = median(U_amp(:)); % median for amplitude normalization
    end
    [Y_amp,~,~] = predictFull(CNN_A,U_amp./me);
    Y_amp = Y_amp.*me;
else
    Y_amp = U_amp;
end

% Filter phase part of U_out with network netP
if ~isempty(CNN_P)
    if isnumeric(CNN_P)
        Y_phs = 0;
    else
        [Y_phs,~,~] = predictFull(CNN_P,m1*angle(U_out)+pi);
        Y_phs = Y_phs-pi;
        Y_phs = m1*Y_phs;
    end
else
    Y_phs = angle(U_out);
end


% Filtered optical field
Y_out0 = Y_amp.*exp(1i.*Y_phs);
% Propagate to camera plane
Y_in0 = AS_propagate_p(Y_out0,-Z,RI,lambda,dx);
% Update amplitude with sqrt(holo)
Y_in1 = sqrt(holo).*exp(1i.*angle(Y_in0));
% Backpropagate to object plane
Y_out = AS_propagate_p(Y_in1,Z,RI,lambda,dx);

U_out = abs(U_out0).*exp(1i.*angle(U_out));

end


function [Y_out,mskT,count] = predictFull(net,img,L,D)
% Function that predicts CNN output on full FOV image by stiching CNN
% results for smaller FOVs (for CNN trained on imSximS images)
%
%   Inputs:
%       net - CNN trained for imSximS images
%       img - input image
%       L - percent of overlay between adjacent small FOVs (Default L = 10)
%       D - parameter for blending (0-1 range) - see drawing below (default
%           D = 0.2)
%   Outputs:
%       Y_out - CNN full FOV output
%       mskT - mask used for blending adjacent images
%       count - number of small FOV CNN predictions
%
% Drawing: A-A reconstructed FOV1, B-B reconstructed FOV2 (both imSximS)
%               Overlay area: B-A = L/100*imS
%  A<-----------------B<----------->A----------------->B
%                        kk
% 1 ___________________|____|
%                           \
%                            \           <- cross section for mask applied to A-A FOV
%                             \____  0        kk = L/100*imS*D/2
%                             | kk |
% 
% Created by:
%   Mikołaj Rogalski,
%   mikolaj.rogalski.dokt@pw.edu.pl
%   Institute of Micromechanics and Photonics,
%   Warsaw University of Technology, 02-525 Warsaw, Poland
%
% Last modified: 01.06.2023

if nargin<3; L = 10; end
if nargin<4; D = 0.2; end

imS = 512; % image size used for CNN training

[Sy,Sx] = size(img);
if Sy < imS || Sx < imS
    % propagate to last layer of CNN (this could also work for Sy and Sx >
    % imS, but for very large images (e.g. 4000x4000) there may be issues
    % with available memory on graphiucs card)
    Y_out = activations(net,img,'relu1_out'); mskT = []; count = [];
elseif Sx == imS && Sy == imS
    % simply preduict the network output
    Y_out = predict(net,img); mskT = []; count = [];
elseif Sx > imS && Sy > imS
    % stitch multiple imSximS results with alpha blending
    
    dd = round(imS*(100-L)/100);
    kk = round((imS-dd)*D/2);
    
    count = 0;
    uu = 1;
    cc = 1;
    Y_out = zeros(Sy,Sx);
    M = zeros(Sy,Sx);
    for yy = 1:dd:(Sy-imS+dd)
        y = yy:yy-1+imS;
        if y(end) > Sy
            y = (Sy-imS+1):Sy;
            if uu==1; yywL = 1-M(Sy-imS+1:Sy,1); uu = 0; end
            yyw = yywL;
        else
            yyw = ones(1,imS);
            if yy > 1
                yyw(1:kk) = 0;
                yyw((kk+1):(imS-dd-kk)) = (0:(imS-dd-2*kk-1))./(imS-dd-2*kk-1);
            end
            yyw(imS-kk+1:imS) = 0;
            yyw(dd+kk+1:imS-kk) = ((imS-dd-2*kk-1):-1:0)./(imS-dd-2*kk-1);
        end
        for xx = 1:dd:(Sx-imS+dd)
            x = xx:xx-1+imS;
            
            if x(end) > Sx
                x = (Sx-imS+1):Sx;
                if yy==1; xxwL = 1-M(1,Sx-imS+1:Sx); end
                xxw = xxwL;
            else
                xxw = ones(1,imS);
                if xx > 1
                    xxw(1:kk) = 0;
                    xxw((kk+1):(imS-dd-kk)) = (0:(imS-dd-2*kk-1))./(imS-dd-2*kk-1);
                end
                xxw(imS-kk+1:imS) = 0;
                xxw(dd+kk+1:imS-kk) = ((imS-dd-2*kk-1):-1:0)./(imS-dd-2*kk-1);
            end
            [xw,yw] = meshgrid(xxw,yyw);
            
            msk = xw.*yw;
            
            if xx > 1 && yy > 1 && cc == 1
                cc = 0;
                mskT = msk;
            end
            
            Y = predict(net,img(y,x));
            count = count + 1;
            Y_out(y,x) = Y_out(y,x)+msk.*Y;
            M(y,x) = M(y,x)+msk;
            %         imagesc(M,[0,1]); colormap jet; axis image; pause(0.01)
        end
    end
end

end

