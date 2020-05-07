% general_cc: estimates the light source of an input_image. 
%
% Depending on the parameters the estimation is equal to Grey-Wolrd, Max-RGB, general Grey-World,
% Shades-of-Gray or Grey-Edge algorithm.
%
% SYNOPSIS:
%    [white_R ,white_G ,white_B,output_data] = general_cc(input_data,njet,mink_norm,sigma,mask_im)
%    
% INPUT :
%   input_data    : color input image (NxMx3)
%	njet          : the order of differentiation (range from 0-2). 
%	mink_norm     : minkowski norm used (if mink_norm==-1 then the max
%                   operation is applied which is equal to minkowski_norm=infinity).
%   mask_im       : binary images with zeros on image positions which
%                   should be considered for illuminant estimation.
% OUTPUT: 
%   [white_R,white_G,white_B]           : illuminant color estimation
%   output_data                         : color corrected image

% LITERATURE :
%
% J. van de Weijer, Th. Gevers, A. Gijsenij
% "Edge-Based Color Constancy"
% IEEE Trans. Image Processing, accepted 2007.
%
% The paper includes references to other Color Constancy algorithms
% included in general_cc.m such as Grey-World, and max-RGB, and
% Shades-of-Gray.

function [white_R ,white_G ,white_B,output_data] = general_cc(input_data,njet,mink_norm,sigma,mask_im)

if(nargin<2), njet=0; end
if(nargin<3), mink_norm=1; end
if(nargin<4), sigma=1; end
if(nargin<5), mask_im=zeros(size(input_data,1),size(input_data,2)); end

% remove all saturated points
saturation_threshold = 255;
mask_im2 = mask_im + (dilation33(double(max(input_data,[],3)>=saturation_threshold)));   
mask_im2=double(mask_im2==0);
mask_im2=set_border(mask_im2,sigma+1,0);
% the mask_im2 contains pixels higher saturation_threshold and which are
% not included in mask_im.

output_data=input_data;

if(njet==0)
   if(sigma~=0)
     for ii=1:3
        input_data(:,:,ii)=gDer(input_data(:,:,ii),sigma,0,0);
     end
   end
end

if(njet>0)
    [Rx,Gx,Bx]=norm_derivative(input_data, sigma, njet);
    
    input_data(:,:,1)=Rx;
    input_data(:,:,2)=Gx;
    input_data(:,:,3)=Bx;    
end

input_data=abs(input_data);

if(mink_norm~=-1)          % minkowski norm = (1,infinity >
    kleur=power(input_data,mink_norm);
    white_R = power(sum(sum(kleur(:,:,1).*mask_im2)),1/mink_norm);
    white_G = power(sum(sum(kleur(:,:,2).*mask_im2)),1/mink_norm);
    white_B = power(sum(sum(kleur(:,:,3).*mask_im2)),1/mink_norm);

    som=sqrt(white_R^2+white_G^2+white_B^2);

    white_R=white_R/som;
    white_G=white_G/som;
    white_B=white_B/som;
else                    %minkowski-norm is infinit: Max-algorithm     
    R=input_data(:,:,1);
    G=input_data(:,:,2);
    B=input_data(:,:,3);
    
    white_R=max(R(:).*mask_im2(:));
    white_G=max(G(:).*mask_im2(:));
    white_B=max(B(:).*mask_im2(:));
    
    som=sqrt(white_R^2+white_G^2+white_B^2);

    white_R=white_R/som;
    white_G=white_G/som;
    white_B=white_B/som;
end
output_data(:,:,1)=output_data(:,:,1)/(white_R*sqrt(3));
output_data(:,:,2)=output_data(:,:,2)/(white_G*sqrt(3));
output_data(:,:,3)=output_data(:,:,3)/(white_B*sqrt(3));