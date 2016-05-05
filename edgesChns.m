function [chnsReg,chnsSim] = edgesChns( I, opts )
% Compute features for structured edge detection.
%
% For an introductory tutorial please see edgesDemo.m.
%
% USAGE
%  [chnsReg,chnsSim] = edgesChns( I, opts )
%
% INPUTS
%  I          - [h x w x 3] color input image
%  opts       - structured edge model options
%
% OUTPUTS
%  chnsReg    - [h x w x nChannel] regular output channels
%  chnsSim    - [h x w x nChannel] self-similarity output channels
%
% EXAMPLE
%
% See also edgesDemo, edgesTrain, edgesDetect, gradientMag
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

shrink=opts.shrink; nTypes=1; chns=cell(1,opts.nChns); k=0;
%if(size(I,3)>3), nTypes=2; Is={I(:,:,1:3),I(:,:,4:end)}; end
%for t=1:nTypes
%  if(nTypes>1), I=Is{t}; end
%  if(size(I,3)==1), cs='gray'; else cs='luv'; end;
  for feat = 1:1
      Io=rgbConvert(I(:,:,feat),'gray');%I = single(I);% 
      Ishrink=imResample(Io,1/shrink); k=k+1; chns{k}=Ishrink;
  end
  for i = 1:2, s=2^(i-1);
    for feat = 1:opts.nChnsColor
        if(s==shrink), I1=imResample(rgbConvert(I(:,:,feat),'gray'),1/shrink); else I1=imResample(rgbConvert(I(:,:,feat),'gray'),1/s); end
        I1_temp = convTri( I1, opts.grdSmooth );
        [M,O] = gradientMag( I1_temp, 0, opts.normRad, .01 );
        H = gradientHist( M, O, max(1,shrink/s), opts.nOrients, 0 );
        k=k+1; chns{k}=imResample(M,s/shrink);
        k=k+1; chns{k}=imResample(H,max(1,s/shrink));
    end
  end
%end
chns=cat(3,chns{1:k}); 
size(chns);
assert(size(chns,3)==opts.nChns);
chnSm=opts.chnSmooth/shrink; if(chnSm>1), chnSm=round(chnSm); end
simSm=opts.simSmooth/shrink; if(simSm>1), simSm=round(simSm); end
chnsReg=convTri(chns,chnSm); chnsSim=convTri(chns,simSm);

end
