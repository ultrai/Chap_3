clear
system('rm -rf Digits')
for temp = 1:3
    for temp2 = 1:15
    mkdir(['Digits/Train/',num2str(temp),'/',num2str(temp2)])
    mkdir(['Digits/Test/',num2str(temp),'/',num2str(temp2)])
    end
end
list = [1:15]';
list = circshift(list,14);
d = pwd;
%% Model layer
addpath(genpath('BM3D'))
addpath(genpath('Piotr_Matlab_Toolbox'))
opts = edgesTrain();
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm=['modelBsds_layer',num2str(8)];  
model = edgesTrain(opts);
%% Normal Data
s1 = 224;
s2 = 224;
dummy = ones(s1,s2);
r = 224;%150
c = 112;%200
dummy = ones(s1,s2);
Images= [];
Sub_idx=[];
Label = []; 
for sub = 1:8
    files = dir([d,'/data/NORMAL',num2str(list(sub)),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,1,numel(files)); % 496,512
    tic
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/NORMAL',num2str(list(sub)),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
        %ii = ii((round(0.7*s1)-r+1+5):round(0.7*s1+5),(round(0.5*s2)-c+1):(round(0.5*s2)+c));
        %ii = imresize(ii(:,:,1),[224,224]);
        [~,ii] = BM3D(dummy,ii);
%       
        images(:,:,1,i) = (ii);
          close all
    end
    mkdir(['Digits/Train/1/',num2str(sub)])
    for temp =1:size(images,4) 
            k=k+1;
            imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp))),['Digits/Train/1/',num2str(sub),'/',num2str(k),'.png'],'png')
     end
    
    toc
   
end

%% AMD Data
k=0;
for sub = 1:8
    files = dir([d,'/data/AMD',num2str(list(sub)),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,1,numel(files)); % 496,512
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/AMD',num2str(list(sub)),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
         %ii = ii((round(0.7*s1)-r+1+5):round(0.7*s1+5),(round(0.5*s2)-c+1):(round(0.5*s2)+c));
        %ii = imresize(ii(:,:,1),[224,224]);
        [~,ii] = BM3D(dummy,ii);
%       
        images(:,:,1,i) = (ii);
          close all
    end
    toc
    if list(sub) == 1
        images(:,:,:,1:14) = [];
    elseif sub==2
        images(:,:,:,1:13) = [];
    end
    toc
   mkdir(['Digits/Train/2/',num2str(sub)])
   for temp =1:size(images,4) 
            k=k+1;
             imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp))),['Digits/Train/2/',num2str(sub),'/',num2str(k),'.png'],'png')
     end 
end
%% DMEData
k=0;
for sub = 1:8
    files = dir([d,'/data/DME',num2str(list(sub)),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,1,numel(files)); % 496,512
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/DME',num2str(list(sub)),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = preprocess(ii);
          ii = imresize(ii(:,:,1),[s1,s2]);
        [~,ii] = BM3D(dummy,ii);
%       
        images(:,:,1,i) = (ii);
        
          close all
    end
    toc
    if list(sub) == 1
        images(:,:,:,1:8) = [];
    elseif list(sub)==2
        images(:,:,:,1:14) = [];
    elseif list(sub) == 3
        images(:,:,:,56:61) = [];
        images(:,:,:,17:26) = [];
    end
   mkdir(['Digits/Train/3/',num2str(sub)])
   for temp =1:size(images,4) 
            k=k+1;
    
                  imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp))),['Digits/Train/3/',num2str(sub),'/',num2str(k),'.png'],'png')
    end 
end
%%



for sub = 9:15
    files = dir([d,'/data/NORMAL',num2str(list(sub)),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,1,numel(files)); % 496,512
    tic
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/NORMAL',num2str(list(sub)),'/TIFFs/8bitTIFFs/',files(i).name]);
       ii(ii==255) =10;
         ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
         [~,ii] = BM3D(dummy,ii);
        images(:,:,1,i) = (ii);  close all
    end
    mkdir(['Digits/Test/1/',num2str(sub)])
     for temp =1:size(images,4) 
            k=k+1;
    
            imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp))),['Digits/Test/1/',num2str(sub),'/',num2str(k),'.png'],'png')
      end
    
    toc
   
end

%% AMD Data
k=0;
for sub = 9:15
    files = dir([d,'/data/AMD',num2str(list(sub)),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,1,numel(files)); % 496,512
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/AMD',num2str(list(sub)),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
        
        [~,ii] = BM3D(dummy,ii);
%       
        images(:,:,1,i) = (ii);
          close all
    end
    toc
   mkdir(['Digits/Test/2/',num2str(sub)])
   for temp =1:size(images,4) 
            k=k+1;
        imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp))),['Digits/Test/2/',num2str(sub),'/',num2str(k),'.png'],'png')
       end 
end
%% DMEData
k=0;
for sub = 9:15
    files = dir([d,'/data/DME',num2str(list(sub)),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,3,numel(files)); % 496,512
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/DME',num2str(list(sub)),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
       ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
         [~,ii] = BM3D(dummy,ii);
%       
        images(:,:,1,i) = (ii);
          close all
    end
    toc
   mkdir(['Digits/Test/3/',num2str(sub)])
   for temp =1:size(images,4) 
            k=k+1;
               imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,1,temp))),['Digits/Test/3/',num2str(sub),'/',num2str(k),'.png'],'png')
   end 
end
%%

system('python data.py')

