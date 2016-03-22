clear
d = pwd;
mkdir('1')
mkdir('2')
mkdir('3')
%% Model layer
addpath(genpath('Piotr_Matlab_Toolbox'))
opts = edgesTrain();
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm=['modelBsds_layer',num2str(8)];  
model = edgesTrain(opts);
%% Normal Data
s1= 496/1;
s2=512/1;
r = 150;%150
c = 200;%200

for sub = 1:15
    files = dir([d,'/Train/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(r,2*c,numel(files)); % 496,512
    label = ones(numel(files),1);
    tic
    mkdir('1/',num2str(sub))
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/Train/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii = imresize(ii(:,:,1),[s1,s2]);ii(ii == 255)=10; ii = preprocess(ii);
        images(:,:,i) = ii((round(0.7*s1)-r+1):round(0.7*s1),(round(0.5*s2)-c+1):(round(0.5*s2)+c));
        
        
    end
    for temp =1:size(images,3) 
            k=k+1;
    
            imwrite(mat2gray(images(:,:,temp)),['1/',num2str(sub),'/',num2str(k),'.png'],'png')
    end 
    toc
       
end
%% AMD Data
k=0;
for sub = 1:15
    files = dir([d,'/Train/AMD',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(r,2*c,numel(files)); % 496,512
    label = 2*ones(numel(files),1);
    mkdir('2/',num2str(sub))
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/Train/AMD',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii = imresize(ii(:,:,1),[s1,s2]);ii(ii == 255)=10; ii = preprocess(ii);
        images(:,:,i) = ii((round(0.7*s1)-r+1):round(0.7*s1),(round(0.5*s2)-c+1):(round(0.5*s2)+c));
       
    end
    
    for temp =1:size(images,3) 
            k=k+1;
    
            imwrite(mat2gray(images(:,:,temp)),['2/',num2str(sub),'/',num2str(k),'.png'],'png')
    end 
end
%% DMEData
k=0;
for sub = 1:15
    files = dir([d,'/Train/DME',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(r,2*c,numel(files)); % 496,512
    label = 3*ones(numel(files),1);
    mkdir('3/',num2str(sub))
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/Train/DME',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii = imresize(ii(:,:,1),[s1,s2]);ii(ii == 255)=10; ii = preprocess(ii);
        images(:,:,i) = ii((round(0.7*s1)-r+1):round(0.7*s1),(round(0.5*s2)-c+1):(round(0.5*s2)+c));
    end
    for temp =1:size(images,3) 
            k=k+1;
    
            imwrite(mat2gray(images(:,:,temp)),['3/',num2str(sub),'/',num2str(k),'.png'],'png')
    end 
    
end
