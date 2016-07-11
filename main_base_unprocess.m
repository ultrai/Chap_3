clear
for temp = 1:3
    for temp2 = 1:15
    mkdir(['Digits/Train/',num2str(temp),'/',num2str(temp2)])
    mkdir(['Digits/Test/',num2str(temp),'/',num2str(temp2)])
    end
end

d = pwd;
%% Normal Data
s1= 224;%496/2;
s2=224;%512/2;
dummy = ones(s1,s2);
Images= [];
Sub_idx=[];
Label = []; 
for sub = 1:8
    files = dir([d,'/data/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,3,numel(files)); % 496,512
    label = ones(numel(files),1);
    sub_idx = ones(numel(files),1)*sub;
    tic
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = imresize(ii(:,:,1),[s1,s2]);
        
        images(:,:,1,i) = ii;
         images(:,:,2,i) = ii;
         images(:,:,3,i) = ii;
          close all
    end
    
    mkdir(['Digits/Train/1/',num2str(sub)])
     for temp =1:size(images,4) 
            k=k+1;
    
            imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,2,temp)),mat2gray(images(:,:,3,temp))),['Digits/Train/1/',num2str(sub),'/',num2str(k),'.png'],'png')
     end
    
    toc
   Images = cat(4,Images,images);
   Sub_idx =[Sub_idx;sub_idx];
   Label =[Label;label];
    
end

%% AMD Data
k=0;
for sub = 1:8
    files = dir([d,'/data/AMD',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,3,numel(files)); % 496,512
    label = 2*ones(numel(files),1);
    sub_idx = ones(numel(files),1)*sub;
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/AMD',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = imresize(ii(:,:,1),[s1,s2]);
        images(:,:,1,i) = ii;
         images(:,:,2,i) = ii;
         images(:,:,3,i) = ii;
          close all
    end
    if sub == 1
        images(:,:,:,1:14) = [];
        label(1:14) = [];
        sub_idx(1:14) = [];
    elseif sub==2
        images(:,:,:,1:13) = [];
        label(1:13) = [];
        sub_idx(1:13) = [];
    end
    toc
   Images = cat(4,Images,images);
   Sub_idx =[Sub_idx;sub_idx];
   Label =[Label;label];
   mkdir(['Digits/Train/2/',num2str(sub)])
   for temp =1:size(images,4) 
            k=k+1;
             imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,2,temp)),mat2gray(images(:,:,3,temp))),['Digits/Train/2/',num2str(sub),'/',num2str(k),'.png'],'png')
    end 
end
%% DMEData
k=0;
for sub = 1:8
    files = dir([d,'/data/DME',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,3,numel(files)); % 496,512
    label = 3*ones(numel(files),1);
    sub_idx = ones(numel(files),1)*sub;
    k=0;
    for i = 1:numel(files)
        ii = imread([d,'/data/DME',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        
         ii = imresize(ii(:,:,1),[s1,s2]);
        
        images(:,:,1,i) = ii;
         images(:,:,2,i) = ii;
         images(:,:,3,i) = ii;
          close all
    end
    toc
    if sub == 1
        images(:,:,:,1:8) = [];
        label(1:8) = [];
        sub_idx(1:8) = [];
    elseif sub==2
        images(:,:,:,1:14) = [];
        label(1:14) = [];
        sub_idx(1:14) = [];
    elseif sub == 3
        images(:,:,:,56:61) = [];
        label(56:61) = [];
        sub_idx(56:61) = [];
        images(:,:,:,17:26) = [];
        label(17:26) = [];
        sub_idx(17:26) = [];
    end
   Images = cat(4,Images,images);
   Sub_idx =[Sub_idx;sub_idx];
   Label =[Label;label];
   mkdir(['Digits/Train/3/',num2str(sub)])
   for temp =1:size(images,4) 
            k=k+1;
    
            imwrite(cat(3,mat2gray(images(:,:,1,temp)),mat2gray(images(:,:,2,temp)),mat2gray(images(:,:,3,temp))),['Digits/Train/3/',num2str(sub),'/',num2str(k),'.png'],'png')
    end 
end
%%
