clear
d = pwd;
%% Model layer
addpath(genpath('BM3D'))
addpath(genpath('Piotr_Matlab_Toolbox'))
opts = edgesTrain();
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm=['modelBsds_layer',num2str(8)];  
model = edgesTrain(opts);
%% Normal Data
s1= 496/2;
s2=512/2;
r = 45;%150
c = 75;%200
dummy = ones(r,2*c);
I1_train = {};
I1_test = {};
for sub = 1:15
    files = dir([d,'/Test/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(r,2*c,numel(files)); % 496,512
    label = ones(numel(files),1);
    tic
    mkdir('1/',num2str(sub))
    k=0;
    parfor i = 1:numel(files)
        ii = imread([d,'/Test/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
       ii(ii == 255)=10; ii = preprocess(ii); ii = imresize(ii(:,:,1),[s1,s2]);
        [~,images(:,:,i)] = BM3D(dummy,ii((round(0.7*s1)-r+1+5):round(0.7*s1+5),(round(0.5*s2)-c+1):(round(0.5*s2)+c)));
%         images(:,:,i) = mat2gray(images(:,:,i));
          close all
    end
    I1_train{sub} = images;
    I1_test{sub} = images;
    toc
end
%% AMD Data
I2_train = {};
I2_test = {};
for sub = 1:15
    files = dir([d,'/Test/AMD',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(r,2*c,numel(files)); % 496,512
    parfor i = 1:numel(files)
        ii = imread([d,'/Test/AMD',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
         ii(ii == 255)=10;ii = preprocess(ii);ii = imresize(ii(:,:,1),[s1,s2]);
        [~,images(:,:,i)] = BM3D(dummy,ii((round(0.7*s1)-r+1+5):round(0.7*s1+5),(round(0.5*s2)-c+1):(round(0.5*s2)+c)));
% images(:,:,i) = mat2gray(images(:,:,i));
        close all
    end
    I2_test{sub} = images;
    if sub == 1
        images(:,:,1:14) = [];
    elseif sub==2
        images(:,:,1:13) = [];
    end
    I2_train{sub} = images;
        
end
%% DMEData
I3_train = {};
I3_test = {};
for sub = 1:15
    files = dir([d,'/Test/DME',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(r,2*c,numel(files)); % 496,512
   parfor i = 1:numel(files)
        ii = imread([d,'/Test/DME',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii == 255)=10; ii = preprocess(ii);ii = imresize(ii(:,:,1),[s1,s2]);
        [~,images(:,:,i)]  = BM3D(dummy,ii((round(0.7*s1)-r+1+5):round(0.7*s1+5),(round(0.5*s2)-c+1):(round(0.5*s2)+c)));
% images(:,:,i) = mat2gray(images(:,:,i));
        close all
    end
    I3_test{sub} = images;
    if sub == 1
        images(:,:,1:8) = [];
    elseif sub==2
        images(:,:,1:14) = [];
    elseif sub == 3
        images(:,:,56:61) = [];
        images(:,:,17:26) = [];
    end
    I3_train{sub} = images;
    
end
save('temp.mat','-v7.3')

n =14;
sub = 1:15;
Train_sub = sub(1:n);
Test_sub = sub(n+1:15);

I_train{1} = I1_train;
I_train{2} = I2_train;
I_train{3} = I3_train;
I_test= {};
I_test{1} = I1_test;
I_test{2} = I2_test;
I_test{3} = I3_test;

Feat_train = [];
Label_train = [];
Sub_idx_train = [];

for disease = 1:3
    for sub = 1:15
        sub
        ls = size(I_train{disease}{sub},3);
        I = I_train{disease}{sub};
        Label_train = cat(1,Label_train,zeros(ls,1)+disease);
        Sub_idx_train = cat(1,Sub_idx_train,zeros(ls,1)+sub);
        for Idx = 1:ls
            I0 = I(:,:,Idx);
            I1 = impyramid(I0, 'reduce'); I2 = impyramid(I1, 'reduce');I3 = impyramid(I2, 'reduce');
            H0 = extractHOGFeatures(I0,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            H1 = extractHOGFeatures(I1,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            H2 = extractHOGFeatures(I2,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            H3 = extractHOGFeatures(I3,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            Feat_train = cat(1,Feat_train,[H0 H1 H2 H3]);
        end
    end
end
Feat_test = [];
Label_test = [];
Sub_idx_test = [];

for disease = 1:3
    for sub = 1:15
        sub
        ls = size(I_test{disease}{sub},3);
        I = I_test{disease}{sub};
        Label_test = cat(1,Label_test,zeros(ls,1)+disease);
        Sub_idx_test = cat(1,Sub_idx_test,zeros(ls,1)+sub);
        for Idx = 1:ls
            I0 = I(:,:,Idx);
            I1 = impyramid(I0, 'reduce'); I2 = impyramid(I1, 'reduce');I3 = impyramid(I2, 'reduce');
            H0 = extractHOGFeatures(I0,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            H1 = extractHOGFeatures(I1,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            H2 = extractHOGFeatures(I2,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            H3 = extractHOGFeatures(I3,'CellSize',[4 4],'BlockSize',[2 2],'BlockOverlap',[1 1] );
            Feat_test = cat(1,Feat_test,[H0 H1 H2 H3]);
        end
    end
end
save('srinivasan.mat','Feat_train','Feat_test','Label_train','Label_test','Sub_idx_train','Sub_idx_test','-v7.3')            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
clear
load('srinivasan.mat')

n = 15;
Val = [];
Decision=[];
Decision2=[];
for n =1:15
    n
    Feat_Train = Feat_train(Sub_idx_train~=n,:);
    Feat_Test= Feat_test(Sub_idx_test==n,:);
    Label_Train = Label_train(Sub_idx_train~=n);
    Label_Test= Label_test(Sub_idx_test==n);
    Mdl = fitcecoc(Feat_Train,Label_Train);
    est = predict(Mdl,Feat_Test);
%     B = TreeBagger(50,Feat_Train,Label_Train);
%     est1 = predict(B,Feat_Test);
%     est = zeros(size(est1));
%     for temp =1:size(est,1)
%         est(temp) = str2double(est1{temp});
%     end
    Val = [Val;sum(est==Label_Test)/numel(est)];
%     est_1 = ones(sum(Label_test==1),1)*mode(est(Label_test==1));
%     est_2 = ones(sum(Label_test==2),1)*mode(est(Label_test==2));
%     est_3 = ones(sum(Label_test==3),1)*mode(est(Label_test==3));
%     Decision =[Decision; sum(([est_1;est_2;est_3])==Label_test)/numel(est)];
    est_1 = (mode(est(Label_Test==1))==1);
    est_2 = (mode(est(Label_Test==2))==2);
    est_3 = (mode(est(Label_Test==3))==3);
    Decision2 =[Decision2; [est_1 est_2 est_3]];
end
n = 8
Feat_train_temp = Feat_train;
Label_train_temp = Label_train;
Feat_test_temp = Feat_test;
Label_test_temp = Label_test;
list = [1:15]';
list = circshift(list,-1);
dd = [];
dd2=[];
for val = 1:15
    Feat_Train = [];
Label_Train = [];
Feat_Test = [];
Label_Test = [];
list = circshift(list,1);
for sub = 1:n
Feat_Train = cat(1,Feat_Train,Feat_train_temp(Sub_idx_train==list(sub),:));
Label_Train = cat(1,Label_Train,Label_train_temp(Sub_idx_train==list(sub)));
end
for sub = (n+1):15
Feat_Test = cat(1,Feat_Test,Feat_test_temp(Sub_idx_test==list(sub),:));
Label_Test = cat(1,Label_Test,Label_test_temp(Sub_idx_test==list(sub)));
end
Mdl = fitcecoc(Feat_Train,Label_Train);
Est = predict(Mdl,Feat_Test);
Err = sum(Est==Label_Test)/numel(Est);
Decision3 = [];
Decision = [];
for sub_test = (n+1):15
    Feat_Test= Feat_test_temp(Sub_idx_test==list(sub_test),:);
    Label_Test_t = Label_test_temp(Sub_idx_test==list(sub_test));
    est = predict(Mdl,Feat_Test);
    temp = ones(numel(est));
    est_1 = (mode(est(Label_Test_t==1))==1);
    est_2 = (mode(est(Label_Test_t==2))==2);
    est_3 = (mode(est(Label_Test_t==3))==3);
   Decision3 =[Decision3; [est_1 est_2 est_3]];
   
   est_1 = mean(est(Label_Test_t==1)==1);
   est_2 = mean(est(Label_Test_t==2)==2);
   est_3 = mean(est(Label_Test_t==3)==3);
%    est_1 = mean((est(Label_Test==1))==(1*temp(Label_Test==1)));%/numel(Label_Test==1);
%     est_2 = mean((est(Label_Test==2))==(2*temp(Label_Test==2)));%/numel(Label_Test==2);
%     est_3 = mean((est(Label_Test==3))==(3*temp(Label_Test==3)));%/numel(Label_Test==3);
   Decision =[Decision; [est_1 est_2 est_3]];
end
dd=[dd;Decision3];
dd2=[dd2;Decision];
end
% SVM = cell(3,1);
% for class = 1:3
%     feat_train = Feat_train;
%     label_train = Label_train;
%     feat_train = feat_train(Label_train~=class,:);
%     label_train = label_train(Label_train~=class);
%     SVM{class} = svmtrain(feat_train,label_train,'kernel_function','linear');                 
% end
