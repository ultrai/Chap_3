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
s1= 224;
s2=224;

dummy = ones(s1,s2);
I1_train = {};
I1_test = {};
for sub = 1:15
    files = dir([d,'/data/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,numel(files)); % 496,512
    label = ones(numel(files),1);
    tic
%     mkdir('1/',num2str(sub))
    k=0;
    parfor i = 1:numel(files)
        ii = imread([d,'/data/NORMAL',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
       ii(ii==255) =10;
        ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
        [~,images(:,:,i)] = BM3D(dummy,ii);
        
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
    files = dir([d,'/data/AMD',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,numel(files)); % 496,512
    parfor i = 1:numel(files)
        ii = imread([d,'/data/AMD',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
        [~,images(:,:,i)] = BM3D(dummy,ii);
        
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
    files = dir([d,'/data/DME',num2str(sub),'/TIFFs/8bitTIFFs/*.tif']);
    images = zeros(s1,s2,numel(files)); % 496,512
   parfor i = 1:numel(files)
        ii = imread([d,'/data/DME',num2str(sub),'/TIFFs/8bitTIFFs/',files(i).name]);
        ii(ii==255) =10;
        ii = preprocess(ii);
        ii = imresize(ii(:,:,1),[s1,s2]);
        [~,images(:,:,i)] = BM3D(dummy,ii);
        
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
load('temp.mat')
% n =14;
% sub = 1:15;
% Train_sub = sub(1:n);
% Test_sub = sub(n+1:15);

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
            Feat_train = cat(1,Feat_train,[H0]);% H1 H2 H3]);
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
            Feat_test = cat(1,Feat_test,[H0]);% H1 H2 H3]);
        end
    end
end
save('srinivasan.mat','Feat_train','Feat_test','Label_train','Label_test','Sub_idx_train','Sub_idx_test','-v7.3')            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
clear
load('srinivasan.mat')

% n = 15;
% Val = [];
% Decision=[];
% Decision2=[];
% for n =1:15
%     n
%     Feat_Train = Feat_train(Sub_idx_train~=n,:);
%     Feat_Test= Feat_test(Sub_idx_test==n,:);
%     Label_Train = Label_train(Sub_idx_train~=n);
%     Label_Test= Label_test(Sub_idx_test==n);
%     Mdl = fitcecoc(Feat_Train,Label_Train);
%     est = predict(Mdl,Feat_Test);
% %     B = TreeBagger(50,Feat_Train,Label_Train);
% %     est1 = predict(B,Feat_Test);
% %     est = zeros(size(est1));
% %     for temp =1:size(est,1)
% %         est(temp) = str2double(est1{temp});
% %     end
%     Val = [Val;sum(est==Label_Test)/numel(est)];
% %     est_1 = ones(sum(Label_test==1),1)*mode(est(Label_test==1));
% %     est_2 = ones(sum(Label_test==2),1)*mode(est(Label_test==2));
% %     est_3 = ones(sum(Label_test==3),1)*mode(est(Label_test==3));
% %     Decision =[Decision; sum(([est_1;est_2;est_3])==Label_test)/numel(est)];
%     est_1 = (mode(est(Label_Test==1))==1);
%     est_2 = (mode(est(Label_Test==2))==2);
%     est_3 = (mode(est(Label_Test==3))==3);
%     Decision2 =[Decision2; [est_1 est_2 est_3]];
% end
%%
Decision2=[];
for cv =1:15
    cv
    n=7;
    list = [1:15]';
    list = circshift(list,cv-1);
    Idx_train = zeros(size(Sub_idx_train,1),1);
    Idx_test = zeros(size(Sub_idx_test,1),1);
    for temp = 1:(15-n)
        Idx_train(Sub_idx_train==list(temp)) = temp;
    end
    for temp = (15-n+1):15
        Idx_test(Sub_idx_test==list(temp)) = temp;
    end
    
    Feat_Train = Feat_train(Idx_train>0,:);
    Feat_Test= Feat_test(Idx_test>0,:);
    Label_Train = Label_train(Idx_train>0);
    %Label_Test= Label_test(abs(Sub_idx_test-cv+1)<=n);
%     Mdl = fitcecoc(Feat_Train,Label_Train);
%     Est = predict(Mdl,Feat_Test);
%       Mdl = TreeBagger(7,Feat_Train,Label_Train);
%       Est_temp = predict(Mdl,Feat_Test);   Est = str2num(cell2mat(Est_temp));
    Mdl = trainSoftmaxLayer(Feat_Train',full(ind2vec(Label_Train')));
    [~,Est] = max(Mdl(Feat_Test'));    Est = Est';

    for sub_test = (15-n+1):15
%     Feat_Test= Feat_test(Sub_idx_test==sub_test,:);
        Label_Test = Label_test(Idx_test==sub_test);
        est = Est(Idx_test(Idx_test>n)==sub_test);
        est_1 = (mode(est(Label_Test==1))==1);
        est_2 = (mode(est(Label_Test==2))==2);
        est_3 = (mode(est(Label_Test==3))==3);
        Decision2 =[Decision2; [est_1 est_2 est_3]];
    end
end
mean(Decision2)

% softmax multiscale  1.0000    0.9048    0.7810
% softmax singiscale  1.0000    0.8952    0.7810
%%

% n = 8
% Feat_Train = Feat_train(Sub_idx_train<=n,:);
% Label_Train = Label_train(Sub_idx_train<=n);
% t1 = templateSVM();
% t2 = templateSVM('KernelFunction','gaussian');
% t3 = templateSVM('KernelFunction','rbf');
% Mdl1 = fitcecoc(Feat_Train,Label_Train,'Learners',t1);
% Mdl2 = fitcecoc(Feat_Train,Label_Train,'Learners',t2);
% Mdl3 = fitcecoc(Feat_Train,Label_Train,'Learners',t3);
% Mdl4 = TreeBagger(10,Feat_Train,Label_Train);
% 
% 
% Feat_Test= Feat_test(Sub_idx_test>n,:);
% Label_Test = Label_test(Sub_idx_test>n);
% Est1 = predict(Mdl1,Feat_Test);
% Est2 = predict(Mdl2,Feat_Test);
% Est3 = predict(Mdl3,Feat_Test);
% Est4_temp = predict(Mdl4,Feat_Test);
% Est4 = [];
% for temp =1:size(Est4_temp,1)
%     Est4 = [Est4;str2num(Est4_temp{temp})];
% end
% 
% ACC1 = mean(Est1==Label_Test)
% ACC2 = mean(Est2==Label_Test)
% ACC3 = mean(Est3==Label_Test)
% ACC4 = mean(Est4==Label_Test)
% 
% Decision1 = [];
% Decision2 = [];
% Decision3 = [];
% Decision4 = [];
% for sub_test = (n+1):15
% %     Feat_Test= Feat_test(Sub_idx_test==sub_test,:);
%     Label_Test = Label_test(Sub_idx_test==sub_test);
%     est = Est1(Sub_idx_test(Sub_idx_test>n)==sub_test);
%     est_1 = (mode(est(Label_Test==1))==1);
%     est_2 = (mode(est(Label_Test==2))==2);
%     est_3 = (mode(est(Label_Test==3))==3);
%    Decision1 =[Decision1; [est_1 est_2 est_3]];
%    est = Est2(Sub_idx_test(Sub_idx_test>n)==sub_test);
%     est_1 = (mode(est(Label_Test==1))==1);
%     est_2 = (mode(est(Label_Test==2))==2);
%     est_3 = (mode(est(Label_Test==3))==3);
%    Decision2 =[Decision2; [est_1 est_2 est_3]];
%    est = Est3(Sub_idx_test(Sub_idx_test>n)==sub_test);
%     est_1 = (mode(est(Label_Test==1))==1);
%     est_2 = (mode(est(Label_Test==2))==2);
%     est_3 = (mode(est(Label_Test==3))==3);
%    Decision3 =[Decision3; [est_1 est_2 est_3]];
%   est = Est4(Sub_idx_test(Sub_idx_test>n)==sub_test);
%     est_1 = (mode(est(Label_Test==1))==1);
%     est_2 = (mode(est(Label_Test==2))==2);
%     est_3 = (mode(est(Label_Test==3))==3);
%    Decision4 =[Decision4; [est_1 est_2 est_3]];
% end
