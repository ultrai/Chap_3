function [out,bias] = preprocess(Img)
opts=edgesTrain();                % default options (good settings)
opts.nChnsColor=1;
opts.modelFnm=['modelBsds_layer8'];        % model name
%opts.modelDir='models_layer_8';        % model name

model=edgesTrain(opts); 
E=edgesDetect(Img,model);
[~,hat]= max(short_path(mat2gray(E)));
tt = zeros(size(Img));
for temp = 1:size(E,2)
    tt(1:hat(temp),temp)=1;
end
[~,hat] = max(1-bwconvhull(tt));
bias = round(0.7*size(Img,1)*ones(1,size(Img,2)))-hat;
Img = Img(:,:,1);
out = Img;
for temp = 1:size(Img,2)
    out(:,temp) = circshift(Img(:,temp),bias(temp),1);
end
%out = cat(3,out,out,out);
end

    
