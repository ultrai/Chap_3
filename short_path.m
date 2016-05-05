function [out] = short_path(inputImg)
%inputImg = E;
szImg = size(inputImg);
img = zeros([szImg(1) szImg(2)+2]);

img(:,2:1+szImg(2)) = inputImg;

% update size of image
szImg = size(img);

% get vertical gradient image
[~,gradImg] = gradient(img,1,1);
gradImg = -1*gradImg;

% normalize gradient
gradImg = (gradImg-min(gradImg(:)))/(max(gradImg(:))-min(gradImg(:)));
gradImg = img;
% get the "invert" of the gradient image.
gradImgMinus = gradImg*-1+1; 
gradImgMinus = img;
%% generate adjacency matrix, see equation 1 in the refered article.

%minimum weight
minWeight = 1E-15;

neighborIterX = [1 1  1 0  0 -1 -1 -1];
neighborIterY = [1 0 -1 1 -1  1  0 -1];

% get location A (in the image as indices) for each weight.
adjMAsub = 1:szImg(1)*szImg(2);

% convert adjMA to subscripts
[adjMAx,adjMAy] = ind2sub(szImg,adjMAsub);

adjMAsub = adjMAsub';
szadjMAsub = size(adjMAsub);

% prepare to obtain the 8-connected neighbors of adjMAsub
% repmat to [1,8]
neighborIterX = repmat(neighborIterX, [szadjMAsub(1),1]);
neighborIterY = repmat(neighborIterY, [szadjMAsub(1),1]);

% repmat to [8,1]
adjMAsub = repmat(adjMAsub,[1 8]);
adjMAx = repmat(adjMAx, [1 8]);
adjMAy = repmat(adjMAy, [1 8]);

% get 8-connected neighbors of adjMAsub
% adjMBx,adjMBy and adjMBsub
adjMBx = adjMAx+neighborIterX(:)';
adjMBy = adjMAy+neighborIterY(:)';

% make sure all locations are within the image.
keepInd = adjMBx > 0 & adjMBx <= szImg(1) & ...
    adjMBy > 0 & adjMBy <= szImg(2);

% adjMAx = adjMAx(keepInd);
% adjMAy = adjMAy(keepInd);
adjMAsub = adjMAsub(keepInd);
adjMBx = adjMBx(keepInd);
adjMBy = adjMBy(keepInd); 

adjMBsub = sub2ind(szImg,adjMBx(:),adjMBy(:))';

% calculate weight
%adjMW = 2 - gradImg(adjMAsub(:)) - gradImg(adjMBsub(:)) + minWeight;
%adjMmW = 2 - gradImgMinus(adjMAsub(:)) - gradImgMinus(adjMBsub(:)) + minWeight;
adjMW = 2 - gradImg(adjMAsub(:)) - gradImg(adjMBsub(:)) + minWeight;

% pad minWeight on the side
imgTmp = nan(size(gradImg));
imgTmp(:,1) = 1;
imgTmp(:,end) = 1;
imageSideInd = ismember(adjMBsub,find(imgTmp(:)==1));
adjMW(imageSideInd) = minWeight;
%adjMmW(imageSideInd) = minWeight;

% build sparse matrices
adjMatrixW = sparse(adjMAsub(:),adjMBsub(:),adjMW(:),numel(img(:)),numel(img(:)));
% build sparse matrices with inverted gradient.
adjMatrixMW = [];%sparse(adjMAsub(:),adjMBsub(:),adjMmW(:),numel(img(:)),numel(img(:)));
[ dist,path{1} ] = graphshortestpath( adjMatrixW, 1, numel(img(:)) );
[pathX,pathY] = ind2sub(szImg,path{1});

% get rid of first and last few points that is by the image borders
out = zeros(szImg);
for temp = 1:numel(pathX)
     out(pathX(temp),pathY(temp)) =1;
end
out = out(:,2:end-1);
pathX =pathX(gradient(pathY)~=0);
pathX = pathX(2:end-1);
pathY =pathY(gradient(pathY)~=0);

%imagesc(img); axis image; colormap('gray'); hold on;
%plot(pathY,pathX,'g--','linewidth',2); hold on;

end