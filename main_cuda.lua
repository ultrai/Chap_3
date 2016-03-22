require 'hdf5'
myFile = hdf5.open('data.h5', 'r')
data = myFile:read(''):all()
myFile:close()
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['y_train']
Y_test = data['y_test']

X_train = X_train:float()
X_test = X_test:float()
Y_train = Y_train:float()--:add(1)
Y_test = Y_test:float()--:add(1)

require 'sys'
require 'cunn'
require 'nn' 
require 'cudnn'
require 'optim'
require 'math'
torch.setdefaulttensortype('torch.FloatTensor')

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
--        f_test = 0
        neval = neval + 1
        for i = 1,Images:size(1) do
            output = model:forward(Images[i]:cuda())
            err = criterion:forward(output:float(),Label[i])
            f = f + err
            df_do = criterion:backward(output:float(),Label[i])
            model:backward(Images[i]:cuda(), df_do:cuda())
 	    collectgarbage()
        end
        --[[out_test = torch.zeros(Images_test:size(1))
        for i = 1,Images_test:size(1) do
            output = model:forward(Images_test[i]:cuda())
            oo,out_test[i] = torch.max(output:float(),1)
 	    collectgarbage()
        end
        acc = torch.sum(torch.eq(out_test, Label_test))/Images_test:size(1)
--        table.insert(train,f/10)
        print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f,  sys:toc(),acc))--gradParameters[1]))]]-- 
      return f/10,gradParameters/10
end
criterion = nn.ClassNLLCriterion()
--criterion:cuda()
model = nn.Sequential()
--model:add(nn.MulConstant(0.00390625))
model:add(cudnn.SpatialConvolution(1, 20, 5, 5,1,1,0)) --146 396
model:add(cudnn.SpatialMaxPooling(3,3,3,3))            --48  132
model:add(cudnn.SpatialConvolution(20, 50, 5, 5,1,1,0))--44  128
model:add(cudnn.SpatialMaxPooling(3,3,3,3))            --14  42
model:add(nn.View(-1):setNumInputDims(3))           -- 50*14*42
model:add(nn.Linear(50*588, 500))
model:add(nn.Sigmoid())
model:add(nn.Linear(500,3))
model:add(nn.LogSoftMax())
model:cuda()
sys:tic()

optimState = {maxIter = 100}
optimMethod = optim.adam
neval = 0
batch = 100
Images_test = X_test
Label_test = Y_test
for epcoh = 1,30 do
    for temp = 1,X_train:size(1)-batch,batch do
        Images = X_train[{{temp,temp+batch},{},{},{}}]
        Label = Y_train[{{temp,temp+batch}}]
        parameters,gradParameters = model:getParameters()
        optimMethod(func, parameters)
    end
out_test = torch.zeros(Images_test:size(1))
        for i = 1,Images_test:size(1) do
            output = model:forward(Images_test[i]:cuda())
            oo,out_test[i] = torch.max(output:float(),1)
 	    collectgarbage()
        end
        acc = torch.sum(torch.eq(out_test, Label_test))/Images_test:size(1)
print(acc)
end
--[[
optimMethod = optim.cg
Images = X_train
Label = Y_train
parameters,gradParameters = model:getParameters()
optimMethod(func, parameters)
torch.save('Model.t7',model)
out_test = torch.zeros(Images_test:size(1))
for i = 1,Images_test:size(1) do
     output = model:forward(Images_test[i]:cuda())
     oo,out_test[i] = torch.max(output:float(),1)
      collectgarbage()
end
acc = torch.sum(torch.eq(out_test, Label_test))/Images_test:size(1)
print(acc)]]--
torch.save('Model.t7',model)
