require 'sys'
require 'cunn'
require 'nn' 
require 'cudnn'
require 'optim'
require 'math'
require 'cutorch'
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(27)
cutorch.setDevice(1)
model = torch.load('inception.t7')

--[[model  
inception1(conv(192,64,1,1):ReLU,conv(192,96,1,1):Relu:conv(96,128,3,3,pad):ReLU,conv(192,16,1,1):Relu:conv(16,32,5,5,pad):ReLU,maxpool(3,3,1,1):conv(192,32):relu)-->256
inception2(conv(256,128,1,1):ReLU,conv(256,128,1,1):Relu:conv(128,192,3,3,pad):ReLU,conv(256,32,1,1):Relu:conv(32,96,5,5,pad):ReLU,maxpool(3,3,1,1):conv(256,64):relu)-->480
inception3(conv(480,192,1,1):ReLU,conv(480,96,1,1):Relu:conv(96,204,3,3,pad):ReLU,conv(480,16,1,1):Relu:conv(16,48,5,5,pad):ReLU,maxpool(3,3,1,1):conv(480,64):relu)-->508
inception4(conv(508,160,1,1):ReLU,conv(508,112,1,1):Relu:conv(112,224,3,3,pad):ReLU,conv(508,24,1,1):Relu:conv(24,64,5,5,pad):ReLU,maxpool(3,3,1,1):conv(508,64):relu)-->512
inception5(conv(512,128,1,1):ReLU,conv(512,128,1,1):Relu:conv(128,256,3,3,pad):ReLU,conv(512,24,1,1):Relu:conv(24,64,5,5,pad):ReLU,maxpool(3,3,1,1):conv(512,64):relu)-->512
inception6(conv(512,112,1,1):ReLU,conv(512,144,1,1):Relu:conv(144,288,3,3,pad):ReLU,conv(512,32,1,1):Relu:conv(32,64,5,5,pad):ReLU,maxpool(3,3,1,1):conv(512,64):relu)-->528
inception7conv(528,256,1,1):ReLU,conv(528,160,1,1):Relu:conv(160,320,3,3,pad):ReLU,conv(528,32,1,1):Relu:conv(32,128,5,5,pad):ReLU,maxpool(3,3,1,1):conv(528,128):relu)-->832
inception8conv(832,256,1,1):ReLU,conv(832,160,1,1):Relu:conv(160,320,3,3,pad):ReLU,conv(832,48,1,1):Relu:conv(48,128,5,5,pad):ReLU,maxpool(3,3,1,1):conv(832,128):relu)-->832
inception9conv(832,384,1,1):ReLU,conv(832,192,1,1):Relu:conv(192,384,3,3,pad):ReLU,conv(832,48,1,1):Relu:conv(48,128,5,5,pad):ReLU,maxpool(3,3,1,1):conv(832,128):relu)-->528

1-- spatial convolution 3--64 stride(2) --> Relu--> maxpooling stride(2) -->LRN
5-- spatial convolution 64--64 --> Relu 
7-- spatial convolution 64--192 --> Relu -->LRN -> maxpooling stride(2)
11-- inception1
12-- inception2 --> maxpool stride(2)
14 --inception3  (loss1 removed for evaluation)
15 --inception4  
16 --inception5
17 --inception6  (loss2 removed for evaluation)
18 --inception7 ---> maxpool stride(2)
20 -- inception8
21 -- inception9 --> averpool --> view(1024)-->(dropout removed for evaluaiton) linear(1024,3)
]]--
model:cuda()
--model = require('weight-init')(model, 'xavier' )
incep =  function(Model) 
local loss1 = nn.Sequential()
loss1:add(cudnn.SpatialAveragePooling(5,5,3,3))
loss1:add(cudnn.SpatialConvolution(508,128,1,1)):add(cudnn.ReLU(true)):add(nn.View(128*3*3))
loss1:add(nn.Linear(128*3*3,1024)):add(cudnn.ReLU(true)):add(nn.Dropout(0.7)):add(nn.Linear(1024,3)):add(cudnn.LogSoftMax())
local loss2 = nn.Sequential()
loss2:add(cudnn.SpatialAveragePooling(5,5,3,3))
loss2:add(cudnn.SpatialConvolution(528,128,1,1)):add(cudnn.ReLU(true)):add(nn.View(128*3*3))
loss2:add(nn.Linear(128*3*3,1024)):add(cudnn.ReLU(true)):add(nn.Dropout(0.7)):add(nn.Linear(1024,3)):add(cudnn.LogSoftMax())
local model1 = nn.Sequential()
for lay =1,14 do
model1:add(Model:get(lay))
end
local model2 = nn.Sequential()
for lay =15,17 do
model2:add(Model:get(lay))
end
local model3 = nn.Sequential()
for lay =18,23 do
model3:add(Model:get(lay))
end
model3:add(nn.View(1024)):add(nn.Dropout(0.4)):add(nn.Linear(1024,3)):add(cudnn.LogSoftMax())

local model_3_l2 = nn.ConcatTable()
model_3_l2:add(model3)
model_3_l2:add(loss2)
model2:add(model_3_l2)
local model_2_l1 = nn.ConcatTable()
model_2_l1:add(model2)
model_2_l1:add(loss1)
model1:add(model_2_l1):add(nn.FlattenTable())
 return model1
end
model = incep(model)
model:cuda()



--model = require('weight-init')(model, 'xavier' )






criteria = nn.ClassNLLCriterion()
--criterion = nn.MultiMarginCriterion()
criteria:cuda()
criterion = nn.ParallelCriterion():add(criteria,1):add(criteria,1):add(criteria,1)
Target =  nn.ConcatTable()
Target:add(nn.Identity())
Target:add(nn.Identity())
Target:add(nn.Identity())
--target = {torch.ones(1):cuda(), torch.ones(1):cuda(), torch.ones(1):cuda()}
--tt = model1:cuda():forward(torch.Tensor(1,3,224,224):cuda())
--oo = losstable:backward(tt,target)

require 'hdf5'
myFile = hdf5.open('data.h5', 'r')
data = myFile:read(''):all()
myFile:close()
X_train = data['X_train'];X_test = data['X_test'];Y_train = data['Y_train'];Y_test = data['Y_test']

X_train = X_train:float():mul(255);X_test = X_test:float():mul(255);Y_train = Y_train:float():reshape(X_train:size(1));Y_test = Y_test:float():reshape(X_test:size(1))


func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f_test = 0
        neval = neval + 1
        output = model:forward(inputs)
         f = criterion:forward(output,Target:forward(outputs))
            df_do = criterion:backward(output,Target:forward(outputs))
            model:backward(inputs, df_do)
 	    collectgarbage()
table.insert(train_loss,f)
if ee==1 then
        out_test = torch.zeros(inputs_test:size(1))
        for i = 1,inputs_test:size(1) do
            output = model:forward(X_test[{{i},{},{},{}}]:cuda())
            oo,out_test[i] = torch.max(output[1]:float(),1)
 	    collectgarbage()
        end

        acc = torch.sum(torch.eq(out_test:float(), outputs_test))/outputs_test:size(1)
table.insert(test,acc)

end


if Acc<acc then
E = neval
Acc = acc
--Model = model:clone():float()
--torch.save('Model_backup.t7',Model:clearstate())
end
print('error  ',f,'   current acc   ',acc,'  iter  ', neval,  'elapsed   ',sys:toc(),'s',' best acc  ',Acc,'   at  ',E,'  iteration' )

--       print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f,  sys:toc(),gradParameters[1]))--acc))--gradParameters[1]))
      return f,gradParameters--f/batch,gradParameters:div(batch)
end



sys:tic()
train_loss = {}
train = {}
test = {}
state = {
maxIter = 100,}
 --  learningRate = 1e-3,
--momentum = 0.9,
--dampening = 0,
--weightDecay = 1e-6,
--nesterov = true,}

optimMethod = optim.adagrad--sgd--adadelta--cg--adam--adagrad--
neval = 0
batch = 64--8--64
inputs_test = X_test
outputs_test = Y_test
Acc = 0
parameters,gradParameters = model:getParameters()
Feat_test = X_test:clone()        
 out_test = torch.zeros(inputs_test:size(1))
        for i = 1,inputs_test:size(1) do
            output = model:forward(X_test[{{i},{},{},{}}]:cuda())
            oo,out_test[i] = torch.max(output[1]:float(),1)
 	    collectgarbage()
        end

        acc = torch.sum(torch.eq(out_test:float(), outputs_test))/outputs_test:size(1)
  table.insert(test,acc)   
for epcoh = 1,50 do
ee = 1
   for temp = 1,X_train:size(1)-batch,batch do
--         print(temp)
        inputs = X_train[{{temp,temp+batch},{},{},{}}]:cuda()
        outputs = Y_train[{{temp,temp+batch}}]:cuda()
        optimMethod(func, parameters,state)
ee=ee+1
   end
inputs = X_train[{{X_train:size(1)-batch,X_train:size(1)},{},{},{}}]:cuda()
outputs = Y_train[{{X_train:size(1)-batch,X_train:size(1)}}]:cuda()
optimMethod(func, parameters,state)
end


idx_test = data['Sub_idx_test']
idx_test = idx_test:float():reshape(X_test:size(1))--:add(1)
out_test = torch.zeros(Y_test:size(1))
        for i = 1,Y_test:size(1) do
            output = model:forward(X_test[{{i},{},{},{}}]:cuda())
            oo,out_test[i] = torch.max(output[1]:float(),1)
 	    collectgarbage()
        end

idx_test = idx_test[torch.ge(idx_test,2)]
D = torch.zeros(15,3)
for sub_test = 9,15 do
    Y_Test = Y_test[torch.eq(idx_test,sub_test)]
    est = out_test[torch.eq(idx_test,sub_test)]
    est_1 = torch.eq(torch.mode(est[torch.eq(Y_Test,1)]),1)
    D[{{sub_test},{1}}] = est_1
    est_2 = torch.eq(torch.mode(est[torch.eq(Y_Test,2)]),2)
    D[{{sub_test},{2}}] = est_2
    est_3 = torch.eq(torch.mode(est[torch.eq(Y_Test,3)]),3)
    D[{{sub_test},{3}}] = est_3
end
print(D)
D2 = torch.zeros(15,3)
for sub_test = 9,15 do
    Y_Test = Y_test[torch.eq(idx_test,sub_test)]
    est = out_test[torch.eq(idx_test,sub_test)]
    est_1 = torch.eq(est[torch.eq(Y_Test,1)],1):float():div(torch.eq(Y_Test,1):sum()):sum()
    D2[{{sub_test},{1}}] = est_1
    est_2 = torch.eq(est[torch.eq(Y_Test,2)],2):float():div(torch.eq(Y_Test,2):sum()):sum()
    D2[{{sub_test},{2}}] = est_2
    est_3 = torch.eq(est[torch.eq(Y_Test,3)],3):float():div(torch.eq(Y_Test,3):sum()):sum()
    D2[{{sub_test},{3}}] = est_3
end
print(D2)
require 'csvigo'
loss = {data = train_loss}
csvigo.save{path='/home/raj/Data/train_proper_loss.txt',data=loss}
accu = {data = test}
csvigo.save{path='/home/raj/Data/test_proper_acc.txt',data=accu}
torch.save('Model.t7',model)
--[[
D = {data = D}
csvigo.save{path='/home/raj/Data/test_proper_D.txt',data=D}
D2 = {data = D2}
csvigo.save{path='/home/raj/Data/test_proper_D2.txt',data=D2}

torch.save('Model_proper.t7',model)
train_loss = torch.Tensor(train_loss)
train = torch.Tensor(train)
test = torch.Tensor(test)
torch.save('train_loss_proper',train_loss)
torch.save('train',train)
torch.save('test_proper',test)
torch.save('Model.t7',model)
]]--
