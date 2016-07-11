require 'sys'
require 'cunn'
require 'nn' 
require 'cudnn'
require 'optim'
require 'math'
require 'cutorch'
--require 'nngraph'
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(27)
cutorch.setDevice(1)
model = torch.load('inception.t7')
--model:add(nn.View(-1))--:setNumInputDims(3))
--model:add(cudnn.LogSoftMax())
--model:add(nn.Linear(1024, 4096)):add(cudnn.ReLU(true)):add(nn.Dropout(0.4)):add(nn.Linear(4096, 3)):add(cudnn.LogSoftMax())
model:add(nn.View(1024)):add(nn.Dropout(0.4)):add(nn.Linear(1024,3)):add(cudnn.LogSoftMax())
--model:add(nn.Linear(1024, 3)):add(cudnn.LogSoftMax())
model:cuda()

--model = require('weight-init')(model, 'xavier' )
--model:forward(torch.Tensor(1,3,224,224):cuda())
--M = nn.Sequential()
--M:add(nn.View(-1))
--M:cuda()
require 'hdf5'
--myFile = hdf5.open('Data.mat', 'r')
myFile = hdf5.open('data.h5', 'r')
data = myFile:read(''):all()
myFile:close()
X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']
idx_test = data['Sub_idx_test']

X_train = X_train:float():mul(255)
X_test = X_test:float():mul(255)
Y_train = Y_train:float():reshape(X_train:size(1))--:add(1)
Y_test = Y_test:float():reshape(X_test:size(1))--:add(1)
idx_test = idx_test:float():reshape(X_test:size(1))--:add(1)


func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
        f_test = 0
        neval = neval + 1
        --[[for i = 1,inputs:size(1) do
            output = model:forward(inputs[{{i},{},{},{}}]:cuda())
            err = criterion:forward(output,outputs[{{i}}]:cuda())
            f = f + err
            df_do = criterion:backward(output,outputs[{{i}}]:cuda())
            model:backward(inputs[{{i},{},{},{}}]:cuda(), df_do:cuda())
 	    collectgarbage()
        end]]--
output = model:forward(inputs:cuda())
            err = criterion:forward(output,outputs:cuda())
            f = err
            df_do = criterion:backward(output,outputs:cuda())
            model:backward(inputs:cuda(), df_do:cuda())
 	    collectgarbage()
table.insert(train_loss,f)
--if (neval%30)==0 then
--model:evaluate()
if ee == 1 then
model:evaluate()
        out_test = torch.zeros(inputs_test:size(1))
        for i = 1,inputs_test:size(1) do
            output = model:forward(X_test[{{i},{},{},{}}]:cuda())
            oo,out_test[i] = torch.max(output:float(),1)
 	    collectgarbage()
        end

acc = torch.sum(torch.eq(out_test:float(), Y_test))/Y_test:size(1)
table.insert(test,acc)
model:training()
end


if Acc<acc then
E = neval
Acc = acc
--Model = model:clone():float()
--torch.save('Model_backup.t7',Model)
end
print('error  ',f,'   current acc   ',acc,'  iter  ', neval,  'elapsed   ',sys:toc(),'s',' best acc  ',Acc,'   at  ',E,'  iteration' )

--end      
--       print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f,  sys:toc(),gradParameters[1]))--acc))--gradParameters[1]))
      return f,gradParameters--f/batch,gradParameters:div(batch)
end
criterion = nn.ClassNLLCriterion()
--criterion = nn.MultiMarginCriterion()
criterion:cuda()

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
batch = 64--8--64--2740--100
inputs_test = X_test
outputs_test = Y_test
Acc = 0
parameters,gradParameters = model:getParameters()
Feat_test = X_test:clone()        
out_test = torch.zeros(inputs_test:size(1))
        for i = 1,inputs_test:size(1) do
            output = model:forward(X_test[{{i},{},{},{}}]:cuda())
            oo,out_test[i] = torch.max(output:float(),1)
 	    collectgarbage()
        end

        acc = torch.sum(torch.eq(out_test:float(), outputs_test))/outputs_test:size(1)
table.insert(test,acc)
   
for epcoh = 1,50 do
ee = 1

   for temp = 1,X_train:size(1)-batch,batch do
        inputs = X_train[{{temp,temp+batch},{},{},{}}]
        outputs = Y_train[{{temp,temp+batch}}]
        optimMethod(func, parameters,state)
ee = ee+1
   end
inputs = X_train[{{X_train:size(1)-batch,X_train:size(1)},{},{},{}}]
outputs = Y_train[{{X_train:size(1)-batch,X_train:size(1)}}]
optimMethod(func, parameters,state)
end

out_test = torch.zeros(Y_test:size(1))
        for i = 1,Y_test:size(1) do
            output = model:forward(X_test[{{i},{},{},{}}]:cuda())
            oo,out_test[i] = torch.max(output:float(),1)
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
csvigo.save{path='/home/raj/Data/train_loss.txt',data=loss}
accu = {data = test}
csvigo.save{path='/home/raj/Data/test_acc.txt',data=accu}
--torch.save('Model.t7',model)
--[[train_loss = torch.Tensor(train_loss)
train = torch.Tensor(train)
test = torch.Tensor(test)
torch.save('train_loss_original_base',train_loss)
torch.save('train',train)
torch.save('test_original_base',test)
torch.save('Model_original.t7',model)]]--
