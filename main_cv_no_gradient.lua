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
require 'hdf5'

Decision = {}
for cv =1,15 do
Model = torch.load('inception.t7')
Model:cuda()
Model:remove(23)
model=nn.Sequential()
model:add(nn.View(1024)):add(nn.Dropout(0.4)):add(nn.Linear(1024,3)):add(cudnn.LogSoftMax())
model:cuda()

criterion = nn.ClassNLLCriterion()--(torch.Tensor({1,1,1}))
--criterion = nn.ClassNLLCriterion(torch.Tensor({0.3,0.3,0.4}))
--criterion = nn.MultiMarginCriterion()
criterion:cuda()

myFile = hdf5.open('data_' .. cv .. '.h5', 'r')
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
        output = model:forward(Model:forward(inputs))
         f = criterion:forward(output,outputs)
            df_do = criterion:backward(output,outputs)
            model:backward(Model:forward(inputs), df_do)
 	    collectgarbage()
table.insert(train_loss,f)
if ee==1 then
        out_test = torch.zeros(inputs_test:size(1))
        for i = 1,inputs_test:size(1)-batch,batch do
            output = model:forward(Model:forward(X_test[{{i,i+batch},{},{},{}}]:cuda()))
            oo,out_test[{{i,i+batch}}] = torch.max(output:float(),2)
 	    collectgarbage()
        end
            output = model:forward(Model:forward(X_test[{{X_test:size(1)-batch,X_test:size(1)},{},{},{}}]:cuda()))
            oo,out_test[{{X_test:size(1)-batch,X_test:size(1)}}] = torch.max(output:float(),2)
 	    collectgarbage()


        acc = torch.sum(torch.eq(out_test:float(), outputs_test))/outputs_test:size(1)
table.insert(test,acc)

end


if Acc<acc then
E = neval
Acc = acc
end
print('error  ',f,'   current acc   ',acc,'  iter  ', neval,  'elapsed   ',sys:toc(),'s',' best acc  ',Acc,'   at  ',E,'  iteration of cv  ' ,cv)
      return f,gradParameters--f/batch,gradParameters:div(batch)
end



sys:tic()
train_loss = {}
train = {}
test = {}
state = {
maxIter = 100,}

optimMethod = optim.adagrad
neval = 0
batch = 64
inputs_test = X_test
outputs_test = Y_test
Acc = 0
parameters,gradParameters = model:getParameters()
Feat_test = X_test:clone()        
 out_test = torch.zeros(inputs_test:size(1))
        for i = 1,inputs_test:size(1)-batch,batch do
            output = model:forward(Model:forward(X_test[{{i,i+batch},{},{},{}}]:cuda()))
            oo,out_test[{{i,i+batch}}] = torch.max(output:float(),2)
 	    collectgarbage()
        end
            output = model:forward(Model:forward(X_test[{{X_test:size(1)-batch,X_test:size(1)},{},{},{}}]:cuda()))
            oo,out_test[{{X_test:size(1)-batch,X_test:size(1)}}] = torch.max(output:float(),2)
 	    collectgarbage()

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
       for i = 1,inputs_test:size(1)-batch,batch do
            output = model:forward(X_test[{{i,i+batch},{},{},{}}]:cuda())
            oo,out_test[{{i,i+batch}}] = torch.max(output:float(),2)
 	    collectgarbage()
        end
            output = model:forward(X_test[{{X_test:size(1)-batch,X_test:size(1)},{},{},{}}]:cuda())
            oo,out_test[{{X_test:size(1)-batch,X_test:size(1)}}] = torch.max(output:float(),2)
 	    collectgarbage()


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
D1 = torch.zeros(15,3)
for sub_test = 9,15 do
    Y_Test = Y_test[torch.eq(idx_test,sub_test)]
    est = out_test[torch.eq(idx_test,sub_test)]
    est_1 = torch.mode(est[torch.eq(Y_Test,1)])
    D1[{{sub_test},{1}}] = est_1
    est_2 = torch.mode(est[torch.eq(Y_Test,2)])
    D1[{{sub_test},{2}}] = est_2
    est_3 = torch.mode(est[torch.eq(Y_Test,3)])
    D1[{{sub_test},{3}}] = est_3
end
Decision[cv] = D1

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
end
torch.save('Decision.t7',Decision)

Decision = torch.load('Decision.t7')
Deci = torch.zeros(105,3)
for temp = 1,15 do
Deci[{{(temp-1)*7+1,(temp-1)*7+7},{}}] = Decision[temp][{{9,15},{}}]
end

Decis =  torch.zeros(105,3)
Decis[{{},{1}}] = torch.eq(Deci[{{},{1}}],1)
Decis[{{},{2}}] = torch.eq(Deci[{{},{2}}],2)
Decis[{{},{3}}] = torch.eq(Deci[{{},{3}}],3)

print(Decis:float():mean(1))
