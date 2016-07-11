require 'sys'
require 'cunn'
require 'nn' 
require 'cudnn'
require 'optim'
require 'math'
require 'hdf5'
require 'image'
require 'cutorch'

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

myFile = hdf5.open('data_error.h5', 'r')
data = myFile:read(''):all()
myFile:close()

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        output = model_2:forward(Images:cuda())
--        err = criterion:forward(output:float(),torch.ones(1))
        df_do = criterion:backward(output,Target:forward(torch.ones(1):cuda()))
        model_2:backward(Images:cuda(), df_do)
 	collectgarbage()
      return err,gradParameters
end

criteria = nn.ClassNLLCriterion()
--criteria = nn.MultiMarginCriterion()
criteria:cuda()
criterion = nn.ParallelCriterion():add(criteria,1):add(criteria,1):add(criteria,1)
Target =  nn.ConcatTable()
Target:add(nn.Identity()):add(nn.Identity()):add(nn.Identity())
optimState = {}
optimMethod = optim.sgd

model = torch.load('Model.t7')
model:cuda():clearState()
--model:forward(torch.Tensor(1,3,224,224):cuda())
model_2 = model:clone()
model_2:cuda()

test_2 = data['test_2']:mul(255)
test_3 = data['test_3']:mul(255)
ind =  (data['ind']):add(1)

--Images = test_2[{{ind[1]},{},{},{}}]:cuda()
Images = test_3[{{11},{},{},{}}]:cuda()
parameters,gradParameters = model_2:getParameters()
state={}
optimMethod(func, parameters,state)--end
--[[
for ind = 1, 61 do
k ,kk = torch.max( model:forward(test_3[{{ind},{},{},{}}]:cuda())[1]:float(),1)
print(kk)
print(ind)
end
]]--
parameters2,gradParameters2 = model_2:parameters()

for temp =1,table.getn(gradParameters2) do
gradParameters2[temp] = gradParameters2[temp]:float() 
end

Grad = {}
n = 1
table.insert(Grad,(gradParameters2[n]):reshape((gradParameters2[n]):size(1),(gradParameters2[n]):size(2)*(gradParameters2[n]):size(3)*(gradParameters2[n]):size(4)))
n = 3
table.insert(Grad,(gradParameters2[n]):reshape((gradParameters2[n]):size(1),(gradParameters2[n]):size(2)*(gradParameters2[n]):size(3)*(gradParameters2[n]):size(4)))
n = 5
table.insert(Grad,(gradParameters2[n]):reshape((gradParameters2[n]):size(1),(gradParameters2[n]):size(2)*(gradParameters2[n]):size(3)*(gradParameters2[n]):size(4)))
Grad[1] = Grad[1]:sum(2)
Grad[2] = Grad[2]:sum(2)
Grad[3] = Grad[3]:sum(2)

for i = 7,114,12 do
n = i
cc = (gradParameters2[n]):reshape((gradParameters2[n]):size(1),(gradParameters2[n]):size(2)*(gradParameters2[n]):size(3)*(gradParameters2[n]):size(4))
n = n+4
cc2 = (gradParameters2[n]):reshape((gradParameters2[n]):size(1),(gradParameters2[n]):size(2)*(gradParameters2[n]):size(3)*(gradParameters2[n]):size(4))
cc = torch.cat(cc:sum(2),cc2:sum(2),1)
n = n+4
cc2 = (gradParameters2[n]):reshape((gradParameters2[n]):size(1),(gradParameters2[n]):size(2)*(gradParameters2[n]):size(3)*(gradParameters2[n]):size(4))
cc = torch.cat(cc,cc2:sum(2),1)
n = n+2
cc2 = (gradParameters2[n]):reshape((gradParameters2[n]):size(1),(gradParameters2[n]):size(2)*(gradParameters2[n]):size(3)*(gradParameters2[n]):size(4))
cc = torch.cat(cc,cc2:sum(2),1)
table.insert(Grad,cc)
end

for temp = 1,table.getn(Grad) do
Grad[temp] = Grad[temp]:reshape(Grad[temp]:size(1))
end


model:evaluate()
Responses = {}
response = model:get(1):forward(Images)
table.insert(Responses,response:float())
for lay = 2,14 do
print(lay)
response = model:get(lay):forward(response)
table.insert(Responses,response:float())
end

response = model:get(15):get(1):get(1):forward(response)
table.insert(Responses,response:float())
response = model:get(15):get(1):get(2):forward(response)
table.insert(Responses,response:float())
response = model:get(15):get(1):get(3):forward(response)
table.insert(Responses,response:float())
response = model:get(15):get(1):get(4):get(1):get(1):forward(response)
table.insert(Responses,response:float())
response = model:get(15):get(1):get(4):get(1):get(2):forward(response)
table.insert(Responses,response:float())
response = model:get(15):get(1):get(4):get(1):get(3):forward(response)
table.insert(Responses,response:float())
response = model:get(15):get(1):get(4):get(1):get(4):forward(response)
table.insert(Responses,response:float())

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

1,2-- spatial convolution 3--64 stride(2) --> Relu--> maxpooling stride(2) -->LRN
3,4-- spatial convolution 64--64 --> Relu 
5,6-- spatial convolution 64--192 --> Relu -->LRN -> maxpooling stride(2)
7,18,-- inception1
19-30-- inception2 --> maxpool stride(2)
31- 42 --inception3  (loss1 removed for evaluation)
43-54 --inception4  
55-66 --inception5
67-78 --inception6  (loss2 removed for evaluation)
79-90 --inception7 ---> maxpool stride(2)
91-102 -- inception8
103-114 -- inception9 --> averpool --> view(1024)-->(dropout removed for evaluaiton) linear(1024,3)
]]--
lis = {1,1,1,1,2,2,3,3,3,3,4,5,5,6,7,8,9,10,10,11,12}
for temp =1,21 do
--print(temp,gradParameters2[lis[temp]]:size())
print(temp)
myFile = hdf5.open('/home/raj/Data/retina2/error/grad_' .. temp .. '.h5', 'w')
myFile:write('/home/raj/Data',Grad[lis[temp]])
myFile:close()
oo,n = torch.max(Grad[lis[temp]],1)
myFile = hdf5.open('/home/raj/Data/retina2/error/resp_' .. temp .. '.h5', 'w')
myFile:write('/home/raj/Data',Responses[temp][1][n[1]])
myFile:close()
myFile = hdf5.open('/home/raj/Data/retina2/error/Resp_' .. temp .. '.h5', 'w')
myFile:write('/home/raj/Data',Responses[temp][1])
myFile:close() 
end
