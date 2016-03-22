require 'sys'
require 'cunn'
require 'nn' 
require 'cudnn'
require 'optim'
require 'math'
require 'hdf5'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

myFile = hdf5.open('data.h5', 'r')
data = myFile:read(''):all()
myFile:close()

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
        neval = neval + 1
        for i = 1,Images:size(1) do
            output = model_2:forward(Images[i]:cuda())
            err = criterion:forward(output:float(),torch.ones(1))
            f = f + err
            df_do = criterion:backward(output:float(),torch.ones(1))
            model_2:backward(Images[i]:cuda(), df_do:cuda())
 	    collectgarbage()
        end
      return f,gradParameters
end

criterion = nn.ClassNLLCriterion()
optimState = {maxIter = 100}
optimMethod = optim.sgd
neval = 0


model = torch.load('Model.t7')
model:cuda()
model_2 = model:clone()

test_2 = data['test_2']
test_3 = data['test_3']
m = data['m']
s = data['s']

ind = 25

Images = test_2[{{ind},{},{},{}}]:cuda()
Images2_1 = data['test_2_images']
Images2 = Images2_1[{{ind},{},{},{}}]
Images2 = Images2:float():reshape(Images:size(3),Images:size(4)):add(-Images2:min()):div(Images2:max())
image.save('I.png', Images2)
--Label = torch.ones(1)
for temp = 1,1 do
parameters,gradParameters = model_2:getParameters()
optimMethod(func, parameters)
end
parameters,gradParameters = model:getParameters()
parameters2,gradParameters2 = model_2:getParameters()
P1,G1 = model_2:float():get(3):parameters()
P2,G2 = model:float():get(3):parameters()
model:remove(9)
model:remove(8)
model:remove(7)
model:remove(6)
model:remove(5)
model:remove(4)

Output =  model:cuda():forward(Images[1]:cuda()):float()
myFile = hdf5.open('/home/mict/2014_Srinivasan/Image.h5', 'w')
myFile:write('/home/mict/2014_Srinivasan', Images2_1[{{ind},{},{},{}}])
myFile:close()
myFile = hdf5.open('/home/mict/2014_Srinivasan/grad.h5', 'w')
myFile:write('/home/mict/2014_Srinivasan', G2[1])
myFile:close()
myFile = hdf5.open('/home/mict/2014_Srinivasan/result.h5', 'w')
myFile:write('/home/mict/2014_Srinivasan', Output)
myFile:close()
myFile = hdf5.open('/home/mict/2014_Srinivasan/weight.h5', 'w')
myFile:write('/home/mict/2014_Srinivasan', P1[1])
myFile:close()
