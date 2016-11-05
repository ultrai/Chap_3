require 'cunn'
require 'cudnn'

local pl = require('pl.import_into')()

local googlenet = dofile('googlenet.lua')
local net = googlenet({
  cudnn.SpatialConvolution,
  cudnn.SpatialMaxPooling,
  cudnn.ReLU,
  cudnn.SpatialCrossMapLRN
})
--net:cuda()
net:remove(24)
net:remove(24)
net:remove(24)
torch.save('../inception.t7',net)
--[[local googlenet = dofile('googlenet2.lua')
local net = googlenet({
  cudnn.SpatialConvolution,
  cudnn.SpatialMaxPooling,
  cudnn.ReLU,
  cudnn.SpatialCrossMapLRN
})
--net:cuda()
net:remove(24)
net:remove(24)
net:remove(24)
torch.save('../inception2.t7',net)
]]--
