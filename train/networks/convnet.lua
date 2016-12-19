--
--  Copyright (c) 2016, DMIS, Digital Mammography DREAM Challenge Team.
--  All rights reserved.
--
--  (Author) Bumsoo Kim, 2016
--  Github : https://github.com/meliketoy/DreamChallenge
--
--  Korea University, Data-Mining Lab
--  Digital Mammography DREAM Challenge Torch Implementation
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local AdaMax = nn.SpatialAdaptiveMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels
   local blocks = {}

   local model = nn.Sequential()
   if opt.dataset == 'dreamChallenge' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      print(' | ConvNet - Challenge Net')
      local nStages = torch.Tensor{256, 512}

      -- The ResNet ImageNet model
      model:add(Convolution(3,nStages[1],7,7,2,2,3,3))                -- 128 x 128
      model:add(SBatchNorm(nStages[1]))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1)) -- 64 x 64
      model:add(Convolution(nStages[1],nStages[2],7,7,2,2,3,3))               -- 32 x 32
      model:add(SBatchNorm(nStages[2]))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1)) -- 16 x 16
      model:add(Avg(16, 16, 1, 1))
      model:add(nn.View(nStages[2]):setNumInputDims(3))
      model:add(nn.Linear(nStages[2], 2))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end

   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
