--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = nn.PReLU --cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function Dropout()
   return nn.Dropout(opt and opt.dropout or 0,nil,true)
end

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels
   local blocks = {}

   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(nInputPlane))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(nBottleneckPlane))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
         end
      end

      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)

      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Creates count residual blocks with specified number of features
   local function wide_layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'dreamChallenge' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      assert((depth-4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth-4)/6
      local k = opt.widen_factor
      print(' | Wide-ResNet-' .. depth .. 'x' .. k .. ' ImageNet')
      local nStages = torch.Tensor{16, 32, 32*k, 64*k, 128*k}

      -- The ResNet ImageNet model
      model:add(Convolution(3,nStages[1],7,7,2,2,3,3)) -- 256 x 256
      model:add(SBatchNorm(nStages[1]))
      model:add(ReLU(nStages[1]))
      model:add(Max(3,3,2,2,1,1)) -- 128 x 128
      model:add(Convolution(nStages[1],nStages[2],7,7,2,2,3,3)) -- 64 x 64
      model:add(wide_layer(wide_basic, nStages[2], nStages[3], n, 2)) -- 32 x 32
      model:add(wide_layer(wide_basic, nStages[3], nStages[4], n, 2)) -- 16 x 16
      model:add(wide_layer(wide_basic, nStages[4], nStages[5], n, 2)) -- 8 x 8
      model:add(SBatchNorm(nStages[5]))
      model:add(ReLU(nStages[5]))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[5]):setNumInputDims(3))
      model:add(nn.Linear(nStages[5], 2))
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