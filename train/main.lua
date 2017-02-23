-- ***********************************************************************
--  Copyright (c) 2016, DMIS, Digital Mammography DREAM Challenge Team.
--  All rights reserved.
--
--  (Author) Bumsoo Kim, 2016
--  Github : https://github.com/meliketoy/DreamChallenge
--
--  Korea University, Data-Mining Lab
--  Digital Mammography DREAM Challenge Torch Implementation
-- ***********************************************************************

require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'networks/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.best(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = 0
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1  = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 > bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      print('==================================================================')
      print(' * Best model (Top1): ', string.format('%5.2f', testTop1)..'%\n')
      print('=> Saving the best model in '..opt.save)
      print('==================================================================\n')
   end

   checkpoints.save(epoch, model, bestModel, opt)
end

print('\n===============[ Test Result Report ]===============')
print(' * Dataset\t: '..opt.dataset)
print(' * Network\t: '..opt.netType..' '..opt.depth)
print(' * Dropout\t: '..opt.dropout)
print(' * nGPU\t\t: '..opt.nGPU)
print(' * Top1\t\t: '..string.format('%6.3f', bestTop1)..'%')
print('=====================================================')
