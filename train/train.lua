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

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)
local elapsed_time = 0

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print(' => Training epoch # ' .. epoch .. " : [LR = " .. self.optimState.learningRate .. "]")
   -- set the batch norm to training mode
   self.model:training()

   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      elapsed_time = elapsed_time + timer:time().real + dataTime
      
      if n % 100 == 0 then print((' | [#%3d][%3d/%d]    Time %.3f  Loss %1.4f  Top1 %7.3f%s')
             :format(epoch, n, trainSize, timer:time().real + dataTime, loss, top1, '%')) end
      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum = 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1*batchSize
      N = N + batchSize
      elpased_time = elapsed_time + timer:time().real + dataTime

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   
   print((' * Finished epoch # %d     top1: %7.3f%s'):format(epoch, top1Sum / N, '%'))
   print(' * Elpased time: '..math.floor(elapsed_time/3600)..' hours '..
                              math.floor((elapsed_time%3600)/60)..' minutes '..
                              math.floor((elapsed_time%3600)%60)..' seconds\n')
   return top1Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = (correct:narrow(2, 1, 1):sum() / batchSize)

   return top1 * 100
end

local function getCudaTensorType(tensorType)
    if tensorType == 'torch.CudaHalfTensor' then
        return cutorch.createCudaHostHalfTensor()
    elseif tensorType == 'torch.CudaDoubleTensor' then
        return cutorch.createCudaHostDoubleTensor()
    else
        return cutorch.createCudaHostTensor()
    end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
        and torch[self.opt.tensorType:match('torch.(%a+)')]()
        or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaTensor() and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'dreamChallenge' then
      decay = math.floor((epoch - 1) / 10)
   end
   return self.opt.LR * math.pow(0.5, decay)
end

return M.Trainer
