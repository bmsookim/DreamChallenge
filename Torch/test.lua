require 'torch'
require 'image'
local optim = require 'optim'
local M = {}
local Tester = torch.class('resnet.Tester', M)
local opts = require 'opts'
local _opt = opts.parse(arg)
local elapsed_time = 0

function Tester:__init(model, criterion, opt, optimState)
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

function Tester:test()
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   self.model:evaluate()

   local test_dir = './preprocessedData/test/'
   images = image.load(test_dir .. '1.jpg')

   interpolation = 'bicubic'
   size = _opt.imageSize

   local w,h = images:size(3), images:size(2)
   if(w <= h and w == size) or (h <= w and h == size) then
      images = images
   end
   if w < h then
      images = image.scale(images, size, h/w * size, interpolation)
   else
      images = image.scale(images, w/h * size, size, interpolation)
   end

   local w1 = math.ceil((images:size(3) - size)/2)
   local h1 = math.ceil((images:size(2) - size)/2)
   images = image.crop(images, w1, h1, w1+size, h1+size)
   self:copyInputs(images)

   print(images:size())
   output = self.model:forward(self.input):float()
   print (output:size())
end

function Tester:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
end

return M.Tester
