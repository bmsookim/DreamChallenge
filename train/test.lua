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

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'image'
local DataLoader = require 'dataloader'
local models = require 'networks/init'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local t = require 'datasets/transforms'
local datasets = require 'datasets/init'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.best(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

local function findImages(dir)
   local imagePaths = torch.CharTensor()
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i = 2, #extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   local f = io.popen('find -L ' .. dir ..findOptions)

   local maxLength = -1
   local imagePaths = {}

   while true do 
      local line = f:read('*line')
      if not line then break end

      local dirname = paths.dirname(line)
      local filename = paths.basename(line)
      local path = dirname .. '/' .. filename

      table.insert(imagePaths, path)
      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   local nImages = #imagePaths
   
   return imagePaths, nImages
end

testImagePath, nImages = findImages('/preprocessedData/dreamCh/test/')

function loadImage(path)
    local ok, input = pcall(function()
        return image.load(path, 3, 'float')
    end)

    if not ok then
        local f = io.open(path, 'r')
        assert(f, 'Error reading: ' .. tostring(path))
        local data = f:read('*a')
        f:close()

        local b = torch.ByteTensor(string.len(data))
        ffi.copy(b:data(), data, b:size(1))

        input = image.decompress(b, 3, 'float')
    end
    return input
end

local meanstd = {
    mean = { 0.492, 0.492, 0.492 },
    std = { 0.229, 0.229, 0.229 },
}

for i=1,nImages do
   test_path = testImagePath[i]
   test_image = loadImage(test_path)
   --test_image:resize(1, 3, 224, 224)

   local dataset = datasets.create(opt, 'val')
   _G.dataset = dataset
   _G.preprocess = dataset:preprocess()
   input_img = _G.preprocess(test_image)

   result = model:forward(input_img):float()
   exp = torch.exp(result)
   exp_sum = exp:sum()
   exp = torch.div(exp, exp_sum)

   print(test_path)
   print('|-', (exp[1][2]), '\n')
   maxs, indices = torch.max(exp, 2)
end
