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
local models = require 'networks/init'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
local ffi = require 'ffi'
local sys = require 'sys'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

labels = {'benign', 'malignant'}

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.best(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)
model:evaluate()

print(opt.resume)

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

-- fd = io.open('/preprocessedData/results.txt', 'w')

count = 0
count_r = 0

for i=1,nImages do
   count = count + 1
   test_path = testImagePath[i]
   test_image = image.load(test_path, 3, 'float')

   for j = 1,3 do
       test_image[j]:add(-0.492)
       test_image[j]:div(0.229)
   end

   test_image:resize(1, 3, 224, 224)

   result = model:forward(test_image):float()
   exp = torch.exp(result)
   exp_sum = exp:sum()
   exp = torch.div(exp, exp_sum)

   print(test_path)
   print('|-', (exp[1][2]))
   if exp[1][2] > 0.5 then
       count_r = count_r + 1
   end
   --fd:write(test_path, '\t', exp[1][2], '\n')
   maxs, indices = torch.max(exp, 2)

   --[[print('The prediction for '..test_path..' is '..
                             sys.COLORS.red .. labels[indices:sum()] ..
                             sys.COLORS.none .. ' by ' .. maxs:sum()
                             .. ' confidence')]]

end

print (count_r / count)
