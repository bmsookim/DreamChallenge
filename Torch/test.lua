--  (Author) Bumsoo Kim, 2016
--  Github : https://github.com/meliketoy/DreamChallenge
--
--  Korea University, Data-Mining Lab
--  Image Recognition Torch Implementation
--

test_path = 'preprocessedData/test/19.jpg'

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'image'
local models = require 'networks/init'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

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

test_image = image.load(test_path)
interpolation = 'bicubic'
size = opt.cropSize

local w, h = test_image:size(3), test_image:size(2)

if(w<=h and w==size) or (h<=w and h==size) then
   test_image = test_image
end

if w < h then
   test_image = image.scale(test_image, size, h/w * size, interpolation)
else
   test_image = image.scale(test_image, w/h * size, size, interpolation)
end

local w1 = math.ceil((test_image:size(3) -size)/2)
local h1 = math.ceil((test_image:size(2) -size)/2)

test_image = image.crop(test_image, w1, h1, w1+size, h1+size)
-- print(test_image:size())

test_image:resize(1, 3, opt.cropSize, opt.cropSize)

result = model:forward(test_image):float()
-- print(result)

sigmoid = torch.sigmoid(result)
-- print(sigmoid)

maxs, indices = torch.max(sigmoid, 2)

print('The prediction for '..test_path..' is ' .. labels[indices:sum()] .. ' by ' .. maxs:sum() .. ' confidence')
