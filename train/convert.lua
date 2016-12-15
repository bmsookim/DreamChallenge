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
local checkpoint, optimState = checkpoints.scratch(opt)

tmp = opt.resume
opt.resume = opt.save

-- Create model
local model, criterion = models.setup(opt, checkpoint)

opt.resume = tmp

large_file = paths.concat(opt.save, checkpoint.modelFile)
small_file = paths.concat(opt.save, checkpoint.modelFile)

os.remove(large_file)

torch.save(small_file, model:clearState())

print("Finished Converting!!")
