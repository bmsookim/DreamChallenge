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

local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text()
   cmd:text('Options:')
   -- General options
   cmd:option('-data',       '/preprocessedData/dreamCh',         'Path to dataset')
   cmd:option('-dataset',    'dreamChallenge',           'Options: dreamChallenge')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       2,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   -- Data options
   cmd:option('-nThreads',        16, 'number of data loading threads')
   -- Training options
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-imageSize',       224,     'Width & Height of input image')
   cmd:option('-featureMap',      0,       'final attention map size')
   cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
   -- Checkpointing options
   cmd:option('-save',            '/scratch',    'Directory in which to save checkpoints')
   cmd:option('-resume',          '/modelState',    'Resume from the latest checkpoint in this directory')
   cmd:option('-modelState',      '/modelState', 'Directory for saving model state')
   -- Optimization options
   cmd:option('-LR',              0.1,     'initial learning rate')
   cmd:option('-momentum',        0.9,     'momentum')
   cmd:option('-weightDecay',     0.0005,  'weight decay')
   -- Model options
   cmd:option('-netType',      'wide-resnet', 'Options: wide-resnet')
   cmd:option('-depth',        34,            'ResNet depth: 6n+4', 'number')
   cmd:option('-widen_factor', 2,             'Wide-Resnet width', 'number')
   cmd:option('-dropout',      0,             'Dropout rate')
   cmd:option('-nExperiment',  1,             'Number of experiment for ensemble')
   cmd:option('-shortcutType', '',            'Options: A | B | C')
   cmd:option('-retrain',      'none',        'fine-tuning, Path to model to retrain with')
   cmd:option('-optimState',   'none',        'Path to an optimState to reload from')
   -- Model options 
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'true',  'Use optnet to reduce memory usage')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.save = opt.save ..'/'..opt.netType..'-'..opt.depth..'x'.. opt.widen_factor..'/'..opt.nExperiment..'/'
   opt.featureMap = math.floor(opt.imageSize/32) -- currently, 7
   if opt.resume ~= '' then 
       opt.resume = opt.resume..'/'..opt.netType..'-'..opt.depth..'x'..opt.widen_factor..'/'..opt.nExperiment..'/'
   end

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create directory in scratch: '..opt.save..'\n')
   end

   if not paths.dirp(opt.resume) and not paths.mkdir(opt.resume) then
      cmd:error('error: unable to create directory in modelState: '..opt.resume..'\n')
   end

   if opt.dataset == 'dreamChallenge' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing DreamChallengeNet data directory')
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
      opt.imageSize = opt.imageSize == 0 and 224 or opt.imageSize
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
