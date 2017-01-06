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

local checkpoint = {}
local before_was_best = false

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState
end

function checkpoint.best(opt)
   if opt.resume == 'none' then
      return nil
   end

   local bestPath = paths.concat(opt.resume, 'best.t7')
   if not paths.filep(bestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. bestPath)
   local best = torch.load(bestPath)

   return best
end

function checkpoint.scratch(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.save, 'best.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Converting checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   os.remove(paths.concat(opt.save, latest.optimFile))

   return latest
end

function checkpoint.save(epoch, model, isBestModel, opt)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network
   model = deepCopy(model):float():clearState()

   local modelFile = 'model_' .. epoch .. '.t7'

   if isBestModel then
      if(epoch ~= 1) then 
         for i=1, (epoch-1) do
            bef_model = 'model_' .. (i) .. '.t7'
            os.remove(paths.concat(opt.resume, bef_model))
            before_was_best = true
         end
      end
      print("is best model!")
      -- model = deepCopy(model):float():clearState()
      torch.save(paths.concat(opt.resume, modelFile), model)
      torch.save(paths.concat(opt.resume, 'best.t7'), {
         epoch = epoch,
         modelFile = modelFile,
         optimFile = optimFile,
      })
   else
      before_was_best = false
   end
end

return checkpoint
