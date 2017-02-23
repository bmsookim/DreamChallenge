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

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local dmisDataset = torch.class('resnet.dmisDataset', M)

function dmisDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function dmisDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      target = class,
   }
end

function dmisDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
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

function dmisDataset:size()
   return self.imageInfo.imageClass:size(1)
end

local meanstd = {
    mean = { 0.492, 0.492, 0.492 },
    std = { 0.229, 0.229, 0.229 },
}

function dmisDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
          t.ColorNormalize(meanstd),
          t.HorizontalFlip(0.5),
          t.VerticalFlip(0.5),
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.dmisDataset
