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

local datasets = require 'datasets/init'
local opts = require 'opts'
local _opt = opts.parse(arg)

datasets.generate(_opt)
