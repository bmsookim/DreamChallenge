require 'torch'
require 'paths'
require 'optim'
require 'nn'

local datasets = require 'datasets/init'
local opts = require 'opts'
local _opt = opts.parse(arg)

datasets.generate(_opt)
