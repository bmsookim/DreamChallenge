# Digital Mammography DREAM Challenge - Torch
Repository for Digital Mammography DREAM Challenge.
Torch implementation.

## Requirements
See the [installation instruction](installation.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html)
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Install 'optnet'
```bash
luarocks install torch
luarocks install cutorch
luarocks install cunn
luarocks install image
luarocks install optnet
```

## Environments
| GPUs       | numbers | nvidia-version | dev    |
|:----------:|:-------:|:--------------:|:------:|
| GTX 980 Ti | 1       | 367.57         | local  |
| GTX 1080   | 2       | 372.20         | server |

## Directories and datasets
- /scratch : All the optimal stages and models will be saved in this directory.
- /modelState : The best optimal stages and models will be saved in this directory.
- datasets : data generation & preprocessing codes are contained.
- networks : directory that contains resnet structure file.
- preprocessedData : we assume that this directory contains the data in the format below.
```bash
preprocessedData
| train
  | class 0
  | class 1
| val
  | class 0
  | class 1

```

## How to run
You can run each dataset which could be either cifar10, cifar100, imagenet, catdog by running the script below.
```bash
./train.sh
```
