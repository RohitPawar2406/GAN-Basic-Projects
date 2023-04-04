import os
import sys

import numpy as np
import utilsFunction

import torch 
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision.utils as vutils


# Paramters
device = "cuda" if torch.cuda.is_available() else "cpu"
dataPath = ''
outPath = "output/"

logFile = os.path.join(outPath,"log.txt")
batchSize = 32
imgChannels = 1
zDimension = 100
gDimension = 64
xDimension = 64
discriminatorDim = 64
epochNumber = 3
realLabel = 1
fakeLabel = 0
lr = 2e-4
seed = 1

utilsFunction.clear_folder(outPath)