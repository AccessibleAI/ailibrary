"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

script.py
==============================================================================
"""
import torch

from torchvision.models import alexnet


model = alexnet(pretrained=True)
model.train()