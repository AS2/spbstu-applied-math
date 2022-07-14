import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Class witch provides CNN training and giving answers
class PSFDetectorCNN(nn.Module):
    def __init__(self):
        # init parent class
        super(PyTorchTestModel2D, self).__init__()

        # Init layers like next plan:
        #
        # Input:  36, 36, 1
        # Output: 35, 35, 1

        # Stage 0: make input: 36x36 -> 35x35
        self.layer_in = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU())

        # Stage 1.1: make resnet block
        self.resnet_blocks1_cnt = 5
        self.resnet_block1 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding='same'),
                nn.ReLU())
        
        # Stage 1.2: pumped-up channels cnt
        self.up_channels = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU())
        
        # Stage 2.1: make resnet block
        self.resnet_blocks2_cnt = 5
        self.resnet_block2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
                nn.ReLU())
        
        # Stage 1.2: down channels cnt
        self.down_channels = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU())
        
        # Stage 3: make resnet block
        self.resnet_blocks3_cnt = 5
        self.resnet_block3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding='same'),
                nn.ReLU())
            
        # Stage 2: makes output
        self.layer_out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding='same'),
            nn.ReLU())
        
        # others helpful layers
        self.relu = nn.ReLU()

        return

    def forward(self, x):
        # Stage 0
        x = self.layer_in(x)

        # Stage 1
        for i in range(self.resnet_blocks1_cnt):
            x_shortcut = x
            x = self.resnet_block1(x)
            x = torch.add(x_shortcut, x)
            x = self.relu(x)

        x = self.up_channels(x)

        # Stage 3
        for i in range(self.resnet_blocks2_cnt):
            x_shortcut = x
            x = self.resnet_block2(x)
            x = torch.add(x_shortcut, x)
            x = self.relu(x)

        x = self.down_channels(x)

        # Stage 3
        for i in range(self.resnet_blocks3_cnt):
            x_shortcut = x
            x = self.resnet_block3(x)
            x = torch.add(x_shortcut, x)
            x = self.relu(x)

        # Stage 2
        x = self.layer_out(x)
        return x
