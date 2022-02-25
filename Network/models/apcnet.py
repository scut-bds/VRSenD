# coding=utf-8
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F
import torchnet.meter as meter
import torch._utils
from PIL import ImageFile
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class ACM(nn.Module):
    def __init__(self, c, layer=4):
        super(ACM, self).__init__()
        self.s1 = 1
        self.s2 = 2
        self.s3 = 4
        self.s4 = 6

        if layer == 4:
            self.conv_1_1 = nn.Conv2d(c, 512, 1)
            print("4")

        elif layer == 3:
            self.conv_1_1 = nn.Conv2d(c, 512, 3, stride=2, bias=False, dilation=1, padding=1)
            print("3")
        else:
            self.conv_1_1 = nn.Sequential(
                nn.Conv2d(c, 512, 3, stride=2, bias=False, dilation=1, padding=1),
                nn.Conv2d(512, 512, 3, stride=2, padding=1),
            )
            print("2")

        self.conv_1_2_s1 = nn.Conv2d(512, self.s1 ** 2, 1)
        self.conv_1_2_s2 = nn.Conv2d(512, self.s2 ** 2, 1)
        self.conv_1_2_s3 = nn.Conv2d(512, self.s3 ** 2, 1)
        self.conv_1_2_s4 = nn.Conv2d(512, self.s4 ** 2, 1)

        self.conv_2_1_s1 = nn.Conv2d(c, 512, 1)
        self.conv_2_1_s2 = nn.Conv2d(c, 512, 1)
        self.conv_2_1_s3 = nn.Conv2d(c, 512, 1)
        self.conv_2_1_s4 = nn.Conv2d(c, 512, 1)

        self.conv_3 = nn.Conv2d(512 * 4, 512, 1)

        self.bn = nn.BatchNorm2d(512)

        # self.relu = torch.nn.LeakyReLU( inplace=True)

    def forward(self, x, x1, IF_USE=True):
        x_1_ = self.conv_1_1(x)
        x_gap = nn.functional.adaptive_avg_pool2d(x_1_, 1)
        x_1 = x_1_ * x_gap

        x_1_s1 = torch.sigmoid(self.conv_1_2_s1(x_1))
        b, c1, w, h = x_1_s1.shape
        x_1_s1 = x_1_s1.view(b, h * w, c1)  # hw x s1**2

        x_1_s2 = torch.sigmoid(self.conv_1_2_s2(x_1))
        b, c2, w, h = x_1_s2.shape
        x_1_s2 = x_1_s2.view(b, h * w, c2)  # hw x s2**2

        x_1_s3 = torch.sigmoid(self.conv_1_2_s3(x_1))
        b, c3, w, h = x_1_s3.shape
        x_1_s3 = x_1_s3.view(b, h * w, c3)  # hw x s1**2

        x_1_s4 = torch.sigmoid(self.conv_1_2_s4(x_1))
        b, c4, w, h = x_1_s4.shape
        x_1_s4 = x_1_s4.view(b, h * w, c4)  # hw x s1**2

        x_2_s1 = nn.functional.adaptive_avg_pool2d(x, self.s1)
        x_2_s1 = self.conv_2_1_s1(x_2_s1)
        b_, c_, w_1, h_1 = x_2_s1.shape
        x_2_s1 = x_2_s1.view(b, h_1 * w_1, c_)  # s1**2x512

        x_2_s2 = nn.functional.adaptive_avg_pool2d(x, self.s2)
        x_2_s2 = self.conv_2_1_s2(x_2_s2)
        b_, c_, w_2, h_2 = x_2_s2.shape
        x_2_s2 = x_2_s2.view(b, h_2 * w_2, c_)  # s2**2x512

        x_2_s3 = nn.functional.adaptive_avg_pool2d(x, self.s3)
        x_2_s3 = self.conv_2_1_s3(x_2_s3)
        b_, c_, w_3, h_3 = x_2_s3.shape
        x_2_s3 = x_2_s3.view(b, h_3 * w_3, c_)  # s3**2x512

        x_2_s4 = nn.functional.adaptive_avg_pool2d(x, self.s4)
        x_2_s4 = self.conv_2_1_s4(x_2_s4)
        b_, c_, w_4, h_4 = x_2_s4.shape
        x_2_s4 = x_2_s4.view(b, h_4 * w_4, c_)  # s3**2x512

        x_s1 = torch.bmm(x_1_s1, x_2_s1).view(b, c_, w, h)
        # x_s1 = self.bn1(x_s1)

        x_s2 = torch.bmm(x_1_s2, x_2_s2).view(b, c_, w, h)
        # x_s2 = self.bn2(x_s2)

        x_s3 = torch.bmm(x_1_s3, x_2_s3).view(b, c_, w, h)

        x_s4 = torch.bmm(x_1_s4, x_2_s4).view(b, c_, w, h)
        # x_s3 = self.bn3(x_s3)

        x_ = torch.cat([x_s1, x_s2, x_s3, x_s4], dim=1)
        x_ = self.conv_3(x_)
        # x_ = x_.view(b,c_,w,h)
        x_ = self.bn(x_)
        if IF_USE:
            print(x_.size())
            print(x1.size())
            x_ = x_ * x1
        else:
            x_ = x_ * x_1_
        return x_, x_1_


class MSCNet(nn.Module):

    def __init__(self, layers):
        super(MSCNet, self).__init__()

        self.layers = layers

        self.avgpool = GlobalAvgPool2d()

        self.fc_ = nn.Linear(2048 + 512 * 3, 3)

        self.layer_2 = nn.Conv2d(512, 512, 3, stride=2, padding=1)

        self.ACM_4 = ACM(2048, 4)
        self.ACM_2 = ACM(512, 2)
        self.ACM_3 = ACM(1024, 3)

        self.gamma1 = nn.Parameter(torch.ones(1) / 2)
        self.gamma2 = nn.Parameter(torch.ones(1) / 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, k=4, C=8):
        # x has 6 patches
        # print(x.shape)
        a, b = x.chunk(2, dim=3)
        # print(a.shape)
        x0, x1, x2 = a.chunk(3, dim=2)
        x3, x4, x5 = b.chunk(3, dim=2)
        # print(x0.shape)

        x0_c2, x0_c3, x0_c4 = self.layers(x0)
        x1_c2, x1_c3, x1_c4 = self.layers(x1)
        x2_c2, x2_c3, x2_c4 = self.layers(x2)
        x3_c2, x3_c3, x3_c4 = self.layers(x3)
        x4_c2, x4_c3, x4_c4 = self.layers(x4)
        x5_c2, x5_c3, x5_c4 = self.layers(x5)

        c2_0 = torch.cat([x0_c2, x1_c2, x2_c2], 2)
        c2_1 = torch.cat([x3_c2, x4_c2, x5_c2], 2)
        c2 = torch.cat([c2_0, c2_1], 3)

        c3_0 = torch.cat([x0_c3, x1_c3, x2_c3], 2)
        c3_1 = torch.cat([x3_c3, x4_c3, x5_c3], 2)
        c3 = torch.cat([c3_0, c3_1], 3)

        c4_0 = torch.cat([x0_c4, x1_c4, x2_c4], 2)
        c4_1 = torch.cat([x3_c4, x4_c4, x5_c4], 2)
        c4 = torch.cat([c4_0, c4_1], 3)

        x_1, x_1_0 = self.ACM_4(c4, c4, False)
        x_4, x_4_0 = self.ACM_3(c3, x_1_0)
        x_2, x_2_0 = self.ACM_2(c2, x_4_0)

        x = torch.cat([x_1, x_2, x_4, c4], 1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc_(x)

        #gamma1 = self.gamma1
        #gamma2 = self.gamma2

        return x

        #return x, gamma1, gamma2
