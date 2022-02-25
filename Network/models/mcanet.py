# coding=utf-8
# author: huang.rong
# date: 2021/04/23

from __future__ import print_function, division
import torch
import torch.nn as nn



class CAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, layer=4):
        super(CAM_Module, self).__init__()
        if layer == 4:
            self.conv_downsampling = nn.Conv2d(in_dim, 512, 1)
            print("layer4 no downsampling...")

        elif layer == 3:
            self.conv_downsampling = nn.Conv2d(in_dim, 512, 3, stride=2, bias=False, dilation=1, padding=1)
            print("layer3 downsampling to 1/2")
        else:
            self.conv_downsampling = nn.Sequential(
                nn.Conv2d(in_dim, 512, 3, stride=2, bias=False, dilation=1, padding=1),
                nn.Conv2d(512, 512, 3, stride=2, padding=1),
            )
            print("layer2 downsampling to 1/4")

        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x1, flag=False, MS=True):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        #print('pam modeule output size is ', out.shape)  #[8, 2048, 16, 16] [8, 1024, 32, 32] [8, 512, 36, 36]
        pam_out = self.conv_downsampling(out)             #[8,512,16,16]
        #print('final pam output shape is ', pam_out.shape)
        if flag == True:
            x1 = self.conv_downsampling(x1)   #only layer4 need to down sampling[8,2048,16,16] [8,512,16,16]
            #print('the add x1 shape is ', x1.shape)
        if MS == True:
            f_out = pam_out + x1  #[8,512,16,16] + [8,512,16,16]
            #print('multi-scale ok')
        else:
            f_out = pam_out
        #print('final 1 output size is ', f_out.shape)
        #print('final 2 output size is', x1.shape)
        return f_out, x1


class CANet(nn.Module):

    def __init__(self, layers):
        super(CANet, self).__init__()

        self.layers = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ = nn.Linear(2048 + 512*3, 2)

        self.CAM_2 = CAM_Module(512, 2)
        self.CAM_3 = CAM_Module(1024, 3)
        self.CAM_4 = CAM_Module(2048, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x has 6 patches
        # print(x.shape)
        _, c2, c3, c4 = self.layers(x)
        cam_4, down_4 = self.CAM_4(c4, c4, True, True) # [8,2048,16,16] [8,2048,16,16] -> [8,512,16,16] [8,512,16,16]
        cam_3, down_3 = self.CAM_3(c3, down_4, False, True)#[8,1024,32,32] [8,512,16,16] -> [8,512,16,16] [8,512,16,16]
        cam_2, down_2 = self.CAM_2(c2, down_3, False, True)#[8,512,64,64] [8,512,16,16] -> [8,512,16,16] [8,512,16,16]

        x = torch.cat([cam_2, cam_3, cam_4, c4], 1)  #[8, 512*3, 16, 16] [8, 2048, 16, 16]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_(x)

        return x
