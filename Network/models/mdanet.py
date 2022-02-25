# coding=utf-8
# author: huang.rong
# date: 2021/04/23

from __future__ import print_function, division
import argparse
import os
import time
import copy

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchnet import meter
import torchvision
from torchvision import models, transforms, datasets

import pandas as pd
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return  c1, c2, c3, c4

#**********************************************************************

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, layer=4):
        super(PAM_Module, self).__init__()
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

        self.chanel_in = in_dim  #512 1024 2048
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        #print('pam modeule output size is ', out.shape)  #[8, 2048, 16, 16] [8, 1024, 32, 32] [8, 512, 36, 36]
        pam_out = self.conv_downsampling(out)             #[8,512,16,16]
        #print('final pam output shape is ', pam_out.shape)
        return pam_out

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

    def forward(self, x):

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
        return pam_out

class DANet(nn.Module):

    def __init__(self, layers):
        super(DANet, self).__init__()

        self.layers = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ = nn.Linear(2048 + 512*3, 2)

        self.PAM_2 = PAM_Module(512, 2)
        self.PAM_3 = PAM_Module(1024, 3)
        self.PAM_4 = PAM_Module(2048, 4)

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
        # print(x.shape)
        _, c2, c3, c4 = self.layers(x)
        pam_4 = self.PAM_4(c4)  # [8,2048,16,16] -> [8,512,16,16]
        pam_3 = self.PAM_3(c3)  # [8,1024,32,32] -> [8,512,16,16]
        pam_2 = self.PAM_2(c2)  # [8,512,64,64]  -> [8,512,16,16]

        cam_4 = self.CAM_4(c4)  # [8,2048,16,16] -> [8,512,16,16]
        cam_3 = self.CAM_3(c3)  # [8,1024,32,32] -> [8,512,16,16]
        cam_2 = self.CAM_2(c2)  # [8,512,64,64]  -> [8,512,16,16]

        dam_4 = pam_4 + cam_4
        dam_3 = pam_3 + cam_3
        dam_2 = pam_2 + cam_2

        x = torch.cat([dam_2, dam_3, dam_4, c4], 1)  #[8, 512*3, 16, 16] [8, 2048, 16, 16]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_(x)

        return x



# ******************set parameters******************************************
# file_name = str(sys.argv[0].split('/')[-1].split('.')[0])
# model_name = 'resnet101'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# data_path = r'/157Dataset/data-huang.rong/all_ODIs'
#
# main_path = r'/157Dataset/data-huang.rong/code/111_five_folder'
# log_save_path = main_path + r'/log_' + file_name + '_4.txt'
# model_save_path = main_path + r'/model_' + file_name + '_4.pth'
#
# test_xlsx = main_path + r'/train_4.xlsx'
# train_xlsx = main_path + r'/test_4.xlsx'
# ***********************************************************************


img_size = 512  # 512
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop((1000,2000)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Resize((1100,2100)),
        # transforms.CenterCrop((1000,2000)),
        transforms.Resize(250),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

# **************************************************************
data_path = r'D:/JupyterNotebook/jupyter_py/pytorch/data/hymenoptera_data'
log_save_path = r'./train_log.txt'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
class_num = len(class_names)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ***************************************************************


class Emo_Dataset():
    def __init__(self, data_path, xlsx, transform):
        all_data = pd.read_excel(xlsx, engine='openpyxl')
        all_filenames = all_data['filename'].values.tolist()

        self.labels_list = all_data['label'].values.tolist()
        self.images_list = [os.path.join(data_path, filename) for filename in all_filenames]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_list[index]
        label = self.labels_list[index]
        pil_img = Image.open(img_path)
        if pil_img.mode == "L":
            pil_img = pil_img.convert("RGB")

        pil_img = self.transform(pil_img)
        return pil_img, label

    def __len__(self):
        return len(self.images_list)


# dataset_train = Emo_Dataset(data_path, train_xlsx, data_transforms['train'])
# dataset_test = Emo_Dataset(data_path, test_xlsx, data_transforms['test'])
# image_datasets = {'train':dataset_train, 'test':dataset_test}
# dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
#                                              shuffle=True, num_workers=0)
#                                           for x in ['train', 'test']}
#
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
# print(dataset_sizes)
# class_names = {0:'negative', 1:'positive'}
# class_num = len(class_names)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    train_loss = meter.AverageValueMeter()
    train_cm = meter.ConfusionMeter(class_num)
    val_cm = meter.ConfusionMeter(class_num)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    log = open(log_save_path, 'a')
    for epoch in range(num_epochs):
        train_loss.reset()
        train_cm.reset()
        val_cm.reset()
        # log.write('{} model begining to train and test...'.format(model_name))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        log.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        log.write('----------------------------------------------------------------\n')
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_loss.add(loss.item())
                        train_cm.add(outputs.detach(), labels.detach())
                    else:
                        val_cm.add(outputs.detach(), labels.detach())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'test':
                cm_value = val_cm.value()
                cm_acc = (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
                recall = cm_value[1][1] / (cm_value[1][0] + cm_value[1][1])
                precise = cm_value[1][1] / (cm_value[0][1] + cm_value[1][1])
                f1 = 2 * recall * precise / (recall + precise)

                # vis.plot('confusion matrix val accuracy', cm_acc)
                # vis.log('epoch:{epoch}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}'.format(
                # epoch=epoch, loss=train_loss.value()[0], train_cm=str(train_cm.value()), val_cm=str(val_cm.value())))  #train_loss.value()[0]
                print(
                    'f1: {:.4f}\n''train_cm:{train_cm}\n''val_cm:{val_cm}\n'.format(f1, train_cm=str(train_cm.value()),
                                                                                    val_cm=str(val_cm.value())))
                output = ('f1: {:.4f}\n''train_cm:{train_cm}\n''val_cm:{val_cm}\n').format(f1, train_cm=str(
                    train_cm.value()), val_cm=str(val_cm.value()))
                log.write(output)
            print('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
            output = ('{} Loss: {:.4f} Acc: {:.4f}\n').format(phase, epoch_loss, epoch_acc)
            log.write(output)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
        print('Best epoch {} and val Acc: {:4f}\n'.format(best_epoch, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 60 // 60,
                                                                  time_elapsed // 60 % 60, time_elapsed % 60))
    print('Best epoch {} and val Acc: {:4f}\n'.format(best_epoch, best_acc))

    output = ('Training complete in {:.0f}h {:.0f}m {:.0f}s\n').format(time_elapsed // 60 // 60,
                                                                       time_elapsed // 60 % 60, time_elapsed % 60)
    log.write(output)
    output = ('Best epoch {} and val Acc: {:4f}\n').format(best_epoch, best_acc)
    log.write(output)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model







model_layer = ResNet(Bottleneck, [3, 4, 23, 3])

model_dict = model_layer.state_dict()
#print(model_dict.keys())
pretrained_dict = models.resnet101(pretrained=True).state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_layer.load_state_dict(model_dict)
# for child in model_layer.children():
#     print(child)
# print("layers is ok")
model = DANet(model_layer)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# train and evaluate

# model_conv = train_model(model, criterion, optimizer_conv, exp_lr_scheduler,
#                          num_epochs=10)
#torch.save(model_conv.state_dict(), model_save_path)
