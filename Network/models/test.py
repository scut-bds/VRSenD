# coding=utf-8
# author: huang.rong
# date: 2021/04/23

from __future__ import print_function, division
import argparse
import os
import time
import copy
import sys
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchnet import meter
import torchvision
from torchvision import models, transforms, datasets

import pandas as pd
import numpy as np
import cv2

# ******************set parameters******************************************
# file_name = str(sys.argv[0].split('/')[-1].split('.')[0])
need_CAM = True
need_result = False  # whether visualize and save the failed predicted images
model_mode = 'CANet'
predict_mode = 'val'
img_size = 512  # 512
info = 'resnet512'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

data_path = r'/157Dataset/data-huang.rong/all_ODIs'
info_xlsx = {'train': r'/157Dataset/data-huang.rong/code/xlsx/111_five_folder/train_4.xlsx',
             'test': r'/157Dataset/data-huang.rong/code/xlsx/111_five_folder/test_4.xlsx',
             'val': r'/157Dataset/data-huang.rong/code/xlsx/val_info.xlsx'}

main_path = r'/157Dataset/data-huang.rong/code/VRED/Result'
log_save_path = main_path + r'/log/' + info + '.txt'
model_save_path = main_path + r'/model/' + info + '.pth'
cam_save_path = main_path + r'/cam/' + info + '_' + predict_mode
if predict_mode == 'val':
    log_save_path = main_path + r'/val/' + info + '.txt'
    model_path = r'/157Dataset/data-huang.rong/code/VRED/Result/model/' + r'only_cam.pth'

# ***********************************************************************


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
        transforms.Resize(550),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}


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

class CAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
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

        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()

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

        return out

class ResNet101(nn.Module):

    def __init__(self, layers):
        super(ResNet101, self).__init__()
        self.layers = layers
        self.conv = nn.Conv2d(2048, 512, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.layers(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CANet(nn.Module):

    def __init__(self, layers):
        super(CANet, self).__init__()

        self.layers = layers
        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU())
        self.cam = CAM_Module(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))
        self.fc_ = nn.Linear(512, 2)

    def forward(self, x):
        # print(x.shape)
        _, _, _, c4 = self.layers(x)  #[8,2048,16,16]
        x = self.conv(c4)             #[8,512,16,16]
        x = self.cam(x)               #[8,512,16,16]
        #x = torch.cat([x, c4], 1)    #[8, 512*3, 16, 16] [8, 2048, 16, 16]
        x = self.avgpool(x)
        #x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc_(x)
        return x

class PANet(nn.Module):

    def __init__(self, layers):
        super(PANet, self).__init__()

        self.layers = layers
        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU())
        self.pam = PAM_Module(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))
        self.fc_ = nn.Linear(512, 2)

    def forward(self, x):
        # print(x.shape)
        _, _, _, c4 = self.layers(x)  #[8,2048,16,16]
        x = self.conv(c4)             #[8,512,16,16]
        x = self.pam(x)               #[8,512,16,16]
        #x = torch.cat([x, c4], 1)  #[8, 512*3, 16, 16] [8, 2048, 16, 16]
        x = self.avgpool(x)
        #x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc_(x)
        return x

class DANet(nn.Module):

    def __init__(self, layers):
        super(DANet, self).__init__()

        self.layers = layers
        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU())
        self.pam = PAM_Module(512)
        self.cam = CAM_Module(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))
        self.fc_ = nn.Linear(512, 2)

    def forward(self, x):
        # print(x.shape)
        _, _, _, c4 = self.layers(x)  #[8,2048,16,16]
        x = self.conv(c4)             #[8,512,16,16]
        pam = self.pam(x)             #[8,512,16,16]
        cam = self.cam(x)
        feat_sum = pam + cam
        #x = torch.cat([x, c4], 1)  #[8, 512*3, 16, 16] [8, 2048, 16, 16]
        x = self.avgpool(feat_sum)
        #x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc_(x)
        return x

# get the model accuracy, loss and confusion matrix and need_result decide whether visualize and save the failed samples
def validate_model(model, criterion, predict_mode='val', need_result=False):
    cm = meter.ConfusionMeter(class_num)
    running_loss = 0.0
    running_corrects = 0
    index = 0
    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for inputs, labels in dataloaders[predict_mode]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        cm.add(outputs.detach(), labels.detach())
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # Visualize and save failed samples
        if need_result:
            for i in range(len(inputs)):
                if preds[i] != labels.data[i]:
                    img_array = inputs.data[i]
                    plt.figure()
                    plt.axis('off')
                    imshow(img_array)
                    img_name = './' + str(labels.data[i].item()) + '_pred_' + str(preds[i].item()) + '_' + str(
                        index) + '.jpg'
                    plt.savefig(img_name)
                    index += 1
    # compute key metrics
    loss = running_loss / dataset_sizes[predict_mode]
    acc = running_corrects.double() / dataset_sizes[predict_mode]
    log = open(log_save_path, 'a')
    output = ('loss:{loss}, acc:{acc}, cm:{cm}\n').format(
        loss=loss, acc=acc, cm=str(cm.value()))
    log.write(output)
    log.flush()
    log.close()
    print('loss:{loss}, acc:{acc}, cm:{cm}'.format(
        loss=loss, acc=acc, cm=str(cm.value())))


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
                    # print(outputs.shape)
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


# show tensor images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.show()      # decide on whether your compiler is interactive
    plt.pause(0.001)  # pause a bit so that plots are updated


dataset_train = Emo_Dataset(data_path, info_xlsx['train'], data_transforms['train'])
dataset_test = Emo_Dataset(data_path, info_xlsx['test'], data_transforms['test'])
dataset_val = Emo_Dataset(data_path, info_xlsx['val'], data_transforms['test'])
image_datasets = {'train': dataset_train, 'test': dataset_test, 'val': dataset_val}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'test', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'val']}
print(dataset_sizes)
class_names = {0: 'negative', 1: 'positive'}
class_num = len(class_names)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_layer = ResNet(Bottleneck, [3, 4, 23, 3])
model_dict = model_layer.state_dict()
pretrained_dict = models.resnet101(pretrained=True).state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_layer.load_state_dict(model_dict)


if model_mode == 'ResNet101':
    model = ResNet101(model_layer)
elif model_mode == 'CANet':
    model = CANet(model_layer)
elif model_mode == 'PANet':
    model = PANet(model_layer)
else:
    model = DANet(model_layer)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# train and save
if predict_mode == 'test':
    model_conv = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=50)
    torch.save(model_conv.state_dict(), model_save_path)

# evaluate
if predict_mode == 'val':
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    validate_model(model, criterion, predict_mode, need_result)

plt.close('all')


# get CAM
# *****************************CAM*****************************************************

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape  # 1,2048,7,7
    output_cam = []
    for idx in class_idx:  # 只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h * w))
        cam = weight_softmax[idx].dot(
            feature_conv.reshape((nc, h * w)))  # (2048, ) * (2048, 7*7) -> (7*7, ) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        cam_img = np.uint8(255 * cam_img)  # Format as CV_8UC1 (as applyColorMap required)
        output_cam.append(cam_img)
    return output_cam

if need_CAM:

    if not os.path.exists(cam_save_path):
        os.mkdir(cam_save_path)

    all_data = pd.read_excel(info_xlsx[predict_mode], engine='openpyxl')
    image_list = all_data['filename'].values.tolist()
    label_list = all_data['label'].values.tolist()
    all_data_dict = dict(zip(image_list, label_list))
    images_path_list = [os.path.join(data_path, img) for img in image_list]

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model_features = nn.Sequential(*list(model.children())[:-3])  #avgpool dropout fc
    #layer_info = [child for child in model.children()]
    #print(layer_info[-4:])

    # get weight matrix of full connection
    fc_weights = model.state_dict()['fc.weight'].cpu().numpy()  # [2,2048]

    #model.to(device)
    #model_features.to(device)

    # hook the feature extractor
    features_blobs = []
    finalconv_name = 'conv'  # this is the last conv layer of the network


    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # set the model to eval mode, it's necessary
    model.eval()
    model_features.eval()

    for i, img_path in enumerate(images_path_list):
        print('*' * 10)
        # img_path = './bee.jpg'  # test single image
        _, img_name = os.path.split(img_path)
        img_label = class_names[all_data_dict[img_name]]
        features_blobs = []
        img = Image.open(img_path).convert('RGB')
        img_tensor = data_transforms['test'](img).unsqueeze(0)  # [1,3,224,224]
        inputs = img_tensor.to(device)
        features = model_features(inputs).detach().cpu().numpy()  # [1,2048,7,7]
        print('feature map layer shape is ', features.shape)
        logit = model(inputs)  # [1,2] -> [ 3.3207, -2.9495]
        h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  # tensor([0.9981, 0.0019])
        probs, idx = h_x.sort(0, True)  # sorted in descending order

        probs = probs.cpu().numpy()  # if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
        idx = idx.cpu().numpy()  # [1, 0]
        for id in range(2):
            # 0.559 -> neg, 0.441 -> pos
            print('{:.3f} -> {}'.format(probs[id], class_names[idx[id]]))

        CAMs = returnCAM(features, fc_weights, [idx[0]])  # output the most probability class activate map
        print(img_name + ' output for the top1 prediction: %s' % class_names[idx[0]])
        img = cv2.imread(img_path)
        height, width, _ = img.shape  # get input image size
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)),
                                    cv2.COLORMAP_JET)  # CAM resize match input image size
        heatmap[np.where(CAMs[0] <= 100)] = 0
        result = heatmap * 0.3 + img * 0.5  # ratio

        text = '%s %.2f%%' % (class_names[idx[0]], probs[0] * 100)
        cv2.putText(result, text, (210, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                    color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

        image_name_ = img_name.split(".")[-2]
        cv2.imwrite(cam_save_path + r'/' + image_name_ + '_' + 'pred_' + class_names[idx[0]] + '.jpg', result)
# **********************************************************************************
