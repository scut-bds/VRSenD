# coding=utf-8
# author:huangr.rong
# date:2021/04/13
# base model

from __future__ import print_function, division
from Network.models.dialted_resnet import ResNet, Bottleneck
from Network.models.mpanet import PANet
from Network.models.mcanet import CANet
import argparse
import os
import sys
import time
import copy
import torchvision.models as models
from ptflops import get_model_complexity_info
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchnet import  meter
import torchvision
from torchvision import  models, transforms, datasets

import pandas as pd
import numpy as np
from thop import profile  #不太行
#******************set parameters******************************************
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
#***********************************************************************


img_size = 224  #512
data_transforms = {
'train':transforms.Compose([
    #transforms.RandomResizedCrop((1000,2000)),
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]),
'test':transforms.Compose([
    #transforms.Resize((1100,2100)),
    #transforms.CenterCrop((1000,2000)),
    transforms.Resize(224),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])}

#**************************************************************
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
#***************************************************************


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

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)     # pause a bit so that plots are updated

# Get a batch of training data
# inputs, classes = next(iter(dataloaders['val']))
# out = torchvision.utils.make_grid(inputs)   # Make a grid from batch
# imshow(out, title=[class_names[x] for x in classes.numpy()])


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
        #log.write('{} model begining to train and test...'.format(model_name))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        log.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        log.write('----------------------------------------------------------------\n')
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                print('f1: {:.4f}\n''train_cm:{train_cm}\n''val_cm:{val_cm}\n'.format(f1, train_cm=str(train_cm.value()), val_cm=str(val_cm.value()))) 
                output = ('f1: {:.4f}\n''train_cm:{train_cm}\n''val_cm:{val_cm}\n').format(f1, train_cm=str(train_cm.value()), val_cm=str(val_cm.value())) 
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
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s\n'.format(time_elapsed//60//60,
        time_elapsed//60%60, time_elapsed % 60))
    print('Best epoch {} and val Acc: {:4f}\n'.format(best_epoch, best_acc))
    
    output = ('Training complete in {:.0f}h {:.0f}m {:.0f}s\n').format(time_elapsed//60//60,
              time_elapsed//60%60, time_elapsed % 60)
    log.write(output)
    output = ('Best epoch {} and val Acc: {:4f}\n').format(best_epoch, best_acc)
    log.write(output)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.numpy()
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# model_conv = models.resnet101(pretrained=True)
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, class_num)

model_layer = ResNet(Bottleneck, [3, 4, 23, 3])
model_dict  = model_layer.state_dict()
pretrained_dict = models.resnet101(pretrained=True).state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict }
model_dict.update(pretrained_dict)
model_layer.load_state_dict(model_dict)
# for child in model_layer.children():
#     print(child)
# print("layers is ok")
model = CANet(model_layer)
flops, params = get_model_complexity_info(model, (3,224,224), as_strings=True, print_per_layer_stat=True)
print("Flops: {}".format(flops))
print("Params: " + params)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


#train and evaluate

model_conv = train_model(model, criterion, optimizer_conv, exp_lr_scheduler,
                      num_epochs=10)
#torch.save(model_conv.state_dict(), model_save_path)

#visualize_model(model_conv, num_images=6)