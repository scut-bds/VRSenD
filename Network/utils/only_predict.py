# coding=utf-8
# author:huangrong
# data:2021/5/09
# information : use val or test dataset to validate model, and need_result=True can save the failed predicted images

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchnet import meter

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

predict_mode = 'val'
need_result = False      # whether visualize and save the failed predicted images

dataset_path = r'/157Dataset/data-huang.rong/all_ODIs'
model_path = r'/157Dataset/data-huang.rong/code/111_five_folder/model_111_4.pth'

info_path = {'test': r'/157Dataset/data-huang.rong/code/111_five_folder/test_4.xlsx',
             'val': r'/157Dataset/data-huang.rong/code/xlsx/val_info.xlsx'}
log_path = { 'val': r'/157Dataset/data-huang.rong/code/val_output.txt',
             'test': r'/157Dataset/data-huang.rong/code/test_output.txt'}

img_size = 512

# Emo_Dataset load our own dataset
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

# transform these input images
data_transforms = transforms.Compose([
        transforms.Resize(550),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])


# get predict dataset information
image_dataset = Emo_Dataset(dataset_path, info_path[predict_mode], data_transforms)
data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=False, num_workers=0)
dataset_size = len(image_dataset)
class_names = {0:'negative', 1:'positive'}
class_num = len(class_names)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


# Make a grid from batch
inputs, classes = next(iter(data_loader))  # Get a batch of test/val data
out = torchvision.utils.make_grid(inputs)             # Make a grid from batch
imshow(out, title=[class_names[x.item()] for x in classes])


# write some useful message to txt
def write_massage(txt_path, massage):
    with open(txt_path, "a") as f:
        f.write(massage)
        f.write('\n')
        f.close()


# give the predict result about num_images random images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):

                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.data[j])

                if images_so_far == num_images:
                    plt.pause(15)
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# get the model accuracy, loss and confusion matrix and need_result decide whether visualize and save the failed samples
def validate_model(model, criterion, need_result=False):
    cm = meter.ConfusionMeter(class_num)
    running_loss = 0.0
    running_corrects = 0
    index = 0
    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for inputs, labels in data_loader:
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
    loss = running_loss / dataset_size
    acc = running_corrects.double() / dataset_size
    log = open(log_path[predict_mode], 'a')
    output = ('loss:{loss}, acc:{acc}, cm:{cm}\n').format(
        loss=loss, acc=acc, cm=str(cm.value()))
    log.write(output)
    log.flush()
    log.close()
    print('loss:{loss}, acc:{acc}, cm:{cm}'.format(
        loss=loss, acc=acc, cm=str(cm.value())))


# load pretrained model every layer parameters
model = models.resnet101(pretrained=False)

num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # [2048,2]
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
criterion = nn.CrossEntropyLoss()
model.to(device)


validate_model(model, criterion, need_result)
plt.close('all')
