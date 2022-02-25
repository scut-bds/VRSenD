# coding=utf-8
# author：huang.rong
# date：21/05/17
# information: get the class activate map(CAM) result

from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import models, transforms

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

model_mode = 'PDANet'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#set and load relevent information
predict_mode = 'val'
img_size = 512
dataset_path = r'/157Dataset/data-huang.rong/all_ODIs'
model_path = r'/157Dataset/data-huang.rong/code/111_five_folder/model_111_4.pth'

info_path = {'test': r'/157Dataset/data-huang.rong/code/111_five_folder/test_4.xlsx',
             'val': r'/157Dataset/data-huang.rong/code/xlsx/val_info.xlsx'}
cam_save_path = { 'val': r'/157Dataset/data-huang.rong/code/val_cam',
                  'test': r'/157Dataset/data-huang.rong/code/test_cam'}
if not os.path.exists(cam_save_path[predict_mode]):
    os.mkdir(cam_save_path[predict_mode])

# get filename and label from excel file
all_data = pd.read_excel(info_path[predict_mode], engine='openpyxl')
image_list = all_data['filename'].values.tolist()
label_list = all_data['label'].values.tolist()
all_data_dict = dict(zip(image_list, label_list))
images_path_list = [os.path.join(dataset_path, img) for img in image_list]


class_ = {0: 'negative', 1: 'positive'}
data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



def returnCAM(feature_conv, weight_softmax, class_idx):
    '''
    get the class activate map
    :param feature_conv:  the feature map by final convolution layer  [1,2048,7,7](resnet101)
    :param weight_softmax: the weight matrix in full connection
    :param class_idx: the predict class sorted in descending order
    :return: class activate map
    '''
    bz, nc, h, w = feature_conv.shape     # 1,2048,7,7
    output_cam = []
    for idx in class_idx:
        feature_conv = feature_conv.reshape((nc, h * w))
        cam = weight_softmax[idx].dot(feature_conv)            # (2048, ) * (2048, 7*7) -> (7*7, )
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        cam_img = np.uint8(255 * cam_img)                      # Format as CV_8UC1 (as applyColorMap required)
        output_cam.append(cam_img)
    return output_cam


# load model and pretrained parameters
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
elif model_mode == 'DANet':
    model = DANet(model_layer)
elif model_mode == 'PDANet':
    model = PDANet(model_layer)
else:
    print('please input right model type.')

model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

# get weight matrix of full connection
fc_weights = model.state_dict()['fc.weight'].cpu().numpy()  # [2,2048]

# set the model to eval mode, it's necessary
model.eval()

for i, img_path in enumerate(images_path_list):
    print('*' * 10)
    # img_path = './bee.jpg'  # test single image
    _, img_name = os.path.split(img_path)
    img_label = class_[all_data_dict[img_name]]
    features_blobs = []
    img = Image.open(img_path).convert('RGB')
    img_tensor = data_transforms['test'](img).unsqueeze(0)  # [1,3,224,224]
    inputs = img_tensor.to(device)

    logit = model(inputs)                                    # [1,2] -> [ 3.3207, -2.9495]
    h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  # tensor([0.9981, 0.0019])
    probs, idx = h_x.sort(0, True)                                  # sorted in descending order

    probs = probs.cpu().numpy()     # if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
    idx = idx.cpu().numpy()         # [1, 0]
    for id in range(2):
        # 0.559 -> neg, 0.441 -> pos
        print('{:.3f} -> {}'.format(probs[id], class_[idx[id]]))

    features = model.finalconv.cpu().numpy()  # [1,2048,7,7]
    print('final feature map layer shape is ', features.shape)

    CAMs = returnCAM(features, fc_weights, [idx[0]])  # output the most probability class activate map
    print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])
    img = cv2.imread(img_path)
    height, width, _ = img.shape                    # get input image size
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)),
                                cv2.COLORMAP_JET)   # CAM resize match input image size
    heatmap[np.where(CAMs[0] <= 100)] = 0
    result = heatmap * 0.3 + img * 0.5              # ratio

    text = '%s %.2f%%' % (class_[idx[0]], probs[0] * 100)
    cv2.putText(result, text, (210, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

    image_name_ = img_name.split(".")[-2]
    cv2.imwrite(cam_save_path[predict_mode] + r'/' + image_name_ + '_' + 'pred_' + class_[idx[0]] + '.jpg', result)