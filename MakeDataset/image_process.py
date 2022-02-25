# coding=utf-8
'''
author：huangrong
date: 2021/2/15
'''
import os
from PIL import Image


def CreateThumbnails(img_path, size):
    path, fname = os.path.split(img_path)
    img = Image.open(img_path)
    img.thumbnail(size)
    save_path = "./thumbnail"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img.save(os.path.join(save_path, fname))


def ResizeImages(img_path, size):
    path, fname = os.path.split(img_path)
    img = Image.open(img_path)
    img = img.resize(size, Image.ANTIALIAS)
    save_path = "./resize"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img.save(os.path.join(save_path, fname))

def RgbaToRgb(img_path):
    path, fname = os.path.split(img_path)
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        img = Image.merge("RGB", (r, g, b))
    elif img.mode != 'RGB':
        img = img.convert("RGBA")
        r, g, b, a = img.split()
        img = Image.merge("RGB", (r, g, b))
    else:
        print(" the mode of %s is %s, which means it doesn't required to process." % (fname, img.mode))
    save_path = "./channel"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img.save(os.path.join(save_path, fname))

path = r'./channel'
img_list = os.listdir(path)

for img in img_list:
    img_path = os.path.join(path, img)
    img_size = (2000, 1000)
    ResizeImages(img_path, img_size)
    print('%s resize %s success~' % (img, img_size))

