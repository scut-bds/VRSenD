# coding=utf-8
'''
author: huang.rong
date: 2020.12.18
'''

import shutil
import pandas as pd
import os


def rename_videos(path):
    '''
    rename 'dataset' or 'bilibili' directory
    :param path: 'dataset' videos path or 'bilibili' videos path
    :return: None
    '''
    _, fname = os.path.split(path)
    src_name_list = []
    dst_name_list = []
    if fname == 'dataset':
        dataset_list = ['VQA-Database', 'VR-HM48', 'VQA-ODV', 'Eye-Tracking']  #according to the order of the dataset
        prefix_list = ['1_31', '1_48', '1_60', '1_208']
        total_num = 0
        for i in range(len(dataset_list)):
            dataset_path = os.path.join(path, dataset_list[i])
            file_list = os.listdir(dataset_path)
            for j in range(len(file_list)):
                if file_list[j].endwith('.mp4'):
                    src_path = os.path.join(dataset_path, file_list[j])
                    dst_name = prefix_list[i] + '_' + str(total_num).zfill(3) + '.mp4'
                    dst_path = os.path.join(dataset_path, dst_name)
                    os.rename(src_path, dst_path)
                    src_name_list.append(file_list[j])
                    dst_name_list.append(dst_name)
                    total_num += 1
    elif fname == 'bilibili':
        i = 0
        for root, dirs, files in os.walk(path, topdown=False):
            for fname in files:
                if fname.endswith('.mp4'):
                    src_path = os.path.join(root, fname)
                    dst_name = '0_' + str(i).zfill(3) + '.mp4'
                    src_name_list.append(fname)
                    dst_name_list.append(dst_name)
                    os.rename(src_path, os.path.join(root, dst_name))
                    i += 1
            fpath, fname = os.path.split(root)
            if fname.startswith('tmp'):       #delete temporary directory cached at download time
                shutil.rmtree(root)
                print(root + 'is deleted.')
    else:
        print('please input directory named "dataset" or "bilibili"')

    if len(src_name_list) != 0 and len(dst_name_list) != 0:
        df = pd.DataFrame([src_name_list, dst_name_list]).T
        xlsx_path = os.path.join(path, 'src_to_dst.xlsx')
        df.to_excel(xlsx_path, index=False)


def rename_images(path):
    '''
    rename dataset images
    :param path: the 360 degree images directory
    :return: None
    '''
    img_list = os.listdir(path)
    _, dataset_name = os.path.split(path)
    img_num = len(img_list)
    print('Total have %d items' % (img_num))

    i = 0
    srcarr = []
    dstarr = []

    for img in img_list:
        src = os.path.join(os.path.abspath(path), img)  # 原图的地址
        dst_name = '1_' + dataset_name + '_' + str(i).zfill(3) + '.jpg'
        dst = os.path.join(os.path.abspath(path), dst_name)
        if img.endswith('.jpg'):
            srcarr.append(img)
            dstarr.append('1_' + str(i).zfill(3))
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))
            i += 1

    df = pd.DataFrame([srcarr, dstarr]).T
    xlsx_path = os.path.join(path, 'rename.xlsx')
    df.to_excel(xlsx_path, index=False)


path = './bilibili'
rename_videos(path)




