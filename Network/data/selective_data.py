# coding=utf-8
'''
author:huang.rong
date:2021/04/23
function: selective different ratio of label confidence{100%, 80%， 60%} to valid the confidence influence of accuracy
'''

import pandas as pd
import random

def spilit_pos_neg(all_data_dict, filelist):
    '''
    get the ratio of positive and negative samples
    :param all_data_dict: the dict like {filename:label}
    :param filelist: 360 deegree iamges list
    :return:None
    '''
    count = 0
    for i in range(len(filelist)):
        if all_data_dict[filelist[i]] == 0:
            count += 1
    print('negative image number is %d' % count)
    print('positive image number is %d' % (len(filelist)-count))


def del_rows(need_data_list, all_data_path, save_data_path):
    '''
    to delete not needed information in all_data_path
    :param need_data_list: information need to be preserved
    :param all_data_path: excel saved all information
    :param save_data_path: excel to save the selective information
    :return: None
    '''
    df = pd.read_excel(all_data_path, engine='openpyxl')
    del_rows = [ ]
    for index, row in  df.iterrows():
        if row['filename'] not in need_data_list:
            del_rows.append(index)
    print('there are %d information need to be deleted.' % len(del_rows))
    for i in range(len(del_rows)):
        df.drop(del_rows[i], inplace=True)
    print('deleted successfully...')
    df.to_excel(save_data_path, index=False)


def selective_confidence(all_data_path, hundred_ratio=1, eighty_ratio=1, sixty_ratio=1):
    '''
    selective different ratio of label confidence{100%, 80%， 60%} to valid the confidence influence of accuracy saved to xlsx
    :param all_data_path: excel about all information
    :param save_path:   to save new ratio of result
    :param hundred_ratio: the polling number of 5/5
    :param eighty_ratio: the polling number of 4/5
    :param sixty_ratio: the polling number of 3/5
    :return: None
    '''

    selective_list = []
    data = pd.read_excel(all_data_path, engine='openpyxl')
    filename_list = data['filename'].values   #视频ID的总条数
    label_list = data['label'].values        #标签的总条数
    confidence_list = data['confidence'].values
    label_dict = dict(zip(filename_list, label_list))
    all_data_dict = dict(zip(filename_list,confidence_list))  #所有的标注信息【ID：label】

    hundred_filename_list = [key for key, value in all_data_dict.items() if value=="100.0%"]
    spilit_pos_neg(label_dict, hundred_filename_list)
    random.shuffle(hundred_filename_list)
    selective_list.extend(hundred_filename_list[0:int(hundred_ratio * len(hundred_filename_list))])  # 786/2=383
    print(len(selective_list))

    eighty_filename_list = [key for key, value in all_data_dict.items() if value=="80.0%"]
    spilit_pos_neg(label_dict, eighty_filename_list)
    random.shuffle(eighty_filename_list)
    selective_list.extend(eighty_filename_list[0:int(eighty_ratio*len(eighty_filename_list))])   #284/2=142
    print(len(selective_list))

    sixth_filename_list = [key for key, value in all_data_dict.items() if value=="60.0%"]
    spilit_pos_neg(label_dict, sixth_filename_list)
    random.shuffle(sixth_filename_list)
    selective_list.extend(sixth_filename_list[0:int(sixty_ratio * len(sixth_filename_list))])  # 284/2=142
    print(len(selective_list))

    save_path = str(hundred_ratio) + '_' + str(eighty_ratio) + '_' + str(sixty_ratio) + '.xlsx'
    del_rows(selective_list, all_data_path, save_path)
    return


random.seed(42)

all_data_path = r'D:\Pycode\VRED\Download\1067_label_timestamps_info.xlsx'
hundred_ratio = 0
eighty_ratio = 0
sixty_ratio = 1

selective_confidence(all_data_path, hundred_ratio, eighty_ratio, sixty_ratio)