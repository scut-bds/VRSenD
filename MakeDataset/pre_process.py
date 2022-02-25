# coding=utf-8
'''
data:2021/4/23
author:huang.rong
information: some scattered function could be used
'''

import  shutil
import os
import pandas as pd
from openpyxl import  load_workbook
from sklearn.model_selection import train_test_split

# img_path = r'J:\Vr_dataset\images\Tangwei45'
# img_name_list = os.listdir(img_path)
# xlsx_save_path = r'D:\Pycode\VRED\0_Download\val_info.xlsx'
# df = pd.DataFrame(img_name_list)
# df.to_excel(xlsx_save_path, index=None, engine='openpyxl')

def make_dataset(dataset_path, bilibili_path, xlsx_path, save_path):
    '''
    make the VRED dataset
    :param dataset_path: the path save dataset 360 degree images
    :param bilibili_path: the path save bilibili 360 degree images
    :param xlsx_path: the final label information format as {filename:label}
    :param save_path: the path to save VRED dataset
    :return: None
    '''

    data = pd.read_excel(xlsx_path, engine='openpyxl')
    filename_list = data['filename'].values
    label_list = data['label'].values
    emotion_dict = dict(zip(filename_list, label_list))  #make dict{filename:label}

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(len(emotion_dict)):
        if filename_list[i].startswith('1'):
            src_file = os.path.join(dataset_path, filename_list[i])
        else:
            src_file = os.path.join(bilibili_path, filename_list[i])
        shutil.copy(src_file, save_path)

# dataset_path = r'D:/VERD/source/dataset/'
# bilibili_path = r'D:/VERD/source/bilibili/'
# xlsx_path = r'./filename2label.xlsx'
# save_path = r'./dataset'

def combine_sheet(part_xlsx_path, total_xlsx_path):
    '''
    we put partial information in single sheet and need to combine them
    :param part_xlsx_path: the xlsx to save partial information
    :param total_xlsx_path: the xlsx to save all information(the file needed to creat before)
    :return: None
    '''
    excel = pd.ExcelFile(part_xlsx_path, engine='openpyxl')
    sheet_name = excel.sheet_names
    sheet_num = len(sheet_name)
    df_rows = 0

    for i in range(sheet_num):
        data = pd.read_excel(part_xlsx_path, sheet_name=sheet_name[i], engine='openpyxl')
        writer = pd.ExcelWriter(total_xlsx_path, engine='openpyxl')
        book = load_workbook(total_xlsx_path)
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        data.to_excel(writer, sheet_name='result', startrow=df_rows, index=False)
        df_rows = df_rows + 1 + data.shape[0]
        writer.save()
        print('the sheet %s written successfully~' % sheet_name[i])

# for example
# part_xlsx_path = r'./part_information.xlsx'
# total_xlsx_path = r'./all_information1.xlsx'
# combine_sheet(part_xlsx_path, total_xlsx_path)


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
            print('the index %s here needed to be deleted...' % index)
            del_rows.append(index)

    print('there are %d information need to be deleted.' % len(del_rows))
    for i in range(len(del_rows)):
        df.drop(del_rows[i], inplace=True)
        print('deleted successfully...')
    df.to_excel(save_data_path, index=False)


def split_dir_train_val(dataset_path, information_path, test_size=0.2):
    '''
    create dataset dir format as: train {positive, negative}; val{positive, negative}
    :param dataset_path: save all labeled 360 degree images
    :param information_path: save all useful information excel
    :param test_size: the proportion of test samples
    :return: None
    '''
    all_data = pd.read_excel(information_path, engine='openpyxl')
    filename_list = all_data['filename'].values
    label_list = all_data['label'].values
    data_dict = dict(zip(filename_list, label_list))

    #create relative directory
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    pos_train_path = os.path.join(train_path, 'positive')
    neg_train_path = os.path.join(train_path, 'negative')
    pos_val_path = os.path.join(val_path, 'positive')
    neg_val_path = os.path.join(val_path, 'negative')
    if not os.path.exists(pos_train_path):
        os.mkdir(pos_train_path)
    if not os.path.exists(neg_train_path):
        os.mkdir(neg_train_path)
    if not os.path.exists(pos_val_path):
        os.mkdir(pos_val_path)
    if not os.path.exists(neg_val_path):
        os.mkdir(neg_val_path)

    #copy images to destine directory
    train_list, test_list = train_test_split(filename_list, test_size=test_size, random_state=42)
    for i in range(len(train_list)):
        if data_dict[train_list[i]] == 'negative':
            shutil.copyfile(os.path.join(dataset_path, train_list[i]), os.path.join(neg_train_path, train_list[i]))
        else:
            shutil.copyfile(os.path.join(dataset_path, train_list[i]), os.path.join(pos_train_path, train_list[i]))
    print('successfully create the train dataset ...')
    for i in range(len(test_list)):
        if data_dict[test_list[i]] == 'negative':
            shutil.copyfile(os.path.join(dataset_path, test_list[i]), os.path.join(neg_val_path, test_list[i]))
        else:
            shutil.copyfile(os.path.join(dataset_path, test_list[i]), os.path.join(pos_val_path, test_list[i]))
    print('successfully create the test dataset ...')
    #print useful information
    neg_train_num = len(os.listdir(neg_train_path))
    print("negative train dir has %s images." % neg_train_num)
    pos_train_num = len(os.listdir(pos_train_path))
    print("positive train dir has %s images." % pos_train_num)
    neg_val_num = len(os.listdir(neg_val_path))
    print("positive dir has %s images." % neg_val_num)
    pos_val_num = len(os.listdir(pos_val_path))
    print("positive dir has %s images." % pos_val_num)
#split_train_val(r'K:\ODIdataset\data\dataset', r'./filename2label.xlsx')


def remove_files(need_data_path, all_data_path):
    need_data = pd.read_excel(need_data_path, engine='openpyxl')
    need_filename_list = need_data['filename'].values.tolist()
    print(len(need_filename_list))
    all_filname_list = os.listdir(all_data_path)
    print(len(all_filname_list))
    for i in range(len(all_filname_list)):
        if all_filname_list[i] not in need_filename_list:
            os.remove(os.path.join(all_data_path, all_filname_list[i]))
# example
# remove_files(r'K:\ODIdataset\data\clear\vote_information.xlsx', r'K:\ODIdataset\data\dataset2')

def copy_files(source_path, source_info_path, destine_path):
    '''
    copy necessary files
    :param source_path: the dir saved need data
    :param source_info_path: the xlsx saved need information
    :param destine_path: the dir put need data
    :return: None
    '''
    if not os.path.exists(destine_path):
        os.mkdir(destine_path)

    all_data = pd.read_excel(source_info_path, engine='openpyxl')
    filename_list = all_data['filename'].values.tolist()

    for i in range(len(filename_list)):
        src_file = os.path.join(source_path, filename_list[i])
        shutil.copy(src_file, destine_path)




need_data_path = r'C:\Users\lab-626\Desktop\补充材料\paper\pdanet'       #the excel need to analysis which saved all annotators label result
all_data_path = r'K:\resnet_test_val'
destine_path = r'C:\Users\lab-626\Desktop\补充材料\paper\resnet'
#need_data_path = r'D:\Pycode\VRED\Download\1067_filename_label.xlsx'    #the excel to save voted relult{filename：label}
#save_data_path = r'D:\Pycode\VRED\Download\1067_label_timestamps_info.xlsx'  #
#need_data_path = r'K:\ODIdataset\data\dataset2'
# data = pd.read_excel(need_data_path, engine='openpyxl')
# all_data_num = len(data)                          #total number
# filename_list = data['filename'].values.tolist()  #filename
# label_list = data['label'].values.tolist()        #label
# person_list = data['person'].values.tolist()      #corresponding person
# all_data_dict = dict(zip(filename_list,label_list))  #{filename:label}

need_data_list = os.listdir(need_data_path)
all_data_list = os.listdir(all_data_path)
for i in range(len(all_data_list)):
    if all_data_list[i] in need_data_list:
        shutil.copy(all_data_path, destine_path)

#del_rows(need_data, all_data_path, save_data_path)