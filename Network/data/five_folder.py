# coding=utf-8
#author:huang.rong
#date:2021/05/12
#get five fold cross validation, ensure train and test no overlap


import pandas as pd
import random



def spilit_pos_neg(filelist):
    '''
    get the ratio of positive and negative samples
    :param filelist: 360 deegree iamges list
    :return:None
    '''
    count = 0
    for i in range(len(filelist)):
        if all_data_dict[filelist[i]] == 0:
            count += 1
    print('negative image number is %d' % count)
    print('positive image number is %d' % (len(filelist)-count))



def get_every_part_list(filelist, folder_num=5):
    '''
    get the prefix for all videos
    :param filelist: all 360 degree images list
    :return: list of every part of the five folder
    '''
    #get the prefix for all videos
    prefix_list = []
    for i in range(len(filelist)):
        part = filelist[i].split('_')
        if part[0] == '0':
            prefix = part[0] + '_' + part[1]
        else:
            prefix = part[0] + '_' + part[1] + '_' + part[2]
        prefix_list.append(prefix)
    unique_prefix_list = list(set(prefix_list))

    #get the same prefix iamges list
    all_same_list = []
    for j in range(len(unique_prefix_list)):
        same_list = []
        for i in range(len(filelist)):
            part = filelist[i].split('_')
            if part[0] == '0':
                prefix = part[0] + '_' + part[1]
            else:
                prefix = part[0] + '_' + part[1] + '_' + part[2]
            if prefix == unique_prefix_list[j]:
                same_list.append(filelist[i])
        all_same_list.append(same_list)
        all_same_list.sort()
    print('all_same_list num is: %d' % len(all_same_list))
    random.shuffle(all_same_list)

    # get the folder_num part list
    part_num = len(all_same_list) // folder_num
    all_folder_list = [all_same_list[i:i + part_num] for i in range(0, len(all_same_list), part_num)]

    if len(all_folder_list) != folder_num:
        all_folder_list[folder_num - 1].extend(all_folder_list[folder_num])
        all_folder_list.pop(folder_num)
    print(len(all_folder_list))

    return all_folder_list


def get_folder_list(all_folder_list, derepitition_flag=False):
    all_train_list = []
    all_test_list = []
    flag = 0
    for k in range(len(all_folder_list)):
        one_train_list = []
        one_test_list = []
        for i in range(len(all_folder_list)):
            if i != flag:
                for j in range(len(all_folder_list[i])):
                    one_train_list.extend(all_folder_list[i][j])
            else:
                #whether testset need to derepitition
                if derepitition_flag:
                    part_test_list = all_folder_list[flag]
                    for n in range(len(part_test_list)):
                        one_test_list.append(part_test_list[n][0])
                else:
                    for n in range(len(all_folder_list[flag])):
                        one_test_list.extend(all_folder_list[flag][n])
        print('%d train list num is %d... ' %(k, len(one_train_list)))
        print('%d test list num is %d... ' % (k, len(one_test_list)))
        print('test list positive and negative ratio is:')
        spilit_pos_neg(one_test_list)
        print('test list positive and negative ratio is:')
        spilit_pos_neg(one_train_list)
        print('*' * 10)
        flag += 1
        all_train_list.append(one_train_list)
        all_test_list.append(one_test_list)

    return all_train_list, all_test_list

def write_folder_xlsx(folder_train_list, folder_test_list, five_folder):
    for i in range(folder_num):
        train_dict = {key:value for key, value in all_data_dict.items() if key in folder_train_list[i]}  # 1041
        train_dict = {'filename':train_dict.keys(),
                            'label':train_dict.values()}

        test_dict = {key:value for key, value in all_data_dict.items() if key in folder_test_list[i]}  # 1041
        test_dict = {'filename':test_dict.keys(),
                            'label':test_dict.values()}
        train_save_path = five_folder + r'/train_'  + str(i) + '.xlsx'
        test_save_path = five_folder + r'/test_'  + str(i) + '.xlsx'
        pd.DataFrame(train_dict).to_excel(train_save_path, index=None)
        pd.DataFrame(test_dict).to_excel(test_save_path, index=None)




random.seed(42)
derepitition_flag = False                       #derepitition_flag decide whether derepitition
folder_num = 5
five_folder = r'./c100'
all_data_path = r'./c100/c100.xlsx'

all_data = pd.read_excel(all_data_path, engine='openpyxl')
all_list = all_data['filename'].values
label_list = all_data['label'].values
all_data_dict = dict(zip(all_list, label_list))  #all information about {filename:label}

all_five_list = get_every_part_list(all_list)
five_train_list, five_test_list = get_folder_list(all_five_list, derepitition_flag)
write_folder_xlsx(five_train_list, five_test_list, five_folder)
